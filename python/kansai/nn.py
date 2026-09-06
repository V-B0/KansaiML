import math

from . import _core as core
from ._core import randn, zeros


class Module:
    # Class-level default: every instance starts in training mode
    # without needing Module.__init__ to run (no subclass here calls
    # super().__init__() -- they set their own attributes directly --
    # so this can't live as an instance attribute set in __init__).
    # train()/eval() below shadow it with a real instance attribute.
    training = True

    def parameters(self):
        return [t for _, t in self.named_parameters()]

    def named_parameters(self, prefix: str = ""):
        """Yields (name, tensor) for every trainable parameter,
        name-qualified by attribute path -- "layers.0.weight" for the
        first sub-layer of a Sequential, say. Same recursion
        parameters() already did, just keeping the path that gets there
        instead of discarding it. That path is exactly what
        kansai.serialize needs to match a saved tensor back to the
        right attribute on load -- a flat, order-dependent list (what
        parameters() alone gives you) has no name to check a checkpoint
        against, only a position, which breaks the moment two model
        definitions differ in ways that don't change parameter count
        (a reordered layer, an extra non-trainable bumper)."""
        for key, value in vars(self).items():
            if isinstance(value, core.Tensor) and value.requires_grad:
                yield (f"{prefix}{key}", value)
            elif isinstance(value, Module):
                yield from value.named_parameters(f"{prefix}{key}.")
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        yield from item.named_parameters(f"{prefix}{key}.{i}.")

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode: bool = True):
        """Sets this module (and every sub-Module reachable the same
        way named_parameters() reaches them) to training mode, for
        BatchNorm's own use -- it behaves differently in the two modes
        (normalizing by the CURRENT batch's own statistics vs by the
        running statistics accumulated over previous training batches)
        and needs a real, explicit signal for which one applies, not an
        implicit one inferred from whether .backward() happens to get
        called afterward."""
        self.training = mode
        for value in vars(self).values():
            if isinstance(value, Module):
                value.train(mode)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        item.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, seed: int = 0):
        std = 1.0 / math.sqrt(in_features)
        self.weight = randn([in_features, out_features], std=std, requires_grad=True, seed=seed)
        self.bias = zeros([out_features], requires_grad=True)

    def forward(self, x):
        return x.matmul(self.weight).add(self.bias)


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, seed: int = 0):
        self.stride = stride
        self.padding = padding
        fan_in = in_channels * kernel_size * kernel_size
        std = 1.0 / math.sqrt(fan_in)
        self.weight = randn([out_channels, in_channels, kernel_size, kernel_size],
                             std=std, requires_grad=True, seed=seed)
        self.bias = zeros([out_channels], requires_grad=True)

    def forward(self, x):
        return x.conv2d(self.weight, self.bias, self.stride, self.padding)


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class GELU(Module):
    """The exact formulation (via erf), not the tanh-based approximation
    some frameworks default to -- see Tensor::gelu's own declaration in
    core/include/kansai/Tensor.hpp for why."""

    def forward(self, x):
        return x.gelu()


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01):
        self.negative_slope = negative_slope

    def forward(self, x):
        return x.leaky_relu(self.negative_slope)


class Softmax(Module):
    def __init__(self, dim: int):
        self.dim = dim

    def forward(self, x):
        return x.softmax(self.dim)


class LayerNorm(Module):
    """Normalizes over the LAST axis only -- (..., num_features) for
    any leading shape -- the common case every transformer's own
    LayerNorm actually uses (normalizing just the embedding dimension,
    not several axes jointly). A pure composition of existing ops
    (mean(dim), sub, mul, div, sqrt, add), all already broadcasting-
    aware, so this needed no new kernel, GradNode, KIR op, or vjp rule
    of its own -- the same "nothing but existing primitives" shape
    softmax/cross_entropy/mean(dim) itself already took."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        self.num_features = num_features
        self.eps = eps
        self.weight = core.ones([num_features], requires_grad=True)
        self.bias = core.zeros([num_features], requires_grad=True)

    def forward(self, x):
        dim = len(x.shape) - 1
        eps_t = core.from_flat([self.eps], [1])
        mean = x.mean(dim, keepdim=True)
        centered = x.sub(mean)
        var = centered.mul(centered).mean(dim, keepdim=True)  # biased (divide by N), matching every real LayerNorm
        normalized = centered.div(var.add(eps_t).sqrt())
        return normalized.mul(self.weight).add(self.bias)


class BatchNorm1d(Module):
    """Normalizes each feature (column) across the batch dimension --
    (batch, num_features) input only; BatchNorm2d for conv activations
    (normalizing per-CHANNEL across N, H, and W jointly) is real,
    unattempted future work, needing a multi-axis reduction this
    project's mean(dim) doesn't do in one call (a transpose+reshape
    detour around it is possible, just not built here yet).

    Training mode (the default -- see Module.train()/eval()) normalizes
    by the CURRENT batch's own mean/variance and folds them into
    running_mean/running_var via an exponential moving average, for
    eval mode to use later; eval mode normalizes by those running
    statistics directly, never the eval batch's own (the whole point --
    a single example at inference time has no batch statistics of its
    own to normalize by).

    running_mean/running_var are buffers, not parameters -- plain
    Tensor attributes with requires_grad=False, deliberately invisible
    to named_parameters()/parameters() (no gradient should ever reach
    them, and no optimizer should ever step them). Updating them by
    round-tripping through tolist()/from_flat() rather than in-place
    Tensor ops is a deliberate choice, not an oversight: Kansai has no
    detach() to cut a value out of an active autograd graph, and the
    batch mean/variance computed on the training path above DO sit
    inside one (they're differentiable, needed for x's own gradient) --
    going through plain Python floats is what actually breaks that
    graph before folding into the buffers, avoiding growing it across
    every training step. The unbiased (n/(n-1)) correction applied to
    the batch variance before folding it into running_var, but NOT to
    the biased variance actually used to normalize this batch, matches
    the convention every real BatchNorm implementation uses.

    Eager-only, deliberately: the training path's tolist()/from_flat()
    running-stats bookkeeping needs real tensor DATA, which a kir.trace()
    TraceValue never carries (it's symbolic -- shape/dtype only). Calling
    kir.trace() on a graph containing a BatchNorm1d in training mode
    fails outright (an AttributeError on TraceValue.tolist()) rather
    than silently tracing something wrong -- a real, stated limitation,
    not a bug to fix here. LayerNorm above has no such restriction: it's
    a pure composition of already-traceable ops with no side-effecting
    bookkeeping, so it traces, fuses, and differentiates through KIR
    exactly like any other op.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = core.ones([num_features], requires_grad=True)
        self.bias = core.zeros([num_features], requires_grad=True)
        self.running_mean = core.zeros([num_features])
        self.running_var = core.ones([num_features])

    def forward(self, x):
        eps_t = core.from_flat([self.eps], [1])

        if self.training:
            mean = x.mean(0, keepdim=True)
            centered = x.sub(mean)
            var = centered.mul(centered).mean(0, keepdim=True)
            normalized = centered.div(var.add(eps_t).sqrt())

            n = x.shape[0]
            correction = n / (n - 1) if n > 1 else 1.0
            mean_flat = mean.tolist()
            var_flat = var.tolist()
            new_running_mean = [(1 - self.momentum) * rm + self.momentum * bm
                                 for rm, bm in zip(self.running_mean.tolist(), mean_flat)]
            new_running_var = [(1 - self.momentum) * rv + self.momentum * bv * correction
                                for rv, bv in zip(self.running_var.tolist(), var_flat)]
            self.running_mean = core.from_flat(new_running_mean, [self.num_features])
            self.running_var = core.from_flat(new_running_var, [self.num_features])
        else:
            centered = x.sub(self.running_mean)
            normalized = centered.div(self.running_var.add(eps_t).sqrt())

        return normalized.mul(self.weight).add(self.bias)


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
