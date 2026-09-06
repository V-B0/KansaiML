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
    them, and no optimizer should ever step them). Updated via
    Tensor.detach() (a real Tensor view -- O(1), no data copy, sharing
    the same Storage -- with requires_grad=False and no grad_node) on
    this batch's own mean/variance, which otherwise DO sit inside the
    active autograd graph (they're differentiable, needed for x's own
    gradient): detaching them before folding into the running buffers
    is what stops that graph from growing across every training step,
    without needing to round-trip through plain Python floats to do it.
    The unbiased (n/(n-1)) correction applied to the batch variance
    before folding it into running_var, but NOT to the biased variance
    actually used to normalize this batch, matches the convention every
    real BatchNorm implementation uses.

    Eager-only, deliberately, for a reason detach() doesn't change:
    updating self.running_mean/self.running_var is a Python-level
    attribute REASSIGNMENT, a side effect a traced graph has no way to
    express regardless of whether the value feeding it is detached --
    detach() itself is eager-only too (no TraceValue.detach() exists),
    so kir.trace() on a graph containing a BatchNorm1d in training mode
    fails outright with a clear AttributeError rather than silently
    tracing something wrong. A real, stated limitation, not a bug to
    fix here. LayerNorm above has no such restriction: it's a pure
    composition of already-traceable ops with no side-effecting
    bookkeeping at all, so it traces, fuses, and differentiates through
    KIR exactly like any other op.
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
            momentum_t = core.from_flat([self.momentum], [1])
            one_minus_momentum_t = core.from_flat([1.0 - self.momentum], [1])
            correction_t = core.from_flat([correction], [1])

            mean_detached = mean.detach().reshape([self.num_features])
            var_detached = var.detach().reshape([self.num_features])
            self.running_mean = self.running_mean.mul(one_minus_momentum_t).add(mean_detached.mul(momentum_t))
            self.running_var = self.running_var.mul(one_minus_momentum_t).add(
                var_detached.mul(correction_t).mul(momentum_t))
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
