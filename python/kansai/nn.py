import math
import random as _random

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


class AvgPool2d(Module):
    """Non-overlapping average pooling only (`stride` fixed equal to
    `kernel_size`, unlike Conv2d's independent stride) -- the common
    case, and the one a pure reshape-based composition can express
    exactly. `(N, C, H, W)` -> reshape to `(N, C, H/k, k, W/k, k)` (a
    real row-major reshape, not a relayout: decomposing `H` into
    `(H/k, k)` this way lands `kh` exactly inside one pooling window,
    verified against a hand-computed 4x4 example before relying on it)
    -> `mean(dim)` over the two `k`-sized axes, one at a time (mean
    over a 2D window separates into two sequential 1D means exactly --
    not an approximation). Composed entirely from already-existing,
    already-traceable ops (`reshape`, `mean(dim)`), so -- unlike
    `MaxPool2d` below -- this needed no new kernel, `GradNode`, or KIR
    work at all: it traces, fuses, and differentiates through `kir.grad`
    for free, the same payoff `LayerNorm` already got from being a pure
    composition. Requires `H` and `W` to divide evenly by
    `kernel_size` -- true for the deliberately-scoped case this covers,
    not handled otherwise (padding to make it true is the caller's own
    job, e.g. via a `Conv2d` upstream sized to land on an exact
    multiple).
    """

    def __init__(self, kernel_size: int):
        self.kernel_size = kernel_size

    def forward(self, x):
        if len(x.shape) != 4:
            raise ValueError(f"AvgPool2d: expected a 4D (N, C, H, W) input, got shape {list(x.shape)}")
        n, c, h, w = x.shape
        k = self.kernel_size
        if h % k != 0 or w % k != 0:
            raise ValueError(f"AvgPool2d: kernel_size={k} must divide both H={h} and W={w} evenly")
        ho, wo = h // k, w // k
        return x.reshape([n, c, ho, k, wo, k]).mean(5).mean(3)


class MaxPool2d(Module):
    """Unlike AvgPool2d above, this is a real, dedicated C++ op
    (Tensor.max_pool2d), not a composition -- see its own declaration in
    core/include/kansai/Tensor.hpp for why reusing the existing (and
    deliberately non-differentiable) max(dim) here would have been a
    correctness trap, not a shortcut. Independent kernel_size/stride
    (unlike AvgPool2d, which only covers the exact-tiling
    stride==kernel_size case) -- stride defaults to kernel_size,
    matching every real MaxPool2d's own default."""

    def __init__(self, kernel_size: int, stride: int = None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x):
        return x.max_pool2d(self.kernel_size, self.stride)


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


class Dropout(Module):
    """Inverted dropout: zeros each element independently with
    probability `p` during training, scaling the survivors by
    `1/(1-p)` so the expected value stays the same either way (the
    modern convention -- eval mode is then a plain identity, not a
    separate `* (1-p)` rescale). Identity in eval mode (see
    Module.train()/eval()) or when `p == 0`, matching the mathematical
    definition exactly, not approximately.

    The mask is a real Tensor with requires_grad=False, so dropout's own
    backward needs no dedicated kernel or GradNode at all: `x.mul(mask)`
    is already correctly differentiable (mul's existing backward routes
    the gradient through the same mask that zeroed the forward pass,
    exactly dropout's own true gradient), the same "composed from an
    already-differentiable op" payoff softmax/LayerNorm/AvgPool2d above
    already got.

    Mask generation goes through Python's stdlib `random` (not a kernel)
    -- fine at the sizes this project's own tests and MNIST benchmark
    run at, a real, un-optimized cost (one Python-level random draw per
    element, every forward call in training mode) at a size large enough
    for that to matter; a C++ RNG-based kernel is real, unattempted
    future work.

    KIR-traceable, with a real caveat worth stating plainly: tracing
    calls this during the ONE trace, generating ONE fixed mask that gets
    baked into the graph as a constant -- correct for a single traced
    run, but a graph cached and re-run multiple times (`kir.run_fused`/
    `kir.run_metal` on the same fused graph object, say) would reuse
    that SAME mask every call, not draw a fresh one -- silently defeating
    dropout's own point across repeated calls to one cached graph. Not a
    concern for ordinary eager use (a fresh Python-level forward() call
    each time, each with a fresh mask), which is how this project's own
    training loops actually call it.
    """

    def __init__(self, p: float = 0.5):
        assert 0.0 <= p < 1.0, f"Dropout: p must be in [0, 1), got {p}"
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        n = 1
        for d in x.shape:
            n *= d
        keep_prob = 1.0 - self.p
        mask_vals = [1.0 / keep_prob if _random.random() > self.p else 0.0 for _ in range(n)]
        mask = core.from_flat(mask_vals, list(x.shape))
        return x.mul(mask)


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


class MultiHeadAttention(Module):
    """Standard scaled dot-product multi-head attention ("Attention Is
    All You Need"): `softmax(Q W_q (K W_k)^T / sqrt(d_k)) (V W_v) W_o`,
    split across `num_heads` independent heads. General cross-attention
    (`query`, `key`, `value` may be different tensors, and `key`/`value`
    may have a different sequence length than `query`) -- ordinary
    self-attention is just `mha(x, x, x)`.

    Only reachable now because two things this project just built landed
    together: batched matmul (this section's own previous entry -- Q/K/V
    here are 4D, `(batch, num_heads, seq, d_k)`, and every matmul below
    is genuinely batched over `(batch, num_heads)`, not a Python-level
    loop over 2D calls) and `softmax(dim)` (a few sections back). Every
    other piece -- splitting/merging heads via `reshape` + `transpose`,
    the `1/sqrt(d_k)` scale via the same shape-`[1]`-broadcast-constant
    idiom `Adam`/`BatchNorm1d` already use, the optional additive mask
    -- was already sitting there waiting to be composed; this class adds
    no new kernel, `GradNode`, or KIR work of its own at all, the same
    "pure composition" payoff `LayerNorm`/`softmax`/`AvgPool2d` above
    already got, now compounding across all of them at once.

    `mask`, if given, is ADDITIVE (broadcast-added to the raw scores
    before `softmax`, typically `0` where allowed and a large negative
    number -- not literal `-inf`, which would produce `NaN` the moment
    every score in a row is masked and `softmax`'s `exp(0)` values all
    still sum to a real, if tiny, number -- where forbidden, e.g. a
    causal mask for a decoder). The additive-mask convention (rather
    than a boolean mask selecting positions) is deliberate, not just
    convenient: Kansai has no `where`/comparison-op/boolean-masking
    primitive yet, so an additive mask -- expressible with `add`, which
    already exists and is already broadcasting-aware -- is what makes
    masking possible AT ALL right now, not a stylistic preference over
    an equally-easy alternative.

    Eager-only in practice, not by an enforced restriction: nothing
    here calls anything that would refuse to trace, but `kir.grad`'s
    matmul vjp rule is 2D-only (previous section's own stated gap), so
    a graph built from this and differentiated via `kir.grad` would hit
    that same limitation. Eager `.backward()` is unaffected and is what
    every test/training loop below actually verifies.
    """

    def __init__(self, d_model: int, num_heads: int, seed: int = 0):
        if d_model % num_heads != 0:
            raise ValueError(f"MultiHeadAttention: d_model={d_model} must be divisible by num_heads={num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        std = 1.0 / math.sqrt(d_model)
        self.w_q = randn([d_model, d_model], std=std, requires_grad=True, seed=seed)
        self.w_k = randn([d_model, d_model], std=std, requires_grad=True, seed=seed + 1)
        self.w_v = randn([d_model, d_model], std=std, requires_grad=True, seed=seed + 2)
        self.w_o = randn([d_model, d_model], std=std, requires_grad=True, seed=seed + 3)

    def _split_heads(self, x, batch, seq):
        # (batch, seq, d_model) -> (batch, num_heads, seq, d_k). Reshape
        # first (splitting d_model into (num_heads, d_k), a real
        # row-major reshape, not a relayout) then transpose seq and
        # num_heads into place -- a genuine data permutation, so the
        # RESULT is freshly contiguous, unlike PyTorch's own transpose
        # (Kansai has no non-contiguous-view concept at all, so there's
        # no separate .contiguous() step needed before the reshape two
        # calls later merges the heads back).
        return x.reshape([batch, seq, self.num_heads, self.d_k]).transpose(1, 2)

    def _merge_heads(self, x, batch, seq):
        # The exact inverse of _split_heads.
        return x.transpose(1, 2).reshape([batch, seq, self.d_model])

    def forward(self, query, key, value, mask=None):
        batch, seq_q, _ = query.shape
        _, seq_k, _ = key.shape

        q = self._split_heads(query.matmul(self.w_q), batch, seq_q)
        k = self._split_heads(key.matmul(self.w_k), batch, seq_k)
        v = self._split_heads(value.matmul(self.w_v), batch, seq_k)

        scale = core.from_flat([1.0 / math.sqrt(self.d_k)], [1])
        # q @ k^T: (batch, heads, seq_q, d_k) @ (batch, heads, d_k, seq_k)
        # -> (batch, heads, seq_q, seq_k) -- a genuinely batched matmul
        # over (batch, heads) both times, not a per-head Python loop.
        scores = q.matmul(k.transpose(2, 3)).mul(scale)
        if mask is not None:
            scores = scores.add(mask)
        weights = scores.softmax(3)  # over seq_k, the last axis
        attended = weights.matmul(v)  # (batch, heads, seq_q, d_k)

        merged = self._merge_heads(attended, batch, seq_q)
        return merged.matmul(self.w_o)


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
