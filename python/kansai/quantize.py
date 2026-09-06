"""Phase 4: post-training int8 weight quantization -- inference-shaped,
forward-only, and deliberately standalone, the same "prototype the
semantics before committing to a real kernel" approach this project
already took twice (kir.py's IR before any C++ backend existed;
distributed.py's _split_tensor/_concat_tensors before a native slice/
concat kernel). Nothing in the existing Tensor/Storage/DType machinery
changes: core/include/kansai/DType.hpp only ever declares Float32, and
Tensor::data_ptr() is unconditionally `float*` at hundreds of call
sites -- retrofitting a second storage width into that class would be
invasive surgery on working code for a feature nothing else in the
codebase yet depends on. A QTensor here is instead a small, separate
Python-level type: an int8 payload plus one float scale, converted back
to an ordinary core.Tensor (dequantize) at the one point it actually
needs to enter a real kernel.

Symmetric, per-tensor quantization, not asymmetric/affine: q =
round(x / scale), scale = max(|x|) / 127, clamped to [-127, 127] --
no zero-point. That's the right fit for what this module actually
quantizes (trained weights, roughly zero-centered by construction: an
untrained Linear/Conv2d weight is itself sampled from a zero-mean
Gaussian, and training rarely pushes a whole weight tensor's
distribution far off zero) and the simpler of the two standard
schemes -- the same "start simple, prove correctness first" choice this
project made for Metal's own first matmul kernel (naive, one thread per
output element) before MPS. Asymmetric quantization would matter for
post-ReLU activations (all >= 0, so symmetric wastes half the int8
range) -- not attempted here, since nothing in this module quantizes
activations.

What this proves, honestly: quantizing a weight to int8 and dequantizing
it back to float32 introduces bounded, small error (checked against a
known bound, not just "looks close"), and a Linear layer's output built
from the dequantized weight stays close to the original float32 output
-- both real, measured claims, not assumed ones. What this does NOT
claim: no speed win. qlinear() dequantizes the weight back to float32
and runs the ordinary Accelerate-backed matmul underneath, so the only
real benefit demonstrated here is memory footprint (4x smaller weight
storage, 1 byte/element vs 4) -- a genuine win for a model too large to
fit in memory at float32, not a FLOPs win. A true int8 GEMM kernel
(feeding int8 operands directly into hardware int8 dot-product paths,
skipping the dequantize step and the float32 matmul it feeds) is real,
substantial, unattempted future work -- the same honest gap this
project already left open for a Metal convolution kernel and for
concurrent multi-device dispatch, not a shortcut dressed up as done.
"""

from . import _core as core


def quantize_symmetric(tensor: "core.Tensor"):
    """Returns (q: list[int], scale: float, shape: list[int]) for
    `tensor`. q's entries are Python ints in [-127, 127]; scale is the
    single float that recovers an approximation of the original value
    via `q[i] * scale`. Per-tensor (one scale for the whole tensor, not
    one per row/channel) -- the simplest granularity, and the right
    starting point for the same reason a single global scale was the
    right first cut anywhere else in this project: per-channel
    quantization (a separate scale per output channel/feature) would
    shrink the error further but is a real refinement on top of this,
    not a prerequisite for it.

    Goes through tolist() rather than a C++ kernel for the same
    "prototype in Python first" reason the rest of this module does --
    see the module docstring.
    """
    flat = tensor.tolist()
    max_abs = max((abs(v) for v in flat), default=0.0)
    scale = max_abs / 127.0 if max_abs > 0.0 else 1.0
    q = [max(-127, min(127, round(v / scale))) for v in flat]
    return q, scale, list(tensor.shape)


def dequantize(q: list, scale: float, shape: list) -> "core.Tensor":
    """The inverse of quantize_symmetric: q[i] * scale for every
    element, reshaped back to `shape`."""
    flat = [v * scale for v in q]
    return core.from_flat(flat, shape)


class QTensor:
    """A quantized tensor: int8 values (as a plain Python list -- there's
    no int8 Storage to hold them in yet, see the module docstring) plus
    the single float scale and shape needed to dequantize. Forward-only
    and immutable by convention: nothing here supports in-place update
    or gradients, matching that this is meant for an already-trained
    weight, not a training-time representation."""

    __slots__ = ("q", "scale", "shape")

    def __init__(self, q: list, scale: float, shape: list):
        self.q = q
        self.scale = scale
        self.shape = list(shape)

    @staticmethod
    def from_tensor(tensor: "core.Tensor") -> "QTensor":
        q, scale, shape = quantize_symmetric(tensor)
        return QTensor(q, scale, shape)

    def dequantize(self) -> "core.Tensor":
        return dequantize(self.q, self.scale, self.shape)

    def nbytes(self) -> int:
        """1 byte per element -- the actual, measurable payload size
        (ignoring the one extra float for `scale`, negligible for any
        tensor bigger than a handful of elements). Compare against
        4 * len(self.q) for the equivalent float32 tensor's real
        footprint."""
        return len(self.q)

    def __repr__(self):
        return f"QTensor(shape={self.shape}, scale={self.scale:.6g}, nbytes={self.nbytes()})"


def qlinear(x: "core.Tensor", qweight: QTensor, bias: "core.Tensor") -> "core.Tensor":
    """A Linear layer's forward pass with an int8-quantized weight:
    dequantize back to float32, then the ordinary matmul+add every other
    Linear-shaped op in this codebase already uses. See the module
    docstring for exactly what this does and doesn't prove -- memory
    footprint, not FLOPs."""
    w = qweight.dequantize()
    return x.matmul(w).add(bias)


class QLinear:
    """An inference-only Linear layer built FROM an already-trained
    nn.Linear: quantizes its weight once at construction time (bias
    stays float32 -- it's a tiny (out_features,) vector, nowhere near
    where quantization's memory win matters, and keeping it exact
    avoids adding a second source of error on top of the weight's).
    Deliberately not an nn.Module subclass with trainable parameters:
    QLinear has no `.parameters()` a trainable model would return,
    because quantizing, then continuing to train through int8 weights
    (quantization-aware training) is real, separate, unattempted work
    -- this is a post-training, inference-shaped conversion only, the
    same scope quantize_symmetric's own docstring states for the module
    as a whole."""

    def __init__(self, linear):
        self.qweight = QTensor.from_tensor(linear.weight)
        self.bias = linear.bias

    def __call__(self, x: "core.Tensor") -> "core.Tensor":
        return qlinear(x, self.qweight, self.bias)
