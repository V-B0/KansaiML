"""Numerical gradient checking: the gold-standard way to verify autograd
correctness, and a stronger check than "training still converges" --
none of the other tests actually verify a gradient against first
principles, they only verify that training behaves as if it were
approximately correct. Written now because matmul's backward pass just
changed (transpose2d + matmul -> matmul_nt/matmul_tn, avoiding a
materialized transpose): this is what actually proves the new kernels
compute the same gradient, not just "doesn't crash and XOR still
converges"."""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai

EPS = 1e-3
TOL = 2e-2  # central differences are approximate -- generous but real


def central_diff_grad(fn, flat_vals, shape):
    """fn: takes a kansai.Tensor (shape `shape`), returns a scalar
    kansai.Tensor. Returns one numerical partial derivative per element,
    via (f(x+eps) - f(x-eps)) / (2*eps)."""
    grads = []
    for i in range(len(flat_vals)):
        plus = list(flat_vals)
        plus[i] += EPS
        minus = list(flat_vals)
        minus[i] -= EPS
        f_plus = fn(kansai.from_flat(plus, shape)).tolist()[0]
        f_minus = fn(kansai.from_flat(minus, shape)).tolist()[0]
        grads.append((f_plus - f_minus) / (2 * EPS))
    return grads


def check(label, analytical, numerical):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        assert abs(a - n) < TOL, f"{label}[{i}]: analytical={a:.5f} numerical={n:.5f}"
    print(f"{label}: OK ({len(analytical)} elements, max diff "
          f"{max(abs(a - n) for a, n in zip(analytical, numerical)):.2e})")


# ---------------------------------------------------------------------
# matmul: the op whose backward just changed (matmul_nt / matmul_tn,
# reading operands transposed via BLAS's flag instead of a materialized
# transpose buffer). Check grad wrt both operands.
# ---------------------------------------------------------------------

rng = random.Random(0)  # a local instance -- see note below on why not the shared module state
M, K, N = 3, 4, 5
a_vals = [rng.uniform(-1, 1) for _ in range(M * K)]
b_vals = [rng.uniform(-1, 1) for _ in range(K * N)]

a = kansai.from_flat(a_vals, [M, K], requires_grad=True)
b = kansai.from_flat(b_vals, [K, N], requires_grad=True)

out = a.matmul(b).sum()
out.backward()

check("matmul grad_a", a.grad.tolist(),
      central_diff_grad(lambda t: t.matmul(kansai.from_flat(b_vals, [K, N])).sum(), a_vals, [M, K]))
check("matmul grad_b", b.grad.tolist(),
      central_diff_grad(lambda t: kansai.from_flat(a_vals, [M, K]).matmul(t).sum(), b_vals, [K, N]))

# ---------------------------------------------------------------------
# Linear forward (matmul + bias-broadcast add) followed by relu, then
# summed -- the exact shape of every layer's forward pass, checked
# end-to-end through all three backward closures at once.
# ---------------------------------------------------------------------

# A fresh, independent RNG rather than continuing the module-level
# stream from the block above: this test originally shared one
# `random.seed(0)` for the whole file, and reordering an earlier block
# silently changed these values -- which is exactly how the first
# version of this test tripped over a coincidence (see below) without
# any code change at all. Each block seeding its own instance means a
# block's data never depends on what ran before it.
rng = random.Random(0)
batch, in_f, out_f = 4, 3, 5
x_vals = [rng.uniform(-1, 1) for _ in range(batch * in_f)]
w_vals = [rng.uniform(-1, 1) for _ in range(in_f * out_f)]
bias_vals = [rng.uniform(-1, 1) for _ in range(out_f)]

x = kansai.from_flat(x_vals, [batch, in_f], requires_grad=True)
w = kansai.from_flat(w_vals, [in_f, out_f], requires_grad=True)
bias = kansai.from_flat(bias_vals, [out_f], requires_grad=True)

pre_activation = x.matmul(w).add(bias)
# ReLU is non-differentiable at exactly 0: perturbing an input by +/-EPS
# can flip a near-zero pre-activation's sign on only one side of the
# central difference, producing a large *correct* discrepancy that has
# nothing to do with whether the engine's gradient is right (this is
# exactly what happened chasing an earlier version of this test, where
# one unit's pre-activation landed at 7e-05). Guard the precondition
# explicitly rather than silently relying on a seed that happens to
# avoid it.
assert all(abs(v) > 0.05 for v in pre_activation.tolist()), (
    "a pre-activation landed too close to ReLU's boundary at 0 -- "
    "pick a different seed rather than debugging a spurious mismatch"
)

y = pre_activation.relu().sum()
y.backward()


def forward_x(t):
    return t.matmul(kansai.from_flat(w_vals, [in_f, out_f])).add(kansai.from_flat(bias_vals, [out_f])).relu().sum()


def forward_w(t):
    return kansai.from_flat(x_vals, [batch, in_f]).matmul(t).add(kansai.from_flat(bias_vals, [out_f])).relu().sum()


def forward_bias(t):
    return kansai.from_flat(x_vals, [batch, in_f]).matmul(kansai.from_flat(w_vals, [in_f, out_f])).add(t).relu().sum()


check("linear+relu grad_x", x.grad.tolist(), central_diff_grad(forward_x, x_vals, [batch, in_f]))
check("linear+relu grad_w", w.grad.tolist(), central_diff_grad(forward_w, w_vals, [in_f, out_f]))
check("linear+relu grad_bias", bias.grad.tolist(), central_diff_grad(forward_bias, bias_vals, [out_f]))

print("\ngradient check passed.")
