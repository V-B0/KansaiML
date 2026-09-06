"""Batched matmul: matmul previously required both operands to be
exactly 2D. Tensor::matmul now handles any rank >= 2, treating every
dim except the trailing two as a NumPy-style broadcastable "batch"
shape (the same right-aligned rule add/sub/mul's own broadcasting
already uses) -- (batch, heads, seq, d_k) @ (batch, heads, d_k, seq),
the shape multi-head attention actually needs, works directly, as does
a single shared 2D weight matrix applied across an entire batch (a
rank-0 batch shape broadcasting against any batch shape).

The exact 2D+2D case keeps its own original, unchanged fast path --
checked here as a regression, not just assumed preserved.

Checked: forward values against an independent from-scratch nested-loop
matmul (not Kansai's own matmul called differently -- that could share
a bug with the implementation under test) for the matching-batch,
broadcast-shared-weight, broadcast-batch-dim-1, and 4D attention-shaped
cases; backward against central differences for both the matching-batch
and the broadcast case (where the smaller operand's gradient has to be
reduced back down, the same reduce_to_shape machinery general
broadcasting's own backward already uses); the full KIR forward path
(a real bug caught and fixed during this work: run_metal's own matmul
dispatch called the 2D-only Metal kernel unconditionally, which would
have crashed on anything batched -- fixed to fall back to the CPU eager
path, the same way reshape/transpose/etc. already do); and the
documented, pre-existing gap this doesn't close: kir.grad's own matmul
vjp rule stays 2D-only (the same gap conv2d already has), confirmed to
fail with a clear error rather than silently computing something wrong.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import kir
from kansai import _core as core

TOL = 1e-4
GRAD_TOL = 2e-2
EPS = 1e-3

rng = random.Random(0)


def check_close(label, a, b, tol=TOL):
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < tol, f"{label}[{i}]: got={x:.6f} expected={y:.6f}"
    print(f"{label}: OK ({len(a)} elements, max diff {max(abs(x - y) for x, y in zip(a, b)):.2e})")


def naive_matmul(a, a_shape, b, b_shape):
    """Independent reference: plain nested Python loops. Batch dims
    must already match exactly between a_shape and b_shape (the actual
    broadcasting, where needed, is done by the CALLER before invoking
    this, by pre-expanding one side -- see how it's used below)."""
    *batch, M, K = a_shape
    *batch2, K2, N = b_shape
    assert batch == batch2 and K == K2, f"{a_shape} vs {b_shape}"
    nbatch = 1
    for d in batch:
        nbatch *= d
    out = [0.0] * (nbatch * M * N)
    for bi in range(nbatch):
        for i in range(M):
            for j in range(N):
                s = 0.0
                for k in range(K):
                    s += a[bi * M * K + i * K + k] * b[bi * K * N + k * N + j]
                out[bi * M * N + i * N + j] = s
    return out


def central_diff_grad(fn, flat_vals, shape):
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


# ---------------------------------------------------------------------
# 1. 2D+2D regression: the original fast path, unchanged.
# ---------------------------------------------------------------------

a2 = kansai.randn([4, 5], std=1.0, seed=1)
b2 = kansai.randn([5, 3], std=1.0, seed=2)
y2 = a2.matmul(b2)
assert list(y2.shape) == [4, 3]
check_close("2D matmul forward (regression)", y2.tolist(), naive_matmul(a2.tolist(), [4, 5], b2.tolist(), [5, 3]))

# ---------------------------------------------------------------------
# 2. 3D+3D matching batch (independent matmuls per batch item) and the
#    real 4D attention shape (batch, heads, seq, d_k) @ (batch, heads,
#    d_k, seq) -> (batch, heads, seq, seq).
# ---------------------------------------------------------------------

a3 = kansai.randn([2, 4, 5], std=1.0, seed=3)
b3 = kansai.randn([2, 5, 3], std=1.0, seed=4)
y3 = a3.matmul(b3)
assert list(y3.shape) == [2, 4, 3]
check_close("3D batched matmul forward (matching batch)", y3.tolist(),
            naive_matmul(a3.tolist(), [2, 4, 5], b3.tolist(), [2, 5, 3]))

batch, heads, seq, d_k = 2, 3, 4, 5
q = kansai.randn([batch, heads, seq, d_k], std=1.0, seed=5)
k = kansai.randn([batch, heads, d_k, seq], std=1.0, seed=6)
scores = q.matmul(k)
assert list(scores.shape) == [batch, heads, seq, seq]
check_close("4D batched matmul (attention QK^T shape)", scores.tolist(),
            naive_matmul(q.tolist(), [batch * heads, seq, d_k], k.tolist(), [batch * heads, d_k, seq]))

# ---------------------------------------------------------------------
# 3. Broadcasting: a shared 2D weight applied across a 3D batch, and a
#    batch dim of size 1 broadcasting up to match the other operand's.
# ---------------------------------------------------------------------

w = kansai.randn([5, 3], std=1.0, seed=7)
x_batch = kansai.randn([2, 4, 5], std=1.0, seed=8)
y_shared = x_batch.matmul(w)
assert list(y_shared.shape) == [2, 4, 3]
ref_shared = []
xb_flat = x_batch.tolist()
for bi in range(2):
    ref_shared.extend(naive_matmul(xb_flat[bi * 20:(bi + 1) * 20], [4, 5], w.tolist(), [5, 3]))
check_close("broadcast: shared 2D weight across a 3D batch", y_shared.tolist(), ref_shared)

a_one = kansai.randn([1, 4, 5], std=1.0, seed=9)
b_two = kansai.randn([2, 5, 3], std=1.0, seed=10)
y_bcast = a_one.matmul(b_two)
assert list(y_bcast.shape) == [2, 4, 3]
b_two_flat = b_two.tolist()
ref_bcast = []
for bi in range(2):
    ref_bcast.extend(naive_matmul(a_one.tolist(), [4, 5], b_two_flat[bi * 15:(bi + 1) * 15], [5, 3]))
check_close("broadcast: batch dim of size 1", y_bcast.tolist(), ref_bcast)

# ---------------------------------------------------------------------
# 4. Backward vs central differences: matching-batch and the broadcast
#    (shared-weight) case, where the smaller operand's gradient needs
#    reducing back down from the full broadcast batch shape.
# ---------------------------------------------------------------------

avals = [rng.uniform(-1, 1) for _ in range(2 * 3 * 4)]
bvals = [rng.uniform(-1, 1) for _ in range(2 * 4 * 3)]
ag = kansai.from_flat(avals, [2, 3, 4], requires_grad=True)
bg = kansai.from_flat(bvals, [2, 4, 3], requires_grad=True)
ag.matmul(bg).sum().backward()
check_close("3D batched matmul backward (a) vs central diff", ag.grad.tolist(),
            central_diff_grad(lambda t: t.matmul(kansai.from_flat(bvals, [2, 4, 3])).sum(), avals, [2, 3, 4]),
            GRAD_TOL)
check_close("3D batched matmul backward (b) vs central diff", bg.grad.tolist(),
            central_diff_grad(lambda t: kansai.from_flat(avals, [2, 3, 4]).matmul(t).sum(), bvals, [2, 4, 3]),
            GRAD_TOL)

wvals = [rng.uniform(-1, 1) for _ in range(5 * 3)]
xvals = [rng.uniform(-1, 1) for _ in range(2 * 4 * 5)]
wg = kansai.from_flat(wvals, [5, 3], requires_grad=True)
xg = kansai.from_flat(xvals, [2, 4, 5], requires_grad=True)
xg.matmul(wg).sum().backward()
check_close("broadcast matmul backward (shared weight) vs central diff", wg.grad.tolist(),
            central_diff_grad(lambda t: kansai.from_flat(xvals, [2, 4, 5]).matmul(t).sum(), wvals, [5, 3]), GRAD_TOL)
check_close("broadcast matmul backward (batched input) vs central diff", xg.grad.tolist(),
            central_diff_grad(lambda t: t.matmul(kansai.from_flat(wvals, [5, 3])).sum(), xvals, [2, 4, 5]), GRAD_TOL)

# ---------------------------------------------------------------------
# 5. Full KIR forward path (trace -> run/run_fused/run_metal), and the
#    documented, pre-existing kir.grad gap (2D-only, same as conv2d)
#    confirmed to fail cleanly rather than silently.
# ---------------------------------------------------------------------

graph = kir.trace(lambda a, b: a.matmul(b).sum(), q, k)
eager_out = q.matmul(k).sum().tolist()
check_close("batched matmul kir.run()", kir.run(graph, q, k).tolist(), eager_out)
check_close("batched matmul kir.run_fused()", kir.run_fused(kir.elementwise_fusion(graph), q, k).tolist(), eager_out)
if core.metal_available():
    check_close("batched matmul kir.run_metal()",
                kir.run_metal(kir.elementwise_fusion(graph), q, k).tolist(), eager_out)

try:
    bwd = kir.grad(graph, graph.inputs)
    kir.run(bwd, q, k)
    raise AssertionError("expected kir.grad to fail on a batched matmul node (documented 2D-only gap)")
except RuntimeError as e:
    print(f"kir.grad correctly fails on batched matmul (documented 2D-only gap, same as conv2d): {e}")

# ---------------------------------------------------------------------
# 6. Incompatible shapes are rejected, not silently misinterpreted.
# ---------------------------------------------------------------------

try:
    kansai.randn([2, 3, 4], std=1.0, seed=1).matmul(kansai.randn([2, 5, 3], std=1.0, seed=2))
    raise AssertionError("expected incompatible inner dims to raise")
except RuntimeError as e:
    print(f"incompatible inner dimensions correctly rejected: {e}")

print("\nBatched matmul test passed.")
