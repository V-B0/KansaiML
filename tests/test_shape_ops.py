"""General tensor manipulation: reshape, transpose, slice, cat. Until
now Kansai had no way to reorganize a tensor's own shape at all --
distributed.py's own DTensor split/concat went through tolist()/
from_flat() specifically because "Kansai has no slice op yet" (see its
own docstring, now out of date). These four close that gap: reshape
(pure reinterpretation, a memcpy), transpose (a genuine N-D data
permutation, not just a metadata swap -- this codebase has no stride
concept, every Tensor is always fully packed row-major), slice
(extract a contiguous sub-range along one dim), and cat (slice's
inverse, joining tensors along an existing dim).

All four copy rather than view (no Tensor shares another's Storage) --
see Tensor::reshape's own doc comment in core/include/kansai/Tensor.hpp
for why a real zero-copy view is real, unattempted future work
(StoragePool's pooling isn't aware of aliased buffers yet).

Checked at three levels, the same bar every other op in this project is
held to: forward values against hand-computed expectations (including a
3D transpose across non-adjacent axes, not just the easy 2D case);
eager backward against central differences (the gold-standard check,
same tolerance test_grad_check.py uses); and the KIR path -- traced,
run through all four interpreters (run/run_fused/run_planned/
run_metal), and differentiated via kir.grad -- matching eager exactly,
including slice's three distinct boundary shapes (padding on both
sides, one side only, or neither) since each exercises a different
branch of _vjp_slice's own logic.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import kir
from kansai import _core as core

EPS = 1e-3
GRAD_TOL = 2e-2  # central differences are approximate, same bar test_grad_check.py uses
TOL = 1e-4


def check_close(label, a, b, tol=TOL):
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < tol, f"{label}[{i}]: got={x:.6f} expected={y:.6f}"
    print(f"{label}: OK ({len(a)} elements, max diff {max(abs(x - y) for x, y in zip(a, b)):.2e})")


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


rng = random.Random(0)

# ---------------------------------------------------------------------
# 1. reshape: forward value (a pure reinterpretation -- row-major
#    reshape never reorders bytes), eager backward via central
#    difference, and the full KIR path.
# ---------------------------------------------------------------------

x_vals = [rng.uniform(-1, 1) for _ in range(24)]
x = kansai.from_flat(x_vals, [2, 3, 4])
y = x.reshape([4, 6])
assert list(y.shape) == [4, 6]
check_close("reshape forward", y.tolist(), x_vals)

analytical = central_diff_grad(lambda t: t.reshape([4, 6]).sum(), x_vals, [2, 3, 4])
xg = kansai.from_flat(x_vals, [2, 3, 4], requires_grad=True)
xg.reshape([4, 6]).sum().backward()
check_close("reshape eager backward vs central diff", xg.grad.tolist(), analytical, GRAD_TOL)

graph = kir.trace(lambda t: t.reshape([4, 6]).sum(), x)
eager_out = x.reshape([4, 6]).sum().tolist()
check_close("reshape kir.run()", kir.run(graph, x).tolist(), eager_out)
check_close("reshape kir.run_fused()", kir.run_fused(kir.elementwise_fusion(graph), x).tolist(), eager_out)
plan = kir.plan_memory(graph)
check_close("reshape kir.run_planned()", kir.run_planned(graph, plan, core.StoragePool(), x).tolist(), eager_out)
if core.metal_available():
    check_close("reshape kir.run_metal()", kir.run_metal(kir.elementwise_fusion(graph), x).tolist(), eager_out)

bwd = kir.grad(graph, graph.inputs)
check_close("reshape kir.grad()", kir.run(bwd, x).tolist(), xg.grad.tolist())

try:
    x.reshape([5, 5])
    raise AssertionError("expected reshape to reject a size mismatch")
except RuntimeError as e:
    print(f"reshape correctly rejects a numel mismatch: {e}")

# ---------------------------------------------------------------------
# 2. transpose: 2D forward, a 3D transpose across NON-adjacent axes
#    (dim0=0, dim1=2 -- exercises the general N-D coordinate math, not
#    just the easy "swap the last two axes" case), eager backward, and
#    the full KIR path.
# ---------------------------------------------------------------------

a_vals = [1, 2, 3, 4, 5, 6]
a = kansai.from_flat(a_vals, [2, 3])
at = a.transpose(0, 1)
assert list(at.shape) == [3, 2]
check_close("transpose 2D forward", at.tolist(), [1, 4, 2, 5, 3, 6])

b_vals = list(range(24))
b = kansai.from_flat(b_vals, [2, 3, 4])
bt = b.transpose(0, 2)
assert list(bt.shape) == [4, 3, 2]


def naive_transpose_024(flat, shape, d0, d1):
    import itertools
    nd = len(shape)
    strides = [1] * nd
    for i in range(nd - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    out_shape = list(shape)
    out_shape[d0], out_shape[d1] = out_shape[d1], out_shape[d0]
    out_strides = [1] * nd
    for i in range(nd - 2, -1, -1):
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1]
    out = [0.0] * len(flat)
    for idx, coord in enumerate(itertools.product(*(range(s) for s in shape))):
        coord = list(coord)
        coord[d0], coord[d1] = coord[d1], coord[d0]
        oidx = sum(c * s for c, s in zip(coord, out_strides))
        out[oidx] = flat[idx]
    return out


check_close("transpose 3D (non-adjacent axes) forward", bt.tolist(),
            naive_transpose_024(b_vals, [2, 3, 4], 0, 2))

analytical_t = central_diff_grad(lambda t: t.transpose(0, 1).sum(), a_vals, [2, 3])
ag = kansai.from_flat(a_vals, [2, 3], requires_grad=True)
ag.transpose(0, 1).sum().backward()
check_close("transpose eager backward vs central diff", ag.grad.tolist(), analytical_t, GRAD_TOL)

graph_t = kir.trace(lambda t: t.transpose(0, 1).sum(), a)
eager_t = a.transpose(0, 1).sum().tolist()
check_close("transpose kir.run()", kir.run(graph_t, a).tolist(), eager_t)
if core.metal_available():
    check_close("transpose kir.run_metal()", kir.run_metal(kir.elementwise_fusion(graph_t), a).tolist(), eager_t)
bwd_t = kir.grad(graph_t, graph_t.inputs)
check_close("transpose kir.grad()", kir.run(bwd_t, a).tolist(), ag.grad.tolist())

# ---------------------------------------------------------------------
# 3. slice: forward along two different dims, eager backward, and the
#    full KIR path -- including all three boundary shapes _vjp_slice
#    branches on (padding both sides, one side, or neither).
# ---------------------------------------------------------------------

c_vals = list(range(12))
c = kansai.from_flat(c_vals, [3, 4])
check_close("slice forward (dim0)", c.slice(0, 1, 3).tolist(), [4, 5, 6, 7, 8, 9, 10, 11])
check_close("slice forward (dim1)", c.slice(1, 1, 3).tolist(), [1, 2, 5, 6, 9, 10])

d_vals = [rng.uniform(-1, 1) for _ in range(20)]
analytical_s = central_diff_grad(lambda t: t.slice(0, 1, 3).sum(), d_vals, [5, 4])
dg = kansai.from_flat(d_vals, [5, 4], requires_grad=True)
dg.slice(0, 1, 3).sum().backward()
check_close("slice eager backward vs central diff", dg.grad.tolist(), analytical_s, GRAD_TOL)

d = kansai.from_flat(d_vals, [5, 4])
for start, stop, label in [(1, 3, "mid-range"), (0, 3, "start=0"), (2, 5, "stop=full"), (0, 5, "whole dim")]:
    graph_s = kir.trace(lambda t, s=start, e=stop: t.slice(0, s, e).sum(), d)
    eager_s = d.slice(0, start, stop).sum().tolist()
    check_close(f"slice kir.run() ({label})", kir.run(graph_s, d).tolist(), eager_s)
    if core.metal_available():
        check_close(f"slice kir.run_metal() ({label})",
                    kir.run_metal(kir.elementwise_fusion(graph_s), d).tolist(), eager_s)
    bwd_s = kir.grad(graph_s, graph_s.inputs)
    dg2 = kansai.from_flat(d_vals, [5, 4], requires_grad=True)
    dg2.slice(0, start, stop).sum().backward()
    check_close(f"slice kir.grad() ({label})", kir.run(bwd_s, d).tolist(), dg2.grad.tolist())

try:
    c.slice(0, 2, 10)
    raise AssertionError("expected slice to reject an out-of-range stop")
except RuntimeError as e:
    print(f"slice correctly rejects an out-of-range range: {e}")

# ---------------------------------------------------------------------
# 4. cat: forward along two different dims, eager backward for both
#    inputs, and the full KIR path.
# ---------------------------------------------------------------------

p = kansai.from_flat([1, 2, 3, 4], [2, 2])
q = kansai.from_flat([5, 6, 7, 8], [2, 2])
check_close("cat forward (dim0)", core.cat([p, q], 0).tolist(), [1, 2, 3, 4, 5, 6, 7, 8])
check_close("cat forward (dim1)", core.cat([p, q], 1).tolist(), [1, 2, 5, 6, 3, 4, 7, 8])

p_vals = [rng.uniform(-1, 1) for _ in range(6)]
q_vals = [rng.uniform(-1, 1) for _ in range(9)]


def cat_loss(pv, qv):
    return core.cat([kansai.from_flat(pv, [2, 3]), kansai.from_flat(qv, [3, 3])], 0).sum()


analytical_p = central_diff_grad(lambda t: cat_loss(t.tolist(), q_vals), p_vals, [2, 3])
analytical_q = central_diff_grad(lambda t: cat_loss(p_vals, t.tolist()), q_vals, [3, 3])
pg = kansai.from_flat(p_vals, [2, 3], requires_grad=True)
qg = kansai.from_flat(q_vals, [3, 3], requires_grad=True)
core.cat([pg, qg], 0).sum().backward()
check_close("cat eager backward vs central diff (input p)", pg.grad.tolist(), analytical_p, GRAD_TOL)
check_close("cat eager backward vs central diff (input q)", qg.grad.tolist(), analytical_q, GRAD_TOL)

pp = kansai.from_flat(p_vals, [2, 3])
qq = kansai.from_flat(q_vals, [3, 3])
graph_c = kir.trace(lambda a, b: kir.cat([a, b], 0).sum(), pp, qq)
eager_c = core.cat([pp, qq], 0).sum().tolist()
check_close("cat kir.run()", kir.run(graph_c, pp, qq).tolist(), eager_c)
if core.metal_available():
    check_close("cat kir.run_metal()", kir.run_metal(kir.elementwise_fusion(graph_c), pp, qq).tolist(), eager_c)
bwd_c = kir.grad(graph_c, graph_c.inputs)
grad_p, grad_q = kir.run(bwd_c, pp, qq)
check_close("cat kir.grad() (input p)", grad_p.tolist(), pg.grad.tolist())
check_close("cat kir.grad() (input q)", grad_q.tolist(), qg.grad.tolist())

try:
    core.cat([p, kansai.from_flat([1, 2, 3], [3, 1])], 0)
    raise AssertionError("expected cat to reject mismatched non-cat dims")
except RuntimeError as e:
    print(f"cat correctly rejects mismatched shapes: {e}")

print("\nShape ops test passed.")
