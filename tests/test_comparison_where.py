"""Comparison ops (gt/lt/eq) and where -- the primitives real boolean-
style masking needs, and a real, stated gap up to now: MultiHeadAttention's
own `mask` argument had to be strictly additive (see its docstring in
python/kansai/nn.py) specifically because Kansai had no way to select
between two tensors by a per-element condition.

`gt`/`lt`/`eq` are new C++ kernels (backend/cpu's `broadcast_binary`
gains three comparison cases alongside its existing Add/Sub/Mul, same
general-broadcasting shape add/sub/mul's own broadcast kernels already
use). They're deliberately, permanently non-differentiable: a
comparison is a step function of its inputs, so Tensor::gt/lt/eq never
attach a GradNode at all (eager), and kir.grad's own vjp rule for them
(_vjp_compare) returns an explicit exact zero for both operands -- the
same "deliberate zero, not an oversight" pattern max(dim)'s own vjp
already established, not "no rule exists" (which would raise instead
of computing the mathematically correct answer).

`where(cond, a, b)` needs no new kernel, KIR node type, or interpreter
dispatch AT ALL -- it's `b + cond * (a - b)`, an algebraic
rearrangement of the obvious `cond*a + (1-cond)*b` chosen specifically
so it only needs sub/mul/add, which already exist identically for both
eager Tensors and traced TraceValues. One plain Python function
(kansai.where) works, unmodified, in both eager code and a kir.trace'd
graph.

Checked: gt/lt/eq forward (same-shape and general-broadcast), the
non-differentiability itself (`requires_grad` never set even when an
input requires it); where's forward against a manual selection, and
backward confirming gradient flows into `a`/`b` at exactly the
positions their branch was selected, zero elsewhere, and NEVER into
`cond`; the combined gt+where full KIR path (trace, all four
interpreters, and kir.grad -- confirming the same zero-into-cond,
correct-into-branches gradient routing survives tracing, not just
eager); and a practical end-to-end test -- fitting a genuinely
piecewise-linear function (different slopes on either side of zero)
by training `where(x > 0, w_pos*x, w_neg*x)` with Adam, which only
converges to the right two slopes if gradient is correctly routed
through the taken branch on every single example.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import kir, optim
from kansai import _core as core

EPS = 1e-3
GRAD_TOL = 2e-2
TOL = 1e-4


def check_close(label, a, b, tol=TOL):
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < tol, f"{label}[{i}]: got={x:.6f} expected={y:.6f}"
    print(f"{label}: OK ({len(a)} elements)")


rng = random.Random(0)

# ---------------------------------------------------------------------
# 1. gt/lt/eq: forward (exact-shape and general-broadcast), and
#    non-differentiability (never attaches a GradNode, even when an
#    input requires grad).
# ---------------------------------------------------------------------

a = kansai.from_flat([1, 2, 3, 4], [2, 2])
b = kansai.from_flat([2, 2, 2, 2], [2, 2])
check_close("gt forward (exact shape)", a.gt(b).tolist(), [0, 0, 1, 1])
check_close("lt forward (exact shape)", a.lt(b).tolist(), [1, 0, 0, 0])
check_close("eq forward (exact shape)", a.eq(b).tolist(), [0, 1, 0, 0])

row = kansai.from_flat([2, 3], [2])
check_close("gt forward (broadcast)", a.gt(row).tolist(), [0, 0, 1, 1])
check_close("lt forward (broadcast)", a.lt(row).tolist(), [1, 1, 0, 0])

ag = kansai.from_flat([1, 2, 3, 4], [2, 2], requires_grad=True)
cmp = ag.gt(b)
assert not cmp.requires_grad, "a comparison must never require grad, even when its input does"

try:
    a.gt(kansai.from_flat([1, 2, 3], [3]))
    raise AssertionError("expected gt to reject a non-broadcastable shape")
except RuntimeError as e:
    print(f"gt correctly rejects a non-broadcastable shape: {e}")

# ---------------------------------------------------------------------
# 2. where: forward against a manual selection, and backward -- grad
#    flows into `a` exactly where cond was 1, into `b` exactly where
#    cond was 0, and NEVER into `cond` itself.
# ---------------------------------------------------------------------

cond = core.from_flat([1, 0, 1, 0], [4])
wa = core.from_flat([10, 20, 30, 40], [4])
wb = core.from_flat([100, 200, 300, 400], [4])
out = kansai.where(cond, wa, wb)
check_close("where forward", out.tolist(), [10, 200, 30, 400])

wag = core.from_flat([10, 20, 30, 40], [4], requires_grad=True)
wbg = core.from_flat([100, 200, 300, 400], [4], requires_grad=True)
kansai.where(cond, wag, wbg).sum().backward()
check_close("where backward: grad routes to a where cond=1", wag.grad.tolist(), [1, 0, 1, 0])
check_close("where backward: grad routes to b where cond=0", wbg.grad.tolist(), [0, 1, 0, 1])

# ---------------------------------------------------------------------
# 3. Combined gt + where through the full KIR path: trace, all four
#    interpreters, and kir.grad -- confirming the gradient routing
#    above (zero into the compared values, correct into the two
#    branches) survives tracing, not just eager.
# ---------------------------------------------------------------------

x = core.from_flat([1, 2, 3, 4], [4])
thresh = core.from_flat([2, 2, 2, 2], [4])
pos = core.from_flat([100, 200, 300, 400], [4])
neg = core.from_flat([-1, -2, -3, -4], [4])

graph = kir.trace(lambda t, th, p, n: kansai.where(t.gt(th), p, n), x, thresh, pos, neg)
eager_out = kansai.where(x.gt(thresh), pos, neg).tolist()
check_close("gt+where kir.run()", kir.run(graph, x, thresh, pos, neg).tolist(), eager_out)
check_close("gt+where kir.run_fused()",
            kir.run_fused(kir.elementwise_fusion(graph), x, thresh, pos, neg).tolist(), eager_out)
plan = kir.plan_memory(graph)
check_close("gt+where kir.run_planned()",
            kir.run_planned(graph, plan, core.StoragePool(), x, thresh, pos, neg).tolist(), eager_out)
if core.metal_available():
    check_close("gt+where kir.run_metal()",
                kir.run_metal(kir.elementwise_fusion(graph), x, thresh, pos, neg).tolist(), eager_out)

graph_sum = kir.trace(lambda t, th, p, n: kansai.where(t.gt(th), p, n).sum(), x, thresh, pos, neg)
bwd = kir.grad(graph_sum, graph_sum.inputs)
grad_t, grad_th, grad_p, grad_n = kir.run(bwd, x, thresh, pos, neg)
check_close("gt+where kir.grad(): zero into compared value t", grad_t.tolist(), [0, 0, 0, 0])
check_close("gt+where kir.grad(): zero into compared value thresh", grad_th.tolist(), [0, 0, 0, 0])
# x > thresh = [1>2, 2>2, 3>2, 4>2] = [0, 0, 1, 1] -- p gets grad there, n elsewhere.
check_close("gt+where kir.grad(): grad into p where cond=1", grad_p.tolist(), [0, 0, 1, 1])
check_close("gt+where kir.grad(): grad into n where cond=0", grad_n.tolist(), [1, 1, 0, 0])

# ---------------------------------------------------------------------
# 4. Practical end-to-end test: fit a genuinely piecewise-linear
#    function -- y = 2x for x>0, y = -x for x<=0 -- by training
#    where(x > 0, w_pos*x, w_neg*x) with Adam. Only converges to the
#    right two slopes if gradient is correctly routed through whichever
#    branch each example actually took.
# ---------------------------------------------------------------------

TRUE_W_POS = 2.0
TRUE_W_NEG = -1.0


def true_fn(xv):
    return TRUE_W_POS * xv if xv > 0 else TRUE_W_NEG * xv


N = 400
xs = [rng.uniform(-3, 3) for _ in range(N)]
ys = [true_fn(v) for v in xs]

w_pos = core.from_flat([0.1], [1], requires_grad=True)
w_neg = core.from_flat([0.1], [1], requires_grad=True)
opt = optim.Adam([w_pos, w_neg], lr=0.1)

zero_scalar = core.from_flat([0.0], [1])

EPOCHS = 300
BATCH = 40
for epoch in range(EPOCHS):
    perm = list(range(N))
    rng.shuffle(perm)
    total_loss = 0.0
    for start in range(0, N, BATCH):
        idx = perm[start:start + BATCH]
        xb = core.from_flat([xs[i] for i in idx], [len(idx)])
        yb = core.from_flat([ys[i] for i in idx], [len(idx)])
        cond_b = xb.gt(core.from_flat([0.0] * len(idx), [len(idx)]))
        pred = kansai.where(cond_b, xb.mul(w_pos), xb.mul(w_neg))
        diff = pred.sub(yb)
        loss = diff.mul(diff).mean()
        w_pos.zero_grad()
        w_neg.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.tolist()[0] * len(idx)
    if epoch % 75 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch}: avg loss {total_loss / N:.5f}, "
              f"w_pos={w_pos.tolist()[0]:.4f}, w_neg={w_neg.tolist()[0]:.4f}")

print(f"learned w_pos={w_pos.tolist()[0]:.4f} (true {TRUE_W_POS}), "
      f"w_neg={w_neg.tolist()[0]:.4f} (true {TRUE_W_NEG})")
assert abs(w_pos.tolist()[0] - TRUE_W_POS) < 0.05, "w_pos didn't converge -- gradient routing through where is wrong"
assert abs(w_neg.tolist()[0] - TRUE_W_NEG) < 0.05, "w_neg didn't converge -- gradient routing through where is wrong"

print("\nComparison ops / where test passed.")
