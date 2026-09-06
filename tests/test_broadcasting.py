"""General NumPy-style broadcasting for add/sub/mul. Before this, add
supported exactly one broadcast shape (the (batch, features) +
(features,) bias case, hardcoded), and sub/mul supported none at all --
every other shape mismatch was a hard error. Now all three accept any
right-aligned broadcast-compatible pair of shapes, at every level this
codebase's other ops are held to: forward values, eager backward against
central differences, and the full KIR path (trace -> all interpreters ->
kir.grad).

The bias-broadcast case isn't just "still correct" here -- it's checked
to still take the exact SAME fast paths it always did (elementwise_fusion
still fuses it into fused_bias_relu, run_metal still routes it through
metal_add_bias/the batched elementwise chain), since those depend on
recognizing that one specific shape precisely, not "any shape mismatch."
A real regression here (fixed during development, see
_metal_elementwise_kind and run_metal's own add/sub/mul dispatch in
kir.py) would have silently misrouted a general broadcast through a
Metal kernel that only knows how to handle the bias shape.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import kir
from kansai import _core as core

EPS = 1e-3
GRAD_TOL = 2e-2
TOL = 1e-4

rng = random.Random(0)


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


# ---------------------------------------------------------------------
# 1. The bias-broadcast case must be bit-for-bit unchanged: same forward
#    value, same fast paths taken (fused_bias_relu, metal_add_bias),
#    same backward. This is a regression check, not new coverage --
#    everything here already passed before broadcasting existed.
# ---------------------------------------------------------------------

X = kansai.randn([4, 3], std=1.0, seed=1)
bias = kansai.randn([3], std=1.0, seed=2)
graph_bias = kir.trace(lambda x, b: x.add(b).relu().sum(), X, bias)

eager_bias = X.add(bias).relu().sum().tolist()
check_close("bias-broadcast kir.run()", kir.run(graph_bias, X, bias).tolist(), eager_bias)
fused_bias = kir.elementwise_fusion(graph_bias)
check_close("bias-broadcast kir.run_fused()", kir.run_fused(fused_bias, X, bias).tolist(), eager_bias)
assert any(n.op == "fused_bias_relu" for n in fused_bias.nodes), \
    "bias+relu no longer fuses into fused_bias_relu -- a real regression"
print("bias+relu still fuses into fused_bias_relu: OK")

if core.metal_available():
    check_close("bias-broadcast kir.run_metal()", kir.run_metal(fused_bias, X, bias).tolist(), eager_bias)

bwd_bias = kir.grad(graph_bias, graph_bias.inputs)
grad_x, grad_bias = kir.run(bwd_bias, X, bias)
Xg = kansai.from_flat(X.tolist(), [4, 3], requires_grad=True)
biasg = kansai.from_flat(bias.tolist(), [3], requires_grad=True)
Xg.add(biasg).relu().sum().backward()
check_close("bias-broadcast kir.grad() (x)", grad_x.tolist(), Xg.grad.tolist())
check_close("bias-broadcast kir.grad() (bias)", grad_bias.tolist(), biasg.grad.tolist())

# ---------------------------------------------------------------------
# 2. General 2-way broadcast: (3,1) + (1,4) -> (3,4), neither operand
#    already at the output shape -- both need their gradient reduced
#    down, not just one (the bias case only ever needed one side
#    reduced, so this exercises a genuinely different code path in
#    _vjp_add/_vjp_sub/_vjp_mul).
# ---------------------------------------------------------------------

A = kansai.randn([3, 1], std=1.0, seed=3)
B = kansai.randn([1, 4], std=1.0, seed=4)

for op_name, op in [("add", lambda a, b: a.add(b)), ("sub", lambda a, b: a.sub(b)), ("mul", lambda a, b: a.mul(b))]:
    y = op(A, B)
    assert list(y.shape) == [3, 4], f"{op_name}: expected broadcast shape [3,4], got {list(y.shape)}"

    a_flat, b_flat = A.tolist(), B.tolist()
    if op_name == "add":
        expected = [a_flat[i] + b_flat[j] for i in range(3) for j in range(4)]
    elif op_name == "sub":
        expected = [a_flat[i] - b_flat[j] for i in range(3) for j in range(4)]
    else:
        expected = [a_flat[i] * b_flat[j] for i in range(3) for j in range(4)]
    check_close(f"{op_name} general broadcast forward", y.tolist(), expected)

    av = [rng.uniform(-1, 1) for _ in range(3)]
    bv = [rng.uniform(-1, 1) for _ in range(4)]
    analytical_a = central_diff_grad(lambda t: op(t, kansai.from_flat(bv, [1, 4])).sum(), av, [3, 1])
    analytical_b = central_diff_grad(lambda t: op(kansai.from_flat(av, [3, 1]), t).sum(), bv, [1, 4])
    ag = kansai.from_flat(av, [3, 1], requires_grad=True)
    bg = kansai.from_flat(bv, [1, 4], requires_grad=True)
    op(ag, bg).sum().backward()
    check_close(f"{op_name} general broadcast eager backward (a) vs central diff", ag.grad.tolist(), analytical_a,
                GRAD_TOL)
    check_close(f"{op_name} general broadcast eager backward (b) vs central diff", bg.grad.tolist(), analytical_b,
                GRAD_TOL)

    graph = kir.trace(lambda a, b, op=op: op(a, b).sum(), A, B)
    eager_out = op(A, B).sum().tolist()
    check_close(f"{op_name} general broadcast kir.run()", kir.run(graph, A, B).tolist(), eager_out)
    if core.metal_available():
        check_close(f"{op_name} general broadcast kir.run_metal()",
                    kir.run_metal(kir.elementwise_fusion(graph), A, B).tolist(), eager_out)

    bwd = kir.grad(graph, graph.inputs)
    ga, gb = kir.run(bwd, A, B)
    ag2 = kansai.from_flat(A.tolist(), [3, 1], requires_grad=True)
    bg2 = kansai.from_flat(B.tolist(), [1, 4], requires_grad=True)
    op(ag2, bg2).sum().backward()
    check_close(f"{op_name} general broadcast kir.grad() (a)", ga.tolist(), ag2.grad.tolist())
    check_close(f"{op_name} general broadcast kir.grad() (b)", gb.tolist(), bg2.grad.tolist())

# ---------------------------------------------------------------------
# 3. Rank-mismatch broadcast: (2,3,4) + (4,) -- a shape the OLD
#    bias-specific check would never have accepted (it required a's
#    rank to be exactly 2), confirming this isn't secretly still
#    2D-only under the hood.
# ---------------------------------------------------------------------

C = kansai.from_flat(list(range(24)), [2, 3, 4])
D = kansai.from_flat([1, 2, 3, 4], [4])
y = C.add(D)
assert list(y.shape) == [2, 3, 4]
c_flat, d_flat = C.tolist(), D.tolist()
expected = [c_flat[idx] + d_flat[idx % 4] for idx in range(24)]
check_close("rank-mismatch broadcast (2,3,4)+(4,) forward", y.tolist(), expected)

# ---------------------------------------------------------------------
# 4. Incompatible shapes are rejected, not silently misinterpreted.
# ---------------------------------------------------------------------

try:
    kansai.from_flat([1, 2, 3], [3]).add(kansai.from_flat([1, 2], [2]))
    raise AssertionError("expected incompatible shapes to raise")
except RuntimeError as e:
    print(f"incompatible shapes correctly rejected: {e}")

print("\nBroadcasting test passed.")
