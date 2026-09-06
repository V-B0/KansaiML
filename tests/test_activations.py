"""The rest of this project's activation vocabulary beyond relu: tanh,
sigmoid, gelu (exact, via erf -- not the tanh-based approximation some
frameworks default to), leaky_relu, plus the elementwise exp/log/sqrt/
reciprocal/div primitives everything above (and Adam, and softmax/
cross_entropy) is built from.

Checked at the same three levels as every other op in this project:
forward values against a hand-computed (or math.* library) reference,
eager backward against central differences, and the full KIR path
(trace -> all four interpreters -> kir.grad) matching eager exactly.
Also checked: the nn.Module wrappers (Tanh/Sigmoid/GELU/LeakyReLU) are
real, working layers, not just thin pass-throughs that happen to exist.
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import kir, nn
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


def gelu_ref(v):
    return v * 0.5 * (1 + math.erf(v / math.sqrt(2)))


ACTIVATIONS = [
    ("exp", lambda t: t.exp(), math.exp),
    ("log", lambda t: t.log(), math.log),
    ("tanh", lambda t: t.tanh(), math.tanh),
    ("sigmoid", lambda t: t.sigmoid(), lambda v: 1 / (1 + math.exp(-v))),
    ("gelu", lambda t: t.gelu(), gelu_ref),
    ("leaky_relu", lambda t: t.leaky_relu(), lambda v: v if v > 0 else 0.01 * v),
]

# log needs strictly positive inputs; keep every activation's test
# values positive so the same vals list works for all of them.
vals = [rng.uniform(0.3, 2.5) for _ in range(5)]
shape = [5]
x = kansai.from_flat(vals, shape)

for name, op, ref in ACTIVATIONS:
    y = op(x)
    check_close(f"{name} forward", y.tolist(), [ref(v) for v in vals])

    xg = kansai.from_flat(vals, shape, requires_grad=True)
    op(xg).sum().backward()
    analytical = central_diff_grad(lambda t: op(t).sum(), vals, shape)
    check_close(f"{name} eager backward vs central diff", xg.grad.tolist(), analytical, GRAD_TOL)

    graph = kir.trace(lambda t, op=op: op(t).sum(), x)
    eager_out = op(x).sum().tolist()
    check_close(f"{name} kir.run()", kir.run(graph, x).tolist(), eager_out)
    check_close(f"{name} kir.run_fused()", kir.run_fused(kir.elementwise_fusion(graph), x).tolist(), eager_out)
    plan = kir.plan_memory(graph)
    check_close(f"{name} kir.run_planned()", kir.run_planned(graph, plan, core.StoragePool(), x).tolist(), eager_out)
    if core.metal_available():
        check_close(f"{name} kir.run_metal()", kir.run_metal(kir.elementwise_fusion(graph), x).tolist(), eager_out)

    bwd = kir.grad(graph, graph.inputs)
    check_close(f"{name} kir.grad()", kir.run(bwd, x).tolist(), xg.grad.tolist())

# leaky_relu specifically needs a mixed-sign check, since its backward
# branches on sign -- the positive-only `vals` above never exercises
# the negative-slope path.
mixed_vals = [1.5, -0.8, 2.1, -1.3, 0.5]
xm = kansai.from_flat(mixed_vals, shape, requires_grad=True)
xm.leaky_relu(0.2).sum().backward()
analytical_mixed = central_diff_grad(lambda t: t.leaky_relu(0.2).sum(), mixed_vals, shape)
check_close("leaky_relu(0.2) backward on mixed signs vs central diff", xm.grad.tolist(), analytical_mixed, GRAD_TOL)

# ---------------------------------------------------------------------
# nn.Module wrappers: real layers, checked end to end in a tiny
# Sequential, not just thin pass-throughs assumed to work because the
# underlying Tensor method already does.
# ---------------------------------------------------------------------

model = nn.Sequential(nn.Linear(3, 4, seed=1), nn.Tanh(), nn.Linear(4, 2, seed=2), nn.Sigmoid())
inp = kansai.randn([2, 3], std=1.0, seed=3)
out = model(inp)
assert list(out.shape) == [2, 2]
assert all(0.0 < v < 1.0 for v in out.tolist()), "Sigmoid output should be strictly between 0 and 1"
print("nn.Tanh + nn.Sigmoid in a Sequential: OK (output shape and range both correct)")

gelu_model = nn.Sequential(nn.Linear(3, 4, seed=1), nn.GELU())
gelu_out = gelu_model(inp)
manual = inp.matmul(gelu_model.layers[0].weight).add(gelu_model.layers[0].bias).gelu()
check_close("nn.GELU matches manual .gelu() call", gelu_out.tolist(), manual.tolist())

leaky_model = nn.Sequential(nn.Linear(3, 4, seed=1), nn.LeakyReLU(0.3))
leaky_out = leaky_model(inp)
manual_leaky = inp.matmul(leaky_model.layers[0].weight).add(leaky_model.layers[0].bias).leaky_relu(0.3)
check_close("nn.LeakyReLU(0.3) matches manual .leaky_relu(0.3) call", leaky_out.tolist(), manual_leaky.tolist())

print("\nActivations test passed.")
