"""LayerNorm and BatchNorm1d -- see python/kansai/nn.py's own docstrings
for the full design (both normalize via mean/var composed from existing
ops; LayerNorm over the last axis per-example, BatchNorm1d over the
batch axis per-feature with running statistics for eval mode) and what
each does and doesn't cover (BatchNorm2d for conv activations is real,
unattempted future work; BatchNorm's training-mode running-stats
bookkeeping makes it eager-only, not kir.trace()-able, unlike LayerNorm).

Checked: forward values against a from-scratch manual normalization
(not kansai's own mean()/var composed a second time), backward against
central differences, LayerNorm's full KIR path (trace -> all
interpreters -> kir.grad), BatchNorm1d's running-stats update and its
eval-mode/training-mode distinction, Module.train()/eval() recursing
through Sequential, and a practical end-to-end training run for each.
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import kir, nn, optim
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


def manual_normalize(row, eps=1e-5):
    m = sum(row) / len(row)
    var = sum((v - m) ** 2 for v in row) / len(row)
    return [(v - m) / math.sqrt(var + eps) for v in row]


# ---------------------------------------------------------------------
# 1. LayerNorm: forward (weight=1/bias=0 at construction, so the raw
#    output IS the normalization) against a from-scratch manual
#    per-row mean/var, on both 2D and 3D input; backward vs central
#    differences, including that weight/bias themselves get gradients;
#    the full KIR path.
# ---------------------------------------------------------------------

ln = nn.LayerNorm(4)
x = kansai.from_flat([1, 2, 3, 4, 5, 6, 7, 8], [2, 4])
expected = manual_normalize([1, 2, 3, 4]) + manual_normalize([5, 6, 7, 8])
check_close("LayerNorm forward (2D) vs manual per-row normalization", ln(x).tolist(), expected)

x3 = kansai.randn([2, 3, 4], std=1.0, seed=1)
out3 = ln(x3)
assert list(out3.shape) == [2, 3, 4]
flat3 = out3.tolist()
for i in range(6):
    row = flat3[i * 4:(i + 1) * 4]
    assert abs(sum(row) / 4) < 1e-3, f"LayerNorm row {i} mean should be ~0, got {sum(row) / 4}"
print("LayerNorm on 3D (batch, seq, features) input: every row normalized to mean~0: OK")

ln2 = nn.LayerNorm(4)
vals = [rng.uniform(-2, 2) for _ in range(8)]
xg = kansai.from_flat(vals, [2, 4], requires_grad=True)
ln2(xg).sum().backward()
analytical = central_diff_grad(lambda t: ln2(t).sum(), vals, [2, 4])
check_close("LayerNorm backward vs central diff", xg.grad.tolist(), analytical, GRAD_TOL)
assert ln2.weight.grad is not None and ln2.bias.grad is not None, "LayerNorm weight/bias must receive gradients"
print("LayerNorm weight/bias gradients populated: OK")

graph = kir.trace(lambda t: ln(t).sum(), x)
eager_ln = ln(x).sum().tolist()
check_close("LayerNorm kir.run()", kir.run(graph, x).tolist(), eager_ln)
check_close("LayerNorm kir.run_fused()", kir.run_fused(kir.elementwise_fusion(graph), x).tolist(), eager_ln)
if core.metal_available():
    check_close("LayerNorm kir.run_metal()", kir.run_metal(kir.elementwise_fusion(graph), x).tolist(), eager_ln)
bwd = kir.grad(graph, graph.inputs)
xg2 = kansai.from_flat(x.tolist(), [2, 4], requires_grad=True)
ln(xg2).sum().backward()
check_close("LayerNorm kir.grad()", kir.run(bwd, x).tolist(), xg2.grad.tolist())

# ---------------------------------------------------------------------
# 2. BatchNorm1d: forward (training mode) against a manual per-COLUMN
#    normalization; running_mean/running_var actually update; eval mode
#    uses the running statistics, not the eval batch's own; backward vs
#    central differences.
# ---------------------------------------------------------------------

bn = nn.BatchNorm1d(3)
xb = kansai.from_flat([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [4, 3])
cols = [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]
normalized_cols = [manual_normalize(c) for c in cols]
expected_bn = []
for i in range(4):
    expected_bn.extend([normalized_cols[0][i], normalized_cols[1][i], normalized_cols[2][i]])
check_close("BatchNorm1d forward (training) vs manual per-column normalization", bn(xb).tolist(), expected_bn)

bn2 = nn.BatchNorm1d(3, momentum=0.1)
before_mean, before_var = bn2.running_mean.tolist(), bn2.running_var.tolist()
bn2(xb)
after_mean, after_var = bn2.running_mean.tolist(), bn2.running_var.tolist()
assert before_mean != after_mean and before_var != after_var, "running stats must update after a training-mode forward"
print(f"BatchNorm1d running_mean updated: {before_mean} -> {[round(v, 4) for v in after_mean]}")

bn3 = nn.BatchNorm1d(3, momentum=0.5)
for _ in range(5):
    bn3(kansai.randn([8, 3], std=2.0, seed=rng.randint(0, 10_000)))
bn3.eval()
single = kansai.from_flat([100.0, 200.0, 300.0], [1, 3])
out_eval = bn3(single)
assert any(abs(v) > 1 for v in out_eval.tolist()), \
    "eval mode must normalize by running stats (nonzero result expected), not this single example's own trivial stats"
print(f"BatchNorm1d eval mode uses running stats, not batch stats: {[round(v, 3) for v in out_eval.tolist()]}")

bn4 = nn.BatchNorm1d(3)
bvals = [rng.uniform(-2, 2) for _ in range(12)]
xbg = kansai.from_flat(bvals, [4, 3], requires_grad=True)
bn4(xbg).sum().backward()

# Each perturbed evaluation needs its OWN fresh BatchNorm1d instance --
# reusing one would let one evaluation's running-stats update (a real,
# intentional side effect of a training-mode forward pass) contaminate
# the next, which would corrupt the finite-difference estimate itself,
# not just be untidy.
analytical_bn = []
for i in range(len(bvals)):
    plus = list(bvals)
    plus[i] += EPS
    minus = list(bvals)
    minus[i] -= EPS
    fp = nn.BatchNorm1d(3)(kansai.from_flat(plus, [4, 3])).sum().tolist()[0]
    fm = nn.BatchNorm1d(3)(kansai.from_flat(minus, [4, 3])).sum().tolist()[0]
    analytical_bn.append((fp - fm) / (2 * EPS))
check_close("BatchNorm1d backward vs central diff", xbg.grad.tolist(), analytical_bn, GRAD_TOL)

# ---------------------------------------------------------------------
# 3. Module.train()/eval() recursion through Sequential.
# ---------------------------------------------------------------------

model = nn.Sequential(nn.Linear(3, 4, seed=1), nn.BatchNorm1d(4), nn.ReLU())
assert model.training and model.layers[1].training
model.eval()
assert not model.training and not model.layers[1].training, "eval() must recurse into Sequential's sub-modules"
model.train()
assert model.training and model.layers[1].training, "train() must also recurse"
print("Module.train()/eval() recursion through Sequential: OK")

# ---------------------------------------------------------------------
# 4. Practical check: train the same 3-class Gaussian-blob classifier
#    test_softmax_cross_entropy.py trains, once with a LayerNorm layer
#    and once with a BatchNorm1d layer inserted, confirming both are
#    genuine working layers in a real training loop, not just
#    forward-correct in isolation.
# ---------------------------------------------------------------------

centers = [(-3.0, -3.0), (3.0, -3.0), (0.0, 3.0)]
num_classes = 3
per_class = 30
xs, ys = [], []
for c, (cx, cy) in enumerate(centers):
    for _ in range(per_class):
        xs.append([cx + rng.gauss(0, 0.5), cy + rng.gauss(0, 0.5)])
        onehot = [0.0] * num_classes
        onehot[c] = 1.0
        ys.append(onehot)
combined = list(zip(xs, ys))
rng.shuffle(combined)
xs, ys = zip(*combined)
X = kansai.from_flat([v for row in xs for v in row], [len(xs), 2])
Y = kansai.from_flat([v for row in ys for v in row], [len(ys), num_classes])

for norm_name, norm_layer in [("LayerNorm", nn.LayerNorm(16)), ("BatchNorm1d", nn.BatchNorm1d(16))]:
    model = nn.Sequential(
        nn.Linear(2, 16, seed=5),
        norm_layer,
        nn.ReLU(),
        nn.Linear(16, num_classes, seed=6),
    )
    opt = optim.Adam(model.parameters(), lr=0.05)
    loss = None
    for step in range(200):
        loss = model(X).cross_entropy(Y)
        model.zero_grad()
        loss.backward()
        opt.step()
    final_loss = loss.tolist()[0]

    if norm_name == "BatchNorm1d":
        model.eval()
    final_logits = model(X).tolist()
    targets_flat = Y.tolist()
    correct = 0
    for i in range(len(xs)):
        row = final_logits[i * num_classes:(i + 1) * num_classes]
        trow = targets_flat[i * num_classes:(i + 1) * num_classes]
        correct += (row.index(max(row)) == trow.index(max(trow)))
    accuracy = correct / len(xs)
    print(f"3-class classifier with {norm_name}: final loss {final_loss:.6f}, accuracy {accuracy:.1%}")
    assert final_loss < 0.2, f"{norm_name} model did not converge (loss {final_loss})"
    assert accuracy > 0.9, f"{norm_name} model accuracy too low: {accuracy:.1%}"

print("\nNormalization test passed.")
