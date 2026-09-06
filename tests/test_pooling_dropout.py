"""AvgPool2d, MaxPool2d, Dropout -- see python/kansai/nn.py's own
docstrings for the full design of each. AvgPool2d is a pure composition
(reshape + mean(dim) x2, verified separable exactly, not approximately)
so it's fully KIR-traceable for free; MaxPool2d is a genuinely new C++
op with a real argmax-routed gradient, deliberately NOT built on the
existing (and deliberately non-differentiable) max(dim), and is
eager-only; Dropout is a real Tensor op (elementwise multiply by a
random mask) needing no dedicated kernel or backward at all.

Checked: forward values against hand-computed references, backward
against central differences (including confirming MaxPool2d's gradient
lands on EXACTLY the winning position in each window, not spread
across it), AvgPool2d's full KIR path, MaxPool2d's documented
eager-only limitation (confirmed to fail with a clear error, not
silently trace something wrong), Dropout's statistical mask ratio and
eval-mode identity, and a practical end-to-end CNN (Conv2d -> ReLU ->
MaxPool2d -> Linear -> cross_entropy) trained to convergence on a small
synthetic image classification task.
"""

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


# ---------------------------------------------------------------------
# 1. AvgPool2d: forward vs hand-computed, backward vs central diff,
#    full KIR path (it's a pure composition, so this should just work).
# ---------------------------------------------------------------------

x = kansai.from_flat(list(range(1, 17)), [1, 1, 4, 4])
ap = nn.AvgPool2d(2)
out = ap(x)
assert list(out.shape) == [1, 1, 2, 2]
check_close("AvgPool2d forward vs hand-computed", out.tolist(), [3.5, 5.5, 11.5, 13.5])

vals = [rng.uniform(-2, 2) for _ in range(16)]
xg = kansai.from_flat(vals, [1, 1, 4, 4], requires_grad=True)
nn.AvgPool2d(2)(xg).sum().backward()
analytical = central_diff_grad(lambda t: nn.AvgPool2d(2)(t).sum(), vals, [1, 1, 4, 4])
check_close("AvgPool2d backward vs central diff", xg.grad.tolist(), analytical, GRAD_TOL)

graph = kir.trace(lambda t: ap(t).sum(), x)
eager_ap = ap(x).sum().tolist()
check_close("AvgPool2d kir.run()", kir.run(graph, x).tolist(), eager_ap)
check_close("AvgPool2d kir.run_fused()", kir.run_fused(kir.elementwise_fusion(graph), x).tolist(), eager_ap)
if core.metal_available():
    check_close("AvgPool2d kir.run_metal()", kir.run_metal(kir.elementwise_fusion(graph), x).tolist(), eager_ap)
bwd = kir.grad(graph, graph.inputs)
xg2 = kansai.from_flat(x.tolist(), [1, 1, 4, 4], requires_grad=True)
ap(xg2).sum().backward()
check_close("AvgPool2d kir.grad()", kir.run(bwd, x).tolist(), xg2.grad.tolist())

try:
    nn.AvgPool2d(3)(x)  # 4 doesn't divide by 3
    raise AssertionError("expected AvgPool2d to reject a kernel_size that doesn't divide evenly")
except ValueError as e:
    print(f"AvgPool2d correctly rejects a non-dividing kernel_size: {e}")

# ---------------------------------------------------------------------
# 2. MaxPool2d: forward vs hand-computed, overlapping windows (stride <
#    kernel_size), backward vs central diff AND confirmed to route
#    gradient to exactly the winning position (not spread across the
#    window), and the documented eager-only limitation.
# ---------------------------------------------------------------------

mp = nn.MaxPool2d(2)
out_mp = mp(x)
assert list(out_mp.shape) == [1, 1, 2, 2]
check_close("MaxPool2d forward vs hand-computed", out_mp.tolist(), [6, 8, 14, 16])

overlap_vals = [1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12]
x_overlap = kansai.from_flat(overlap_vals, [1, 1, 3, 4])
out_overlap = nn.MaxPool2d(2, stride=1)(x_overlap)
assert list(out_overlap.shape) == [1, 1, 2, 3]
print(f"MaxPool2d overlapping windows (kernel=2, stride=1): shape OK {list(out_overlap.shape)}")

xg3 = kansai.from_flat(vals, [1, 1, 4, 4], requires_grad=True)
mp(xg3).sum().backward()
analytical_mp = central_diff_grad(lambda t: nn.MaxPool2d(2)(t).sum(), vals, [1, 1, 4, 4])
check_close("MaxPool2d backward vs central diff", xg3.grad.tolist(), analytical_mp, GRAD_TOL)
nonzero = sum(1 for g in xg3.grad.tolist() if abs(g) > 1e-6)
assert nonzero == 4, f"expected exactly 4 nonzero gradient entries (one per 2x2 window), got {nonzero}"
print(f"MaxPool2d gradient routed to exactly the winning position in each window ({nonzero}/16 nonzero): OK")

try:
    kir.trace(lambda t: mp(t).sum(), x)
    raise AssertionError("expected MaxPool2d to fail tracing (documented eager-only limitation)")
except AttributeError as e:
    print(f"MaxPool2d correctly fails to trace (documented, eager-only): {e}")

# ---------------------------------------------------------------------
# 3. Dropout: eval mode and p=0 are both exact identities; training
#    mode zeros roughly the expected fraction and scales survivors by
#    1/(1-p) (checked statistically, over enough elements that a wrong
#    ratio would clearly show, not enough to be flaky); backward routes
#    correctly through the same mask.
# ---------------------------------------------------------------------

big = kansai.randn([2000], std=1.0, seed=1)
drop = nn.Dropout(0.3)
drop.eval()
check_close("Dropout eval mode is exact identity", drop(big).tolist(), big.tolist())

drop_zero = nn.Dropout(0.0)
check_close("Dropout(p=0) is exact identity even in training mode", drop_zero(big).tolist(), big.tolist())

drop.train()
out_drop = drop(big).tolist()
zeros = sum(1 for v in out_drop if v == 0.0)
zero_frac = zeros / len(out_drop)
print(f"Dropout(p=0.3) training mode: {zero_frac:.1%} zeroed (expected ~30%)")
assert 0.20 < zero_frac < 0.40, f"dropout zero fraction {zero_frac:.1%} far from expected ~30%"
survivors = [v for v, orig in zip(out_drop, big.tolist()) if v != 0.0]
orig_survivors = [orig for v, orig in zip(out_drop, big.tolist()) if v != 0.0]
ratios = [s / o for s, o in zip(survivors, orig_survivors) if abs(o) > 1e-6]
avg_ratio = sum(ratios) / len(ratios)
print(f"Dropout(p=0.3) survivor scale: average {avg_ratio:.4f} (expected 1/(1-0.3)={1 / 0.7:.4f})")
assert abs(avg_ratio - 1 / 0.7) < 0.01, f"survivor scaling {avg_ratio} doesn't match 1/(1-p)"

xd = kansai.from_flat([1.0, 2.0, 3.0, 4.0], [4], requires_grad=True)
drop_half = nn.Dropout(0.5)
drop_half.train()
out_d = drop_half(xd)
out_d.sum().backward()
# every surviving (nonzero-output) element's gradient must equal its own scale factor (2.0),
# and every dropped element's gradient must be exactly 0 -- backward must route through the SAME mask.
out_d_vals = out_d.tolist()
grad_vals = xd.grad.tolist()
for i in range(4):
    if out_d_vals[i] == 0.0:
        assert grad_vals[i] == 0.0, f"dropped element {i} should have zero gradient, got {grad_vals[i]}"
    else:
        assert abs(grad_vals[i] - 2.0) < 1e-5, f"surviving element {i} should have gradient 2.0, got {grad_vals[i]}"
print("Dropout backward correctly routes through the same mask used in forward: OK")

# ---------------------------------------------------------------------
# 4. Practical check: a small CNN (Conv2d -> ReLU -> MaxPool2d ->
#    reshape -> Linear -> cross_entropy) trained on a synthetic 3-class
#    8x8 image task -- proves MaxPool2d works inside a real gradient
#    flow through Conv2d, not just in isolation.
# ---------------------------------------------------------------------

def make_image(cls):
    """An 8x8 image with a bright 4x4 block in one of three positions
    (top-left / top-right / bottom) plus noise -- class = which
    quadrant is bright."""
    img = [rng.uniform(0.0, 0.2) for _ in range(64)]
    if cls == 0:
        rows, cols = range(0, 4), range(0, 4)
    elif cls == 1:
        rows, cols = range(0, 4), range(4, 8)
    else:
        rows, cols = range(4, 8), range(0, 8, 1)
    for r in rows:
        for c in cols:
            img[r * 8 + c] = rng.uniform(0.8, 1.0)
    return img


examples = [(make_image(c), c) for c in range(3) for _ in range(40)]
rng.shuffle(examples)
imgs = [e[0] for e in examples]
labels = [e[1] for e in examples]
onehot = []
for label in labels:
    row = [0.0, 0.0, 0.0]
    row[label] = 1.0
    onehot.extend(row)

X = kansai.from_flat([v for img in imgs for v in img], [len(imgs), 1, 8, 8])
Y = kansai.from_flat(onehot, [len(imgs), 3])

cnn = nn.Sequential(
    nn.Conv2d(1, 4, kernel_size=3, padding=1, seed=1),
    nn.ReLU(),
    nn.MaxPool2d(2),  # 8x8 -> 4x4
)


class Flatten:
    def __call__(self, x):
        n = x.shape[0]
        size = x.shape[1] * x.shape[2] * x.shape[3]
        return x.reshape([n, size])


model = nn.Sequential(cnn, Flatten(), nn.Linear(4 * 4 * 4, 3, seed=2))
opt = optim.Adam(model.parameters(), lr=0.01)

loss = None
for step in range(150):
    loss = model(X).cross_entropy(Y)
    model.zero_grad()
    loss.backward()
    opt.step()

final_loss = loss.tolist()[0]
final_logits = model(X).tolist()
correct = sum(
    1 for i in range(len(imgs))
    if final_logits[i * 3:(i + 1) * 3].index(max(final_logits[i * 3:(i + 1) * 3])) == labels[i]
)
accuracy = correct / len(imgs)
print(f"CNN (Conv2d -> ReLU -> MaxPool2d -> Linear) on synthetic 3-class images: "
      f"loss {final_loss:.6f}, accuracy {accuracy:.1%}")
assert final_loss < 0.2, f"CNN with MaxPool2d did not converge (loss {final_loss})"
assert accuracy > 0.9, f"CNN with MaxPool2d accuracy too low: {accuracy:.1%}"

print("\nPooling/Dropout test passed.")
