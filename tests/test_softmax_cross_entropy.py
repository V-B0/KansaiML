"""sum(dim)/mean(dim)/max(dim) -- reduction along one axis rather than
every axis the way sum()/mean() alone do -- plus softmax and
cross_entropy, both composed entirely from those reductions and the
elementwise ops in test_activations.py/test_broadcasting.py, with no
dedicated kernel or vjp rule of their own for softmax/cross_entropy
themselves.

Checked: sum(dim)/mean(dim)/max(dim) forward and backward (including
max(dim) deliberately having NO gradient -- see Tensor::max's own
declaration in core/include/kansai/Tensor.hpp for exactly why); softmax
forward, that it sums to 1, and that it stays finite on logits large
enough to overflow a naive exp() (the entire reason for the max-
subtraction trick); cross_entropy against an independent from-scratch
Python implementation of log-sum-exp, not kansai's own softmax+log
composed a second time (which would share any bug the real
implementation has); and, practically, training a real 3-class
classifier to convergence -- the same "does it still actually work" bar
test_xor.py and test_adam.py already hold themselves to, now for
classification instead of regression.
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import kir, nn, optim
from kansai import _core as core

TOL = 1e-4
GRAD_TOL = 2e-2
EPS = 1e-3

rng = random.Random(0)


def check_close(label, a, b, tol=TOL):
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < tol, f"{label}[{i}]: got={x:.6f} expected={y:.6f}"
    print(f"{label}: OK ({len(a)} elements, max diff {max(abs(x - y) for x, y in zip(a, b)):.2e})")


# ---------------------------------------------------------------------
# 1. sum(dim)/mean(dim)/max(dim): forward values, and backward -- with
#    max(dim) checked to have NO gradient, by design, not by omission.
# ---------------------------------------------------------------------

M = kansai.from_flat([1, 2, 3, 4, 5, 6], [2, 3])
check_close("sum(dim=0)", M.sum(0).tolist(), [5, 7, 9])
check_close("sum(dim=1)", M.sum(1).tolist(), [6, 15])
check_close("sum(dim=0, keepdim=True) shape", [float(d) for d in M.sum(0, True).shape], [1.0, 3.0])
check_close("mean(dim=0)", M.mean(0).tolist(), [2.5, 3.5, 4.5])
check_close("mean(dim=1)", M.mean(1).tolist(), [2.0, 5.0])
check_close("max(dim=0)", M.max(0).tolist(), [4, 5, 6])
check_close("max(dim=1)", M.max(1).tolist(), [3, 6])

Mg = kansai.from_flat([1, 2, 3, 4, 5, 6], [2, 3], requires_grad=True)
Mg.sum(1).sum().backward()
check_close("sum(dim=1) backward (all ones)", Mg.grad.tolist(), [1.0] * 6)

Mg2 = kansai.from_flat([1, 2, 3, 4, 5, 6], [2, 3], requires_grad=True)
Mg2.mean(1).sum().backward()
check_close("mean(dim=1) backward (1/3 each)", Mg2.grad.tolist(), [1 / 3.0] * 6)

Mg3 = kansai.from_flat([1, 2, 3, 4, 5, 6], [2, 3], requires_grad=True)
max_result = Mg3.max(1).sum()
assert not max_result.requires_grad, "max(dim) output must never require grad -- it's a deliberate stop-gradient"
print("max(dim) correctly has no gradient path: OK")

# ---------------------------------------------------------------------
# 2. softmax: sums to 1, matches a direct exp/sum computation at small
#    magnitudes, and stays finite (no NaN/Inf) at logit magnitudes that
#    would overflow float32's exp() without the max-subtraction trick
#    -- the entire reason this exists as more than "just exp then
#    divide".
# ---------------------------------------------------------------------

logits = kansai.from_flat([1, 2, 3], [1, 3])
sm = logits.softmax(1)
exp_vals = [math.exp(v) for v in [1, 2, 3]]
total = sum(exp_vals)
check_close("softmax forward", sm.tolist(), [v / total for v in exp_vals])
check_close("softmax sums to 1", [sum(sm.tolist())], [1.0], tol=1e-5)

big_logits = kansai.from_flat([1000.0, 1001.0, 1002.0], [1, 3])
sm_big = big_logits.softmax(1)
big_vals = sm_big.tolist()
assert all(v == v and abs(v) != float("inf") for v in big_vals), f"softmax produced NaN/Inf: {big_vals}"
check_close("softmax numerically stable on logits ~1000", [sum(big_vals)], [1.0], tol=1e-4)
print(f"softmax on logits [1000,1001,1002]: {[round(v, 6) for v in big_vals]} (finite, sums to 1)")

# ---------------------------------------------------------------------
# 3. cross_entropy against an independent from-scratch log-sum-exp
#    implementation in plain Python -- not kansai's own softmax().log()
#    composed a second time, which could share a bug with the real
#    implementation under test.
# ---------------------------------------------------------------------

def reference_cross_entropy(logits_flat, targets_flat, batch, classes):
    total_loss = 0.0
    for i in range(batch):
        row = logits_flat[i * classes:(i + 1) * classes]
        trow = targets_flat[i * classes:(i + 1) * classes]
        m = max(row)
        lse = m + math.log(sum(math.exp(v - m) for v in row))
        picked = sum(v * t for v, t in zip(row, trow))
        total_loss += lse - picked
    return total_loss / batch


logits2 = kansai.from_flat([2.0, 1.0, 0.1, 0.5, 2.5, 1.0], [2, 3])
targets2 = kansai.from_flat([1, 0, 0, 0, 1, 0], [2, 3])
ce = logits2.cross_entropy(targets2)
ref_ce = reference_cross_entropy([2.0, 1.0, 0.1, 0.5, 2.5, 1.0], [1, 0, 0, 0, 1, 0], 2, 3)
check_close("cross_entropy vs independent log-sum-exp reference", ce.tolist(), [ref_ce], tol=1e-4)

# backward vs central differences
logit_vals = [rng.uniform(-2, 2) for _ in range(6)]
target_vals = [1, 0, 0, 0, 0, 1]  # one-hot: row 0 -> class 0, row 1 -> class 2


def ce_loss(flat_logits):
    return kansai.from_flat(flat_logits, [2, 3]).cross_entropy(kansai.from_flat(target_vals, [2, 3])).tolist()[0]


analytical = []
for i in range(6):
    plus = list(logit_vals)
    plus[i] += EPS
    minus = list(logit_vals)
    minus[i] -= EPS
    analytical.append((ce_loss(plus) - ce_loss(minus)) / (2 * EPS))

logits_g = kansai.from_flat(logit_vals, [2, 3], requires_grad=True)
logits_g.cross_entropy(kansai.from_flat(target_vals, [2, 3])).backward()
check_close("cross_entropy backward vs central diff", logits_g.grad.tolist(), analytical, GRAD_TOL)

# ---------------------------------------------------------------------
# 4. Practical check: train a real 3-class classifier (2D points drawn
#    from three separated Gaussian blobs) with softmax + cross_entropy
#    + Adam to convergence, then check classification accuracy -- the
#    same "does it still actually work" bar test_xor.py/test_adam.py
#    already hold themselves to, now for classification.
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

# shuffle so classes aren't grouped in the batch (irrelevant for full-
# batch gradient descent, but matches how real data is never sorted)
combined = list(zip(xs, ys))
rng.shuffle(combined)
xs, ys = zip(*combined)

X = kansai.from_flat([v for row in xs for v in row], [len(xs), 2])
Y = kansai.from_flat([v for row in ys for v in row], [len(ys), num_classes])

model = nn.Sequential(
    nn.Linear(2, 16, seed=5),
    nn.ReLU(),
    nn.Linear(16, num_classes, seed=6),
)
opt = optim.Adam(model.parameters(), lr=0.05)

loss = None
for step in range(200):
    logits_out = model(X)
    loss = logits_out.cross_entropy(Y)
    model.zero_grad()
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.tolist()[0]:.6f}")

final_loss = loss.tolist()[0]
print(f"3-class classifier: final loss {final_loss:.6f}")
assert final_loss < 0.1, "3-class classifier did not converge"

# accuracy: argmax of the final logits vs the true one-hot class
final_logits = model(X).tolist()
targets_flat = Y.tolist()
correct = 0
for i in range(len(xs)):
    row = final_logits[i * num_classes:(i + 1) * num_classes]
    trow = targets_flat[i * num_classes:(i + 1) * num_classes]
    pred_class = row.index(max(row))
    true_class = trow.index(max(trow))
    correct += (pred_class == true_class)
accuracy = correct / len(xs)
print(f"3-class classifier: accuracy {accuracy:.1%} ({correct}/{len(xs)})")
assert accuracy > 0.95, f"3-class classifier accuracy too low: {accuracy:.1%}"

print("\nSoftmax/cross-entropy test passed.")
