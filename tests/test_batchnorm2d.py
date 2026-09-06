"""BatchNorm2d -- see python/kansai/nn.py's own docstring for the
transpose+reshape trick (`(N,C,H,W)` -> `(N,H,W,C)` -> `(N*H*W,C)`,
BatchNorm1d's own forward completely unmodified, then reversed) that
closes the gap BatchNorm1d's own docstring used to flag as real,
unattempted future work.

A genuinely subtle pitfall this test exists specifically to document
and avoid: `.sum()` of a BatchNorm's output is a mathematical
IDENTITY, constant with respect to the input, regardless of what the
input actually is -- normalization zero-centers each channel, so
`sum_i (x_i - mean)/std` is algebraically exactly 0 for ANY x with
nonzero variance (`sum_i x_i - n*mean = 0`, always). Sum-of-SQUARES
is the same trap one level up: normalizing to unit variance makes
`sum_i normalized_i^2` exactly `n` (the population-variance identity),
also constant in x. Both being constant means their GRADIENT with
respect to x is also exactly zero -- true, correct, but a completely
uninformative check: an implementation with a real backward bug and a
correct one would both pass a central-difference check built on either
loss (both sides read ~0). This test's gradient check instead uses a
position-WEIGHTED sum (different, fixed, non-uniform weights on each
element before summing) -- breaking the symmetry that made sum/
sum-of-squares degenerate, since it depends on more than just the
normalized distribution's first and second moments -- which is what
actually exercises the real gradient path.

Checked: forward (training mode) against a hand-computed manual
per-CHANNEL mean/variance over N, H, AND W jointly (not just N -- the
actual thing BatchNorm2d has to get right that BatchNorm1d's own
per-column-over-batch-only case doesn't exercise); backward against
central differences using the position-weighted loss above (a fresh
BatchNorm2d instance per perturbed evaluation, with `weight`/`bias`
copied over -- reusing one instance would let one evaluation's
running-stats update contaminate the next); eval mode using the
running statistics rather than a fresh batch's; and, practically, that
a small `Conv2d` -> `BatchNorm2d` -> `ReLU` -> `Linear` network
actually trains a real classification task to convergence, the same
"does it still actually work" bar every layer addition in this project
is held to.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import nn, optim
from kansai import _core as core

EPS = 1e-3
GRAD_TOL = 2e-2
TOL = 1e-4

rng = random.Random(0)


def check_close(label, a, b, tol=TOL):
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < tol, f"{label}[{i}]: got={x:.6f} expected={y:.6f}"
    print(f"{label}: OK ({len(a)} elements, max diff {max(abs(x - y) for x, y in zip(a, b)):.2e})")


# ---------------------------------------------------------------------
# 1. Forward (training mode) vs. a manual per-channel mean/var computed
#    over N, H, AND W jointly.
# ---------------------------------------------------------------------

N, C, H, W = 4, 3, 5, 5
x_vals = [rng.uniform(-2, 2) for _ in range(N * C * H * W)]
x = core.from_flat(x_vals, [N, C, H, W])

bn = nn.BatchNorm2d(C)
out = bn(x).tolist()


def xget(n, c, h, w):
    return x_vals[((n * C + c) * H + h) * W + w]


def oget(vals, n, c, h, w):
    return vals[((n * C + c) * H + h) * W + w]


EPS_NORM = 1e-5
expected = [0.0] * (N * C * H * W)
for c in range(C):
    vals = [xget(n, c, h, w) for n in range(N) for h in range(H) for w in range(W)]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    denom = (var + EPS_NORM) ** 0.5
    for n in range(N):
        for h in range(H):
            for w in range(W):
                idx = ((n * C + c) * H + h) * W + w
                expected[idx] = (xget(n, c, h, w) - mean) / denom  # weight=1, bias=0 by default

check_close("BatchNorm2d forward (training) vs manual per-channel (N,H,W) normalization", out, expected)

# ---------------------------------------------------------------------
# 2. Backward vs. central differences, using a position-weighted loss
#    (see this file's own docstring for why plain sum()/sum-of-squares
#    would be a degenerate, uninformative check here).
# ---------------------------------------------------------------------

Ng, Cg, Hg, Wg = 2, 3, 2, 2
xg_vals = [rng.uniform(-2, 2) for _ in range(Ng * Cg * Hg * Wg)]
xg = core.from_flat(xg_vals, [Ng, Cg, Hg, Wg], requires_grad=True)
posw_vals = [rng.uniform(0.5, 2.0) for _ in range(Ng * Cg * Hg * Wg)]
posw = core.from_flat(posw_vals, [Ng, Cg, Hg, Wg])

bn_g = nn.BatchNorm2d(Cg)
bn_weight_vals = bn_g.weight.tolist()
bn_bias_vals = bn_g.bias.tolist()

loss = bn_g(xg).mul(posw).sum()
loss.backward()
analytic = xg.grad.tolist()


def eval_loss(vals):
    xt = core.from_flat(vals, [Ng, Cg, Hg, Wg])
    bn2 = nn.BatchNorm2d(Cg)
    bn2.weight = core.from_flat(bn_weight_vals, [Cg])
    bn2.bias = core.from_flat(bn_bias_vals, [Cg])
    return bn2(xt).mul(posw).sum().tolist()[0]


central_diff = []
for i in range(len(xg_vals)):
    plus = list(xg_vals)
    plus[i] += EPS
    minus = list(xg_vals)
    minus[i] -= EPS
    central_diff.append((eval_loss(plus) - eval_loss(minus)) / (2 * EPS))

check_close("BatchNorm2d backward vs central diff (position-weighted loss)", analytic, central_diff, GRAD_TOL)

# ---------------------------------------------------------------------
# 3. Eval mode uses running statistics, not the eval batch's own --
#    confirmed with a wildly out-of-distribution input after training,
#    the same check test_normalization.py's own BatchNorm1d test uses.
# ---------------------------------------------------------------------

bn_eval = nn.BatchNorm2d(2, momentum=0.5)
train_vals = [rng.uniform(-1, 1) for _ in range(3 * 2 * 4 * 4)]
bn_eval(core.from_flat(train_vals, [3, 2, 4, 4]))  # one training-mode call folds real stats in
bn_eval.eval()

ood_vals = [50.0] * (1 * 2 * 4 * 4)  # wildly out-of-distribution: a fresh batch's own stats would be ~0 variance
out_ood = bn_eval(core.from_flat(ood_vals, [1, 2, 4, 4])).tolist()
assert any(abs(v) > 0.5 for v in out_ood), (
    "eval mode must normalize by the RUNNING statistics, not a fresh (near-zero-variance) batch's own")
print("BatchNorm2d eval mode uses running statistics, not the eval batch's own: OK")

# ---------------------------------------------------------------------
# 4. Practical: Conv2d -> BatchNorm2d -> ReLU -> Linear actually trains
#    a real classification task to convergence.
# ---------------------------------------------------------------------

rng2 = random.Random(7)


def make_example():
    # Two well-separated Gaussian-ish blobs over a 1x6x6 image, labeled
    # by which blob the image belongs to -- the same style of
    # synthetic-but-real task test_conv2d.py's own practical check uses.
    label = rng2.randint(0, 1)
    center = 2.0 if label == 1 else -2.0
    pixels = [center + rng2.uniform(-0.5, 0.5) for _ in range(6 * 6)]
    return pixels, label


examples = [make_example() for _ in range(200)]
X = kansai.tensor([p for p, _ in examples])
X = X.reshape([200, 1, 6, 6])
Y = kansai.tensor([[1.0, 0.0] if lbl == 0 else [0.0, 1.0] for _, lbl in examples])

conv = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1, seed=3)
bnorm = nn.BatchNorm2d(4)
fc = nn.Linear(4 * 6 * 6, 2, seed=4)


class ConvBNNet(nn.Module):
    def __init__(self):
        self.conv = conv
        self.bn = bnorm
        self.relu = nn.ReLU()
        self.fc = fc

    def forward(self, x):
        h = self.relu(self.bn(self.conv(x)))
        bsz, c, hh, ww = h.shape
        return self.fc(h.reshape([bsz, c * hh * ww]))


model = ConvBNNet()
opt = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(150):
    logits = model(X)
    loss = logits.cross_entropy(Y)
    model.zero_grad()
    loss.backward()
    opt.step()

final_loss = loss.tolist()[0]
model.eval()
preds = model(X).tolist()
correct = sum(1 for i in range(200) if (preds[i * 2] > preds[i * 2 + 1]) == (examples[i][1] == 0))
accuracy = correct / 200
print(f"Conv2d->BatchNorm2d->ReLU->Linear trained: final loss {final_loss:.4f}, accuracy {accuracy:.2%}")
assert accuracy >= 0.95, f"expected >=95% accuracy, got {accuracy:.2%}"

print("\nBatchNorm2d test passed.")
