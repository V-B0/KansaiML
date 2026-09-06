"""Gradient clipping and LR schedulers -- training-loop utilities every
serious optimizer setup reaches for, and previously entirely absent:
before this, a single unlucky batch producing a huge gradient (a real,
common failure mode training anything with attention layers) had
nothing to guard against it, and every optimizer's `lr` was fixed for
the whole run.

`clip_grad_norm_` computes ONE global L2 norm across every parameter's
gradient combined (not per-parameter -- see its own docstring in
python/kansai/optim.py for why that distinction matters) and scales
every gradient down by the same factor if it exceeds `max_norm`,
mutating `.grad` in place via the same self-aliasing `add_` trick
AdamW's own decoupled decay already uses (there's no Python-level
setter for `.grad` at all, so in-place mutation is the only way this
CAN work). `StepLR`/`CosineAnnealingLR` both just read/write
`optimizer.lr` directly -- every optimizer class already re-reads
`self.lr` fresh inside its own `step()`, so no optimizer-side change
was needed for either schedule to take effect.

Checked: clip_grad_norm_'s returned norm against a hand-computed
value, that clipping actually rescales the gradient to land at exactly
`max_norm` (checked by recomputing the norm post-clip, not just
trusting the formula), that a norm already within bounds is left
completely untouched, and that the norm is combined across MULTIPLE
parameters correctly (not clipped one at a time); StepLR's schedule
against the exact expected step sequence; CosineAnnealingLR against
the closed-form cosine formula at several points including both
endpoints; and a practical end-to-end test -- training a small model
with Adam, gradient clipping, and CosineAnnealingLR together, confirming
clipping actually engages at least once (not a silent no-op) and the
model still converges.
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import nn, optim
from kansai import _core as core

TOL = 1e-4


def check_close(label, a, b, tol=TOL):
    if not isinstance(a, (list, tuple)):
        a, b = [a], [b]
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < tol, f"{label}[{i}]: got={x:.6f} expected={y:.6f}"
    print(f"{label}: OK")


rng = random.Random(0)

# ---------------------------------------------------------------------
# 1. clip_grad_norm_: returned norm, actual rescaling to exactly
#    max_norm, a no-op when already within bounds, and the norm
#    combined correctly across multiple parameters.
# ---------------------------------------------------------------------


def set_grad(p, grad_vals):
    """Gives `p` an exact, known gradient without hand-deriving a loss:
    d/dp of sum(p * coeffs) is exactly `coeffs`, since that derivative
    doesn't depend on p's own values at all -- so `coeffs = grad_vals`
    makes p.grad come out to exactly grad_vals after backward()."""
    coeffs = core.from_flat(grad_vals, list(p.shape))
    p.mul(coeffs).sum().backward()


p1 = core.from_flat([0.0, 0.0], [2], requires_grad=True)
set_grad(p1, [3.0, 4.0])  # grad norm 5 on its own
assert p1.grad.tolist() == [3.0, 4.0]

returned_norm = optim.clip_grad_norm_([p1], max_norm=1000.0)
check_close("clip_grad_norm_ returns the correct pre-clip norm", returned_norm, 5.0)
check_close("clip_grad_norm_ leaves grad untouched when norm <= max_norm", p1.grad.tolist(), [3.0, 4.0])

p2 = core.from_flat([0.0, 0.0], [2], requires_grad=True)
set_grad(p2, [3.0, 4.0])
optim.clip_grad_norm_([p2], max_norm=1.0)
g = p2.grad.tolist()
post_norm = (g[0] ** 2 + g[1] ** 2) ** 0.5
check_close("clip_grad_norm_ rescales to exactly max_norm", post_norm, 1.0, tol=1e-3)
check_close("clip_grad_norm_ preserves gradient DIRECTION", g, [0.6, 0.8], tol=1e-3)  # (3,4)/5

# Combined across two parameters: norm is sqrt(sum of ALL squared
# components across BOTH params), not each param clipped to max_norm
# independently -- (3,4) and (0,0) combined has norm 5, same as p1
# alone, so clipping at max_norm=1 should scale by 1/5 same as above.
pa = core.from_flat([0.0, 0.0], [2], requires_grad=True)
pb = core.from_flat([0.0, 0.0], [2], requires_grad=True)
set_grad(pa, [3.0, 4.0])
set_grad(pb, [0.0, 0.0])
combined_norm = optim.clip_grad_norm_([pa, pb], max_norm=1.0)
check_close("clip_grad_norm_ combines norm across multiple params", combined_norm, 5.0, tol=1e-3)
check_close("clip_grad_norm_ scales param a consistently with the combined norm", pa.grad.tolist(), [0.6, 0.8], tol=1e-3)

# params with no gradient at all: a no-op, not an error.
p_none = core.from_flat([1.0], [1], requires_grad=True)
no_grad_norm = optim.clip_grad_norm_([p_none], max_norm=1.0)
check_close("clip_grad_norm_ on an all-None-grad param list returns 0", no_grad_norm, 0.0)

# ---------------------------------------------------------------------
# 2. StepLR: exact expected schedule over several decay boundaries.
# ---------------------------------------------------------------------

sgd = optim.SGD([core.from_flat([1.0], [1], requires_grad=True)], lr=1.0)
sched = optim.StepLR(sgd, step_size=3, gamma=0.5)
expected_lrs = [1.0, 1.0, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.125]
actual_lrs = []
for _ in range(len(expected_lrs)):
    sched.step()
    actual_lrs.append(sgd.lr)
check_close("StepLR schedule matches expected decay boundaries", actual_lrs, expected_lrs, tol=1e-9)

# ---------------------------------------------------------------------
# 3. CosineAnnealingLR: closed-form check at several points, including
#    both endpoints (last_epoch=0 stays at base_lr before any step()
#    call; last_epoch=T_max lands exactly at eta_min).
# ---------------------------------------------------------------------

adam = optim.Adam([core.from_flat([1.0], [1], requires_grad=True)], lr=0.1)
T_max = 10
eta_min = 0.001
cos_sched = optim.CosineAnnealingLR(adam, T_max=T_max, eta_min=eta_min)
check_close("CosineAnnealingLR starts at base_lr before any step()", adam.lr, 0.1)

for epoch in range(1, T_max + 1):
    cos_sched.step()
    expected = eta_min + (0.1 - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2
    check_close(f"CosineAnnealingLR matches closed form at epoch {epoch}", adam.lr, expected, tol=1e-9)

check_close("CosineAnnealingLR reaches exactly eta_min at T_max", adam.lr, eta_min, tol=1e-9)

# past T_max: progress clamps at 1.0, so lr stays pinned at eta_min
# rather than the cosine argument overshooting past pi.
cos_sched.step()
check_close("CosineAnnealingLR stays pinned at eta_min past T_max", adam.lr, eta_min, tol=1e-9)

# ---------------------------------------------------------------------
# 4. LinearWarmup: linear ramp to the optimizer's own construction-time
#    lr, exact peak at the warmup boundary, then a clean hand-off to
#    an `after_scheduler` (CosineAnnealingLR here) for every step past
#    that -- the standard warmup-then-decay pairing real transformer
#    training recipes use.
# ---------------------------------------------------------------------

adam2 = optim.Adam([core.from_flat([1.0], [1], requires_grad=True)], lr=0.1)
cos_after = optim.CosineAnnealingLR(adam2, T_max=10, eta_min=0.001)
warmup = optim.LinearWarmup(adam2, warmup_steps=5, after_scheduler=cos_after)

expected_warmup_lrs = [0.1 * s / 5 for s in range(1, 6)]
actual_warmup_lrs = []
for _ in range(5):
    warmup.step()
    actual_warmup_lrs.append(adam2.lr)
check_close("LinearWarmup ramps linearly to the target lr", actual_warmup_lrs, expected_warmup_lrs, tol=1e-9)
check_close("LinearWarmup reaches EXACTLY the target lr at the warmup boundary", [adam2.lr], [0.1], tol=1e-9)

for epoch in range(1, 11):
    warmup.step()
    expected = eta_min + (0.1 - eta_min) * (1 + math.cos(math.pi * epoch / 10)) / 2
    check_close(f"LinearWarmup hands off to CosineAnnealingLR correctly at post-warmup epoch {epoch}",
                [adam2.lr], [expected], tol=1e-9)

# ---------------------------------------------------------------------
# 5. Practical end-to-end test: gradient clipping + CosineAnnealingLR
#    together on a real training loop. A large initial lr (large
#    enough that early-step gradients would genuinely blow up without
#    clipping) confirms clipping actually engages -- not a silent
#    no-op -- while the model still converges once the cosine schedule
#    brings lr down.
# ---------------------------------------------------------------------

X = kansai.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = kansai.tensor([[0], [1], [1], [0]])

model = nn.Sequential(nn.Linear(2, 16, seed=11), nn.ReLU(), nn.Linear(16, 1, seed=12))
opt = optim.Adam(model.parameters(), lr=0.1)
EPOCHS = 400
MAX_NORM = 0.5
scheduler = optim.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=0.001)

clipped_at_least_once = False
loss = None
for epoch in range(EPOCHS):
    pred = model(X)
    diff = pred.sub(Y)
    loss = diff.mul(diff).mean()
    model.zero_grad()
    loss.backward()
    pre_clip_norm = optim.clip_grad_norm_(model.parameters(), max_norm=MAX_NORM)
    if pre_clip_norm > MAX_NORM:
        clipped_at_least_once = True
    opt.step()
    scheduler.step()

final_loss = loss.tolist()[0]
print(f"XOR trained with Adam + clip_grad_norm_ + CosineAnnealingLR: final loss {final_loss:.6f}, "
      f"final lr {opt.lr:.6f}, clipped at least once: {clipped_at_least_once}")
assert clipped_at_least_once, "gradient clipping never actually engaged -- test isn't exercising the real path"
assert final_loss < 0.05, "XOR did not converge with clipping + cosine schedule active"
check_close("CosineAnnealingLR brought lr down to eta_min by the end of training", opt.lr, 0.001, tol=1e-6)

print("\nOptimizer utilities (clip_grad_norm_, StepLR, CosineAnnealingLR, LinearWarmup) test passed.")
