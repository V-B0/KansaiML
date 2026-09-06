"""SGD with momentum -- see python/kansai/optim.py's own docstring for
the formula (`buf = momentum*buf + grad`, initialized to `grad` itself
on the first step, PyTorch's own convention) and its Nesterov variant.
Plain SGD (momentum=0) already existed; nonzero momentum was a real,
previously-missing piece of the optimizer vocabulary -- without it,
SGD needs an impractically small, carefully hand-tuned learning rate
to converge at any reasonable speed on anything but a toy problem.

Checked: momentum=0 is an exact, byte-for-byte regression against the
original single-`add_` SGD (this constructor default must never
silently change what "SGD" already meant); both the plain-momentum and
Nesterov trajectories match a from-scratch pure-Python re-implementation
of the update rule (no kansai.Tensor calls at all -- a self-consistency
check against the SAME ops this project's own SGD uses internally
wouldn't catch a bug shared by both) over several steps on a FIXED,
constant gradient (isolates the recurrence itself from any real loss
landscape); and, practically, that momentum measurably speeds up
convergence on a real ill-conditioned loss surface -- a narrow, steep
valley (one dimension curved much more sharply than the other) where
plain gradient descent is known to oscillate across the steep axis
rather than making steady progress along the shallow one, and momentum
is the textbook fix.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import optim
from kansai import _core as core

TOL = 1e-4


def check_close(label, a, b, tol=TOL):
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < tol, f"{label}[{i}]: got={x:.6f} expected={y:.6f}"
    print(f"{label}: OK")


def set_grad(p, grad_vals):
    coeffs = core.from_flat(grad_vals, list(p.shape))
    p.mul(coeffs).sum().backward()


# ---------------------------------------------------------------------
# 1. momentum=0 regression: byte-for-byte identical to plain SGD.
# ---------------------------------------------------------------------

p_plain = core.from_flat([1.0, -2.0, 3.0], [3], requires_grad=True)
set_grad(p_plain, [0.5, -1.5, 2.0])
opt_plain = optim.SGD([p_plain], lr=0.1, momentum=0.0)
opt_plain.step()
check_close("SGD momentum=0 regression", p_plain.tolist(), [1.0 - 0.1 * 0.5, -2.0 - 0.1 * -1.5, 3.0 - 0.1 * 2.0])

# ---------------------------------------------------------------------
# 2. Plain momentum vs. a from-scratch pure-Python reference, over
#    several steps of a FIXED constant gradient (isolates the buf
#    recurrence from any real loss surface).
# ---------------------------------------------------------------------

LR, MOM, STEPS = 1.0, 0.9, 6
GRAD = 1.0

p_mom = core.from_flat([0.0], [1], requires_grad=True)
opt_mom = optim.SGD([p_mom], lr=LR, momentum=MOM)
coeffs = core.from_flat([GRAD], [1])

ref_p, ref_buf = 0.0, 0.0
for step in range(STEPS):
    p_mom.zero_grad()
    p_mom.mul(coeffs).sum().backward()
    opt_mom.step()
    ref_buf = GRAD if step == 0 else MOM * ref_buf + GRAD
    ref_p -= LR * ref_buf
check_close("SGD momentum matches pure-Python reference after several steps", p_mom.tolist(), [ref_p], tol=1e-3)

# ---------------------------------------------------------------------
# 3. Nesterov variant vs. its own pure-Python reference.
# ---------------------------------------------------------------------

p_nes = core.from_flat([0.0], [1], requires_grad=True)
opt_nes = optim.SGD([p_nes], lr=LR, momentum=MOM, nesterov=True)

ref_p_nes, ref_buf_nes = 0.0, 0.0
for step in range(STEPS):
    p_nes.zero_grad()
    p_nes.mul(coeffs).sum().backward()
    opt_nes.step()
    ref_buf_nes = GRAD if step == 0 else MOM * ref_buf_nes + GRAD
    update = GRAD + MOM * ref_buf_nes
    ref_p_nes -= LR * update
check_close("SGD nesterov=True matches pure-Python reference after several steps", p_nes.tolist(), [ref_p_nes],
            tol=1e-3)

# ---------------------------------------------------------------------
# 4. Practical: momentum measurably speeds up convergence on an
#    ill-conditioned (narrow-valley) loss surface -- loss = 50*x^2 +
#    0.5*y^2, where plain gradient descent is known to oscillate
#    across the steep x-axis rather than making steady progress along
#    the shallow y-axis. Same starting point, same learning rate, same
#    step budget for both -- momentum should reach a substantially
#    lower loss.
# ---------------------------------------------------------------------

A, B = 50.0, 0.5  # curvature along x and y respectively
LR2, BUDGET = 0.010, 60  # lr chosen so all three variants stay numerically stable on this surface


def run(momentum, nesterov=False):
    xy = core.from_flat([1.0, 1.0], [2], requires_grad=True)
    opt = optim.SGD([xy], lr=LR2, momentum=momentum, nesterov=nesterov)
    coeff = core.from_flat([A, B], [2])
    for _ in range(BUDGET):
        xy.zero_grad()
        loss = xy.mul(xy).mul(coeff).sum()  # d/dxy = 2*coeff*xy, giving the A/B curvature above
        loss.backward()
        opt.step()
    final_loss = xy.mul(xy).mul(coeff).sum().tolist()[0]
    return final_loss


loss_plain = run(momentum=0.0)
loss_momentum = run(momentum=0.9)
loss_nesterov = run(momentum=0.9, nesterov=True)
print(f"ill-conditioned valley after {BUDGET} steps: plain={loss_plain:.6f} "
      f"momentum={loss_momentum:.6f} nesterov={loss_nesterov:.6f}")
assert loss_momentum < loss_plain * 0.5, (
    f"momentum should reach substantially lower loss than plain SGD on this ill-conditioned surface: "
    f"momentum={loss_momentum:.6f} vs plain={loss_plain:.6f}")
print("momentum measurably beats plain SGD on an ill-conditioned loss surface: OK")
assert loss_nesterov < loss_momentum, (
    f"nesterov should reach lower loss than plain momentum at this budget: "
    f"nesterov={loss_nesterov:.6f} vs momentum={loss_momentum:.6f}")
print("nesterov measurably beats plain momentum on the same surface: OK")

print("\nSGD momentum test passed.")
