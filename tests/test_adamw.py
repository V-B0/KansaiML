"""AdamW (Loshchilov & Hutter, 2019) -- Adam with DECOUPLED weight
decay. See python/kansai/optim.py's own docstring for the real
distinction from L2 regularization (which this project doesn't
implement either): L2 folds weight_decay*p into the gradient itself,
so it gets divided by the second-moment estimate along with everything
else; decoupled decay shrinks the parameter directly and completely
outside that machinery, `p *= (1 - lr*weight_decay)`.

Two checks, chosen to isolate exactly what's new here (the decay term)
from what AdamW inherits unchanged (Adam's own moment bookkeeping,
already checked step-by-step against an independent reference in
test_adam.py): (1) with weight_decay=0, AdamW.step() must produce the
IDENTICAL parameter trajectory to plain Adam.step() given the same
gradients -- confirming the subclass didn't change anything about the
inherited update; (2) with a gradient that's an honest, real Tensor
but analytically exactly zero everywhere (so Adam's own update
contributes exactly nothing -- m and v both stay at zero, so
m_hat/(sqrt(v_hat)+eps) = 0/eps = 0), decay is isolated as the ONLY
thing moving the parameter at all, checked against the closed-form
`p_0 * (1 - lr*weight_decay)^steps` after N steps. Finally, a practical
check: AdamW trains the same XOR model test_adam.py trains with plain
Adam, to the same convergence bar, confirming it's a genuine drop-in
optimizer and not just a formula that happens to type-check.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import nn, optim

TOL = 1e-4


def check_close(label, a, b, tol=TOL):
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < tol, f"{label}[{i}]: got={x:.8f} expected={y:.8f}"
    print(f"{label}: OK ({len(a)} elements, max diff {max(abs(x - y) for x, y in zip(a, b)):.2e})")


# ---------------------------------------------------------------------
# 1. weight_decay=0 must match plain Adam exactly, step by step -- the
#    subclass's extra decay step is a no-op at wd=0, so nothing else
#    about the inherited update should have changed.
# ---------------------------------------------------------------------

n = 5
init_vals = [0.5, -1.2, 3.0, -0.3, 2.1]
grad_sequence = [
    [0.1, -0.2, 0.05, 0.3, -0.1],
    [0.08, -0.15, 0.02, 0.25, -0.05],
    [-0.05, 0.1, -0.1, 0.2, 0.15],
    [0.2, 0.05, -0.3, -0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.1],
]

param_adam = kansai.from_flat(init_vals, [n], requires_grad=True)
param_adamw = kansai.from_flat(init_vals, [n], requires_grad=True)
opt_adam = optim.Adam([param_adam], lr=0.01)
opt_adamw = optim.AdamW([param_adamw], lr=0.01, weight_decay=0.0)

for step, grads in enumerate(grad_sequence):
    coeffs = kansai.from_flat(grads, [n])

    param_adam.zero_grad()
    param_adam.mul(coeffs).sum().backward()
    opt_adam.step()

    param_adamw.zero_grad()
    param_adamw.mul(coeffs).sum().backward()
    opt_adamw.step()

    check_close(f"AdamW(weight_decay=0) step {step + 1} matches plain Adam exactly",
                param_adamw.tolist(), param_adam.tolist())

# ---------------------------------------------------------------------
# 2. Isolate the decay term: a gradient that's a real Tensor (not None
#    -- decay only applies when p.grad is not None) but analytically
#    exactly zero everywhere, so Adam's own update contributes nothing
#    and decay is the ONLY thing moving the parameter. Checked against
#    the closed-form p_0 * (1 - lr*weight_decay)^steps.
# ---------------------------------------------------------------------

p0 = 4.0
lr, wd = 0.1, 0.2
p = kansai.from_flat([p0], [1], requires_grad=True)
opt = optim.AdamW([p], lr=lr, weight_decay=wd)

STEPS = 10
for step in range(STEPS):
    p.zero_grad()
    zero_coeff = kansai.from_flat([0.0], [1])
    p.mul(zero_coeff).sum().backward()  # grad is a real zero Tensor, not None
    assert p.grad is not None and p.grad.tolist() == [0.0]
    opt.step()

expected = p0 * (1.0 - lr * wd) ** STEPS
check_close(f"AdamW decoupled decay in isolation ({STEPS} steps, zero gradient)", p.tolist(), [expected], tol=1e-5)

# weight_decay=0 must leave a zero-gradient parameter completely
# unchanged (no decay applied, and the Adam update itself is a no-op
# for an all-zero gradient).
p_nodecay = kansai.from_flat([p0], [1], requires_grad=True)
opt_nodecay = optim.AdamW([p_nodecay], lr=lr, weight_decay=0.0)
for step in range(STEPS):
    p_nodecay.zero_grad()
    p_nodecay.mul(kansai.from_flat([0.0], [1])).sum().backward()
    opt_nodecay.step()
check_close("AdamW(weight_decay=0) leaves a zero-gradient parameter unchanged",
            p_nodecay.tolist(), [p0], tol=1e-6)

# ---------------------------------------------------------------------
# 3. Practical check: AdamW trains the same XOR model test_adam.py
#    trains with plain Adam, to the same convergence bar.
# ---------------------------------------------------------------------

X = kansai.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = kansai.tensor([[0], [1], [1], [0]])

model = nn.Sequential(
    nn.Linear(2, 8, seed=3),
    nn.ReLU(),
    nn.Linear(8, 1, seed=4),
)
adamw = optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.001)

loss = None
for step in range(500):
    pred = model(X)
    diff = pred.sub(Y)
    loss = diff.mul(diff).mean()
    model.zero_grad()
    loss.backward()
    adamw.step()

final_loss = loss.tolist()[0]
print(f"XOR trained with AdamW: final loss {final_loss:.6f}")
print("predictions:", [round(v, 3) for v in model(X).tolist()])
assert final_loss < 0.05, "XOR did not converge with AdamW"
print("XOR converged with AdamW.")

print("\nAdamW test passed.")
