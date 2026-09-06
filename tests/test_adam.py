"""Adam (Kingma & Ba, 2014) -- see python/kansai/optim.py's own
docstring for what's implemented (the original paper's algorithm, no
weight decay -- AdamW, a separate class, has that) and why it was
implemented in pure Python over existing Tensor ops rather than a
dedicated kernel.

Two independent checks: a step-by-step comparison against a pure-Python
re-implementation of Adam's update rule (no kansai.Tensor calls at all,
just floats and lists -- a self-consistency check against the SAME
Tensor ops wouldn't catch a bug shared by both, so this is deliberately
built from scratch against the formula in the original paper), and a
practical check that Adam actually trains a real model -- the same "does
it still actually work" bar test_xor.py's own SGD-trained model is held
to, run here with Adam instead to prove it's a genuine drop-in optimizer,
not just a formula that happens to type-check.
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


class ReferenceAdam:
    """Adam's update rule from the original paper, in plain Python --
    no kansai.Tensor anywhere, so this can't share a bug with the
    implementation under test."""

    def __init__(self, n, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.n = n
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [0.0] * n
        self.v = [0.0] * n

    def step(self, params, grads):
        self.t += 1
        bc1 = 1 - self.beta1 ** self.t
        bc2 = 1 - self.beta2 ** self.t
        for i in range(self.n):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            params[i] -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)
        return params


# ---------------------------------------------------------------------
# 1. Step-by-step comparison against the independent reference, across
#    several steps (so bias correction and the running averages'
#    accumulation over time are both actually exercised, not just a
#    single step where m/v start at zero and the difference between a
#    subtly-wrong and a correct implementation might not show up yet).
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

param = kansai.from_flat(init_vals, [n], requires_grad=True)
opt = optim.Adam([param], lr=0.01)
ref = ReferenceAdam(n, lr=0.01)
ref_params = list(init_vals)

for step, grads in enumerate(grad_sequence):
    # Feed the same known gradient sequence into Kansai's Adam without
    # re-deriving autograd (already covered elsewhere): a synthetic loss
    # sum(grads[i] * param[i]) has exactly `grads` as its own gradient
    # wrt param, since d/dparam[i] of that sum is just grads[i].
    param.zero_grad()
    coeffs = kansai.from_flat(grads, [n])
    loss = param.mul(coeffs).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()

    ref_params = ref.step(ref_params, grads)
    check_close(f"Adam step {step + 1} vs independent reference", param.tolist(), ref_params)

# ---------------------------------------------------------------------
# 2. Practical check: Adam actually trains the same XOR model
#    test_xor.py trains with SGD, converging to the same near-zero loss
#    bar -- proving this is a working drop-in optimizer on a real model,
#    not just a formula that matches in isolation.
# ---------------------------------------------------------------------

X = kansai.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = kansai.tensor([[0], [1], [1], [0]])

model = nn.Sequential(
    nn.Linear(2, 8, seed=3),
    nn.ReLU(),
    nn.Linear(8, 1, seed=4),
)
adam = optim.Adam(model.parameters(), lr=0.05)

loss = None
for step in range(500):
    pred = model(X)
    diff = pred.sub(Y)
    loss = diff.mul(diff).mean()
    model.zero_grad()
    loss.backward()
    adam.step()

final_loss = loss.tolist()[0]
print(f"XOR trained with Adam: final loss {final_loss:.6f}")
print("predictions:", [round(v, 3) for v in model(X).tolist()])
assert final_loss < 0.05, "XOR did not converge with Adam"
print("XOR converged with Adam.")

print("\nAdam test passed.")
