import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import nn, optim

X = kansai.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = kansai.tensor([[0], [1], [1], [0]])

# Seeded rather than left to draw from random_device: with ReLU + plain
# SGD, an unlucky init can push every hidden unit permanently negative
# within ~20 steps ("dying ReLU"), which then gets stuck predicting the
# mean forever -- lr=0.5 hit this on roughly half of random inits when
# checked over a 20-trial sweep. lr=0.1 plus these two seeds is verified
# (by that same sweep) to converge cleanly and, being seeded, always
# reproduces the same run rather than passing or failing by chance.
model = nn.Sequential(
    nn.Linear(2, 8, seed=3),
    nn.ReLU(),
    nn.Linear(8, 1, seed=4),
)

opt = optim.SGD(model.parameters(), lr=0.1)

loss = None
for step in range(500):
    pred = model(X)
    diff = pred.sub(Y)
    loss = diff.mul(diff).mean()

    model.zero_grad()
    loss.backward()
    opt.step()

    if step % 50 == 0:
        print(f"step {step:4d}  loss {loss.tolist()[0]:.6f}")

final_loss = loss.tolist()[0]
print(f"final loss: {final_loss:.6f}")
print("predictions:", [round(v, 3) for v in model(X).tolist()])

assert final_loss < 0.05, "XOR did not converge"
print("XOR converged.")
