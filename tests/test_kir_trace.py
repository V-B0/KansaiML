import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import nn, optim, kir

# ---------------------------------------------------------------------
# 1. trace + replay must match eager exactly, on the same MLP shape used
#    for the Milestone 0 XOR test.
# ---------------------------------------------------------------------

model = nn.Sequential(nn.Linear(2, 8, seed=3), nn.ReLU(), nn.Linear(8, 1, seed=4))
X = kansai.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])

eager_out = model(X).tolist()

jit_forward = kir.jit(lambda x: model(x))
jit_out = jit_forward(X).tolist()

assert eager_out == jit_out, f"mismatch: eager={eager_out} jit={jit_out}"
print("trace/replay matches eager:", jit_out)

graph = jit_forward.trace(X)
print("\ntraced graph (Linear -> ReLU -> Linear):")
print(graph)

# ---------------------------------------------------------------------
# 2. dead_code_elimination must drop an unused branch and leave the
#    numeric result unchanged.
# ---------------------------------------------------------------------

def fn_with_dead_branch(x, y):
    unused = x.mul(x)      # dead: never reaches the output
    live = x.add(y)
    return live.relu()

a = kansai.tensor([[1.0, -2.0]])
b = kansai.tensor([[0.5, 0.5]])

g = kir.trace(fn_with_dead_branch, a, b)
before = len(g.nodes)
g_opt = kir.dead_code_elimination(g)
after = len(g_opt.nodes)

print(f"\nDCE: {before} nodes -> {after} nodes")
assert after < before, "DCE should have removed the dead mul node"

out_before = kir.run(g, a, b).tolist()
out_after = kir.run(g_opt, a, b).tolist()
assert out_before == out_after, "DCE changed the result"
print("DCE preserves output:", out_after)

# ---------------------------------------------------------------------
# 3. DCE must never drop a placeholder, even an entirely unused one --
#    dropping it would silently change the graph's calling convention.
# ---------------------------------------------------------------------

def fn_ignores_y(x, y):
    return x.relu()

g2 = kir.trace(fn_ignores_y, a, b)
g2_opt = kir.dead_code_elimination(g2)
assert len(g2_opt.inputs) == 2, "DCE must not drop an unused placeholder argument"
out2 = kir.run(g2_opt, a, b).tolist()
print("unused placeholder argument preserved through DCE; output:", out2)

# ---------------------------------------------------------------------
# 4. The real test: train through the jit'd forward pass instead of
#    calling model(x) directly, and confirm it converges exactly like
#    the Milestone 0 XOR test. This works because run() dispatches back
#    to the real (autograd-tracked) Tensor ops in traced order, so
#    backward() and the optimizer see the exact same graph they would in
#    eager mode -- proving trace -> IR -> replay is sound across an
#    entire training loop, not just a single forward call.
# ---------------------------------------------------------------------

Y = kansai.tensor([[0], [1], [1], [0]])
# Seeded for the same reason as test_xor.py: lr=0.5 lets "dying ReLU"
# permanently kill every hidden unit on an unlucky init; seeds=(3,4) with
# lr=0.1 is verified to converge cleanly and deterministically.
train_model = nn.Sequential(nn.Linear(2, 8, seed=3), nn.ReLU(), nn.Linear(8, 1, seed=4))
train_forward = kir.jit(lambda x: train_model(x))
opt = optim.SGD(train_model.parameters(), lr=0.1)

loss = None
for step in range(500):
    pred = train_forward(X)
    diff = pred.sub(Y)
    loss = diff.mul(diff).mean()

    train_model.zero_grad()
    loss.backward()
    opt.step()

    if step % 100 == 0:
        print(f"step {step:4d}  loss {loss.tolist()[0]:.6f}")

final_loss = loss.tolist()[0]
print(f"final loss (via jit'd forward): {final_loss:.6f}")
assert final_loss < 0.05, "XOR did not converge through the jit'd forward pass"

print("\nKIR Phase 2 smoke test passed.")
