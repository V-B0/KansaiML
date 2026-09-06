"""Conv1d -- see python/kansai/nn.py's own docstring for the mechanism
(1D convolution is exactly 2D convolution with the height axis pinned
at 1, so this reuses Conv2d's own kernel/backward/KIR support entirely
via reshape, with manual zero-padding on the length axis handled
before calling conv2d with padding=0 -- Conv2d's own single-scalar
padding argument would otherwise pad the dummy height axis too and
silently grow a second spatial dimension this class never asked for).

The one place this needed real, deliberate handling rather than
falling out "for free" the way AvgPool2d/LayerNorm did: the manual
`cat` used to pad the length axis is module-level and n-ary, so it
does NOT participate in the automatic real-Tensor/TraceValue coercion
every BINARY op (add/sub/mul/div) gets via TraceValue._coerce --
forward() branches explicitly on which world `x` is in.

Checked: forward against a direct nested-loop reference (independent
of Conv2d's own implementation, catching a reshape/index bug a
self-consistency check against Conv2d couldn't); backward against
central differences, and that weight/bias actually receive nonzero
gradients; the full KIR path -- trace, all four interpreters, and
kir.grad -- for BOTH the padding>0 branch (exercising the cat-based
padding under tracing specifically) and the padding=0 branch;
run_metal (the Metal-fallback conv2d path underneath it); and,
practically, a small Conv1d-based sequence classifier trained to
convergence.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import nn, optim, kir
from kansai import _core as core

EPS = 1e-3
TOL_FWD = 1e-4
TOL_GRAD = 2e-2


def check_close(label, a, b, tol=TOL_FWD):
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < tol, f"{label}[{i}]: got={x:.6f} expected={y:.6f}"
    print(f"{label}: OK ({len(a)} elements, max diff {max(abs(x - y) for x, y in zip(a, b)):.2e})")


rng = random.Random(0)


def naive_conv1d(x, w, b, Cin, L, Cout, K, stride, padding):
    Lp = L + 2 * padding
    xp = [0.0] * (Cin * Lp)
    for c in range(Cin):
        for i in range(L):
            xp[c * Lp + padding + i] = x[c * L + i]
    Lout = (Lp - K) // stride + 1
    out = [0.0] * (Cout * Lout)
    for co in range(Cout):
        for o in range(Lout):
            s = b[co]
            for ci in range(Cin):
                for k in range(K):
                    s += w[((co * Cin + ci) * K) + k] * xp[ci * Lp + o * stride + k]
            out[co * Lout + o] = s
    return out, Lout


# ---------------------------------------------------------------------
# 1. Forward vs. a direct nested-loop reference (deliberately
#    independent of Conv2d's own implementation).
# ---------------------------------------------------------------------

Cin, L, Cout, K, stride, padding = 3, 10, 4, 3, 2, 1
conv1d = nn.Conv1d(Cin, Cout, K, stride=stride, padding=padding, seed=1)
x_vals = [rng.uniform(-1, 1) for _ in range(Cin * L)]
x = core.from_flat(x_vals, [1, Cin, L])
out = conv1d(x)

w_vals = conv1d.weight.tolist()
b_vals = conv1d.bias.tolist()
expected, Lout = naive_conv1d(x_vals, w_vals, b_vals, Cin, L, Cout, K, stride, padding)
assert list(out.shape) == [1, Cout, Lout], f"shape mismatch: {list(out.shape)} vs expected Lout={Lout}"
check_close("Conv1d forward vs naive nested-loop reference", out.tolist(), expected)

# ---------------------------------------------------------------------
# 2. Backward vs. central differences (input, weight, and bias all
#    receive real, nonzero gradients).
# ---------------------------------------------------------------------

Cin2, L2, Cout2, K2, stride2, padding2 = 2, 7, 3, 3, 1, 1
conv1d_g = nn.Conv1d(Cin2, Cout2, K2, stride=stride2, padding=padding2, seed=3)
xg_vals = [rng.uniform(-1, 1) for _ in range(Cin2 * L2)]
xg = core.from_flat(xg_vals, [1, Cin2, L2], requires_grad=True)
loss = conv1d_g(xg).sum()
loss.backward()
grad_x = xg.grad.tolist()


def eval_loss(vals):
    xt = core.from_flat(vals, [1, Cin2, L2])
    return conv1d_g(xt).sum().tolist()[0]


central_diff = []
for i in range(len(xg_vals)):
    plus = list(xg_vals)
    plus[i] += EPS
    minus = list(xg_vals)
    minus[i] -= EPS
    central_diff.append((eval_loss(plus) - eval_loss(minus)) / (2 * EPS))
check_close("Conv1d backward grad_x vs central diff", grad_x, central_diff, TOL_GRAD)

assert conv1d_g.weight.grad is not None and any(abs(v) > 1e-6 for v in conv1d_g.weight.grad.tolist()), \
    "Conv1d weight must receive a real, nonzero gradient"
assert conv1d_g.bias.grad is not None and any(abs(v) > 1e-6 for v in conv1d_g.bias.grad.tolist()), \
    "Conv1d bias must receive a real, nonzero gradient"
print("Conv1d weight/bias receive nonzero gradients: OK")

# ---------------------------------------------------------------------
# 3. Full KIR path, for BOTH the padding>0 branch (exercising the
#    cat-based manual padding specifically under tracing -- the one
#    part of this class that needed deliberate eager/traced handling)
#    and the padding=0 branch.
# ---------------------------------------------------------------------

for label, layer in [("padding>0", nn.Conv1d(Cin, Cout, K, stride=stride, padding=padding, seed=1)),
                      ("padding=0", nn.Conv1d(Cin, Cout, K, stride=2, padding=0, seed=1))]:
    x_k = core.from_flat(x_vals, [1, Cin, L])
    x_k_grad = core.from_flat(x_vals, [1, Cin, L], requires_grad=True)
    eager_out = layer(x_k).tolist()

    graph = kir.trace(lambda t: layer(t), x_k)
    check_close(f"Conv1d ({label}) kir.run()", kir.run(graph, x_k).tolist(), eager_out)
    check_close(f"Conv1d ({label}) kir.run_fused()",
                kir.run_fused(kir.elementwise_fusion(graph), x_k).tolist(), eager_out)
    plan = kir.plan_memory(graph)
    check_close(f"Conv1d ({label}) kir.run_planned()",
                kir.run_planned(graph, plan, core.StoragePool(), x_k).tolist(), eager_out)
    if core.metal_available():
        check_close(f"Conv1d ({label}) kir.run_metal()",
                    kir.run_metal(kir.elementwise_fusion(graph), x_k).tolist(), eager_out)

    scalar_graph = kir.trace(lambda t: layer(t).sum(), x_k)
    bwd = kir.grad(scalar_graph, scalar_graph.inputs)
    loss_eager = layer(x_k_grad).sum()
    loss_eager.backward()
    check_close(f"Conv1d ({label}) kir.grad()", kir.run(bwd, x_k).tolist(), x_k_grad.grad.tolist(), TOL_GRAD)

# ---------------------------------------------------------------------
# 4. Practical: a small Conv1d-based sequence classifier trains to
#    convergence -- two classes distinguished by whether a short
#    "spike" pattern appears anywhere in a 1D signal, the kind of task
#    Conv1d exists for (local pattern detection along a sequence).
# ---------------------------------------------------------------------

SEQ_LEN = 16


def make_example():
    label = rng.randint(0, 1)
    signal = [rng.uniform(-0.2, 0.2) for _ in range(SEQ_LEN)]
    if label == 1:
        pos = rng.randint(0, SEQ_LEN - 3)
        signal[pos] = signal[pos + 1] = signal[pos + 2] = 3.0
    return signal, label


examples = [make_example() for _ in range(200)]
X = kansai.tensor([[s] for s, _ in examples])  # (200, 1, SEQ_LEN)
Y = kansai.tensor([[1.0, 0.0] if lbl == 0 else [0.0, 1.0] for _, lbl in examples])


class SeqClassifier(nn.Module):
    def __init__(self):
        self.conv = nn.Conv1d(1, 4, kernel_size=3, stride=1, padding=1, seed=2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4 * SEQ_LEN, 2, seed=5)

    def forward(self, x):
        h = self.relu(self.conv(x))
        n, c, length = h.shape
        return self.fc(h.reshape([n, c * length]))


model = SeqClassifier()
opt = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(150):
    logits = model(X)
    loss = logits.cross_entropy(Y)
    model.zero_grad()
    loss.backward()
    opt.step()

final_loss = loss.tolist()[0]
preds = model(X).tolist()
correct = sum(1 for i in range(200) if (preds[i * 2] > preds[i * 2 + 1]) == (examples[i][1] == 0))
accuracy = correct / 200
print(f"Conv1d spike-detection classifier: final loss {final_loss:.4f}, accuracy {accuracy:.2%}")
assert accuracy >= 0.95, f"expected >=95% accuracy, got {accuracy:.2%}"

print("\nConv1d test passed.")
