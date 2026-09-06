"""MultiHeadAttention -- see its own docstring in python/kansai/nn.py
for the full design (standard scaled dot-product attention, general
cross-attention, an additive mask since Kansai has no boolean-masking
primitive yet) and why it was only reachable now: it needs batched
matmul (Q/K/V are 4D, every matmul inside is genuinely batched over
(batch, heads)) and softmax(dim), both landing in this same session.

Checked: output shapes for both self-attention and cross-attention
(different query/key sequence lengths); forward values against a
from-scratch single-head (num_heads=1) attention implementation in
plain Python (not Kansai's own ops called a different way, which could
share a bug with the implementation under test); backward against
central differences, including that every projection weight
(w_q/w_k/w_v/w_o) receives a gradient; an additive causal mask
confirmed to zero out attention to every future position while each
row's weights still sum to 1 (not just "the loss looks reasonable");
and a practical end-to-end check -- a small attention-based sequence
classifier (MultiHeadAttention -> mean-pool -> Linear -> cross_entropy)
trained to convergence on a synthetic "find the class marker among
distractors, at a random position" task.
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
GRAD_TOL = 3e-2
EPS = 1e-3

rng = random.Random(0)


def check_close(label, a, b, tol=TOL):
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < tol, f"{label}[{i}]: got={x:.6f} expected={y:.6f}"
    print(f"{label}: OK ({len(a)} elements, max diff {max(abs(x - y) for x, y in zip(a, b)):.2e})")


# ---------------------------------------------------------------------
# 1. Shapes: self-attention, and cross-attention with different
#    query/key sequence lengths.
# ---------------------------------------------------------------------

mha = nn.MultiHeadAttention(d_model=8, num_heads=2, seed=1)
x = kansai.randn([2, 5, 8], std=1.0, seed=2)
out = mha(x, x, x)
assert list(out.shape) == [2, 5, 8], list(out.shape)
print(f"self-attention output shape: {list(out.shape)}: OK")

q_in = kansai.randn([2, 3, 8], std=1.0, seed=3)
kv_in = kansai.randn([2, 7, 8], std=1.0, seed=4)
out_cross = mha(q_in, kv_in, kv_in)
assert list(out_cross.shape) == [2, 3, 8], list(out_cross.shape)
print(f"cross-attention output shape (seq_q=3, seq_k=7): {list(out_cross.shape)}: OK")

# ---------------------------------------------------------------------
# 2. Forward values against a from-scratch single-head (num_heads=1)
#    attention implementation in plain Python.
# ---------------------------------------------------------------------


def manual_attention(x_flat, seq, d, wq, wk, wv, wo):
    def matmul(a, arows, acols, b, brows, bcols):
        assert acols == brows
        result = [[0.0] * bcols for _ in range(arows)]
        for i in range(arows):
            for j in range(bcols):
                result[i][j] = sum(a[i][k] * b[k][j] for k in range(acols))
        return result

    xm = [x_flat[i * d:(i + 1) * d] for i in range(seq)]
    wq_m = [wq[i * d:(i + 1) * d] for i in range(d)]
    wk_m = [wk[i * d:(i + 1) * d] for i in range(d)]
    wv_m = [wv[i * d:(i + 1) * d] for i in range(d)]
    wo_m = [wo[i * d:(i + 1) * d] for i in range(d)]
    q = matmul(xm, seq, d, wq_m, d, d)
    k = matmul(xm, seq, d, wk_m, d, d)
    v = matmul(xm, seq, d, wv_m, d, d)
    scale = 1.0 / math.sqrt(d)
    scores = [[sum(q[i][t] * k[j][t] for t in range(d)) * scale for j in range(seq)] for i in range(seq)]
    weights = []
    for row in scores:
        m = max(row)
        exps = [math.exp(v_ - m) for v_ in row]
        s = sum(exps)
        weights.append([e / s for e in exps])
    attended = matmul(weights, seq, seq, v, seq, d)
    result = matmul(attended, seq, d, wo_m, d, d)
    return [v_ for row in result for v_ in row]


mha1 = nn.MultiHeadAttention(d_model=4, num_heads=1, seed=5)
xs = kansai.randn([1, 3, 4], std=1.0, seed=6)
out1 = mha1(xs, xs, xs)
ref1 = manual_attention(xs.tolist(), 3, 4, mha1.w_q.tolist(), mha1.w_k.tolist(), mha1.w_v.tolist(),
                         mha1.w_o.tolist())
check_close("single-head attention vs from-scratch manual reference", out1.tolist(), ref1)

# ---------------------------------------------------------------------
# 3. Backward vs central differences -- input and every projection
#    weight.
# ---------------------------------------------------------------------

mha2 = nn.MultiHeadAttention(d_model=4, num_heads=2, seed=7)
xvals = [rng.uniform(-1, 1) for _ in range(1 * 3 * 4)]
xg = kansai.from_flat(xvals, [1, 3, 4], requires_grad=True)
mha2(xg, xg, xg).sum().backward()


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


analytical = central_diff_grad(lambda t: mha2(t, t, t).sum(), xvals, [1, 3, 4])
check_close("MultiHeadAttention backward (input) vs central diff", xg.grad.tolist(), analytical, GRAD_TOL)

for name in ("w_q", "w_k", "w_v", "w_o"):
    assert getattr(mha2, name).grad is not None, f"{name} should receive a gradient"
print("MultiHeadAttention projection weight gradients (w_q/w_k/w_v/w_o) all populated: OK")

# ---------------------------------------------------------------------
# 4. Additive causal mask: zeros attention to every future position,
#    while each row's weights still sum to 1.
# ---------------------------------------------------------------------

mha3 = nn.MultiHeadAttention(d_model=4, num_heads=1, seed=8)
xs3 = kansai.randn([1, 4, 4], std=1.0, seed=9)
seq = 4
NEG = -1e9
mask_vals = [0.0 if j <= i else NEG for i in range(seq) for j in range(seq)]
mask = kansai.from_flat(mask_vals, [1, 1, seq, seq])

q = mha3._split_heads(xs3.matmul(mha3.w_q), 1, seq)
k = mha3._split_heads(xs3.matmul(mha3.w_k), 1, seq)
scale = core.from_flat([1.0 / math.sqrt(mha3.d_k)], [1])
scores = q.matmul(k.transpose(2, 3)).mul(scale).add(mask)
weights = scores.softmax(3)
w = weights.tolist()

violations = sum(1 for i in range(seq) for j in range(seq) if j > i and w[i * seq + j] > 1e-6)
assert violations == 0, f"causal mask failed to zero out {violations} future-position attention weight(s)"
for i in range(seq):
    row_sum = sum(w[i * seq:(i + 1) * seq])
    assert abs(row_sum - 1.0) < 1e-4, f"row {i} attention weights should sum to 1, got {row_sum}"
print(f"causal mask: 0 violations out of {seq * seq} entries, every row still sums to 1: OK")

# ---------------------------------------------------------------------
# 5. Practical check: a small attention-based sequence classifier
#    (MultiHeadAttention -> mean-pool -> Linear -> cross_entropy)
#    trained on a synthetic task -- each sequence has a class-specific
#    "marker" vector inserted at a RANDOM position among distractors;
#    the model must find it regardless of where it lands.
# ---------------------------------------------------------------------

d_model, seq_len, num_classes = 8, 5, 3
markers = [[rng.uniform(0.8, 1.2) if (i % num_classes) == c else rng.uniform(-0.2, 0.2) for i in range(d_model)]
           for c in range(num_classes)]


def make_example(cls):
    sequence = [[rng.uniform(-0.3, 0.3) for _ in range(d_model)] for _ in range(seq_len)]
    sequence[rng.randrange(seq_len)] = markers[cls]
    return sequence


examples = [(make_example(c), c) for c in range(num_classes) for _ in range(30)]
rng.shuffle(examples)
seqs = [e[0] for e in examples]
labels = [e[1] for e in examples]
onehot = []
for lab in labels:
    row = [0.0] * num_classes
    row[lab] = 1.0
    onehot.extend(row)

X = kansai.from_flat([v for sequence in seqs for row in sequence for v in row], [len(seqs), seq_len, d_model])
Y = kansai.from_flat(onehot, [len(seqs), num_classes])


class AttnClassifier(nn.Module):
    def __init__(self, mha, classifier):
        self.mha = mha
        self.classifier = classifier

    def forward(self, x):
        attended = self.mha(x, x, x)
        pooled = attended.mean(1)
        return self.classifier(pooled)


model = AttnClassifier(nn.MultiHeadAttention(d_model=d_model, num_heads=2, seed=10),
                        nn.Linear(d_model, num_classes, seed=20))
opt = optim.Adam(model.parameters(), lr=0.01)

loss = None
for step in range(200):
    loss = model(X).cross_entropy(Y)
    model.zero_grad()
    loss.backward()
    opt.step()

final_loss = loss.tolist()[0]
final_logits = model(X).tolist()
correct = sum(
    1 for i in range(len(seqs))
    if final_logits[i * num_classes:(i + 1) * num_classes].index(max(final_logits[i * num_classes:(i + 1) * num_classes]))
    == labels[i]
)
accuracy = correct / len(seqs)
print(f"attention-based sequence classifier: loss {final_loss:.6f}, accuracy {accuracy:.1%}")
assert final_loss < 0.3, f"attention classifier did not converge (loss {final_loss})"
assert accuracy > 0.85, f"attention classifier accuracy too low: {accuracy:.1%}"

print("\nMultiHeadAttention test passed.")
