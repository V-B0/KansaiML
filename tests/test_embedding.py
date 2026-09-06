"""index_select and nn.Embedding -- the two pieces needed to look up
rows of a tensor by a list of integer positions, and to turn that into
a token-embedding lookup table. Until now Kansai had no indexing
primitive at all: every existing op picks apart a tensor by shape
(reshape/transpose/slice/cat) or by value (add/mul/...), never by an
arbitrary, possibly-repeating list of positions.

`indices` is a plain Python list of ints, not a core.Tensor -- Kansai
has no integer dtype, and an index into a lookup table isn't a
differentiable quantity anyway (see Tensor::index_select's own doc
comment in core/include/kansai/Tensor.hpp). The one real subtlety is
that indices can repeat: index_select_backward has to ACCUMULATE
(+=) into every repeated position's gradient, not overwrite it, or a
token that appears twice in a batch would silently only get credit for
one of its two occurrences.

Checked at the usual bar: index_select's forward against hand-computed
expectations (both a row-selection and a column-selection case, plus
repeated indices), eager backward against central differences
(confirming the accumulation, not just "some gradient flows"), the
full KIR path (trace, run/run_fused/run_planned/run_metal, and
kir.grad, including the repeated-index case through the *traced*
backward graph, not just eager), out-of-range rejection; then
Embedding's forward shape/values against a manual row lookup, backward
gradient accumulation when the same token id appears more than once
across a batch, and a practical end-to-end test -- a tiny "does this
sequence contain the marker token" classifier (Embedding -> mean-pool
-> Linear -> cross_entropy) trained with Adam to convergence.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import kir, nn, optim
from kansai import _core as core

EPS = 1e-3
GRAD_TOL = 2e-2
TOL = 1e-4


def check_close(label, a, b, tol=TOL):
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < tol, f"{label}[{i}]: got={x:.6f} expected={y:.6f}"
    print(f"{label}: OK ({len(a)} elements)")


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


rng = random.Random(0)

# ---------------------------------------------------------------------
# 1. index_select: forward (row selection with repeats, and column
#    selection), eager backward vs. central differences (confirming
#    repeated-index accumulation, not just "a gradient exists"), the
#    full KIR path, and out-of-range rejection.
# ---------------------------------------------------------------------

x_vals = [float(v) for v in range(12)]  # 4x3: rows [0,1,2], [3,4,5], [6,7,8], [9,10,11]
x = kansai.from_flat(x_vals, [4, 3])

y = x.index_select(0, [2, 0, 2])
assert list(y.shape) == [3, 3]
check_close("index_select forward (dim0, repeated index)", y.tolist(), [6, 7, 8, 0, 1, 2, 6, 7, 8])

y_col = x.index_select(1, [2, 0])
assert list(y_col.shape) == [4, 2]
check_close("index_select forward (dim1)", y_col.tolist(), [2, 0, 5, 3, 8, 6, 11, 9])

analytical = central_diff_grad(lambda t: t.index_select(0, [2, 0, 2]).sum(), x_vals, [4, 3])
xg = kansai.from_flat(x_vals, [4, 3], requires_grad=True)
xg.index_select(0, [2, 0, 2]).sum().backward()
check_close("index_select eager backward vs central diff", xg.grad.tolist(), analytical, GRAD_TOL)
check_close("index_select eager backward accumulates repeats", xg.grad.tolist(),
            [1, 1, 1, 0, 0, 0, 2, 2, 2, 0, 0, 0])

graph = kir.trace(lambda t: t.index_select(0, [2, 0, 2]).sum(), x)
eager_out = x.index_select(0, [2, 0, 2]).sum().tolist()
check_close("index_select kir.run()", kir.run(graph, x).tolist(), eager_out)
check_close("index_select kir.run_fused()", kir.run_fused(kir.elementwise_fusion(graph), x).tolist(), eager_out)
plan = kir.plan_memory(graph)
check_close("index_select kir.run_planned()", kir.run_planned(graph, plan, core.StoragePool(), x).tolist(), eager_out)
if core.metal_available():
    check_close("index_select kir.run_metal()", kir.run_metal(kir.elementwise_fusion(graph), x).tolist(), eager_out)

bwd = kir.grad(graph, graph.inputs)
check_close("index_select kir.grad()", kir.run(bwd, x).tolist(), xg.grad.tolist())
if core.metal_available():
    check_close("index_select kir.grad() via run_metal (accumulation through traced backward)",
                kir.run_metal(kir.elementwise_fusion(bwd), x).tolist(), xg.grad.tolist())

try:
    x.index_select(0, [4])
    raise AssertionError("expected index_select to reject an out-of-range index")
except RuntimeError as e:
    print(f"index_select correctly rejects an out-of-range index: {e}")

# ---------------------------------------------------------------------
# 2. Embedding: forward shape/values against a manual row lookup (both
#    a flat sequence and a (batch, seq_len) nested lookup), and
#    backward gradient accumulation when a token id repeats within a
#    batch.
# ---------------------------------------------------------------------

vocab_size, embed_dim = 6, 4
emb = nn.Embedding(vocab_size, embed_dim, seed=1)
w_rows = [emb.weight.tolist()[i * embed_dim:(i + 1) * embed_dim] for i in range(vocab_size)]

seq_ids = [3, 0, 5, 3]
out = emb(seq_ids)
assert list(out.shape) == [4, embed_dim]
expected_flat = [v for i in seq_ids for v in w_rows[i]]
check_close("Embedding forward (flat sequence)", out.tolist(), expected_flat)

batch_ids = [[1, 2], [0, 1]]
out_batch = emb(batch_ids)
assert list(out_batch.shape) == [2, 2, embed_dim]
expected_batch_flat = [v for row in batch_ids for i in row for v in w_rows[i]]
check_close("Embedding forward (batched nested ids)", out_batch.tolist(), expected_batch_flat)

# Backward: token id 3 appears twice in seq_ids (positions 0 and 3) --
# weight.grad's row 3 must be the SUM of both positions' output
# gradients, not just one of them.
emb2 = nn.Embedding(vocab_size, embed_dim, seed=1)
loss = emb2(seq_ids).sum()
loss.backward()
grad_rows = [emb2.weight.grad.tolist()[i * embed_dim:(i + 1) * embed_dim] for i in range(vocab_size)]
check_close("Embedding backward: row 3 (id used twice) accumulates", grad_rows[3], [2.0] * embed_dim)
check_close("Embedding backward: row 0 (id used once)", grad_rows[0], [1.0] * embed_dim)
check_close("Embedding backward: row 5 (id used once)", grad_rows[5], [1.0] * embed_dim)
check_close("Embedding backward: unused rows stay zero", grad_rows[1] + grad_rows[2] + grad_rows[4],
            [0.0] * (embed_dim * 3))


# Central-difference check on Embedding's backward, done directly
# against the weight tensor (token ids aren't differentiable so can't
# be perturbed) -- confirms the analytical gradient above against the
# gold-standard numerical check every other op in this project is held
# to, not just an internal self-consistency check.
def embed_loss(w_tensor):
    class _Tmp(nn.Module):
        def __init__(self, weight):
            self.weight = weight
            self.embed_dim = embed_dim

        def forward(self, ids):
            return nn.Embedding.forward(self, ids)

    return _Tmp(w_tensor)(seq_ids).sum()


w_vals = emb2.weight.tolist()
analytical_emb = central_diff_grad(embed_loss, w_vals, [vocab_size, embed_dim])
check_close("Embedding backward vs central diff", emb2.weight.grad.tolist(), analytical_emb, GRAD_TOL)

# ---------------------------------------------------------------------
# 3. Practical end-to-end test: a tiny "does this sequence contain the
#    marker token" binary classifier -- Embedding -> mean-pool over the
#    sequence -> Linear -> cross_entropy -- trained with Adam to
#    convergence on synthetic data. The marker token (id 0) appears at
#    a random position in half the sequences and never in the other
#    half, with distractor tokens (ids 1..VOCAB-1) filling the rest --
#    forces the model to actually learn token identity through the
#    embedding table, not just sequence statistics.
# ---------------------------------------------------------------------

VOCAB = 10
SEQ_LEN = 6
EMBED_DIM = 8
MARKER = 0


def make_example():
    has_marker = rng.random() < 0.5
    seq = [rng.randint(1, VOCAB - 1) for _ in range(SEQ_LEN)]
    if has_marker:
        seq[rng.randint(0, SEQ_LEN - 1)] = MARKER
    return seq, (1 if has_marker else 0)


class MarkerClassifier(nn.Module):
    def __init__(self, seed=0):
        self.embed = nn.Embedding(VOCAB, EMBED_DIM, seed=seed)
        self.fc = nn.Linear(EMBED_DIM, 2, seed=seed + 1)

    def forward(self, batch_ids):
        e = self.embed(batch_ids)  # (batch, seq_len, embed_dim)
        pooled = e.mean(1)  # (batch, embed_dim) -- mean over the sequence
        return self.fc(pooled)  # (batch, 2)


model = MarkerClassifier(seed=7)
opt = optim.Adam(model.parameters(), lr=0.05)

N_TRAIN = 200
train_seqs, train_labels = zip(*(make_example() for _ in range(N_TRAIN)))
train_seqs, train_labels = list(train_seqs), list(train_labels)

EPOCHS = 60
BATCH = 20
for epoch in range(EPOCHS):
    perm = list(range(N_TRAIN))
    rng.shuffle(perm)
    total_loss = 0.0
    for start in range(0, N_TRAIN, BATCH):
        idx = perm[start:start + BATCH]
        batch_ids = [train_seqs[i] for i in idx]
        batch_labels = [train_labels[i] for i in idx]
        targets = core.from_flat(
            [1.0 if c == lbl else 0.0 for lbl in batch_labels for c in range(2)], [len(idx), 2])

        logits = model(batch_ids)
        loss = logits.cross_entropy(targets)
        model.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.tolist()[0] * len(idx)
    if epoch % 15 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch}: avg loss {total_loss / N_TRAIN:.4f}")

N_TEST = 100
correct = 0
for _ in range(N_TEST):
    seq, label = make_example()
    logits = model([seq])
    pred = 0 if logits.tolist()[0] > logits.tolist()[1] else 1
    correct += int(pred == label)

accuracy = correct / N_TEST
print(f"MarkerClassifier test accuracy: {accuracy:.2%}")
assert accuracy >= 0.9, f"expected the marker classifier to reach >=90% test accuracy, got {accuracy:.2%}"

print("\nindex_select / Embedding test passed.")
