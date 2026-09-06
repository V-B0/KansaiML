"""TransformerBlock -- see its own docstring in python/kansai/nn.py for
the full design (pre-LN self-attention + residual, then a position-wise
feed-forward network + its own residual: `x = x + attn(LN(x))`,
`x = x + ffn(LN(x))`) and why it was only reachable now: kir.grad's own
batched-matmul gap (closed just before this, see the devlog entry of
the same name) was a real blocker for tracing and differentiating a
full block, even though eager training through it was always fine.

Checked: output shape; that BOTH residual connections are actually
wired correctly -- not inferred from "training seems to work" but
proven structurally, by zeroing exactly the two sublayers' own output
projections (attn's w_o, ffn's fc2) so each branch contributes EXACTLY
zero, and confirming the block's output equals its input exactly (not
approximately) and the gradient of a sum-reduction is exactly all-ones
-- the signature of an identity-plus-something residual path, not just
"the shapes happen to match"; backward against central differences for
the input, and that every parameter across both sublayers (ln1, attn's
w_q/w_k/w_v/w_o, ln2, fc1, fc2) receives a gradient; a causal mask
confirmed to make each position's output genuinely INDEPENDENT of every
LATER position's input token (changing a future token must not change
an earlier position's output at all, checked to exact equality -- the
real behavioral guarantee a causal mask is supposed to provide, not
just "the loss looks reasonable"); the full KIR path including
kir.grad matching eager exactly; and a practical end-to-end test --
stacking two TransformerBlocks into a tiny causal model trained to
solve a genuine attention task (predict the FIRST token's identity at
every later position, which a per-position-only network provably
cannot solve without looking back through attention) to convergence.
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import kir, nn, optim
from kansai import _core as core

TOL = 1e-4
GRAD_TOL = 3e-2
EPS = 1e-3

rng = random.Random(0)


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


# ---------------------------------------------------------------------
# 1. Output shape.
# ---------------------------------------------------------------------

block = nn.TransformerBlock(d_model=8, num_heads=2, d_ff=16, seed=1)
x0 = kansai.randn([2, 5, 8], std=1.0, seed=2)
out0 = block(x0)
assert list(out0.shape) == [2, 5, 8]
print("TransformerBlock output shape: OK")

# ---------------------------------------------------------------------
# 2. Both residual connections wired correctly -- proven structurally,
#    not inferred. Zeroing attn's w_o and ffn's fc2 (weight AND bias)
#    makes both sublayer branches contribute EXACTLY zero regardless of
#    everything upstream of them, so the block reduces to a pure
#    identity: output must equal input exactly, and d(sum(output))/d
#    (input) must be exactly all-ones.
# ---------------------------------------------------------------------

block_id = nn.TransformerBlock(d_model=6, num_heads=2, d_ff=12, seed=3)
zero_wo = core.zeros(list(block_id.attn.w_o.shape))
block_id.attn.w_o = core.from_flat(zero_wo.tolist(), list(zero_wo.shape))
block_id.fc2.weight = core.from_flat([0.0] * (12 * 6), [12, 6])
block_id.fc2.bias = core.from_flat([0.0] * 6, [6])

x_id = kansai.randn([1, 4, 6], std=1.0, seed=4)
out_id = block_id(x_id)
check_close("TransformerBlock reduces to exact identity when both sublayer outputs are zeroed",
            out_id.tolist(), x_id.tolist(), tol=1e-6)

x_id_g = kansai.from_flat(x_id.tolist(), [1, 4, 6], requires_grad=True)
block_id(x_id_g).sum().backward()
check_close("TransformerBlock identity case: gradient is exactly all-ones (both residuals present)",
            x_id_g.grad.tolist(), [1.0] * (1 * 4 * 6), tol=1e-6)

# ---------------------------------------------------------------------
# 3. Backward vs central differences (input), and every parameter
#    across both sublayers receives a gradient.
# ---------------------------------------------------------------------

block2 = nn.TransformerBlock(d_model=4, num_heads=2, d_ff=8, seed=7)
xvals = [rng.uniform(-1, 1) for _ in range(1 * 3 * 4)]
xg = kansai.from_flat(xvals, [1, 3, 4], requires_grad=True)
block2(xg).sum().backward()

analytical = central_diff_grad(lambda t: block2(t).sum(), xvals, [1, 3, 4])
check_close("TransformerBlock backward (input) vs central diff", xg.grad.tolist(), analytical, GRAD_TOL)

for name, p in block2.named_parameters():
    assert p.grad is not None, f"{name} should receive a gradient"
print("TransformerBlock: every parameter (ln1, attn.*, ln2, fc1, fc2) received a gradient: OK")

# ---------------------------------------------------------------------
# 3b. kir.grad through a full TransformerBlock forward.
# ---------------------------------------------------------------------

x_trace = core.from_flat(xvals, [1, 3, 4])
graph = kir.trace(lambda t: block2(t).sum(), x_trace)
eager_val = block2(x_trace).sum().tolist()
check_close("TransformerBlock kir.run() matches eager forward", kir.run(graph, x_trace).tolist(), eager_val)

bwd = kir.grad(graph, graph.inputs)
grad_x_kir = kir.run(bwd, x_trace)
check_close("TransformerBlock kir.grad() matches eager backward exactly", grad_x_kir.tolist(), xg.grad.tolist())
if core.metal_available():
    grad_x_metal = kir.run_metal(kir.elementwise_fusion(bwd), x_trace)
    check_close("TransformerBlock kir.grad() via run_metal matches eager backward",
                grad_x_metal.tolist(), xg.grad.tolist())

# ---------------------------------------------------------------------
# 4. Causal mask: each position's output must be genuinely INDEPENDENT
#    of every LATER position's input token -- checked to exact
#    equality by changing a future token and confirming an earlier
#    position's own output doesn't move at all.
# ---------------------------------------------------------------------

block3 = nn.TransformerBlock(d_model=4, num_heads=1, d_ff=8, seed=9)
seq = 5
NEG = -1e9
mask_vals = [0.0 if j <= i else NEG for i in range(seq) for j in range(seq)]
mask = kansai.from_flat(mask_vals, [1, 1, seq, seq])

base_vals = [rng.uniform(-1, 1) for _ in range(1 * seq * 4)]
xa = kansai.from_flat(base_vals, [1, seq, 4])
out_a = block3(xa, mask=mask).tolist()

changed_vals = list(base_vals)
for j in range(4):  # perturb every feature of the LAST (future-most) position
    changed_vals[(seq - 1) * 4 + j] += rng.uniform(-5, 5)
xb = kansai.from_flat(changed_vals, [1, seq, 4])
out_b = block3(xb, mask=mask).tolist()

# Every position EXCEPT the last (the one that was perturbed) must be
# completely unaffected -- causal masking means no earlier position can
# see it.
for pos in range(seq - 1):
    check_close(f"causal TransformerBlock: position {pos}'s output unaffected by a later token's change",
                out_a[pos * 4:(pos + 1) * 4], out_b[pos * 4:(pos + 1) * 4], tol=1e-6)
# Sanity: the perturbed position's OWN output should generally change
# (confirms the perturbation was large enough to matter and the mask
# isn't accidentally blocking everything).
last_a, last_b = out_a[(seq - 1) * 4:], out_b[(seq - 1) * 4:]
assert any(abs(p - q) > 1e-4 for p, q in zip(last_a, last_b)), \
    "expected the perturbed position's own output to change -- perturbation had no effect at all"
print("causal TransformerBlock: perturbed position's own output correctly DID change: OK")

# ---------------------------------------------------------------------
# 5. Practical end-to-end test: stack two TransformerBlocks into a
#    tiny causal model trained to predict the FIRST token's identity
#    at every later position -- a task a per-position-only network
#    (no attention) provably cannot solve, since position 0's identity
#    isn't locally visible anywhere else in the sequence.
# ---------------------------------------------------------------------

VOCAB, SEQ_LEN, D_MODEL, D_FF, NUM_HEADS = 8, 6, 16, 32, 2


class TinyCausalModel(nn.Module):
    def __init__(self, seed=0):
        self.tok_embed = nn.Embedding(VOCAB, D_MODEL, seed=seed)
        self.pos_embed = nn.Embedding(SEQ_LEN, D_MODEL, seed=seed + 1)
        self.block1 = nn.TransformerBlock(D_MODEL, NUM_HEADS, D_FF, seed=seed + 2)
        self.block2 = nn.TransformerBlock(D_MODEL, NUM_HEADS, D_FF, seed=seed + 3)
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, seed=seed + 4)

        causal_vals = [0.0 if j <= i else -1e9 for i in range(SEQ_LEN) for j in range(SEQ_LEN)]
        self.mask = core.from_flat(causal_vals, [1, 1, SEQ_LEN, SEQ_LEN])

    def forward(self, batch_ids):
        batch_n = len(batch_ids)
        tok = self.tok_embed(batch_ids)
        positions = [list(range(SEQ_LEN))] * batch_n
        pos = self.pos_embed(positions)
        x = tok.add(pos)
        x = self.block1(x, mask=self.mask)
        x = self.block2(x, mask=self.mask)
        x = self.ln_f(x)
        return self.head(x)  # (batch, seq, VOCAB)


def make_example():
    return [rng.randint(0, VOCAB - 1) for _ in range(SEQ_LEN)]


model = TinyCausalModel(seed=11)
opt = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

N_TRAIN = 300
EPOCHS = 150
train_examples = [make_example() for _ in range(N_TRAIN)]

for epoch in range(EPOCHS):
    rng.shuffle(train_examples)
    total_loss = 0.0
    BATCH = 30
    for start in range(0, N_TRAIN, BATCH):
        batch = train_examples[start:start + BATCH]
        logits = model(batch)  # (batch, seq, vocab)
        bsz = len(batch)
        # Target at every position: the first token of that same
        # sequence -- flatten (batch, seq) logits/targets to
        # (batch*seq, vocab) for cross_entropy.
        targets_onehot = []
        for seq_ids in batch:
            first = seq_ids[0]
            for _pos in range(SEQ_LEN):
                targets_onehot.extend([1.0 if c == first else 0.0 for c in range(VOCAB)])
        targets = core.from_flat(targets_onehot, [bsz * SEQ_LEN, VOCAB])
        logits_flat = logits.reshape([bsz * SEQ_LEN, VOCAB])
        loss = logits_flat.cross_entropy(targets)
        model.zero_grad()
        loss.backward()
        optim.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        total_loss += loss.tolist()[0] * bsz
    if epoch % 30 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch}: avg loss {total_loss / N_TRAIN:.4f}")

N_TEST = 200
correct_positions = 0
total_positions = 0
for _ in range(N_TEST):
    seq_ids = make_example()
    logits = model([seq_ids])
    flat = logits.tolist()
    first = seq_ids[0]
    for pos in range(SEQ_LEN):
        row = flat[pos * VOCAB:(pos + 1) * VOCAB]
        pred = row.index(max(row))
        correct_positions += int(pred == first)
        total_positions += 1

accuracy = correct_positions / total_positions
print(f"TinyCausalModel (2x TransformerBlock) 'copy first token' accuracy: {accuracy:.2%}")
assert accuracy >= 0.9, f"expected >=90% per-position accuracy on the copy-first-token task, got {accuracy:.2%}"

print("\nTransformerBlock test passed.")
