"""Kansai's second real end-to-end benchmark, and its first on genuine
sequence data: a small decoder-only, character-level transformer
(`nn.Embedding` -> `nn.TransformerBlock` x N -> `nn.LayerNorm` ->
`nn.Linear`) trained on the real "tiny Shakespeare" corpus -- not MNIST,
not a synthetic marker-token task, not a hand-picked "copy the first
token" toy. This is the actual claim every piece landed this session
(`Embedding`, comparison ops/`where`, `AdamW`, `clip_grad_norm_`,
`CosineAnnealingLR`, `Dataset`/`DataLoader`, the `kir.grad` batched-
matmul fix, `TransformerBlock`) has to back up TOGETHER, on real text,
not in isolation.

Deliberately small (a few hundred thousand parameters, not millions):
the point here is finding out where a from-scratch framework genuinely
breaks at a real (if modest) scale, honestly, not producing a
publishable language model. Whatever happens -- if it trains cleanly,
if it's slower than expected, if generation quality is rough at this
size -- gets reported as observed, not smoothed over.

Run: python3 train_shakespeare.py (downloads the ~1.1MB corpus on
first run via download_shakespeare.py; cached under
examples/tinyshakespeare/data/ afterward, gitignored).
"""

import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, os.path.dirname(__file__))

import kansai
from kansai import nn, optim
from kansai import _core as core
from kansai.data import DataLoader, Dataset
from download_shakespeare import load_shakespeare

SEED = 0
BLOCK_SIZE = 48
D_MODEL = 96
NUM_HEADS = 4
D_FF = 256
NUM_LAYERS = 3
BATCH_SIZE = 48
TOTAL_STEPS = 2000
LR = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
LOG_EVERY = 100
VAL_EVERY = 500
TRAIN_CHARS = 900_000  # leaves the corpus's own final ~200K characters as a genuine held-out tail
VAL_CHARS = 50_000


class CharDataset(Dataset):
    """Every possible `block_size`-length window of `tokens`, starting
    at every position -- `__getitem__(idx)` returns
    `(tokens[idx:idx+block_size], tokens[idx+1:idx+block_size+1])`, the
    standard next-character-prediction (x, y) pair language-model
    training always uses. `DataLoader`'s own shuffling picks which
    START positions get visited each epoch and in what order; the
    windows themselves overlap heavily by construction (adjacent
    indices share all but one token), which is normal for this kind of
    dataset, not a bug."""

    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


class TinyGPT(nn.Module):
    """Token embedding + learned positional embedding, summed
    (the standard GPT-style approach -- not a fixed sinusoidal
    encoding, which Kansai doesn't implement either, but a real,
    trainable `nn.Embedding` over position indices, exactly as
    reusable as the token embedding table right next to it), N stacked
    `TransformerBlock`s under one shared causal mask, a final
    `LayerNorm`, then a `Linear` projection back to vocabulary logits.
    """

    def __init__(self, vocab_size: int, seed: int = 0):
        self.vocab_size = vocab_size
        self.tok_embed = nn.Embedding(vocab_size, D_MODEL, seed=seed)
        self.pos_embed = nn.Embedding(BLOCK_SIZE, D_MODEL, seed=seed + 1)
        self.blocks = [nn.TransformerBlock(D_MODEL, NUM_HEADS, D_FF, seed=seed + 10 + i)
                       for i in range(NUM_LAYERS)]
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size, seed=seed + 99)

        causal_vals = [0.0 if j <= i else -1e9 for i in range(BLOCK_SIZE) for j in range(BLOCK_SIZE)]
        self.mask = core.from_flat(causal_vals, [1, 1, BLOCK_SIZE, BLOCK_SIZE])

    def forward(self, batch_ids):
        bsz = len(batch_ids)
        seq_len = len(batch_ids[0])
        tok = self.tok_embed(batch_ids)
        positions = [list(range(seq_len))] * bsz
        x = tok.add(self.pos_embed(positions))
        for block in self.blocks:
            mask = self.mask if seq_len == BLOCK_SIZE else None
            x = block(x, mask=mask)
        x = self.ln_f(x)
        return self.head(x)  # (batch, seq_len, vocab_size)


def compute_loss(model, batch_x, batch_y, vocab_size):
    bsz, seq_len = len(batch_x), len(batch_x[0])
    logits = model(batch_x).reshape([bsz * seq_len, vocab_size])
    targets_flat = []
    for seq_ids in batch_y:
        for tid in seq_ids:
            targets_flat.extend([1.0 if c == tid else 0.0 for c in range(vocab_size)])
    targets = core.from_flat(targets_flat, [bsz * seq_len, vocab_size])
    return logits.cross_entropy(targets)


def estimate_val_loss(model, val_tokens, vocab_size, n_batches=10, seed=1234):
    """A few random windows from the held-out tail, no gradient
    tracking needed -- deliberately not wrapped in any "no_grad"
    context (Kansai has no such context manager yet), so backward()
    is simply never called on these; the graph gets built and
    discarded, a real but cheap-at-this-scale inefficiency."""
    rng = random.Random(seed)
    losses = []
    for _ in range(n_batches):
        xs, ys = [], []
        for _ in range(BATCH_SIZE):
            i = rng.randint(0, len(val_tokens) - BLOCK_SIZE - 1)
            chunk = val_tokens[i:i + BLOCK_SIZE + 1]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        loss = compute_loss(model, xs, ys, vocab_size)
        losses.append(loss.tolist()[0])
    return sum(losses) / len(losses)


def generate(model, encode, decode, prompt: str, max_new_tokens: int, vocab_size: int,
             temperature: float = 0.8, seed: int = 0) -> str:
    """Autoregressive sampling, one character at a time: run the model
    on the current (right-truncated-to-BLOCK_SIZE) context, softmax
    the LAST position's logits at `temperature`, sample, append,
    repeat. A prompt shorter than BLOCK_SIZE is left-PADDED with token
    0 -- a real, stated simplification (no learned start/pad token;
    Kansai has no such convention yet) rather than a proper mechanism,
    acceptable here since prompts in practice are much shorter than
    BLOCK_SIZE and the padding falls out of the context almost
    immediately as generation proceeds."""
    rng = random.Random(seed)
    ids = encode(prompt)
    for _ in range(max_new_tokens):
        context = ids[-BLOCK_SIZE:]
        pad = BLOCK_SIZE - len(context)
        if pad > 0:
            context = [0] * pad + context
        logits = model([context])
        last_logits = logits.tolist()[(BLOCK_SIZE - 1) * vocab_size: BLOCK_SIZE * vocab_size]
        scaled = [v / temperature for v in last_logits]
        m = max(scaled)
        exps = [math.exp(v - m) for v in scaled]
        total = sum(exps)
        probs = [e / total for e in exps]
        r = rng.random()
        cum = 0.0
        next_id = vocab_size - 1
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                next_id = i
                break
        ids.append(next_id)
    return decode(ids)


def main():
    print("Loading tiny Shakespeare...")
    text, vocab, encode, decode = load_shakespeare()
    vocab_size = len(vocab)
    print(f"  corpus: {len(text)} characters, vocab size {vocab_size}")

    train_tokens = encode(text[:TRAIN_CHARS])
    val_tokens = encode(text[-VAL_CHARS:])  # the corpus's own final tail -- never seen in a training window

    dataset = CharDataset(train_tokens, BLOCK_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)

    model = TinyGPT(vocab_size, seed=SEED)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"  model: {NUM_LAYERS} TransformerBlocks, d_model={D_MODEL}, {NUM_HEADS} heads, "
          f"d_ff={D_FF}, block_size={BLOCK_SIZE} -- {nparams:,} parameters")

    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.CosineAnnealingLR(opt, T_max=TOTAL_STEPS, eta_min=LR * 0.1)

    print(f"\nTraining for {TOTAL_STEPS} steps (batch size {BATCH_SIZE}, "
          f"random-uniform baseline loss = ln({vocab_size}) = {math.log(vocab_size):.3f})\n")

    step = 0
    running_loss = 0.0
    train_start = time.perf_counter()
    while step < TOTAL_STEPS:
        for batch_x, batch_y in loader:
            if step >= TOTAL_STEPS:
                break
            loss = compute_loss(model, batch_x, batch_y, vocab_size)
            model.zero_grad()
            loss.backward()
            optim.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            opt.step()
            scheduler.step()
            running_loss += loss.tolist()[0]
            step += 1

            if step % LOG_EVERY == 0:
                elapsed = time.perf_counter() - train_start
                avg_loss = running_loss / LOG_EVERY
                running_loss = 0.0
                print(f"step {step:5d}/{TOTAL_STEPS}  train loss {avg_loss:.4f}  "
                      f"lr {opt.lr:.2e}  ({elapsed:.1f}s elapsed, {elapsed / step:.3f}s/step)")

            if step % VAL_EVERY == 0:
                val_loss = estimate_val_loss(model, val_tokens, vocab_size)
                print(f"           held-out val loss {val_loss:.4f} (perplexity {math.exp(val_loss):.2f})")

    total_time = time.perf_counter() - train_start
    final_val_loss = estimate_val_loss(model, val_tokens, vocab_size, n_batches=30)
    print(f"\nDone in {total_time:.1f}s total ({total_time / TOTAL_STEPS:.3f}s/step average).")
    print(f"Final held-out val loss: {final_val_loss:.4f} (perplexity {math.exp(final_val_loss):.2f})")

    print("\n--- Sample generation (temperature 0.8) ---")
    for prompt in ["ROMEO:", "\n", "The king"]:
        sample = generate(model, encode, decode, prompt, max_new_tokens=300, vocab_size=vocab_size, seed=SEED)
        print(f"\nPrompt: {prompt!r}\n{sample}")

    return final_val_loss, total_time


if __name__ == "__main__":
    main()
