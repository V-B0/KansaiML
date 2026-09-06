"""A second, deliberately larger run of the same tiny-Shakespeare
transformer as train_shakespeare.py -- not a different task, the same
one at roughly 7.6x the parameter count (277,601 -> 2,110,913: wider
`d_model` 96->192, more heads 4->6, wider `d_ff` 256->512, twice the
depth 3->6 layers, longer context 48->64, bigger batches 48->64), the
same number of optimizer steps.

Why a separate script rather than just bumping train_shakespeare.py's
own constants: that file's specific numbers (277,601 params, 2,000
steps, 562s, perplexity 10.5->8.28) are the ones DEVLOG.md and
README.md actually cite and expect a reader to be able to reproduce
quickly. Overwriting them would break that documented, fast-to-run
reference point. This script exists purely to ask the honest follow-up
question a single data point can't answer: does the memory-leak fix
(see DEVLOG.md's own account) and the rest of this session's work
actually hold up at a meaningfully larger scale, or was 277K parameters
just small enough to get lucky? Before running this for real, memory
was explicitly re-checked at this config over 150 steps of synthetic
batches (resident memory climbed slightly over the first ~60 steps,
2.75GB -> 2.96GB, then sat completely flat through step 150) --
confirming the fix generalizes before spending the ~65+ minutes a real
run at this scale takes on Apple Silicon CPU.

Run: python3 train_shakespeare_large.py (reuses the same cached corpus
train_shakespeare.py downloads, via download_shakespeare.py).
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
from kansai.data import DataLoader
from download_shakespeare import load_shakespeare
from train_shakespeare import CharDataset

SEED = 0
BLOCK_SIZE = 64
D_MODEL = 192
NUM_HEADS = 6
D_FF = 512
NUM_LAYERS = 6
BATCH_SIZE = 64
TOTAL_STEPS = 2000
LR = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
LOG_EVERY = 100
VAL_EVERY = 500
TRAIN_CHARS = 900_000
VAL_CHARS = 50_000


class BigGPT(nn.Module):
    """Identical architecture to train_shakespeare.py's own TinyGPT --
    token + learned positional Embedding, summed -> N TransformerBlocks
    under one shared causal mask -> LayerNorm -> Linear -- at this
    file's own (larger) D_MODEL/NUM_HEADS/D_FF/NUM_LAYERS/BLOCK_SIZE."""

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
        return self.head(x)


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
    rng = random.Random(seed)
    losses = []
    with kansai.no_grad():
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
    rng = random.Random(seed)
    ids = encode(prompt)
    with kansai.no_grad():
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
    val_tokens = encode(text[-VAL_CHARS:])

    dataset = CharDataset(train_tokens, BLOCK_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)

    model = BigGPT(vocab_size, seed=SEED)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"  model: {NUM_LAYERS} TransformerBlocks, d_model={D_MODEL}, {NUM_HEADS} heads, "
          f"d_ff={D_FF}, block_size={BLOCK_SIZE} -- {nparams:,} parameters "
          f"(train_shakespeare.py's own TinyGPT has 277,601 -- this is {nparams / 277_601:.1f}x)")

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
