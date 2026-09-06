"""Dataset/DataLoader -- see python/kansai/data.py's own docstring for
what this replaces (the "shuffle a list of indices, slice into chunks"
pattern every training loop in this project had been writing by hand)
and why it's map-style only with no worker-process plumbing.

Checked: DataLoader with shuffle=False produces batches in the exact
same order as manual slicing (the baseline every other test's manual
batching loop already trusts); shuffle=True is reproducible across two
independently constructed loaders sharing a seed, uses its OWN
random.Random instance rather than the global random module (confirmed
by perturbing global random state in between and getting the same
order anyway), and still visits every example exactly once per epoch
(set equality against range(n), not just "looks shuffled");
drop_last's effect on both the trailing partial batch and __len__();
_default_collate's tuple-transposition and plain-value cases; and a
practical end-to-end test -- retraining test_embedding.py's own
"marker token" classifier, but through a real Dataset/DataLoader
instead of the hand-rolled batching that test uses, to the same
convergence bar, proving this is a genuine drop-in replacement for a
real Embedding-based model, not just correct in isolation.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from kansai import nn, optim
from kansai.data import DataLoader, Dataset, _default_collate
from kansai import _core as core

# ---------------------------------------------------------------------
# 1. shuffle=False: exact order match against manual slicing.
# ---------------------------------------------------------------------

items = list(range(23))  # deliberately not a multiple of batch_size
loader = DataLoader(items, batch_size=5, shuffle=False)
expected_batches = [items[i:i + 5] for i in range(0, 23, 5)]
actual_batches = list(loader)
assert actual_batches == expected_batches, f"shuffle=False order mismatch: {actual_batches} != {expected_batches}"
print(f"DataLoader(shuffle=False) matches manual slicing exactly: OK ({len(actual_batches)} batches)")

# ---------------------------------------------------------------------
# 2. shuffle=True: reproducible across two loaders sharing a seed, uses
#    its own RNG (not the global `random` module -- perturbed in
#    between to prove it), and visits every example exactly once.
# ---------------------------------------------------------------------

loader_a = DataLoader(items, batch_size=5, shuffle=True, seed=42)
epoch_a = [b for batch in loader_a for b in batch]

random.seed(999)
random.shuffle([1, 2, 3, 4, 5])  # perturb the GLOBAL random module's state

loader_b = DataLoader(items, batch_size=5, shuffle=True, seed=42)
epoch_b = [b for batch in loader_b for b in batch]

assert epoch_a == epoch_b, "same seed must produce the identical shuffled order, independent of global random state"
print("DataLoader(shuffle=True) is reproducible across instances sharing a seed, "
      "independent of global random module state: OK")

assert sorted(epoch_a) == sorted(items), "shuffled epoch must still visit every example exactly once"
print("DataLoader(shuffle=True) visits every example exactly once per epoch: OK")

# Re-iterating the SAME loader reshuffles each epoch (per-epoch
# reshuffle semantics), so two successive epochs from one loader
# instance should generally differ in order while still covering every
# example.
loader_c = DataLoader(items, batch_size=5, shuffle=True, seed=7)
epoch1 = [b for batch in loader_c for b in batch]
epoch2 = [b for batch in loader_c for b in batch]
assert sorted(epoch1) == sorted(items) and sorted(epoch2) == sorted(items)
assert epoch1 != epoch2, "successive epochs from the same loader should reshuffle, not repeat the identical order"
print("DataLoader(shuffle=True) reshuffles on every new __iter__() (per-epoch reshuffle): OK")

# ---------------------------------------------------------------------
# 3. drop_last: trailing partial batch dropped or kept, and __len__()
#    matches in both modes.
# ---------------------------------------------------------------------

loader_keep = DataLoader(items, batch_size=5, shuffle=False, drop_last=False)
loader_drop = DataLoader(items, batch_size=5, shuffle=False, drop_last=True)
batches_keep = list(loader_keep)
batches_drop = list(loader_drop)
assert len(batches_keep) == 5 and len(batches_keep[-1]) == 3, "23 items / batch_size=5 should keep a final batch of 3"
assert len(batches_drop) == 4 and all(len(b) == 5 for b in batches_drop), "drop_last=True must drop the partial batch"
assert len(loader_keep) == 5 and len(loader_drop) == 4, "__len__() must match the actual number of batches yielded"
print("DataLoader drop_last (both the trailing batch and __len__()): OK")

# ---------------------------------------------------------------------
# 4. _default_collate: tuple-of-(x,y) transposes into (list_x, list_y);
#    plain (non-tuple) items collate to a plain list.
# ---------------------------------------------------------------------

collated = _default_collate([(1, "a"), (2, "b"), (3, "c")])
assert collated == ([1, 2, 3], ["a", "b", "c"]), f"tuple collation mismatch: {collated}"
print("_default_collate transposes a batch of (x, y) tuples correctly: OK")

collated_plain = _default_collate([10, 20, 30])
assert collated_plain == [10, 20, 30], f"plain-value collation mismatch: {collated_plain}"
print("_default_collate handles a batch of plain (non-tuple) values: OK")


# ---------------------------------------------------------------------
# 5. Practical end-to-end test: the same "does this sequence contain
#    the marker token" classifier test_embedding.py trains by hand,
#    retrained here through a real Dataset/DataLoader instead --
#    proving this is a genuine drop-in replacement, not just correct
#    in isolation.
# ---------------------------------------------------------------------

rng = random.Random(0)

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


class MarkerDataset(Dataset):
    def __init__(self, n):
        self.examples = [make_example() for _ in range(n)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class MarkerClassifier(nn.Module):
    def __init__(self, seed=0):
        self.embed = nn.Embedding(VOCAB, EMBED_DIM, seed=seed)
        self.fc = nn.Linear(EMBED_DIM, 2, seed=seed + 1)

    def forward(self, batch_ids):
        e = self.embed(batch_ids)
        pooled = e.mean(1)
        return self.fc(pooled)


train_ds = MarkerDataset(200)
train_loader = DataLoader(train_ds, batch_size=20, shuffle=True, seed=3)

model = MarkerClassifier(seed=7)
opt = optim.Adam(model.parameters(), lr=0.05)

EPOCHS = 60
for epoch in range(EPOCHS):
    total_loss = 0.0
    n_seen = 0
    for batch_ids, batch_labels in train_loader:
        targets = core.from_flat(
            [1.0 if c == lbl else 0.0 for lbl in batch_labels for c in range(2)], [len(batch_ids), 2])
        logits = model(batch_ids)
        loss = logits.cross_entropy(targets)
        model.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.tolist()[0] * len(batch_ids)
        n_seen += len(batch_ids)
    if epoch % 15 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch}: avg loss {total_loss / n_seen:.4f}")

test_ds = MarkerDataset(100)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
correct = 0
for batch_ids, batch_labels in test_loader:
    logits = model(batch_ids)
    pred = 0 if logits.tolist()[0] > logits.tolist()[1] else 1
    correct += int(pred == batch_labels[0])

accuracy = correct / len(test_ds)
print(f"MarkerClassifier (trained via Dataset/DataLoader) test accuracy: {accuracy:.2%}")
assert accuracy >= 0.9, f"expected >=90% test accuracy training through DataLoader, got {accuracy:.2%}"

print("\nDataset/DataLoader test passed.")
