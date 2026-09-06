"""Dataset/DataLoader -- replaces the hand-rolled "shuffle a list of
indices, slice it into chunks" boilerplate every training loop in this
project (test_embedding.py's MarkerClassifier, test_optim_utils.py's
practical check, examples/mnist/train_mnist.py) has been writing by
hand up to now.

Map-style only (an object with `__len__`/`__getitem__`, indexed by
plain Python int) -- no iterable-style datasets, no multi-process
worker pool, no pinned memory, no custom samplers. Kansai has no
threading/IPC infrastructure that would make out-of-process workers
meaningful yet (see distributed.py's own concurrent-dispatch work for
what actually exists: real threads across DeviceMesh devices, not
data-loading workers), and nothing in this project's own training
loops has ever been slow enough for data loading itself to be the
bottleneck -- adding worker-process plumbing now would be exactly the
kind of premature generality this project's own conventions avoid.

Collation deliberately stops at "grouped into per-field Python lists,"
not built into a Tensor: different model inputs want different
shapes/types from the exact same DataLoader (a flat `core.Tensor` for
a Linear model's features, a plain nested list of ints for
`nn.Embedding`'s token ids -- see Tensor::index_select's own doc
comment for why indices are plain ints, never a Tensor), so forcing
one Tensor-construction convention into the loader would be wrong for
half of what it needs to feed.
"""

import random


class Dataset:
    """Base class for a map-style dataset: override `__len__` and
    `__getitem__`. Not strictly required to subclass this -- DataLoader
    only ever calls `len(dataset)` and `dataset[i]`, so any object
    implementing those two works -- but subclassing documents intent
    and gives a clear NotImplementedError instead of a confusing
    TypeError if a method is missing."""

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


def _default_collate(items):
    """A batch of `(x, y)` tuples becomes `(list_of_x, list_of_y)` --
    transposing rows into columns, the same shape every training loop
    in this project already builds by hand (`[train_seqs[i] for i in
    idx]`, `[train_labels[i] for i in idx]` in test_embedding.py, say)
    -- via `zip(*items)`. A batch of plain (non-tuple) items collates
    to a plain list, for a Dataset whose `__getitem__` returns a single
    value rather than an (x, y) pair."""
    if items and isinstance(items[0], tuple):
        return tuple(list(col) for col in zip(*items))
    return list(items)


class DataLoader:
    """Iterates a Dataset in batches, optionally shuffled every epoch.
    `shuffle=True` reshuffles on EVERY `__iter__()` call (i.e. every
    `for batch in loader:` loop, matching PyTorch's own per-epoch
    reshuffle semantics) using this loader's own `random.Random(seed)`
    instance, not the global `random` module -- so two loaders built
    with the same `seed` produce identical batch orderings regardless
    of what other code has called `random.*` in the meantime, and a
    fixed `seed` makes a run reproducible.
    """

    def __init__(self, dataset, batch_size: int = 1, shuffle: bool = False,
                 seed: int = 0, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._rng = random.Random(seed)

    def __iter__(self):
        n = len(self.dataset)
        indices = list(range(n))
        if self.shuffle:
            self._rng.shuffle(indices)
        for start in range(0, n, self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break
            items = [self.dataset[i] for i in batch_indices]
            yield _default_collate(items)

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
