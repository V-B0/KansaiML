"""Kansai's first real end-to-end benchmark on an actual, standard
dataset -- every other test in this project trains on XOR, a tiny
synthetic conv net, or a hand-generated 2D blob classifier. Real MNIST
(60,000 training images, 10,000 held-out test images, 28x28 grayscale
digits 0-9), a real mini-batch training loop, and a real test-set
accuracy number reported at the end -- the actual claim this project's
whole toolchain (Linear, BatchNorm1d, ReLU, cross_entropy, Adam) has to
back up together, not in isolation.

Run: python3 train_mnist.py (downloads MNIST on first run, ~11MB;
cached under examples/mnist/data/ afterward, gitignored).
"""

import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.path.insert(0, os.path.dirname(__file__))

import kansai
from kansai import nn, optim
from download_mnist import load_mnist

SEED = 0
BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-3
HIDDEN1 = 256
HIDDEN2 = 64
NUM_CLASSES = 10


def shuffle_once(pixels, onehot, labels, n, pixel_stride, label_stride, seed):
    """One fixed shuffle applied before training starts, not re-shuffled
    between epochs -- a real, stated simplification (Kansai has no
    gather/index_select op, so a genuinely random per-epoch shuffle
    would mean rebuilding the full flat dataset every epoch, an O(n)
    cost paid EVERY epoch instead of once). MNIST's own canonical file
    order isn't sorted by class already, so sequential mini-batches
    after this one shuffle still see a reasonable mix of digits each
    batch -- not identical to per-epoch reshuffling, but a defensible,
    honestly-stated middle ground for a benchmark script."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    new_pixels = [0.0] * (n * pixel_stride)
    new_onehot = [0.0] * (n * label_stride)
    new_labels = [0] * n
    for new_i, old_i in enumerate(idx):
        new_pixels[new_i * pixel_stride:(new_i + 1) * pixel_stride] = \
            pixels[old_i * pixel_stride:(old_i + 1) * pixel_stride]
        new_onehot[new_i * label_stride:(new_i + 1) * label_stride] = \
            onehot[old_i * label_stride:(old_i + 1) * label_stride]
        new_labels[new_i] = labels[old_i]
    return new_pixels, new_onehot, new_labels


def accuracy(model, images_flat, labels, n, pixel_stride, eval_batch=1000):
    """Runs in eval mode (so BatchNorm1d uses its running statistics,
    not this data's own -- see nn.BatchNorm1d's own docstring for why
    that distinction is real, not cosmetic), batched only to keep any
    one matmul at a modest size, not because a single 10000x784 matmul
    would actually be a problem for Accelerate."""
    model.eval()
    correct = 0
    for start in range(0, n, eval_batch):
        end = min(start + eval_batch, n)
        batch_n = end - start
        xb = kansai.from_flat(images_flat[start * pixel_stride:end * pixel_stride], [batch_n, pixel_stride])
        preds = model(xb).tolist()
        for i in range(batch_n):
            row = preds[i * NUM_CLASSES:(i + 1) * NUM_CLASSES]
            if row.index(max(row)) == labels[start + i]:
                correct += 1
    model.train()
    return correct / n


def main():
    print("Loading MNIST...")
    t0 = time.perf_counter()
    data = load_mnist()
    print(f"  {data['train_n']} train / {data['test_n']} test images, "
          f"{data['rows']}x{data['cols']} pixels, loaded in {time.perf_counter() - t0:.2f}s")

    pixel_stride = data["rows"] * data["cols"]
    train_images, train_onehot, train_labels = shuffle_once(
        data["train_images"], data["train_labels_onehot"], data["train_labels"],
        data["train_n"], pixel_stride, NUM_CLASSES, seed=SEED,
    )

    model = nn.Sequential(
        nn.Linear(pixel_stride, HIDDEN1, seed=SEED + 1),
        nn.BatchNorm1d(HIDDEN1),
        nn.ReLU(),
        nn.Linear(HIDDEN1, HIDDEN2, seed=SEED + 2),
        nn.ReLU(),
        nn.Linear(HIDDEN2, NUM_CLASSES, seed=SEED + 3),
    )
    opt = optim.Adam(model.parameters(), lr=LR)

    n_batches = data["train_n"] // BATCH_SIZE
    print(f"\nTraining: {EPOCHS} epochs x {n_batches} batches of {BATCH_SIZE} "
          f"({model.__class__.__name__}: Linear({pixel_stride}->{HIDDEN1}) -> BatchNorm1d -> ReLU "
          f"-> Linear({HIDDEN1}->{HIDDEN2}) -> ReLU -> Linear({HIDDEN2}->{NUM_CLASSES}))\n")

    train_start = time.perf_counter()
    for epoch in range(EPOCHS):
        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        for b in range(n_batches):
            start = b * BATCH_SIZE
            end = start + BATCH_SIZE
            xb = kansai.from_flat(train_images[start * pixel_stride:end * pixel_stride], [BATCH_SIZE, pixel_stride])
            yb = kansai.from_flat(train_onehot[start * NUM_CLASSES:end * NUM_CLASSES], [BATCH_SIZE, NUM_CLASSES])

            loss = model(xb).cross_entropy(yb)
            model.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.tolist()[0]

        epoch_time = time.perf_counter() - epoch_start
        avg_loss = epoch_loss / n_batches
        test_acc = accuracy(model, data["test_images"], data["test_labels"], data["test_n"], pixel_stride)
        print(f"epoch {epoch + 1:2d}/{EPOCHS}  loss {avg_loss:.4f}  test acc {test_acc:.2%}  "
              f"({epoch_time:.2f}s)")

    total_time = time.perf_counter() - train_start
    final_train_acc = accuracy(model, train_images, train_labels, data["train_n"], pixel_stride)
    final_test_acc = accuracy(model, data["test_images"], data["test_labels"], data["test_n"], pixel_stride)

    print(f"\nDone in {total_time:.1f}s total ({total_time / EPOCHS:.2f}s/epoch average).")
    print(f"Final train accuracy: {final_train_acc:.2%}")
    print(f"Final test accuracy:  {final_test_acc:.2%}  ({data['test_n']} held-out images, never trained on)")

    return final_test_acc, total_time


if __name__ == "__main__":
    main()
