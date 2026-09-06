"""Downloads and caches the real MNIST dataset (60,000 training images,
10,000 test images, 28x28 grayscale handwritten digits, labels 0-9) from
the same mirror torchvision itself uses -- the original host
(yann.lecun.com) has been unreliable for years, this one hasn't.

Deliberately not a dependency of the framework itself, or even of this
example beyond this one file: train_mnist.py imports load_mnist() from
here, but nothing in python/kansai/ knows this module exists. Downloaded
files are cached in this directory (gitignored -- see the project's own
.gitignore -- so the ~11MB compressed / ~55MB decompressed dataset never
ends up in git history) and re-used on every subsequent run.
"""

import gzip
import os
import struct
import urllib.request

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _download(name: str, filename: str) -> str:
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return path
    os.makedirs(DATA_DIR, exist_ok=True)
    url = MIRROR + filename
    print(f"downloading {url} ...")
    urllib.request.urlretrieve(url, path)
    return path


def _read_images(path: str):
    """Parses the IDX3 ubyte image format: a 16-byte header (magic,
    count, rows, cols, all big-endian uint32) followed by count*rows*cols
    raw pixel bytes (0-255). Returns (flat pixel values in [0,1], count,
    rows, cols) -- flat, not nested, since that's what core.from_flat
    wants directly."""
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic number in {path}: {magic}"
        raw = f.read(count * rows * cols)
    return [b / 255.0 for b in raw], count, rows, cols


def _read_labels(path: str):
    """IDX1 ubyte label format: an 8-byte header (magic, count) followed
    by count raw label bytes (0-9). Returns a plain list of ints."""
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic number in {path}: {magic}"
        raw = f.read(count)
    return list(raw)


def _one_hot(labels: list, num_classes: int = 10) -> list:
    """Flat one-hot encoding -- cross_entropy takes one-hot targets, not
    class indices (see Tensor::cross_entropy's own declaration in
    core/include/kansai/Tensor.hpp for why: no integer gather op exists
    yet)."""
    flat = [0.0] * (len(labels) * num_classes)
    for i, label in enumerate(labels):
        flat[i * num_classes + label] = 1.0
    return flat


def load_mnist():
    """Returns a dict with train_images/train_labels_onehot/train_labels
    (raw ints, for accuracy reporting)/test_images/test_labels_onehot/
    test_labels, plus rows/cols -- everything train_mnist.py needs, as
    plain Python lists (flat pixel/one-hot values) ready for
    kansai.from_flat, not yet wrapped as Tensors (so this module has no
    dependency on kansai itself, just stdlib)."""
    train_img_path = _download("train_images", FILES["train_images"])
    train_lbl_path = _download("train_labels", FILES["train_labels"])
    test_img_path = _download("test_images", FILES["test_images"])
    test_lbl_path = _download("test_labels", FILES["test_labels"])

    train_pixels, train_n, rows, cols = _read_images(train_img_path)
    train_labels = _read_labels(train_lbl_path)
    test_pixels, test_n, _, _ = _read_images(test_img_path)
    test_labels = _read_labels(test_lbl_path)

    assert len(train_labels) == train_n
    assert len(test_labels) == test_n

    return {
        "train_images": train_pixels,
        "train_labels": train_labels,
        "train_labels_onehot": _one_hot(train_labels),
        "train_n": train_n,
        "test_images": test_pixels,
        "test_labels": test_labels,
        "test_labels_onehot": _one_hot(test_labels),
        "test_n": test_n,
        "rows": rows,
        "cols": cols,
    }


if __name__ == "__main__":
    data = load_mnist()
    print(f"train: {data['train_n']} images, test: {data['test_n']} images, "
          f"{data['rows']}x{data['cols']} pixels each")
    print(f"first training label: {data['train_labels'][0]}")
