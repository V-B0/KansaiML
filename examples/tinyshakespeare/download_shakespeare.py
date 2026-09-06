"""Downloads and caches the "tiny Shakespeare" text corpus (~1.1MB,
Shakespeare's plays concatenated into one plain-text file) -- the
standard toy dataset for character-level language modeling (Karpathy's
char-rnn, later nanoGPT, both train on exactly this file), from the
same repo those projects pull it from.

Character-level tokenization only: the vocabulary is built from every
UNIQUE character actually present in the corpus, not a fixed ASCII
range or a subword tokenizer -- Kansai has neither a byte-pair-encoding
implementation nor any tokenizer dependency at all, and a real,
integer-per-character vocabulary is the simplest thing that actually
works with `nn.Embedding` (indices are plain Python ints -- see
Tensor::index_select's own doc comment for why that's not a Tensor).

Deliberately not a dependency of the framework itself, or even of this
example beyond this one file: train_shakespeare.py imports
load_shakespeare() from here, but nothing in python/kansai/ knows this
module exists. The downloaded file is cached in this directory
(gitignored -- see the project's own .gitignore -- so the ~1.1MB
corpus never ends up in git history) and re-used on every subsequent
run.
"""

import os
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_PATH = os.path.join(DATA_DIR, "input.txt")


def _download():
    if os.path.exists(DATA_PATH):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"downloading {URL} ...")
    urllib.request.urlretrieve(URL, DATA_PATH)


def load_shakespeare():
    """Returns (text, vocab, encode, decode): `text` is the raw corpus
    string; `vocab` is the sorted list of unique characters actually
    present (its length is the model's vocab size); `encode(str) ->
    list[int]` and `decode(list[int]) -> str` map characters to/from
    their position in `vocab`, the plain-int token ids nn.Embedding
    and Tensor.index_select both expect."""
    _download()
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    return text, vocab, encode, decode


if __name__ == "__main__":
    text, vocab, encode, decode = load_shakespeare()
    print(f"corpus length: {len(text)} characters, vocab size: {len(vocab)}")
    print(f"vocab: {''.join(vocab)!r}")
    print(f"first 200 characters:\n{text[:200]}")
    roundtrip = decode(encode(text[:200]))
    assert roundtrip == text[:200], "encode/decode roundtrip mismatch"
    print("\nencode/decode roundtrip: OK")
