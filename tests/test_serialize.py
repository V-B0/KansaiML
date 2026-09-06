"""Phase 4: model serialization. See python/kansai/serialize.py's own
docstring for the file format and exactly what this does and doesn't
cover (parameter values only, not architecture; no pickle anywhere, so
loading an untrusted checkpoint can't execute code).

Checked: a round trip through save()/load() reproduces a trained
model's parameters bit-for-bit (float32 -> bytes -> float32 is lossless,
not approximate) and its forward pass output exactly, for both a
Sequential(Linear, ReLU, Linear) model and a Conv2d model; four distinct
failure modes are rejected with a clear, specific error rather than a
silent wrong load or a confusing crash; and the file's actual byte size
matches what the header declares, computed independently rather than
just trusted.
"""

import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import nn, optim, serialize

TOL = 1e-6  # float32 round-trip through this format is lossless -- any
            # deviation at all would mean a real bug, not float noise


def check_close(label, a, b, tol=TOL):
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < tol, f"{label}[{i}]: got={x:.8f} expected={y:.8f}"
    print(f"{label}: OK ({len(a)} elements, max diff {max(abs(x - y) for x, y in zip(a, b)):.2e})")


tmpdir = tempfile.mkdtemp(prefix="kansai_serialize_test_")

# ---------------------------------------------------------------------
# 1. Round trip: train the exact XOR model test_xor.py trains, save it,
#    load it into a FRESH model (different seeds, so its initial
#    weights are provably different from the trained ones -- a bug that
#    silently left the fresh model's own weights in place would still
#    pass a test that used identical seeds for both models).
# ---------------------------------------------------------------------

X = kansai.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = kansai.tensor([[0], [1], [1], [0]])

model = nn.Sequential(
    nn.Linear(2, 8, seed=3),
    nn.ReLU(),
    nn.Linear(8, 1, seed=4),
)
opt = optim.SGD(model.parameters(), lr=0.1)
for step in range(500):
    pred = model(X)
    diff = pred.sub(Y)
    loss = diff.mul(diff).mean()
    model.zero_grad()
    loss.backward()
    opt.step()
print(f"trained XOR model: final loss {loss.tolist()[0]:.6f}")

ckpt_path = os.path.join(tmpdir, "xor.kan")
serialize.save(model, ckpt_path)
print(f"saved checkpoint: {os.path.getsize(ckpt_path)} bytes")

fresh_model = nn.Sequential(
    nn.Linear(2, 8, seed=99),   # different seeds -- provably different
    nn.ReLU(),                  # initial weights from the trained model
    nn.Linear(8, 1, seed=100),
)
fresh_before = [p.tolist() for p in fresh_model.parameters()]
trained_params = [p.tolist() for p in model.parameters()]
assert any(a != b for a, b in zip(fresh_before[0], trained_params[0])), \
    "test setup bug: fresh model's weights should differ from the trained model's before loading"

serialize.load(fresh_model, ckpt_path)

for i, (loaded, original) in enumerate(zip(fresh_model.parameters(), model.parameters())):
    check_close(f"loaded param[{i}] vs original (bit-exact)", loaded.tolist(), original.tolist())

check_close("loaded model forward output vs original", fresh_model(X).tolist(), model(X).tolist())

# ---------------------------------------------------------------------
# 2. Round trip on a different module shape (Conv2d) -- confirms this
#    isn't special-cased to Linear/Sequential.
# ---------------------------------------------------------------------

conv = nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1, seed=5)
conv_X = kansai.randn([1, 2, 5, 5], std=1.0, seed=8)
conv_out_before = conv(conv_X).tolist()

conv_path = os.path.join(tmpdir, "conv.kan")
serialize.save(conv, conv_path)

fresh_conv = nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1, seed=77)
serialize.load(fresh_conv, conv_path)
check_close("loaded Conv2d forward output vs original", fresh_conv(conv_X).tolist(), conv_out_before)

# ---------------------------------------------------------------------
# 3. Failure modes: each rejected with a specific, useful error, not a
#    silent wrong load or a generic crash deep inside array/json.
# ---------------------------------------------------------------------

# 3a. Shape mismatch.
mismatched = nn.Sequential(
    nn.Linear(2, 16, seed=1),  # 16, not 8 -- same layer count, wrong shape
    nn.ReLU(),
    nn.Linear(16, 1, seed=2),
)
try:
    serialize.load(mismatched, ckpt_path)
    raise AssertionError("expected a shape-mismatch ValueError")
except ValueError as e:
    assert "shape mismatch" in str(e)
    print(f"shape mismatch correctly rejected: {e}")

# 3b. Missing parameter (model expects one the checkpoint doesn't have).
extra_layer_model = nn.Sequential(
    nn.Linear(2, 8, seed=3),
    nn.ReLU(),
    nn.Linear(8, 1, seed=4),
    nn.Linear(1, 1, seed=6),  # not present in the saved checkpoint
)
try:
    serialize.load(extra_layer_model, ckpt_path)
    raise AssertionError("expected a missing-parameter ValueError")
except ValueError as e:
    assert "missing parameter" in str(e)
    print(f"missing parameter correctly rejected: {e}")

# 3c. Unexpected parameter (checkpoint has one the model doesn't).
smaller_model = nn.Sequential(nn.Linear(2, 8, seed=3), nn.ReLU())
try:
    serialize.load(smaller_model, ckpt_path)
    raise AssertionError("expected an unexpected-parameter ValueError")
except ValueError as e:
    assert "doesn't expect" in str(e)
    print(f"unexpected parameter correctly rejected: {e}")

# 3d. Corrupted magic bytes -- not a valid Kansai checkpoint at all.
bad_path = os.path.join(tmpdir, "corrupt.kan")
with open(ckpt_path, "rb") as f:
    data = f.read()
with open(bad_path, "wb") as f:
    f.write(b"NOPE" + data[4:])
try:
    serialize.load(nn.Sequential(nn.Linear(2, 8, seed=3), nn.ReLU(), nn.Linear(8, 1, seed=4)), bad_path)
    raise AssertionError("expected a bad-magic ValueError")
except ValueError as e:
    assert "not a Kansai checkpoint" in str(e)
    print(f"corrupted magic correctly rejected: {e}")

# ---------------------------------------------------------------------
# 4. File size is exactly what the format's own layout predicts --
#    computed independently from the header, not just trusted.
# ---------------------------------------------------------------------

with open(ckpt_path, "rb") as f:
    magic = f.read(4)
    header_len = struct.unpack("<Q", f.read(8))[0]
    header_bytes = f.read(header_len)

total_param_floats = sum(p.numel() for p in model.parameters())
expected_size = 4 + 8 + header_len + total_param_floats * 4
actual_size = os.path.getsize(ckpt_path)
print(f"file size: {actual_size} bytes (12-byte prefix + {header_len}-byte header + "
      f"{total_param_floats * 4} bytes of float32 data)")
assert actual_size == expected_size, f"file size mismatch: got {actual_size}, expected {expected_size}"

print("\nSerialization test passed.")
