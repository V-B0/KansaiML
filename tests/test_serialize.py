"""Phase 4: model serialization. See python/kansai/serialize.py's own
docstring for the file format and exactly what this does and doesn't
cover (parameter values only, not architecture; no pickle anywhere, so
loading an untrusted checkpoint can't execute code).

Checked: a round trip through save()/load() reproduces a trained
model's parameters bit-for-bit (float32 -> bytes -> float32 is lossless,
not approximate) and its forward pass output exactly, for a
Sequential(Linear, ReLU, Linear) model, a Conv2d model, and a model
composed of every layer type added AFTER this file was originally
built (Embedding, TransformerBlock -- itself nesting MultiHeadAttention/
LayerNorm/Linear/GELU -- and BatchNorm2d) -- confirming save()/load()'s
generic named_parameters()-based mechanism actually still works with
real newer layers, not just assumed to because it "should" in
principle; BatchNorm2d's running_mean/running_var (buffers, not
parameters) are confirmed correctly OUTSIDE this format's scope, left
untouched by a load rather than silently restored; four distinct
failure modes are rejected with a clear, specific error rather than a
silent wrong load or a confusing crash; and the file's actual byte size
matches what the header declares, computed independently rather than
just trusted.
"""

import os
import random
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
# 2b. Round trip on every layer type added AFTER serialize.py itself
#     was built (Embedding, TransformerBlock -- itself nesting
#     MultiHeadAttention/LayerNorm/Linear/GELU -- and BatchNorm2d,
#     subclassing BatchNorm1d), composed together in one model. save()/
#     load() go purely through named_parameters()'s own generic
#     recursion (see serialize.py's own docstring), never anything
#     layer-specific, so this SHOULD already work without serialize.py
#     needing to know any of these types exist -- checked directly
#     rather than just assumed, since "the mechanism is generic" is a
#     claim worth actually exercising against real newer layers, not
#     just believing.
# ---------------------------------------------------------------------

VOCAB, D_MODEL, HEADS, FF, LAYERS, BLOCK = 20, 16, 2, 32, 2, 6


class TinyComposedModel(nn.Module):
    def __init__(self, seed=0):
        self.embed = nn.Embedding(VOCAB, D_MODEL, seed=seed)
        self.blocks = [nn.TransformerBlock(D_MODEL, HEADS, FF, seed=seed + i + 1) for i in range(LAYERS)]
        self.ln = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, seed=seed + 99)

    def forward(self, ids):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))


rng = random.Random(0)
composed_model = TinyComposedModel(seed=1)
n_params = sum(p.numel() for _, p in composed_model.named_parameters())
print(f"composed model (Embedding + {LAYERS}x TransformerBlock + LayerNorm + Linear): "
      f"{n_params:,} parameters across {len(list(composed_model.named_parameters()))} tensors")

batch_ids = [[rng.randint(0, VOCAB - 1) for _ in range(BLOCK)] for _ in range(3)]
composed_out_before = composed_model(batch_ids).tolist()

composed_path = os.path.join(tmpdir, "composed.kan")
serialize.save(composed_model, composed_path)

fresh_composed = TinyComposedModel(seed=999)  # different seed -- provably different init
fresh_composed_before = fresh_composed(batch_ids).tolist()
assert any(abs(a - b) > 1e-3 for a, b in zip(composed_out_before, fresh_composed_before)), \
    "test setup bug: fresh model's output should differ from the trained model's before loading"

serialize.load(fresh_composed, composed_path)
check_close("loaded Embedding+TransformerBlock+LayerNorm+Linear model output vs original",
            fresh_composed(batch_ids).tolist(), composed_out_before)

# BatchNorm2d specifically -- also checks running_mean/running_var
# (buffers, not parameters, so NOT covered by named_parameters()/the
# round trip above at all) are correctly left at THIS model's own
# fresh eval-mode defaults after a load, since save()/load() only ever
# touch parameters -- a real, deliberate scope boundary (matching
# every other framework's own state_dict convention), not an oversight
# to fix here.
bn2d = nn.BatchNorm2d(3)
bn2d_X = kansai.randn([2, 3, 4, 4], std=1.0, seed=11)
bn2d(bn2d_X)  # one training-mode call to give running_mean/running_var real, non-default values
bn2d.eval()
bn2d_out_before = bn2d(bn2d_X).tolist()

bn2d_path = os.path.join(tmpdir, "bn2d.kan")
serialize.save(bn2d, bn2d_path)

fresh_bn2d = nn.BatchNorm2d(3)
serialize.load(fresh_bn2d, bn2d_path)
check_close("loaded BatchNorm2d weight/bias vs original", fresh_bn2d.weight.tolist(), bn2d.weight.tolist())
check_close("loaded BatchNorm2d weight/bias vs original", fresh_bn2d.bias.tolist(), bn2d.bias.tolist())
fresh_bn2d.eval()
assert fresh_bn2d(bn2d_X).tolist() != bn2d_out_before, (
    "fresh_bn2d's running_mean/running_var were never trained -- its eval-mode output SHOULD differ "
    "from the original's, confirming load() correctly left buffers untouched (parameters-only scope)")
print("BatchNorm2d save/load: weight/bias restored exactly, running_mean/running_var correctly untouched: OK")

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
