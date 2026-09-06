"""Phase 4: post-training int8 weight quantization. See
python/kansai/quantize.py's own docstring for the full design (symmetric
per-tensor int8, forward-only, dequantize-then-matmul) and exactly what
it does and doesn't prove -- memory footprint, not FLOPs.

Three checks, each independent: quantize/dequantize error is bounded by
the known quantization step (not just "looks close"), the memory
footprint claim is an actual measured 4x, not an assumed one, and a
model trained to convergence keeps working (same qualitative behavior)
after its weights are quantized -- the same "does it still actually
work" bar test_xor.py and test_conv2d.py's tiny conv net already hold
their own claims to.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import nn, optim
from kansai.quantize import QTensor, QLinear, quantize_symmetric, dequantize

TOL = 1e-4


def check_close(label, a, b, tol=TOL):
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < tol, f"{label}[{i}]: got={x:.6f} expected={y:.6f}"
    print(f"{label}: OK ({len(a)} elements, max diff {max(abs(x - y) for x, y in zip(a, b)):.2e})")


# ---------------------------------------------------------------------
# 1. Round-trip error is bounded by the known quantization step
#    (scale/2 -- half the gap between adjacent representable values),
#    not just eyeballed as "small". Checked at a size big enough that a
#    lucky small sample couldn't hide a real bug.
# ---------------------------------------------------------------------

rng = random.Random(0)
vals = [rng.uniform(-3.0, 3.0) for _ in range(1000)]
t = kansai.from_flat(vals, [1000])

q, scale, shape = quantize_symmetric(t)
assert shape == [1000]
assert all(-127 <= v <= 127 for v in q), "quantized values must stay in int8's usable range"

recovered = dequantize(q, scale, shape).tolist()
max_err = max(abs(a - b) for a, b in zip(vals, recovered))
print(f"quantize/dequantize: scale={scale:.6f}, max round-trip error={max_err:.6f} "
      f"(bound: scale/2={scale / 2:.6f})")
assert max_err <= scale / 2 + 1e-6, "round-trip error exceeded the known quantization step bound"

# A value of exactly zero must quantize to exactly zero -- symmetric
# quantization's whole point is a zero-point-free scheme, which only
# holds if 0.0 maps to q=0 exactly (not off by a rounding fluke).
q_zero, _, _ = quantize_symmetric(kansai.from_flat([0.0, 1.0, -1.0], [3]))
assert q_zero[0] == 0, f"expected exact zero to quantize to q=0, got {q_zero[0]}"
print("exact zero quantizes to exactly q=0: OK")

# ---------------------------------------------------------------------
# 2. Memory footprint: an actual measured 4x, not an assumed one.
# ---------------------------------------------------------------------

big = kansai.randn([256, 256], std=1.0, seed=1)
qbig = QTensor.from_tensor(big)
float32_bytes = 4 * big.numel()
int8_bytes = qbig.nbytes()
ratio = float32_bytes / int8_bytes
print(f"memory: float32={float32_bytes} bytes, int8={int8_bytes} bytes, ratio={ratio:.2f}x")
assert abs(ratio - 4.0) < 1e-9, f"expected exactly 4x (1 byte/elem vs 4), got {ratio}x"

# ---------------------------------------------------------------------
# 3. A real trained model keeps working after quantization. Reuses
#    test_xor.py's exact architecture and training setup (same seeds,
#    same convergence bar) so this isolates quantization's own effect
#    rather than re-testing whether XOR training itself works.
# ---------------------------------------------------------------------

X = kansai.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = kansai.tensor([[0], [1], [1], [0]])

model = nn.Sequential(
    nn.Linear(2, 8, seed=3),
    nn.ReLU(),
    nn.Linear(8, 1, seed=4),
)
opt = optim.SGD(model.parameters(), lr=0.1)

loss = None
for step in range(500):
    pred = model(X)
    diff = pred.sub(Y)
    loss = diff.mul(diff).mean()
    model.zero_grad()
    loss.backward()
    opt.step()

final_loss = loss.tolist()[0]
assert final_loss < 0.05, "XOR did not converge -- can't test quantization on a model that didn't train"
print(f"trained XOR model: final loss {final_loss:.6f}")

float_preds = model(X).tolist()

qmodel = nn.Sequential(
    QLinear(model.layers[0]),
    nn.ReLU(),
    QLinear(model.layers[2]),
)
quant_preds = qmodel(X).tolist()

print(f"float32 predictions:   {[round(v, 4) for v in float_preds]}")
print(f"quantized predictions: {[round(v, 4) for v in quant_preds]}")

# Not bit-exact (that's the whole point of measuring quantization error
# rather than claiming zero), but close enough that every prediction
# still rounds to the correct XOR class -- the same practical bar
# test_xor.py's own float32 model is held to.
check_close("quantized vs float32 XOR predictions", quant_preds, float_preds, tol=0.1)
quant_classes = [round(v) for v in quant_preds]
expected_classes = [0, 1, 1, 0]
assert quant_classes == expected_classes, \
    f"quantized model's predictions no longer classify XOR correctly: {quant_classes}"
print("quantized model still classifies XOR correctly")

print("\nQuantization test passed.")
