"""Phase 4: DeviceMesh / DTensor. Every shard's computation in this file
runs through the real backend it's dispatched to (kir.run for "cpu",
kir.run_metal for "metal") -- nothing here is simulated.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import nn, kir
from kansai import _core as core
from kansai.distributed import DeviceMesh, Shard, Replicate, DTensor, dtensor_run, _split_tensor, _concat_tensors

TOL = 1e-4


def check_close(label, a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < TOL, f"{label}[{i}]: got={x:.6f} expected={y:.6f}"
    print(f"{label}: OK ({len(a)} elements)")


# ---------------------------------------------------------------------
# 1. _split_tensor / _concat_tensors round-trip, including an UNEVEN
#    split (7 rows across 3 pieces -- 3,2,2) to prove the chunk-size
#    logic is correct, not just the common evenly-divisible case.
# ---------------------------------------------------------------------

rng = random.Random(0)
vals = [rng.uniform(-1, 1) for _ in range(7 * 4)]
t = kansai.from_flat(vals, [7, 4])

pieces = _split_tensor(t, 0, 3)
piece_shapes = [list(p.shape) for p in pieces]
assert piece_shapes == [[3, 4], [2, 4], [2, 4]], f"expected [3,4],[2,4],[2,4], got {piece_shapes}"
print("uneven split (7 -> 3+2+2): shapes OK", piece_shapes)

rebuilt = _concat_tensors(pieces, 0)
check_close("split+concat round-trip (uneven)", rebuilt.tolist(), vals)

# Also split along a non-leading dimension, to exercise the outer/inner
# stride bookkeeping (not just the simple dim-0 case).
vals2 = [rng.uniform(-1, 1) for _ in range(3 * 8)]
t2 = kansai.from_flat(vals2, [3, 8])
pieces2 = _split_tensor(t2, 1, 4)
assert [list(p.shape) for p in pieces2] == [[3, 2]] * 4
rebuilt2 = _concat_tensors(pieces2, 1)
check_close("split+concat round-trip (dim=1)", rebuilt2.tolist(), vals2)

# ---------------------------------------------------------------------
# 2. DeviceMesh validation: rejects an unknown device name outright.
# ---------------------------------------------------------------------

try:
    DeviceMesh(["cpu", "tpu"])
    raise AssertionError("DeviceMesh should have rejected an unknown device name")
except ValueError as e:
    assert "unknown device" in str(e)
    print("DeviceMesh correctly rejects an unknown device name")

if not core.metal_available():
    print("\nNo Metal device available -- skipping the two-backend dispatch tests.")
    sys.exit(0)

# ---------------------------------------------------------------------
# 3. The real test: a Linear+ReLU forward pass, batch sharded across
#    [cpu, metal] (Shard(0)) with weight/bias replicated on both -- the
#    actual shape of data-parallel training (parameters replicated,
#    batch split). Gathered result checked against running the exact
#    same graph unsharded through kir.run().
# ---------------------------------------------------------------------

mesh = DeviceMesh(["cpu", "metal"])
print(f"\nmesh: {mesh}")

model = nn.Linear(4, 3, seed=7)
batch = 8  # divides evenly across 2 devices -- see dtensor_run's docstring for why that matters
X = kansai.randn([batch, 4], std=1.0, seed=1)

graph = kir.trace(lambda x: model(x).relu(), X)

X_dt = DTensor.from_tensor(X, mesh, Shard(0))
print(f"X sharded: {X_dt}")

w_dt = DTensor.from_tensor(model.weight, mesh, Replicate())
b_dt = DTensor.from_tensor(model.bias, mesh, Replicate())

# dtensor_run's graph was traced against a single placeholder (X) --
# the weight/bias are "constant" nodes baked in at trace time (same
# capture kir.jit already relies on), so only X needs to be passed as
# the traced placeholder argument here; each shard's constant weight/
# bias values are whatever model.weight/model.bias already are.
result_dt = dtensor_run(graph, Shard(0), X_dt)

for i, (device, shard) in enumerate(zip(mesh.devices, result_dt.shards)):
    print(f"  shard {i} ({device}): shape={list(shard.shape)}")

gathered = result_dt.to_tensor()
expected = kir.run(graph, X).tolist()
check_close("dtensor_run (sharded, cpu+metal) vs unsharded kir.run", gathered.tolist(), expected)

# ---------------------------------------------------------------------
# 4. Full Replicate(): every device computes the whole thing
#    independently: both shards' results must be identical to each
#    other and to the unsharded result.
# ---------------------------------------------------------------------

X_rep = DTensor.from_tensor(X, mesh, Replicate())
result_rep = dtensor_run(graph, Replicate(), X_rep)
check_close("Replicate() shard 0 (cpu) vs unsharded", result_rep.shards[0].tolist(), expected)
check_close("Replicate() shard 1 (metal) vs unsharded", result_rep.shards[1].tolist(), expected)
gathered_rep = result_rep.to_tensor()
check_close("Replicate() gathered (= shard 0) vs unsharded", gathered_rep.tolist(), expected)

print("\nDeviceMesh/DTensor test passed.")
