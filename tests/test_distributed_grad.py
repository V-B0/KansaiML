"""Phase 4: distributed gradients. dtensor_grad(graph, wrt, wrt_placements,
*dtensor_args) closes the gap dtensor_run's own docstring left open --
auto-inserted collectives during backward, all-reduce (sum) for a
Replicate()'d gradient and all-gather (concat) for a Shard()'d one --
run on the same two real backends (kir.run for "cpu", kir.run_metal for
"metal") dtensor_run already proves forward, not a simulation of either.

The correctness bar here is exactly the standard data-parallel-training
claim: the gradient computed from N devices each seeing 1/N of the batch,
combined by these collectives, must equal the gradient computed from one
device seeing the whole batch at once. That's checked directly against
kir.grad() run on the unsharded graph -- not re-deriving kir.grad() vs
eager agreement, which test_kir_grad.py already covers independently.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import nn, kir
from kansai import _core as core
from kansai.distributed import DeviceMesh, Shard, Replicate, DTensor, dtensor_grad

TOL = 1e-4

if not core.metal_available():
    print("No Metal device available -- skipping the distributed gradient test.")
    sys.exit(0)


def check_close(label, a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < TOL, f"{label}[{i}]: got={x:.6f} expected={y:.6f}"
    print(f"{label}: OK ({len(a)} elements, max diff {max(abs(x - y) for x, y in zip(a, b)):.2e})")


# ---------------------------------------------------------------------
# 1. The real test: a Linear+ReLU model, batch sharded [cpu, metal] via
#    Shard(0), weight/bias replicated (baked into the graph as
#    constants, per find_constant() -- the ordinary data-parallel
#    shape). Loss is sum(), not mean() -- see dtensor_grad's own
#    docstring for exactly why that matters: sum() over the full batch
#    is additive across shards, so summing each device's local gradient
#    reconstructs the true full-batch gradient with no extra scaling.
# ---------------------------------------------------------------------

mesh = DeviceMesh(["cpu", "metal"])
print(f"mesh: {mesh}")

model = nn.Linear(4, 3, seed=7)
batch = 8  # divides evenly across 2 devices
X = kansai.randn([batch, 4], std=1.0, seed=1)

graph = kir.trace(lambda x: model(x).relu().sum(), X)

wrt = [kir.find_constant(graph, model.weight), kir.find_constant(graph, model.bias), graph.inputs[0]]
wrt_placements = [Replicate(), Replicate(), Shard(0)]

X_dt = DTensor.from_tensor(X, mesh, Shard(0))
print(f"X sharded: {X_dt}")

dW, db, dX = dtensor_grad(graph, wrt, wrt_placements, X_dt)
print(f"  dW shape={list(dW.shape)}  db shape={list(db.shape)}  dX shape={list(dX.shape)}")

bwd_full = kir.grad(graph, wrt)
dW_ref, db_ref, dX_ref = kir.run(bwd_full, X)

check_close("dtensor_grad dW (all-reduce) vs full-batch kir.grad", dW.tolist(), dW_ref.tolist())
check_close("dtensor_grad db (all-reduce) vs full-batch kir.grad", db.tolist(), db_ref.tolist())
check_close("dtensor_grad dX (all-gather) vs full-batch kir.grad", dX.tolist(), dX_ref.tolist())

# ---------------------------------------------------------------------
# 2. Single wrt (len(wrt) == 1): grad() returns a single-output graph,
#    not a "tuple" node -- exercises that dtensor_grad's per-device
#    tuple-vs-single-Tensor handling covers both shapes of grad()'s
#    return value, on both backends.
# ---------------------------------------------------------------------

(dW_only,) = dtensor_grad(graph, [wrt[0]], [Replicate()], X_dt)
check_close("dtensor_grad single-wrt dW vs full-batch kir.grad", dW_only.tolist(), dW_ref.tolist())

# ---------------------------------------------------------------------
# 3. mean()-reduced loss, not just sum(): dtensor_grad's docstring
#    explains why plain all-reduce-summing is *already* correct here
#    with no rescale -- mean()'s vjp bakes in a scale computed from
#    `graph`'s own trace-time (full-batch) shape, so every device's
#    local gradient already carries the right 1/(full batch size)
#    factor, not a per-shard one. Checked directly rather than taken on
#    faith: the distributed result must match full-batch kir.grad()
#    exactly, the same bar section 1 held sum() to.
# ---------------------------------------------------------------------

mean_graph = kir.trace(lambda x: model(x).relu().mean(), X)
mean_wrt = [kir.find_constant(mean_graph, model.weight)]

(dW_mean_distributed,) = dtensor_grad(mean_graph, mean_wrt, [Replicate()], X_dt)

full_batch_mean_bwd = kir.grad(mean_graph, mean_wrt)
dW_mean_full = kir.run(full_batch_mean_bwd, X)

check_close("dtensor_grad dW with mean() loss vs full-batch kir.grad",
            dW_mean_distributed.tolist(), dW_mean_full.tolist())

print("\nDistributed gradients test passed.")
