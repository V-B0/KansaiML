import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import nn, kir
from kansai import _core as core

# ---------------------------------------------------------------------
# Build a representative graph: the same Linear -> ReLU -> Linear
# forward shape used throughout this project, traced once.
# ---------------------------------------------------------------------

model = nn.Sequential(nn.Linear(2, 8, seed=3), nn.ReLU(), nn.Linear(8, 1, seed=4))
X = kansai.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])

graph = kir.trace(lambda x: model(x), X)
plan = kir.plan_memory(graph)

print(f"graph: {len(graph.nodes)} nodes, {plan.num_computed_nodes} need a buffer, "
      f"{plan.num_slots} concurrent slots (peak)")
assert plan.num_slots < plan.num_computed_nodes, "the plan should find some reuse on this graph"

# ---------------------------------------------------------------------
# 1. Plan correctness: two nodes sharing a slot must never have
#    overlapping liveness. i is produced at its own index and last read
#    at plan.last_use[i]; j (produced later) may only reuse i's slot if
#    i is already dead (last_use[i] <= j's own index) by the time j is
#    produced.
# ---------------------------------------------------------------------

index_of = {n.id: idx for idx, n in enumerate(graph.nodes)}

for i, si in plan.slot_of.items():
    for j, sj in plan.slot_of.items():
        if i == j or si != sj or index_of[i] >= index_of[j]:
            continue
        assert plan.last_use[i] <= index_of[j], (
            f"slot {si}: node {j} reuses it while node {i} "
            f"(last used at {plan.last_use[i]}) is still live"
        )
print("plan correctness: no overlapping-liveness nodes share a slot")

# ---------------------------------------------------------------------
# 2. run_planned matches run() exactly.
# ---------------------------------------------------------------------

pool = core.StoragePool()
out_plain = kir.run(graph, X).tolist()
out_planned = kir.run_planned(graph, plan, pool, X).tolist()
assert out_plain == out_planned, f"mismatch: {out_plain} vs {out_planned}"
print("run_planned matches run():", out_planned)

# ---------------------------------------------------------------------
# 3. Safety across many iterations with DIFFERENT inputs: reusing the
#    same pool must never corrupt a later call's result. This is the
#    load-bearing correctness check for the whole approach.
# ---------------------------------------------------------------------

rng = random.Random(7)
mismatches = 0
for _ in range(20):
    vals = [rng.uniform(0, 1) for _ in range(8)]
    x_i = kansai.from_flat(vals, [4, 2])
    expected = kir.run(graph, x_i).tolist()
    actual = kir.run_planned(graph, plan, pool, x_i).tolist()
    if expected != actual:
        mismatches += 1
assert mismatches == 0, f"{mismatches}/20 runs corrupted by pooled reuse"
print("20 runs with fresh random inputs, pool reused throughout: all correct")

# ---------------------------------------------------------------------
# 4. The pool actually stops allocating once warm: after enough
#    iterations for every distinct buffer size to have been seen once,
#    num_allocated() should sit at (or very near) plan.num_slots, not
#    keep growing with iteration count.
# ---------------------------------------------------------------------

# Realistic usage: read the value out, then hand the output buffer back
# -- exactly what an inference loop does each step (read a prediction,
# move on). Without that release call the *output* buffer alone would
# still malloc fresh every iteration, which is a real thing this test
# tripped over the first time it ran (see run_planned()'s docstring).
warm_pool = core.StoragePool()
for _ in range(50):
    result = kir.run_planned(graph, plan, warm_pool, X)
    result.tolist()
    core.release_to_pool(warm_pool, result)
print(f"after 50 iterations: pool made {warm_pool.num_allocated()} real allocations total "
      f"(plan ceiling: {plan.num_slots} concurrent slots, +1 for the output buffer)")
assert warm_pool.num_allocated() <= plan.num_slots + 2, "pool should have converged, not kept growing"

# ---------------------------------------------------------------------
# 5. Safety guard: run_planned must refuse a requires_grad input rather
#    than silently risk corrupting a live backward graph.
# ---------------------------------------------------------------------

X_grad = kansai.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], requires_grad=True)
try:
    kir.run_planned(graph, plan, pool, X_grad)
    raise AssertionError("run_planned should have refused a requires_grad input")
except AssertionError as e:
    assert "forward-only" in str(e)
    print("safety guard: requires_grad input correctly refused")

# ---------------------------------------------------------------------
# 6. Performance, on a realistically-sized graph. At the tiny XOR scale
#    above, pooling is measured to be a net *loss* (~0.8-0.95x): each
#    call's Python-level release bookkeeping plus its handful of extra
#    nanobind calls (set_active_pool, one release_to_pool per freed
#    node, clear_active_pool) costs more than the malloc/free it avoids
#    for a few dozen floats. That crossed over, measured, around
#    feature-dim ~1000 at this batch size: large enough that a fresh
#    buffer's page faults (touching never-before-mapped memory) cost
#    more than this function's fixed per-call overhead, so reusing an
#    already-resident buffer starts winning outright. Same lesson as
#    elementwise fusion's first (bytecode-interpreter) attempt: measure
#    the actual crossover, don't assume "avoids work" means "faster".
# ---------------------------------------------------------------------

big_model = nn.Sequential(nn.Linear(1024, 1024, seed=5), nn.ReLU(), nn.Linear(1024, 1024, seed=6))
big_X = kansai.randn([128, 1024], std=1.0, seed=2)
big_graph = kir.trace(lambda x: big_model(x), big_X)
big_plan = kir.plan_memory(big_graph)

iters = 300

t0 = time.perf_counter()
for _ in range(iters):
    kir.run(big_graph, big_X)
t_plain = time.perf_counter() - t0

bench_pool = core.StoragePool()
t0 = time.perf_counter()
for _ in range(iters):
    result = kir.run_planned(big_graph, big_plan, bench_pool, big_X)
    core.release_to_pool(bench_pool, result)
t_planned = time.perf_counter() - t0

print(f"\nbenchmark: {iters} iterations, Linear(1024,1024)->ReLU->Linear(1024,1024), batch 128")
print(f"  run() (fresh malloc every node):  {t_plain*1000:8.1f} ms  ({t_plain/iters*1e6:7.1f} us/iter)")
print(f"  run_planned() (pooled):           {t_planned*1000:8.1f} ms  ({t_planned/iters*1e6:7.1f} us/iter)")
print(f"  speedup: {t_plain/t_planned:.2f}x")
assert t_planned < t_plain, "pooling should win outright at this scale, not just be numerically equal"

print("\nKIR memory planner smoke test passed.")
