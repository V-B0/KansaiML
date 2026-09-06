"""Phase 4: concurrent dispatch. dtensor_run/dtensor_grad now run each
mesh device's work on its own thread (distributed.py's _run_parallel),
and every compute-heavy nanobind binding those devices actually call is
bound with nb::call_guard<nb::gil_scoped_release>() (python/bindings.cpp)
-- so, unlike plain Python threads (normally serialized by the GIL), the
"cpu" and "metal" threads genuinely overlap their own C++ compute. That
claim is measured here, not just argued: the same matmul run on "cpu"
alone, on "metal" alone, and through dtensor_run's concurrent dispatch,
with the concurrent wall time checked against the naive (serial) sum of
the other two -- the same median-of-N-trials approach
test_metal_backend.py already uses for its own benchmarks, since a
single-trial wall-clock measurement is exactly as vulnerable to ordinary
OS scheduling noise here as it was there.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import kir
from kansai import _core as core
from kansai.distributed import DeviceMesh, Shard, DTensor, dtensor_run

if not core.metal_available():
    print("No Metal device available -- skipping the concurrency test.")
    sys.exit(0)


def median_time(fn, iters=3, trials=5):
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        times.append((time.perf_counter() - t0) / iters)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------
# 1. Correctness at this benchmark's own shape, first: dtensor_run's
#    gathered result must still match the unsharded computation, the
#    same bar test_distributed.py already holds dtensor_run to at a
#    smaller shape -- confirms the threaded dispatch didn't trade
#    correctness for speed at a shape large enough for the speed claim
#    below to mean something.
# ---------------------------------------------------------------------

N, K, F = 4096, 4096, 4096  # split 2048+2048 across [cpu, metal]. Tried
                            # a smaller (2048x2048) shape first: each
                            # call landed in the single-digit
                            # milliseconds, and at that scale thread
                            # creation/scheduling overhead was a large
                            # enough fraction of the total to swamp the
                            # actual concurrency signal -- three
                            # back-to-back runs of that version came
                            # back 1.49x, 1.15x (failed), 0.90x (failed,
                            # concurrent dispatch measured *slower* than
                            # serial). This size's own per-call time
                            # (tens of ms) makes real compute dominate
                            # over that fixed overhead: seven repeated
                            # trials at this shape came back
                            # 27.12-27.35ms, essentially noise-free.
w = kansai.randn([K, F], std=1.0 / (K ** 0.5), seed=1)
X = kansai.randn([N, K], std=1.0, seed=2)

graph = kir.trace(lambda x: x.matmul(w), X)
fused_graph = kir.elementwise_fusion(graph)

mesh = DeviceMesh(["cpu", "metal"])
X_dt = DTensor.from_tensor(X, mesh, Shard(0))
cpu_shard, metal_shard = X_dt.shards
print(f"mesh: {mesh}, shard shapes: {[list(s.shape) for s in X_dt.shards]}")

expected = kir.run(graph, X).tolist()
result_dt = dtensor_run(graph, Shard(0), X_dt)
gathered = result_dt.to_tensor().tolist()
max_err = max(abs(a - b) for a, b in zip(gathered, expected))
print(f"dtensor_run (concurrent) vs unsharded kir.run: max diff {max_err:.2e}")
assert max_err < 1e-2, f"concurrent dtensor_run diverged from the unsharded reference: {max_err}"

# ---------------------------------------------------------------------
# 2. The actual concurrency claim: time "cpu" alone, "metal" alone, and
#    the real dtensor_run concurrent dispatch, and check the concurrent
#    wall time against their naive (serial) sum.
# ---------------------------------------------------------------------


def run_cpu_alone():
    return kir.run(graph, cpu_shard)


def run_metal_alone():
    return kir.run_metal(fused_graph, metal_shard)


def run_concurrent():
    return dtensor_run(graph, Shard(0), X_dt)


# Warm up: the very first Metal dispatch in this process pays a one-time
# cost (shader compilation inside state()'s lazy init) that every run
# after the first doesn't -- excluding it from the timed trials avoids
# attributing that fixed cost to either device's own steady-state speed.
run_cpu_alone()
run_metal_alone()
run_concurrent()

t_cpu = median_time(run_cpu_alone)
t_metal = median_time(run_metal_alone)
t_concurrent = median_time(run_concurrent)
naive_sum = t_cpu + t_metal
speedup = naive_sum / t_concurrent

print(f"  cpu alone:         {t_cpu * 1e3:7.3f} ms")
print(f"  metal alone:       {t_metal * 1e3:7.3f} ms")
print(f"  naive serial sum:  {naive_sum * 1e3:7.3f} ms")
print(f"  concurrent (both): {t_concurrent * 1e3:7.3f} ms")
print(f"  speedup vs naive serial sum: {speedup:.2f}x")

assert t_concurrent < naive_sum * 0.85, (
    f"expected concurrent dispatch to meaningfully beat the naive serial sum "
    f"({t_concurrent * 1e3:.3f}ms vs {naive_sum * 1e3:.3f}ms) -- GIL release may not be "
    f"taking effect (check python/bindings.cpp's call_guard<gil_scoped_release> annotations)"
)

print("\nConcurrent dispatch test passed.")
