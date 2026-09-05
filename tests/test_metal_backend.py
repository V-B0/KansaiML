"""Phase 3: a real Metal compute backend. Every check here runs actual
GPU kernels (compiled at runtime via MTLDevice::newLibraryWithSource,
since this machine has only Command Line Tools, not the offline
metal/metallib compilers) on this machine's own GPU -- nothing here is
simulated or stubbed.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import nn, kir
from kansai import _core as core

if not core.metal_available():
    print("No Metal device available on this system -- skipping.")
    sys.exit(0)

TOL = 1e-3  # GPU reductions sum in a different order than the CPU BLAS call;
            # exact bit-for-bit agreement isn't expected, close agreement is


def max_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


# ---------------------------------------------------------------------
# 1. Correctness: metal_matmul / metal_bias_relu / metal_add_bias
#    against the CPU (Accelerate) implementations, at a size too small
#    to prove anything about performance but enough to prove the
#    kernels themselves are right.
# ---------------------------------------------------------------------

a = kansai.randn([16, 32], std=1.0, seed=1)
b = kansai.randn([32, 24], std=1.0, seed=2)
d1 = max_diff(core.metal_matmul(a, b).tolist(), a.matmul(b).tolist())
assert d1 < TOL, f"metal_matmul mismatch: {d1}"
print(f"metal_matmul matches CPU (max diff {d1:.2e})")

x = kansai.randn([8, 16], std=1.0, seed=3)
bias = kansai.randn([16], std=1.0, seed=4)
d2 = max_diff(core.metal_bias_relu(x, bias).tolist(), x.add(bias).relu().tolist())
assert d2 < TOL, f"metal_bias_relu mismatch: {d2}"
print(f"metal_bias_relu matches CPU (max diff {d2:.2e})")

d3 = max_diff(core.metal_add_bias(x, bias).tolist(), x.add(bias).tolist())
assert d3 < TOL, f"metal_add_bias mismatch: {d3}"
print(f"metal_add_bias matches CPU (max diff {d3:.2e})")

# ---------------------------------------------------------------------
# 2. Correctness end-to-end: the full Linear->ReLU->Linear forward pass,
#    run_metal (post-fusion graph, real GPU dispatch) against run()
#    (CPU/eager) -- the actual architecture claim (same graph, same op
#    vocabulary, a different backend underneath) proven on real data.
# ---------------------------------------------------------------------

model = nn.Sequential(nn.Linear(2, 8, seed=3), nn.ReLU(), nn.Linear(8, 1, seed=4))
X = kansai.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])

graph = kir.trace(lambda x: model(x), X)
fused = kir.elementwise_fusion(graph)
assert any(n.op == "fused_bias_relu" for n in fused.nodes), "fusion should have produced a fused_bias_relu node"

out_cpu = kir.run(graph, X).tolist()
out_metal = kir.run_metal(fused, X).tolist()
d4 = max_diff(out_cpu, out_metal)
assert d4 < TOL, f"run_metal mismatch: {d4}"
print(f"run_metal matches run() end-to-end (max diff {d4:.2e}): {[round(v, 4) for v in out_metal]}")

# ---------------------------------------------------------------------
# 3. Performance: does a naive (untiled, one-thread-per-output-element)
#    Metal matmul kernel actually beat Accelerate's highly-optimized CPU
#    BLAS? Not assumed -- measured across a size sweep, same approach as
#    the memory planner's crossover search. A naive GPU kernel losing to
#    a good CPU BLAS implementation at modest sizes would be a
#    completely normal, expected result, not a bug.
# ---------------------------------------------------------------------

print("\nmatmul benchmark: CPU (Accelerate) vs Metal (naive kernel), square matrices")
for dim in (64, 256, 512, 1024, 2048):
    m1 = kansai.randn([dim, dim], std=1.0, seed=10)
    m2 = kansai.randn([dim, dim], std=1.0, seed=11)
    iters = max(3, min(50, 200_000_000 // (dim ** 3) + 1))

    t0 = time.perf_counter()
    for _ in range(iters):
        m1.matmul(m2)
    t_cpu = (time.perf_counter() - t0) / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        core.metal_matmul(m1, m2)
    t_metal = (time.perf_counter() - t0) / iters

    print(f"  {dim:5d}x{dim:<5d}  cpu={t_cpu*1e3:9.3f} ms  metal={t_metal*1e3:9.3f} ms  "
          f"speedup={t_cpu/t_metal:5.2f}x  ({iters} iters)")

print("\nMetal backend test passed.")
