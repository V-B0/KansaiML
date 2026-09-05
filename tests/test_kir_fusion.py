import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import kir

TOL = 1e-4


def assert_close(a, b, label):
    for u, f in zip(a, b):
        assert abs(u - f) < TOL, f"{label}: fusion changed the result ({u} vs {f})"


# ---------------------------------------------------------------------
# 1. bias_add -> relu: the pattern in every Linear+ReLU forward pass.
#    Exercises the plain "producer feeds sole consumer" chain and the
#    bias-broadcast load.
# ---------------------------------------------------------------------

def bias_relu(matmul_out, bias):
    return matmul_out.add(bias).relu()

mo = kansai.randn([4, 8], std=1.0, seed=1)
bias = kansai.randn([8], std=1.0, seed=2)

g1 = kir.trace(bias_relu, mo, bias)
g1_fused = kir.elementwise_fusion(g1)
assert len(g1_fused.nodes) < len(g1.nodes), "bias_add+relu should fuse into one node"

out1_unfused = kir.run(g1, mo, bias).tolist()
out1_fused = kir.run_fused(g1_fused, mo, bias).tolist()
assert_close(out1_unfused, out1_fused, "bias_add+relu")
print(f"bias_add+relu fusion: {len(g1.nodes)} nodes -> {len(g1_fused.nodes)}, matches unfused")

# ---------------------------------------------------------------------
# 2. sub -> mul, self-multiply: the shape of MSE loss (diff*diff).
#    Exercises the `dup` instruction -- both operands of the fused mul
#    are the same upstream value.
# ---------------------------------------------------------------------

def mse_like(pred, target):
    diff = pred.sub(target)
    return diff.mul(diff)

p = kansai.tensor([[1.0, -2.0, 3.0, 0.5]])
t = kansai.tensor([[0.0, -1.0, 3.0, 1.5]])

g2 = kir.trace(mse_like, p, t)
g2_fused = kir.elementwise_fusion(g2)
assert len(g2_fused.nodes) < len(g2.nodes), "sub+mul(self) should fuse into one node"

out2_unfused = kir.run(g2, p, t).tolist()
out2_fused = kir.run_fused(g2_fused, p, t).tolist()
assert_close(out2_unfused, out2_fused, "sub+mul(self)")
print(f"sub+mul(self, dup) fusion: {len(g2.nodes)} nodes -> {len(g2_fused.nodes)}, matches unfused")
print("  values:", out2_fused)

# ---------------------------------------------------------------------
# 3. Performance: the actual point of fusion. Unfused sub+mul allocates
#    and fully writes an N-length intermediate (`diff`) before mul reads
#    it back; fused does one pass, no intermediate. Big enough N that
#    memory traffic dominates Python/graph-walk overhead.
# ---------------------------------------------------------------------

N = 2_000_000
big_a = kansai.randn([N], std=1.0, seed=3)
big_b = kansai.randn([N], std=1.0, seed=4)

g3 = kir.trace(mse_like, big_a, big_b)
g3_fused = kir.elementwise_fusion(g3)

iters = 200

t0 = time.perf_counter()
for _ in range(iters):
    kir.run(g3, big_a, big_b)
t_unfused = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(iters):
    kir.run_fused(g3_fused, big_a, big_b)
t_fused = time.perf_counter() - t0

print(f"\nbenchmark: N={N}, {iters} iterations of sub+mul(self)")
print(f"  unfused: {t_unfused*1000:8.1f} ms total  ({t_unfused/iters*1e6:7.1f} us/iter)")
print(f"  fused:   {t_fused*1000:8.1f} ms total  ({t_fused/iters*1e6:7.1f} us/iter)")
print(f"  speedup: {t_unfused/t_fused:.2f}x")

assert t_fused < t_unfused, "fused path should be faster, not just numerically equal"

print("\nKIR elementwise fusion smoke test passed.")
