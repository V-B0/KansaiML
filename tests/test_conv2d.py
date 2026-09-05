"""Conv2d: forward via im2col + the same matmul kernel every other op
already uses, backward via matmul_nt/matmul_tn + col2im. NCHW only --
see the project README for why (no layout optimizer exists yet to pick
between layouts).

Correctness here is checked two genuinely independent ways, deliberately
not just one: forward against a direct/naive nested-loop reference
(catches an indexing bug that a self-consistency check like gradient
checking against the *same* forward implementation never could, since
both sides would reflect the same mistake), then backward against
central differences.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import nn, optim, kir
from kansai import _core as core

EPS = 1e-3
TOL_FWD = 1e-4
TOL_GRAD = 2e-2


def reference_conv2d(x_flat, x_shape, w_flat, w_shape, b_flat, stride, padding):
    """Direct nested-loop convolution -- the textbook definition, with
    no im2col/matmul anywhere in it -- as an independent check on the
    real implementation's forward pass."""
    N, Cin, H, W = x_shape
    Cout, Cin_w, kH, kW = w_shape
    assert Cin == Cin_w
    Hout = (H + 2 * padding - kH) // stride + 1
    Wout = (W + 2 * padding - kW) // stride + 1

    def xget(n, c, h, w):
        if h < 0 or h >= H or w < 0 or w >= W:
            return 0.0
        return x_flat[((n * Cin + c) * H + h) * W + w]

    out = [0.0] * (N * Cout * Hout * Wout)
    for n in range(N):
        for co in range(Cout):
            for oh in range(Hout):
                for ow in range(Wout):
                    acc = b_flat[co]
                    for ci in range(Cin):
                        for kh in range(kH):
                            for kw in range(kW):
                                ih = oh * stride + kh - padding
                                iw = ow * stride + kw - padding
                                widx = ((co * Cin_w + ci) * kH + kh) * kW + kw
                                acc += xget(n, ci, ih, iw) * w_flat[widx]
                    out[((n * Cout + co) * Hout + oh) * Wout + ow] = acc
    return out, [N, Cout, Hout, Wout]


def check_close(label, a, b, tol):
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < tol, f"{label}[{i}]: got={x:.6f} expected={y:.6f}"
    print(f"{label}: OK ({len(a)} elements, max diff "
          f"{max(abs(x - y) for x, y in zip(a, b)):.2e})")


# ---------------------------------------------------------------------
# 1. Forward correctness against the independent naive reference, at a
#    shape that exercises multiple channels, stride > 1, and padding > 0
#    all at once.
# ---------------------------------------------------------------------

rng = random.Random(0)
N, Cin, H, W = 2, 3, 7, 7
Cout, k, stride, padding = 4, 3, 2, 1

x_vals = [rng.uniform(-1, 1) for _ in range(N * Cin * H * W)]
w_vals = [rng.uniform(-1, 1) for _ in range(Cout * Cin * k * k)]
b_vals = [rng.uniform(-1, 1) for _ in range(Cout)]

x = kansai.from_flat(x_vals, [N, Cin, H, W])
w = kansai.from_flat(w_vals, [Cout, Cin, k, k])
b = kansai.from_flat(b_vals, [Cout])

out = x.conv2d(w, b, stride, padding)
ref_out, ref_shape = reference_conv2d(x_vals, [N, Cin, H, W], w_vals, [Cout, Cin, k, k], b_vals, stride, padding)

assert list(out.shape) == ref_shape, f"shape mismatch: {out.shape} vs {ref_shape}"
check_close("conv2d forward vs naive reference", out.tolist(), ref_out, TOL_FWD)

# ---------------------------------------------------------------------
# 2. Backward, checked against central differences -- a separate,
#    smaller shape (finite differences need one full forward pass per
#    perturbed element, so this stays small deliberately).
# ---------------------------------------------------------------------

rng = random.Random(1)
N2, Cin2, H2, W2 = 1, 2, 5, 5
Cout2, k2, stride2, padding2 = 2, 3, 1, 1

x2_vals = [rng.uniform(-1, 1) for _ in range(N2 * Cin2 * H2 * W2)]
w2_vals = [rng.uniform(-1, 1) for _ in range(Cout2 * Cin2 * k2 * k2)]
b2_vals = [rng.uniform(-1, 1) for _ in range(Cout2)]

x2 = kansai.from_flat(x2_vals, [N2, Cin2, H2, W2], requires_grad=True)
w2 = kansai.from_flat(w2_vals, [Cout2, Cin2, k2, k2], requires_grad=True)
b2 = kansai.from_flat(b2_vals, [Cout2], requires_grad=True)

loss = x2.conv2d(w2, b2, stride2, padding2).sum()
loss.backward()


def eval_loss(xv, wv, bv):
    xt = kansai.from_flat(xv, [N2, Cin2, H2, W2])
    wt = kansai.from_flat(wv, [Cout2, Cin2, k2, k2])
    bt = kansai.from_flat(bv, [Cout2])
    return xt.conv2d(wt, bt, stride2, padding2).sum().tolist()[0]


def central_diff(base_vals, make_eval):
    grads = []
    for i in range(len(base_vals)):
        plus = list(base_vals)
        minus = list(base_vals)
        plus[i] += EPS
        minus[i] -= EPS
        grads.append((make_eval(plus) - make_eval(minus)) / (2 * EPS))
    return grads


check_close("conv2d grad_x", x2.grad.tolist(),
            central_diff(x2_vals, lambda v: eval_loss(v, w2_vals, b2_vals)), TOL_GRAD)
check_close("conv2d grad_weight", w2.grad.tolist(),
            central_diff(w2_vals, lambda v: eval_loss(x2_vals, v, b2_vals)), TOL_GRAD)
check_close("conv2d grad_bias", b2.grad.tolist(),
            central_diff(b2_vals, lambda v: eval_loss(x2_vals, w2_vals, v)), TOL_GRAD)

# ---------------------------------------------------------------------
# 3. A practical sanity check beyond "the math checks out": a tiny conv
#    net actually learns something via ordinary SGD, the same way XOR
#    proved the eager engine end to end. Task: given a 1-channel 6x6
#    image split into four non-overlapping 3x3 patches (stride=3,
#    kernel=3 exactly tiles it), predict each patch's own pixel sum.
#    A single 3x3 filter of all 1s and zero bias reproduces this
#    exactly -- unlike a classification-shaped target, this has a real,
#    checkable near-zero MSE floor to converge to, the same role XOR's
#    "loss -> 0" bar played for the eager engine, rather than a
#    threshold picked by guessing (an early version of this test tried
#    a sign-classification target, whose achievable MSE floor for a
#    linear-only model turns out to be ~0.365 by symmetry -- an
#    unreachable bar that had nothing to do with whether conv2d's
#    gradients were correct).
# ---------------------------------------------------------------------

rng = random.Random(2)
model = nn.Conv2d(1, 1, kernel_size=3, stride=3, padding=0, seed=7)


def make_example():
    img = [rng.uniform(-1, 1) for _ in range(36)]

    def patch_sum(r0, c0):
        return sum(img[(r0 + dr) * 6 + (c0 + dc)] for dr in range(3) for dc in range(3))

    targets = [patch_sum(oh * 3, ow * 3) for oh in range(2) for ow in range(2)]
    return img, targets


examples = [make_example() for _ in range(64)]
X = kansai.from_flat([v for img, _ in examples for v in img], [64, 1, 6, 6])
Y = kansai.from_flat([t for _, targets in examples for t in targets], [64, 1, 2, 2])

opt = optim.SGD(model.parameters(), lr=0.05)

loss = None
for step in range(300):
    pred = model(X)
    diff = pred.sub(Y)
    loss = diff.mul(diff).mean()
    model.zero_grad()
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.tolist()[0]:.6f}")

final_loss = loss.tolist()[0]
print(f"tiny conv net: final loss {final_loss:.6f}, learned weight {[round(v, 3) for v in model.weight.tolist()]}")
assert final_loss < 0.01, "conv net did not learn the patch-sum task"

# ---------------------------------------------------------------------
# 4. KIR integration: trace() captures conv2d as a first-class node, and
#    run()/run_planned() both dispatch it correctly. Not yet wired into
#    elementwise_fusion (no fusable pattern involves conv2d) or
#    run_metal (no Metal conv kernel exists) -- both would raise a clear
#    error rather than silently mishandling a conv2d node, since neither
#    has a case for it.
# ---------------------------------------------------------------------

kir_model = nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1, seed=5)
kir_X = kansai.randn([2, 2, 5, 5], std=1.0, seed=8)

kir_graph = kir.trace(lambda t: kir_model(t), kir_X)
assert kir_graph.nodes[-1].op == "conv2d", f"expected conv2d as the traced op, got {kir_graph.nodes[-1].op}"

out_eager = kir_model(kir_X).tolist()
out_run = kir.run(kir_graph, kir_X).tolist()
check_close("kir.run() conv2d vs eager", out_run, out_eager, TOL_FWD)

kir_plan = kir.plan_memory(kir_graph)
kir_pool = core.StoragePool()
out_planned = kir.run_planned(kir_graph, kir_plan, kir_pool, kir_X).tolist()
check_close("kir.run_planned() conv2d vs eager", out_planned, out_eager, TOL_FWD)

print("\nConv2d test passed.")
