"""Source-transform autograd: kir.grad(graph, wrt) returns a new Graph
computing gradients, built once from the forward graph's structure --
not a tape replayed at runtime. This is the strongest kind of test for
that claim: every check below either (a) validates kir.grad()'s output
against pure numerical differentiation with the eager autograd engine
never involved at all, or (b) cross-checks it against that eager engine
as an independent second implementation, or (c) trains XOR to
convergence using only kir.grad()-produced graphs, with .backward()
never called anywhere in the loop.
"""

import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import nn, optim, kir

TOL_NUM = 2e-2   # vs. central differences: approximate, generous but real
TOL_EXACT = 5e-4  # vs. the eager engine: both are analytical, should nearly agree


def check(label, a, b, tol):
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < tol, f"{label}[{i}]: got={x:.6f} expected={y:.6f}"
    print(f"{label}: OK ({len(a)} elements, max diff "
          f"{max(abs(x - y) for x, y in zip(a, b)):.2e})")


def central_diff_placeholder(graph, arg_idx, args, eps=1e-3):
    """Numerically differentiates kir.run(graph, *args) wrt args[arg_idx]
    -- the forward pass runs entirely through kir.run(), never touching
    the eager engine's autograd."""
    target = args[arg_idx]
    flat, shape = target.tolist(), target.shape
    grads = []
    for i in range(len(flat)):
        plus, minus = list(flat), list(flat)
        plus[i] += eps
        minus[i] -= eps
        args_plus = list(args)
        args_plus[arg_idx] = kansai.from_flat(plus, shape)
        args_minus = list(args)
        args_minus[arg_idx] = kansai.from_flat(minus, shape)
        f_plus = kir.run(graph, *args_plus).tolist()[0]
        f_minus = kir.run(graph, *args_minus).tolist()[0]
        grads.append((f_plus - f_minus) / (2 * eps))
    return grads


def central_diff_retrace(make_and_run, base_tensor, eps=1e-3):
    """Numerically differentiates wrt a *constant*-captured tensor (a
    weight/bias): since it isn't a placeholder, checking it means
    retracing the whole graph with a perturbed copy substituted in."""
    flat, shape = base_tensor.tolist(), base_tensor.shape
    grads = []
    for i in range(len(flat)):
        plus, minus = list(flat), list(flat)
        plus[i] += eps
        minus[i] -= eps
        f_plus = make_and_run(kansai.from_flat(plus, shape))
        f_minus = make_and_run(kansai.from_flat(minus, shape))
        grads.append((f_plus - f_minus) / (2 * eps))
    return grads


# ---------------------------------------------------------------------
# 1. matmul, checked against central differences with the eager
#    autograd engine never involved: kir.trace -> kir.grad -> kir.run,
#    start to finish.
# ---------------------------------------------------------------------

rng = random.Random(0)
M, K, N = 3, 4, 5
a_vals = [rng.uniform(-1, 1) for _ in range(M * K)]
b_vals = [rng.uniform(-1, 1) for _ in range(K * N)]
a0 = kansai.from_flat(a_vals, [M, K])
b0 = kansai.from_flat(b_vals, [K, N])


def f_matmul(a, b):
    return a.matmul(b).sum()


graph1 = kir.trace(f_matmul, a0, b0)
bwd1 = kir.grad(graph1, graph1.inputs)
grad_a, grad_b = kir.run(bwd1, a0, b0)

check("matmul (KIR-only) grad_a", grad_a.tolist(),
      central_diff_placeholder(graph1, 0, [a0, b0]), TOL_NUM)
check("matmul (KIR-only) grad_b", grad_b.tolist(),
      central_diff_placeholder(graph1, 1, [a0, b0]), TOL_NUM)

# ---------------------------------------------------------------------
# 2. Linear + ReLU: gradients wrt the input (a placeholder) AND the
#    weight/bias (constants captured at trace time), checked against
#    central differences -- again with no eager backward anywhere.
# ---------------------------------------------------------------------

rng = random.Random(1)
batch, in_f, out_f = 4, 3, 5
x_vals = [rng.uniform(-1, 1) for _ in range(batch * in_f)]
w_vals = [rng.uniform(-1, 1) for _ in range(in_f * out_f)]
bias_vals = [rng.uniform(-1, 1) for _ in range(out_f)]
x0 = kansai.from_flat(x_vals, [batch, in_f])
w0 = kansai.from_flat(w_vals, [in_f, out_f])
bias0 = kansai.from_flat(bias_vals, [out_f])

pre = x0.matmul(w0).add(bias0)
assert all(abs(v) > 0.05 for v in pre.tolist()), (
    "a pre-activation landed too close to ReLU's boundary -- pick a different seed "
    "rather than debugging a spurious mismatch (see test_grad_check.py for why)"
)


def linear_relu(x):
    return x.matmul(w0).add(bias0).relu().sum()


graph2 = kir.trace(linear_relu, x0)
wrt2 = [graph2.inputs[0], kir.find_constant(graph2, w0), kir.find_constant(graph2, bias0)]
bwd2 = kir.grad(graph2, wrt2)
grad_x, grad_w, grad_bias = kir.run(bwd2, x0)

check("linear+relu (KIR-only) grad_x", grad_x.tolist(),
      central_diff_placeholder(graph2, 0, [x0]), TOL_NUM)


def eval_with_w(w_perturbed):
    def fn(x):
        return x.matmul(w_perturbed).add(bias0).relu().sum()
    return kir.run(kir.trace(fn, x0), x0).tolist()[0]


def eval_with_bias(bias_perturbed):
    def fn(x):
        return x.matmul(w0).add(bias_perturbed).relu().sum()
    return kir.run(kir.trace(fn, x0), x0).tolist()[0]


check("linear+relu (KIR-only) grad_w", grad_w.tolist(),
      central_diff_retrace(eval_with_w, w0), TOL_NUM)
check("linear+relu (KIR-only) grad_bias", grad_bias.tolist(),
      central_diff_retrace(eval_with_bias, bias0), TOL_NUM)

# ---------------------------------------------------------------------
# 3. Cross-check against the eager engine: an independent second
#    implementation of the same math, same inputs, tight tolerance
#    (both are analytical -- only floating-point rounding should
#    separate them, not the O(eps) slop central differences allow).
# ---------------------------------------------------------------------

x1 = kansai.from_flat(x_vals, [batch, in_f], requires_grad=True)
w1 = kansai.from_flat(w_vals, [in_f, out_f], requires_grad=True)
bias1 = kansai.from_flat(bias_vals, [out_f], requires_grad=True)
(x1.matmul(w1).add(bias1).relu().sum()).backward()

check("KIR grad_x vs eager grad_x", grad_x.tolist(), x1.grad.tolist(), TOL_EXACT)
check("KIR grad_w vs eager grad_w", grad_w.tolist(), w1.grad.tolist(), TOL_EXACT)
check("KIR grad_bias vs eager grad_bias", grad_bias.tolist(), bias1.grad.tolist(), TOL_EXACT)

# ---------------------------------------------------------------------
# 4. The real test: train XOR to convergence using only kir.grad()-
#    produced graphs. .backward() is never called anywhere in this
#    loop -- gradients come entirely from running bwd_graph, which
#    kir.grad() built once, ahead of time, from graph's structure.
# ---------------------------------------------------------------------

X = kansai.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = kansai.tensor([[0], [1], [1], [0]])
model = nn.Sequential(nn.Linear(2, 8, seed=3), nn.ReLU(), nn.Linear(8, 1, seed=4))


def forward_and_loss(x):
    pred = model(x)
    diff = pred.sub(Y)
    return diff.mul(diff).mean()


graph3 = kir.trace(forward_and_loss, X)
params = model.parameters()
wrt3 = [kir.find_constant(graph3, p) for p in params]
bwd3 = kir.grad(graph3, wrt3)

lr = 0.1
loss = None
for step in range(500):
    loss = kir.run(graph3, X)
    grads = kir.run(bwd3, X)
    for p, g in zip(params, grads):
        p.add_(g, -lr)
    if step % 100 == 0:
        print(f"step {step:4d}  loss {loss.tolist()[0]:.6f}")

final_loss = loss.tolist()[0]
print(f"final loss (KIR-only, source-transform grad): {final_loss:.6f}")
assert final_loss < 0.05, "XOR did not converge through kir.grad()-only training"

# ---------------------------------------------------------------------
# What this costs, honestly: bwd3 recomputes the whole forward pass
# internally (vjp rules need primal values), so a training step through
# kir.grad() does roughly 2x the forward work eager backward() does in
# one pass. Measured, not assumed.
# ---------------------------------------------------------------------

eager_model = nn.Sequential(nn.Linear(2, 8, seed=3), nn.ReLU(), nn.Linear(8, 1, seed=4))
eager_opt = optim.SGD(eager_model.parameters(), lr=0.1)

t0 = time.perf_counter()
for _ in range(500):
    pred = eager_model(X)
    l = pred.sub(Y).mul(pred.sub(Y)).mean()
    eager_model.zero_grad()
    l.backward()
    eager_opt.step()
t_eager = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(500):
    kir.run(graph3, X)
    g = kir.run(bwd3, X)
    for p, gi in zip(params, g):
        p.add_(gi, -lr)
t_kir = time.perf_counter() - t0

print(f"\n500 training steps: eager backward()={t_eager*1000:.1f}ms  "
      f"kir.grad()-only={t_kir*1000:.1f}ms  ({t_kir/t_eager:.2f}x)")

print("\nKIR source-transform autograd test passed.")
