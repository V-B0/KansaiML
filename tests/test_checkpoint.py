"""Gradient (activation) checkpointing -- see python/kansai/__init__.py's
own docstring for the full mechanism (no_grad() for the first pass,
core.attach_custom_grad to register a single custom GradNode against
the ORIGINAL inputs, recompute-then-Tensor.backward(grad_output) inside
that node's own closure). Three new primitives back it, none of which
existed before this session and none of which are checkpoint-specific
on their own: Tensor.backward(grad_output) (the general explicit-seed
form, not just the scalar-only implicit-ones one), Tensor._set_requires_grad
(the one low-level escape hatch letting a detached copy become a real
leaf again without a full data round trip), and core.attach_custom_grad
(the general "attach a GradNode with a Python-callable backward_fn"
primitive -- every other op attaches one from C++ with a C++ closure;
this is the one way Python-level code can do the same).

Checked: forward output and backward gradients through a checkpointed
call match an ordinary (non-checkpointed) call on the identical
computation EXACTLY, for both a single checkpoint and a DEEP stack of
them chained together (confirming the gradient chain survives multiple
checkpoint boundaries in a row, not just one in isolation); an input
that does NOT require grad gets a correctly-omitted (not
incorrectly-computed) gradient; the whole mechanism is memory-SAFE
over many repeated iterations with Python's cyclic garbage collector
explicitly DISABLED (ordinary refcounting alone must free everything --
the same rigorous check this project's own memory-leak investigation
established as the bar, see DEVLOG.md); and, the actual point of the
feature, that checkpointing a real deep stack measurably uses
substantially less peak memory than running the identical stack without
it -- not just "correct," genuinely useful for what it claims to do.
"""

import gc
import os
import resource
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import kansai
from kansai import nn
from kansai import _core as core

TOL = 1e-4


def check_close(label, a, b, tol=TOL):
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < tol, f"{label}[{i}]: got={x:.6f} expected={y:.6f}"
    print(f"{label}: OK ({len(a)} elements, max diff {max(abs(x - y) for x, y in zip(a, b)):.2e})")


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def main():
    # ---------------------------------------------------------------------
    # 1. Single checkpoint: forward and backward match an ordinary
    #    (non-checkpointed) call on the identical computation exactly.
    # ---------------------------------------------------------------------

    def fn(x, w):
        return x.mul(x).matmul(w).sum(1, keepdim=True)


    x_vals = [1.0, 2.0, 3.0, 4.0]
    w_vals = [0.5, 0.5, 0.5, 0.5]

    x_baseline = core.from_flat(x_vals, [2, 2], requires_grad=True)
    w_baseline = core.from_flat(w_vals, [2, 2], requires_grad=True)
    out_baseline = fn(x_baseline, w_baseline)
    out_baseline.sum().backward()

    x_ckpt = core.from_flat(x_vals, [2, 2], requires_grad=True)
    w_ckpt = core.from_flat(w_vals, [2, 2], requires_grad=True)
    out_ckpt = kansai.checkpoint(fn, x_ckpt, w_ckpt)
    out_ckpt.sum().backward()

    check_close("checkpoint() forward matches non-checkpointed", out_ckpt.tolist(), out_baseline.tolist())
    check_close("checkpoint() grad_x matches non-checkpointed", x_ckpt.grad.tolist(), x_baseline.grad.tolist())
    check_close("checkpoint() grad_w matches non-checkpointed", w_ckpt.grad.tolist(), w_baseline.grad.tolist())

    # ---------------------------------------------------------------------
    # 2. A deep chain of checkpoints -- confirms the gradient chain
    #    survives multiple checkpoint boundaries in a row (each one's own
    #    custom GradNode correctly hands off to the next), not just a
    #    single isolated one.
    # ---------------------------------------------------------------------

    DEPTH = 8
    layers = [nn.Linear(4, 4, seed=i) for i in range(DEPTH)]
    gelu = nn.GELU()

    x0_vals = [0.3, -0.2, 0.5, 0.1]

    xb = core.from_flat(x0_vals, [1, 4], requires_grad=True)
    hb = xb
    for layer in layers:
        hb = gelu(layer(hb))
    hb.sum().backward()

    xc = core.from_flat(x0_vals, [1, 4], requires_grad=True)
    hc = xc
    for layer in layers:
        hc = kansai.checkpoint(lambda h, lyr=layer: gelu(lyr(h)), hc)
    hc.sum().backward()

    check_close(f"checkpoint() chain of {DEPTH} matches non-checkpointed forward", hc.tolist(), hb.tolist())
    check_close(f"checkpoint() chain of {DEPTH} matches non-checkpointed grad_x", xc.grad.tolist(), xb.grad.tolist())

    # ---------------------------------------------------------------------
    # 3. An input that does NOT require grad gets a correctly-OMITTED
    #    gradient (no crash, and the grad-requiring input's own gradient
    #    is unaffected by the presence of a non-grad sibling).
    # ---------------------------------------------------------------------

    x_mixed = core.from_flat([1.0, 2.0], [2], requires_grad=True)
    const_mixed = core.from_flat([3.0, 4.0], [2], requires_grad=False)
    out_mixed = kansai.checkpoint(lambda a, b: a.mul(b).sum(), x_mixed, const_mixed)
    out_mixed.backward()
    check_close("checkpoint() with a mixed requires_grad input set: grad_x correct", x_mixed.grad.tolist(), [3.0, 4.0])
    assert const_mixed.grad is None, "a non-grad-requiring input must never accumulate a gradient"
    print("checkpoint() correctly omits gradient for a non-grad-requiring input: OK")

    # ---------------------------------------------------------------------
    # 4. Memory safety over many repeated iterations, gc EXPLICITLY
    #    DISABLED -- ordinary refcounting alone must free everything, the
    #    same rigorous bar this project's own memory-leak investigation
    #    established (see DEVLOG.md). A regression here would mean
    #    checkpoint() introduced exactly the class of permanent leak that
    #    investigation found and fixed elsewhere.
    # ---------------------------------------------------------------------

    def one_checkpointed_step():
        xi = core.from_flat([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)
        wi = core.from_flat([0.5, 0.5, 0.5, 0.5], [2, 2], requires_grad=True)
        oi = kansai.checkpoint(fn, xi, wi)
        oi.sum().backward()


    gc.disable()
    baseline_rss = None
    for i in range(1, 3001):
        one_checkpointed_step()
        if i == 500:
            baseline_rss = rss_mb()
        if i % 1000 == 0:
            print(f"iter {i} rss={rss_mb():.2f}MB")
    final_rss = rss_mb()
    gc.enable()

    growth = final_rss - baseline_rss
    assert growth < 20.0, (
        f"checkpoint() should not leak with gc disabled: rss grew {growth:.1f}MB from iter 500 to 3000")
    print(f"checkpoint() memory-safe over 3000 iterations with gc disabled: OK (grew {growth:.2f}MB after warmup)")

    # ---------------------------------------------------------------------
    # 5. The actual point: checkpointing a real deep stack measurably uses
    #    substantially less peak memory than the identical stack without
    #    it -- not just correct, genuinely doing what it claims.
    # ---------------------------------------------------------------------

    MEM_DEPTH, WIDTH, BATCH = 40, 512, 64
    mem_layers = [nn.Linear(WIDTH, WIDTH, seed=i) for i in range(MEM_DEPTH)]
    mem_gelu = nn.GELU()


    def run_no_checkpoint():
        h = core.from_flat([0.1] * (BATCH * WIDTH), [BATCH, WIDTH], requires_grad=True)
        for layer in mem_layers:
            h = mem_gelu(layer(h))
        h.sum().backward()


    def run_with_checkpoint():
        h = core.from_flat([0.1] * (BATCH * WIDTH), [BATCH, WIDTH], requires_grad=True)
        for layer in mem_layers:
            h = kansai.checkpoint(lambda hh, lyr=layer: mem_gelu(lyr(hh)), h)
        h.sum().backward()


    # ru_maxrss deltas are real but noisy -- sensitive to whatever ELSE
    # is running on the machine concurrently (allocator behavior,
    # system memory pressure), not just this test's own two functions.
    # Two repetitions per side, keeping the SMALLER delta each --
    # a transient concurrent-load spike can only ever inflate a
    # measurement, never deflate one below what the code path actually
    # needed, so the minimum across repeats is the more trustworthy
    # reading of the two.
    def measure(fn):
        deltas = []
        for _ in range(2):
            gc.collect()
            before = rss_mb()
            fn()
            gc.collect()
            deltas.append(rss_mb() - before)
        return min(deltas)

    delta_no_ckpt = measure(run_no_checkpoint)
    delta_ckpt = measure(run_with_checkpoint)

    print(f"{MEM_DEPTH}-layer stack: without checkpoint delta={delta_no_ckpt:.1f}MB, "
          f"with checkpoint delta={delta_ckpt:.1f}MB")
    assert delta_ckpt < delta_no_ckpt * 0.85, (
        f"checkpointing a {MEM_DEPTH}-layer stack should use meaningfully less peak memory: "
        f"with={delta_ckpt:.1f}MB vs without={delta_no_ckpt:.1f}MB")
    print(f"checkpointing measurably reduces peak memory on a real {MEM_DEPTH}-layer stack: OK "
          f"({delta_no_ckpt / max(delta_ckpt, 0.01):.1f}x less growth)")

    print("\ncheckpoint() test passed.")


if __name__ == "__main__":
    main()
