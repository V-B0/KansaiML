"""Regression test for a real, permanent memory leak found while training
an actual transformer on real text (examples/tinyshakespeare) -- the
kind of bug small synthetic tests never exercise long enough to catch,
and exactly why the project's own capstone benchmark matters as more
than a demo.

The bug: Tensor::exp/sqrt/reciprocal/tanh/sigmoid (core/src/Tensor.cpp)
each attach a GradNode whose backward_fn closure captured `out` -- the
SAME Tensor object that GradNode was about to be attached to, via
`out.set_grad_node(node)` right after. That closure capturing `out` BY
VALUE creates a genuine std::shared_ptr CYCLE: out.impl_ (TensorData)
owns the GradNode, the GradNode's backward_fn closure owns a copy of
`out` (same impl_), which owns the SAME TensorData again. Nothing in
C++'s shared_ptr model ever breaks a cycle like that -- refcounting
alone can't, and (crucially) this is a pure C++-side cycle invisible
to Python's own cyclic garbage collector too (it only walks PyObject
reference graphs, not shared_ptrs opaque to it), so EVERY differentiable
call to any of these five ops leaked its entire output permanently, for
the life of the process. softmax composes exp() internally, and
Adam/AdamW call sqrt() on every single optimizer step -- so this had
been leaking during ordinary training the entire time, just slowly
enough (or masked by short-enough test runs) that nothing in this
project's existing test suite, none of which trains for more than a
few hundred steps, had a long enough loop to notice.

The fix: each of the five now captures `out.to_vector()` (a plain,
disconnected std::vector<float> copy of the values, via the existing
to_vector() helper) instead of `out` itself -- semantically identical,
since none of these outputs are ever mutated after construction, but
with no back-reference into out's own TensorData/GradNode at all,
breaking the cycle.

This test doesn't re-derive correctness (test_activations.py already
covers forward/backward correctness for all five exhaustively) -- it
checks specifically that repeated differentiable calls to each of the
five ops do NOT grow the process's resident memory without bound.
Measured via ru_maxrss (macOS reports this in bytes, unlike Linux's
KB), which is monotonically non-decreasing within a process -- exactly
the right instrument for catching a REGRESSION of a leak like this
one: if the cycle ever comes back, thousands of iterations will drive
this number up by hundreds of MB; fixed, it should barely move past
its post-warmup baseline.
"""

import gc
import os
import resource
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from kansai import _core as core

# Threshold chosen well above ordinary allocator noise but WAY below
# what the actual bug produced: at its measured rate (~0.28MB/step for
# exp() alone, on a much larger tensor than this test even uses), 4000
# iterations of a real leak would add hundreds of MB, not tens.
LEAK_THRESHOLD_MB = 40.0
ITERS = 4000
SHAPE = [8, 4, 48, 48]  # the actual attention-scores shape that surfaced this


def rss_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is bytes on macOS/BSD, kilobytes on Linux.
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return ru.ru_maxrss / divisor


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def check_bounded(label, fn):
    x = core.from_flat([0.3] * numel(SHAPE), SHAPE, requires_grad=True)

    # Warm up: let the allocator settle into its steady-state working
    # set before taking the baseline measurement, so normal (non-leak)
    # allocator fragmentation doesn't get misread as a leak.
    for _ in range(200):
        fn(x)
    gc.collect()
    baseline = rss_mb()

    for i in range(ITERS):
        fn(x)
        if i % 200 == 0:
            gc.collect()

    gc.collect()
    growth = rss_mb() - baseline
    assert growth < LEAK_THRESHOLD_MB, (
        f"{label}: ru_maxrss grew {growth:.1f}MB over {ITERS} iterations "
        f"(threshold {LEAK_THRESHOLD_MB}MB) -- the GradNode/out reference "
        f"cycle this test guards against may have come back")
    print(f"{label}: OK (ru_maxrss grew {growth:.1f}MB over {ITERS} iterations, well under "
          f"the {LEAK_THRESHOLD_MB}MB threshold)")


check_bounded("exp()", lambda x: x.exp())
check_bounded("sqrt()", lambda x: x.sqrt())
check_bounded("reciprocal()", lambda x: x.reciprocal())
check_bounded("tanh()", lambda x: x.tanh())
check_bounded("sigmoid()", lambda x: x.sigmoid())
check_bounded("softmax() (composes exp() internally)", lambda x: x.softmax(3))

print("\nMemory leak regression test passed -- no unbounded growth across "
      "exp/sqrt/reciprocal/tanh/sigmoid/softmax.")
