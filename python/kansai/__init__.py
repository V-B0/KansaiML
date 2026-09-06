import contextlib

from ._core import Tensor, zeros, ones, randn, from_flat
from . import _core as core


def _flatten(nested):
    shape = []
    node = nested
    while isinstance(node, list):
        shape.append(len(node))
        node = node[0] if node else None

    flat = []

    def _walk(x):
        if isinstance(x, list):
            for item in x:
                _walk(item)
        else:
            flat.append(float(x))

    _walk(nested)
    return flat, shape


def tensor(data, requires_grad=False):
    """Build a Tensor from a (possibly nested) Python list."""
    flat, shape = _flatten(data)
    return from_flat(flat, shape, requires_grad)


def where(cond, a, b):
    """Elementwise select: `a` where `cond` is 1.0 (typically the
    output of Tensor.gt/lt/eq), `b` where it's 0.0. Written as
    `b + cond * (a - b)` rather than the more obvious `cond*a +
    (1-cond)*b` specifically to avoid needing a `ones_like(cond)` --
    that algebraic rearrangement makes this composable from ONLY
    sub/mul/add, all three of which already exist for both eager
    Tensors and traced TraceValues, so this one function works
    unmodified in either context: no new C++ kernel, no new KIR node
    type, nothing to add to any interpreter's dispatch. Real gradient
    still flows into `a`/`b` (never into `cond`, which is correct --
    see Tensor::gt/lt/eq's own doc comment for why a comparison's
    gradient is a deliberate exact zero) through the ordinary sub/mul/
    add vjps, with no special-casing needed there either."""
    return b.add(cond.mul(a.sub(b)))


@contextlib.contextmanager
def no_grad():
    """Suppresses autograd graph-building for every op run inside this
    block, regardless of any input's own `requires_grad` -- the standard
    "I'm about to run inference/validation and never call backward()"
    escape hatch every DL framework has, and this one previously
    didn't: `estimate_val_loss` in examples/tinyshakespeare/
    train_shakespeare.py built a full backward graph on every single
    validation call, immediately discarded without ever calling
    `.backward()` on it -- correct, but real, needless work (and, worse,
    every leaf-adjacent intermediate along the way used to hold a
    now-fixed GradNode reference cycle -- see DEVLOG.md's own account of
    that bug -- so building graphs nothing will ever differentiate was
    genuinely costly before that fix, not just wasteful in principle).

    Implemented as a single per-thread flag (core.set_grad_enabled,
    thread_local in C++ -- see Tensor.hpp's own comment on
    grad_enabled() for why: DeviceMesh dispatches real, concurrently-
    overlapping threads that must not share this state) that every op's
    own `if (x.requires_grad())` check in Tensor.cpp now also requires.
    Reentrant/nestable: restores whatever grad-tracking state was
    active before this block (not unconditionally re-enabling it), so
    a `no_grad()` block nested inside another one doesn't incorrectly
    turn tracking back on when the inner block exits.
    """
    previous = core.grad_enabled()
    core.set_grad_enabled(False)
    try:
        yield
    finally:
        core.set_grad_enabled(previous)


def checkpoint(fn, *inputs):
    """Gradient (activation) checkpointing: trades extra compute for
    less memory by NOT keeping `fn`'s intermediate activations around
    for backward() -- recomputing them from scratch instead, the
    moment they're actually needed. For a deep stack of layers (the
    kind of workload `examples/tinyshakespeare/train_shakespeare_large.py`
    exists to stress -- see that script's own docstring on why memory
    at real depth is a real, live concern here, not a hypothetical
    one), the dominant memory cost during training is exactly these
    stored activations, one full set per layer, all held simultaneously
    until backward() finally consumes them. Wrapping a layer's forward
    call in `checkpoint()` collapses that to O(1) per checkpointed
    segment: only `fn`'s OUTPUT and its (detached) inputs are kept,
    at the cost of running `fn` a second time during backward().

    `fn` must take Tensor arguments only and return a SINGLE Tensor
    (matching what a single `nn.Module.forward()` call typically
    returns) -- close over anything else (a fixed mask, a scalar
    hyperparameter) as an ordinary Python closure rather than passing
    it through `*inputs`, the same way `TransformerBlock.forward`
    already closes over its own `mask` argument.

    Mechanics: the FIRST call runs under `no_grad()` (see that
    function's own docstring) -- no graph built at all, every
    intermediate immediately eligible for collection the instant `fn`
    returns, keeping only the final output. A single custom GradNode
    (`core.attach_custom_grad`, the one general escape hatch letting
    Python-level code attach a GradNode the same way every C++ op
    already does internally) is attached to that output, registered
    against the ORIGINAL (not detached) `inputs` -- critical for
    correctness: registering detached copies instead would sever the
    graph's own topological walk right at this checkpoint boundary,
    silently losing whatever gradient chain existed further upstream
    of `inputs` before this call. When backward() actually reaches
    this node, its closure re-detaches fresh copies of `inputs`
    (marking the ones that originally required grad via
    `Tensor._set_requires_grad`, the one low-level setter that exists
    specifically for this), re-runs `fn` on THOSE -- this time with
    grad tracking on, building a real (but small, single-segment)
    graph -- and calls `Tensor.backward(grad_output)` (the explicit-seed
    form, needed here since the incoming gradient is whatever actually
    flowed in from downstream, not an implicit all-ones scalar seed)
    to get real gradients for the recomputed leaves, which are what
    gets returned as this checkpoint's own contribution to `inputs`'
    gradients.
    """
    detached_inputs = [inp.detach() for inp in inputs]
    with no_grad():
        output = fn(*detached_inputs)

    needs_grad = [inp.requires_grad for inp in inputs]
    if not any(needs_grad):
        return output

    def backward_fn(grad_output):
        recompute_inputs = []
        for inp, needed in zip(inputs, needs_grad):
            r = inp.detach()
            if needed:
                r._set_requires_grad(True)
            recompute_inputs.append(r)

        recomputed_output = fn(*recompute_inputs)
        recomputed_output.backward(grad_output)

        return [r.grad if needed else core.zeros(list(r.shape))
                for r, needed in zip(recompute_inputs, needs_grad)]

    core.attach_custom_grad(output, list(inputs), backward_fn)
    return output


__all__ = ["Tensor", "zeros", "ones", "randn", "tensor", "from_flat", "where", "no_grad", "checkpoint"]
