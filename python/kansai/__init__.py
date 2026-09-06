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


__all__ = ["Tensor", "zeros", "ones", "randn", "tensor", "from_flat", "where", "no_grad"]
