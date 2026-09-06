from ._core import Tensor, zeros, ones, randn, from_flat


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


__all__ = ["Tensor", "zeros", "ones", "randn", "tensor", "from_flat", "where"]
