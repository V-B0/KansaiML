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


__all__ = ["Tensor", "zeros", "ones", "randn", "tensor", "from_flat"]
