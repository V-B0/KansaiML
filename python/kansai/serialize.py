"""Phase 4: model serialization -- save a Module's trained parameters to
disk, load them back into a freshly-constructed Module of matching
architecture. The scope is deliberately narrow: parameter *values* only,
not architecture. Reconstructing a model's hyperparameters (layer sizes,
activation choices, how Sequential's children compose) from a saved file
is a real, separate, harder problem -- it needs a stable schema for
describing a model's structure itself, not just its numbers -- so the
contract here matches PyTorch's own state_dict/load_state_dict split:
the caller builds the model first (with the right constructor arguments,
already known at call time -- the same way a training script always
already knows them), then this fills in the numbers. Saving/loading a
traced KIR Graph's own structure (nodes, edges, attrs) would be the
other real option Phase 4 could have picked here -- a stable node-id/
op-name schema, harder still -- and is unattempted, separate future
work, not a gap this quietly folds in.

File format, invented for this project rather than reusing an existing
one, but structurally similar to a well-known one (HuggingFace's
safetensors: a length-prefixed JSON header describing each tensor's
name/shape/byte-range, followed by one flat data blob) -- NOT claimed
to be binary-compatible with it, since this omits things the real
safetensors spec requires (8-byte offset alignment, an "__metadata__"
key, specific dtype string casing) that nothing here needs:

    bytes  0- 3   magic:      b"KAN1" (format tag + version 1)
    bytes  4-11   header_len: uint64, little-endian
    bytes 12..    header:     UTF-8 JSON, {name: {shape, dtype, offset, nbytes}}
    remaining     data:       every tensor's raw float32 bytes,
                              concatenated; tensor `name`'s bytes are at
                              data[offset : offset + nbytes]

Deliberately not pickle, PyTorch's own default and the reason
loading an arbitrary .pt file off the internet is a real security
concern (unpickling can execute arbitrary code as a side effect of
reconstructing an object graph). This format has no such path: `json`
parses data, never executes it, and the tensor payload is unpacked as
raw floats via the `array` module, not deserialized as objects. The
worst an adversarial or merely corrupted file can do here is fail a
loud, specific check (bad magic, unparseable header JSON, a byte range
that doesn't match the declared shape's element count) -- never run
code. Still worth being honest about the remaining, narrower risk any
format sharing memory this directly carries: a header lying about
`shape` while `nbytes` matches self-consistently would still produce a
Tensor of the (wrong but requested) shape without a crash -- caught
here indirectly, because load() separately checks the loaded shape
against the ALREADY-CONSTRUCTED model's own parameter shape and refuses
a mismatch, not because the file format enforces shape/byte-count
consistency on its own.

Goes through Tensor.tolist()/core.from_flat() rather than a C++ bulk
reader/writer, the same "prototype in Python first" tradeoff this
project already made for DTensor's split/concat and quantize.py's
(de)quantization -- fine at this project's own model sizes, a real,
un-optimized cost (one Python float object per tensor element, both
directions) at a size large enough for that to matter, which nothing
in this codebase's own test suite yet is.
"""

import array
import json
import struct

from . import _core as core

_MAGIC = b"KAN1"


def _tensor_to_bytes(tensor: "core.Tensor") -> bytes:
    return array.array("f", tensor.tolist()).tobytes()


def _bytes_to_floats(raw: bytes) -> list:
    a = array.array("f")
    a.frombytes(raw)
    return a.tolist()


def _get_by_path(root, path: str):
    """The inverse of _set_by_path -- used only to read the CURRENT
    value at a path for error messages (e.g. reporting the shape
    mismatch load() found), never for the load itself."""
    parts = path.split(".")
    obj = root
    for part in parts[:-1]:
        obj = obj[int(part)] if isinstance(obj, (list, tuple)) else getattr(obj, part)
    return getattr(obj, parts[-1])


def _set_by_path(root, path: str, tensor: "core.Tensor"):
    """Walks a dotted name like "layers.0.weight" back down through
    `root` and assigns `tensor` at the end -- the exact inverse of how
    Module.named_parameters() built that name in the first place (an
    attribute name, optionally followed by a list index whenever that
    attribute held a list of sub-Modules, repeated down to the final
    attribute name). Reassigns the attribute directly (`setattr`)
    rather than mutating the existing Tensor's own data in place,
    because Tensor exposes no generic in-place "copy these values into
    me" operation -- only add_(), which isn't a natural fit for "replace
    this parameter's value from a checkpoint" without going through an
    awkward subtract-then-add. Perfectly fine here: nothing else in this
    codebase keeps a second reference to a model's own parameter Tensor
    across a load() call the way, say, an optimizer keeps its own
    reference to the *list* of parameters (SGD.params) -- and that list
    holds Tensor handles by identity, so replacing model.weight with a
    new Tensor object here would NOT be reflected in an optimizer built
    against the model's parameters() from before the load. Call load()
    before constructing an optimizer, not after -- the same ordering
    PyTorch's own load_state_dict expects for exactly the same reason.
    """
    parts = path.split(".")
    obj = root
    for part in parts[:-1]:
        obj = obj[int(part)] if isinstance(obj, (list, tuple)) else getattr(obj, part)
    setattr(obj, parts[-1], tensor)


def save(module, path: str) -> None:
    """Writes every trainable parameter in `module` (via
    named_parameters(), so nested Sequential/Module structure is
    captured by name) to `path` in the format this module's own
    docstring describes."""
    named = list(module.named_parameters())

    header = {}
    blob = bytearray()
    offset = 0
    for name, tensor in named:
        data = _tensor_to_bytes(tensor)
        header[name] = {
            "dtype": "float32",
            "shape": list(tensor.shape),
            "offset": offset,
            "nbytes": len(data),
        }
        blob += data
        offset += len(data)

    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(blob)


def load(module, path: str) -> None:
    """Loads parameters from `path` into `module` IN PLACE, by name --
    `module` must already be constructed with the matching architecture
    (see the module docstring for why this doesn't reconstruct one from
    the file). Every name and shape must match exactly, in either
    direction (a checkpoint missing an expected parameter, or carrying
    one `module` doesn't have, are both rejected, not silently
    ignored) -- a partial or silently-mismatched load is exactly the
    kind of bug that would surface later as a confusingly wrong forward
    pass, not here where the actual cause is still visible."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != _MAGIC:
            raise ValueError(f"{path}: not a Kansai checkpoint (expected magic {_MAGIC!r}, got {magic!r})")
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        blob = f.read()

    named = dict(module.named_parameters())

    missing = sorted(set(named) - set(header))
    unexpected = sorted(set(header) - set(named))
    if missing:
        raise ValueError(f"{path}: checkpoint is missing parameter(s) the model expects: {missing}")
    if unexpected:
        raise ValueError(f"{path}: checkpoint has parameter(s) the model doesn't expect: {unexpected}")

    for name, tensor in named.items():
        meta = header[name]
        if list(tensor.shape) != meta["shape"]:
            raise ValueError(
                f"{path}: {name!r} shape mismatch -- model has {list(tensor.shape)}, "
                f"checkpoint has {meta['shape']}"
            )

        n = 1
        for d in meta["shape"]:
            n *= d
        expected_nbytes = n * 4  # float32
        if meta["nbytes"] != expected_nbytes:
            raise ValueError(
                f"{path}: {name!r} has an inconsistent header -- shape {meta['shape']} implies "
                f"{expected_nbytes} bytes but the header declares {meta['nbytes']}"
            )

        raw = blob[meta["offset"]:meta["offset"] + meta["nbytes"]]
        if len(raw) != meta["nbytes"]:
            raise ValueError(
                f"{path}: {name!r}'s declared byte range runs past the end of the file "
                f"(wanted {meta['nbytes']} bytes at offset {meta['offset']}, file has {len(raw)} there)"
            )

        flat = _bytes_to_floats(raw)
        _set_by_path(module, name, core.from_flat(flat, meta["shape"], tensor.requires_grad))
