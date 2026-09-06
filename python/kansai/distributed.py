"""Phase 4, started: DeviceMesh and DTensor -- the data model for
sharding a tensor's computation across Kansai's real backends.

Deferred until now on purpose (see the project history): DeviceMesh
only means something once there's more than one real backend to shard
across, and before the Metal backend existed and reached genuine
parity with Accelerate, this would have been a data model with nothing
real underneath it. It isn't a simulation now -- each shard's forward
pass really is computed by that device's actual interpreter
(kir.run for "cpu", kir.run_metal for "metal").

What "device" means here, precisely, matters: every kansai.Tensor
already lives in ordinary host memory, and Metal's own NoCopy path
computes directly on that same memory rather than moving it anywhere.
So a DeviceMesh's entries don't name a physical location a tensor gets
moved to -- they name which backend's interpreter processes a given
shard's graph. That's a real, honest distinction from a mesh spanning
actual separate memory spaces (multiple discrete GPUs, or multiple
machines), and it's why splitting/gathering here goes through
tolist()/from_flat() rather than anything resembling a network
transfer -- there's nothing to transfer, only a decision about which
compute path runs on which slice of already-shared memory.

Forward-only, deliberately, same as fusion/pooling/run_metal before it:
there's no autograd through a DTensor. A real implementation needs
auto-inserted collectives during backward -- all-reduce for a
Replicate()'d gradient, all-gather for a Shard()'d one -- which is
exactly the "auto-comm insertion" line from the project's own Phase 4
roadmap, and genuinely separate, substantial work from the data model
itself. Not attempted here.

No real concurrency either: dtensor_run dispatches to each device in a
plain Python loop, one after another. nanobind doesn't release the GIL
around any binding in this codebase, so even calling the CPU and Metal
paths from separate Python threads wouldn't overlap their execution
today -- true concurrent dispatch is a real, separate piece of work
(releasing the GIL around the blocking Metal calls specifically, since
they already do their own synchronous wait), not something this module
does for free.
"""

from . import _core as core
from . import kir


class DeviceMesh:
    """A 1-D declaration of which backend computes each shard of a
    DTensor. "cpu" dispatches through kir.run (Accelerate underneath);
    "metal" dispatches through kir.run_metal (MPS/tiled Metal kernels
    underneath). Nothing here assumes there are exactly two entries or
    that they're these two specific names beyond what Kansai's backends
    currently answer to -- a third real backend would just be a third
    valid device string, not a redesign."""

    _VALID_DEVICES = ("cpu", "metal")

    def __init__(self, devices):
        devices = list(devices)
        for d in devices:
            if d not in self._VALID_DEVICES:
                raise ValueError(f"unknown device {d!r} -- kansai has {self._VALID_DEVICES} backends only")
        if "metal" in devices and not core.metal_available():
            raise RuntimeError("DeviceMesh includes 'metal' but no Metal device is available on this system")
        self.devices = devices

    def __len__(self):
        return len(self.devices)

    def __repr__(self):
        return f"DeviceMesh({self.devices})"


class Shard:
    """Placement: split along dimension `dim` into len(mesh) pieces, as
    equal as possible (the remainder distributed to the first few
    pieces, the same convention numpy/torch's own chunk() uses)."""

    def __init__(self, dim: int):
        self.dim = dim

    def __repr__(self):
        return f"Shard({self.dim})"

    def __eq__(self, other):
        return isinstance(other, Shard) and self.dim == other.dim


class Replicate:
    """Placement: every device gets the same full tensor."""

    def __repr__(self):
        return "Replicate()"

    def __eq__(self, other):
        return isinstance(other, Replicate)


def _chunk_sizes(total: int, n: int) -> list:
    base, remainder = divmod(total, n)
    return [base + 1 if i < remainder else base for i in range(n)]


def _split_tensor(tensor, dim: int, n: int) -> list:
    """Splits `tensor` into `n` pieces along `dim`. Reference-level, not
    a native kernel: goes through tolist()/from_flat() since Kansai has
    no slice op yet -- the same "prototype the semantics in Python
    before committing to a real kernel" approach kir.py itself took for
    the IR, not an oversight."""
    shape = list(tensor.shape)
    flat = tensor.tolist()
    sizes = _chunk_sizes(shape[dim], n)

    outer = 1
    for d in shape[:dim]:
        outer *= d
    inner = 1
    for d in shape[dim + 1:]:
        inner *= d
    dim_size = shape[dim]

    pieces = []
    offset = 0
    for size in sizes:
        piece_shape = list(shape)
        piece_shape[dim] = size
        piece_flat = [0.0] * (outer * size * inner)
        for o in range(outer):
            src_start = o * dim_size * inner + offset * inner
            dst_start = o * size * inner
            length = size * inner
            piece_flat[dst_start:dst_start + length] = flat[src_start:src_start + length]
        pieces.append(core.from_flat(piece_flat, piece_shape))
        offset += size
    return pieces


def _concat_tensors(tensors: list, dim: int):
    """The inverse of _split_tensor."""
    shapes = [list(t.shape) for t in tensors]
    dim_sizes = [s[dim] for s in shapes]
    out_shape = list(shapes[0])
    out_shape[dim] = sum(dim_sizes)

    outer = 1
    for d in out_shape[:dim]:
        outer *= d
    inner = 1
    for d in out_shape[dim + 1:]:
        inner *= d

    out_flat = [0.0] * (outer * out_shape[dim] * inner)
    flats = [t.tolist() for t in tensors]

    offset = 0
    for flat, size in zip(flats, dim_sizes):
        for o in range(outer):
            src_start = o * size * inner
            dst_start = o * out_shape[dim] * inner + offset * inner
            length = size * inner
            out_flat[dst_start:dst_start + length] = flat[src_start:src_start + length]
        offset += size
    return core.from_flat(out_flat, out_shape)


class DTensor:
    """A tensor sharded or replicated across a DeviceMesh. `.shards[i]`
    is a real, local kansai.Tensor holding mesh.devices[i]'s actual
    data -- not a view or a placeholder."""

    def __init__(self, mesh: DeviceMesh, placement, shards: list):
        if len(shards) != len(mesh):
            raise ValueError(f"DTensor needs one shard per mesh device ({len(mesh)}), got {len(shards)}")
        self.mesh = mesh
        self.placement = placement
        self.shards = list(shards)

    @staticmethod
    def from_tensor(tensor, mesh: DeviceMesh, placement) -> "DTensor":
        if isinstance(placement, Replicate):
            shards = [tensor for _ in mesh.devices]
        elif isinstance(placement, Shard):
            shards = _split_tensor(tensor, placement.dim, len(mesh))
        else:
            raise TypeError(f"unknown placement {placement!r}")
        return DTensor(mesh, placement, shards)

    def to_tensor(self):
        if isinstance(self.placement, Replicate):
            return self.shards[0]
        if isinstance(self.placement, Shard):
            return _concat_tensors(self.shards, self.placement.dim)
        raise TypeError(f"unknown placement {self.placement!r}")

    def __repr__(self):
        shapes = [list(s.shape) for s in self.shards]
        return f"DTensor(mesh={self.mesh}, placement={self.placement}, shard_shapes={shapes})"


def dtensor_run(graph, output_placement, *dtensor_args) -> DTensor:
    """Runs a traced KIR graph once per mesh device, feeding each device
    its own shard of every DTensor argument, dispatched to that
    device's real backend -- kir.run for "cpu", kir.run_metal (on an
    elementwise_fusion'd copy of the graph, matching run_metal's own
    contract) for "metal". Each shard's result comes from that backend's
    actual interpreter, not a simulation of one.

    All DTensor arguments must already share the same mesh, and (since
    `graph` was traced once, against one representative shape) every
    argument's shards must all match that traced shape -- true whenever
    the sharded dimension divides evenly across the mesh, not handled
    otherwise; tracing a separate graph per differently-shaped shard is
    real, unattempted future work.

    output_placement is supplied by the caller rather than inferred:
    working out how a placement propagates through an arbitrary graph
    (batch-sharded in -> batch-sharded out, replicated in -> replicated
    out, and everything in between) is its own real design problem, not
    attempted generically here.
    """
    if not dtensor_args:
        raise ValueError("dtensor_run needs at least one DTensor argument")
    mesh = dtensor_args[0].mesh
    for dt in dtensor_args:
        if dt.mesh is not mesh:
            raise ValueError("all DTensor arguments must share the same DeviceMesh")

    fused_graph = None
    results = []
    for i, device in enumerate(mesh.devices):
        shard_inputs = [dt.shards[i] for dt in dtensor_args]
        if device == "cpu":
            results.append(kir.run(graph, *shard_inputs))
        elif device == "metal":
            if fused_graph is None:
                fused_graph = kir.elementwise_fusion(graph)
            results.append(kir.run_metal(fused_graph, *shard_inputs))
        else:
            raise ValueError(f"unknown device {device!r}")

    return DTensor(mesh, output_placement, results)
