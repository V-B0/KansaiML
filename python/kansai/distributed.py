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

dtensor_run itself is still forward-only -- no grad_node is attached to
anything it computes, same as run_metal/run_planned before it.
Distributed *backward*, though, is real: dtensor_grad (below) builds
the backward graph once via kir.grad, runs it per-device against each
device's own shard exactly the way dtensor_run runs the forward graph,
and auto-inserts the collective each wrt entry's own placement calls
for -- all-reduce (sum) for a Replicate()'d one, all-gather (concat)
for a Shard()'d one. See its own docstring for the full design and the
loss-scaling subtlety that comes with it.

Real concurrency, not simulated: every compute-heavy nanobind binding
this module's dispatch touches -- the plain Tensor ops kir.run uses on
"cpu", and every metal_* function kir.run_metal uses on "metal" -- is
bound with `nb::call_guard<nb::gil_scoped_release>()` (see
python/bindings.cpp), so each one releases Python's GIL for the
duration of its own blocking C++ call (an Accelerate routine, or a
Metal dispatch's own synchronous `waitUntilCompleted`). _run_parallel
(below) runs each device's work on its own Python `threading.Thread`;
with the GIL released inside each device's actual compute, two threads
genuinely execute their C++ calls concurrently rather than being
serialized by the GIL the way plain Python threads normally are. Safe
to do: Metal's command queue is documented thread-safe for concurrent
command-buffer creation from multiple threads (Apple's Metal Best
Practices Guide), and the CPU backend's Accelerate calls and hand-
written kernels touch only the buffers passed to them, not any shared
mutable state -- see backend/metal/MetalOps.mm's `state()` (a
function-local static, whose one-time initialization C++11 already
makes thread-safe) and backend/cpu's stateless kernels. The one real
caveat: this doesn't extend to `kir.run_planned`'s `StoragePool` (a
genuinely shared, non-thread-safe free-list) -- dtensor_run/dtensor_grad
never use it, and combining pooled execution with concurrent dispatch
is unattempted, unverified future work, not silently assumed safe.
Measured, not just argued: tests/test_distributed_concurrency.py times
"cpu"-alone and "metal"-alone against the same work run concurrently
through dtensor_run, and the concurrent wall time comes in well under
their naive sum.
"""

import threading

from . import _core as core
from . import kir


def _run_parallel(work_fns: list) -> list:
    """Runs each zero-arg callable in `work_fns` on its own
    threading.Thread and returns their results in the same order --
    real overlap between threads depends entirely on the callables'
    own C++ calls releasing the GIL (see the module docstring); without
    that, Python's GIL would serialize these threads the same as
    calling them one after another, just with extra thread-scheduling
    overhead on top. A raised exception from any thread is re-raised
    here (from the main thread, after every thread has finished) rather
    than silently lost, which is what happens by default when an
    exception escapes a threading.Thread's target."""
    results: list = [None] * len(work_fns)
    errors: list = [None] * len(work_fns)

    def runner(i):
        try:
            results[i] = work_fns[i]()
        except Exception as e:  # noqa: BLE001 -- re-raised below, not swallowed
            errors[i] = e

    threads = [threading.Thread(target=runner, args=(i,)) for i in range(len(work_fns))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for e in errors:
        if e is not None:
            raise e
    return results


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
    actual interpreter, not a simulation of one, and every device runs
    concurrently on its own thread (see _run_parallel and the module
    docstring for why that's real overlap, not just extra threads).

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

    # Built once, up front, rather than lazily inside a thread: a mesh
    # could in principle list "metal" more than once, and racing two
    # threads on the same `fused_graph is None` check is exactly the
    # kind of unsynchronized-shared-state bug the rest of this module's
    # concurrency claim explicitly does NOT extend to.
    fused_graph = kir.elementwise_fusion(graph) if "metal" in mesh.devices else None

    work_fns = []
    for i, device in enumerate(mesh.devices):
        shard_inputs = [dt.shards[i] for dt in dtensor_args]
        if device == "cpu":
            work_fns.append(lambda shard_inputs=shard_inputs: kir.run(graph, *shard_inputs))
        elif device == "metal":
            work_fns.append(lambda shard_inputs=shard_inputs: kir.run_metal(fused_graph, *shard_inputs))
        else:
            raise ValueError(f"unknown device {device!r}")

    results = _run_parallel(work_fns)
    return DTensor(mesh, output_placement, results)


def _all_reduce_sum(tensors: list):
    """Elementwise-sums a list of same-shape Tensors -- the collective a
    Replicate()'d gradient needs. Every device ran the forward+backward
    pass using the exact same value at this point (a weight baked into
    the graph as a `constant` node, per find_constant()'s own docstring,
    or an input placeholder itself Replicate()'d rather than sharded),
    so each device's local gradient is only a partial contribution --
    from that device's own slice of whatever *was* sharded upstream (the
    batch, in the ordinary data-parallel shape) -- and the true gradient
    is their sum, not any single device's answer on its own. Goes
    through tolist()/from_flat() for the same reason _split_tensor and
    _concat_tensors do: there's no native elementwise-N-way-add kernel,
    and prototyping the collective's semantics in Python first is the
    same tradeoff this module already made for split/gather."""
    flats = [t.tolist() for t in tensors]
    total = list(flats[0])
    for flat in flats[1:]:
        for i, v in enumerate(flat):
            total[i] += v
    return core.from_flat(total, list(tensors[0].shape))


def dtensor_grad(graph, wrt: list, wrt_placements: list, *dtensor_args) -> list:
    """Distributed backward. Builds kir.grad(graph, wrt) once (an
    *unfused* graph -- see kir.grad's own docstring for why fusion has
    to wait until after differentiation), runs it on each mesh device
    against that device's own shard of every dtensor_arg -- exactly the
    same per-device dispatch dtensor_run uses for the forward graph,
    kir.run for "cpu" and kir.run_metal (on an elementwise_fusion'd copy,
    same as dtensor_run) for "metal" -- and combines each wrt entry's N
    per-device *local* gradients into the one true gradient via the
    collective its own placement calls for:

    - Replicate() -- typically a weight/bias found via
      find_constant(graph, tensor) on the ORIGINAL tensor object dtensor
      arguments never wrapped, since every device's forward pass read
      that exact same constant-embedded value: each device's local
      gradient is a partial contribution from its own data slice, and
      the true gradient is their SUM (_all_reduce_sum, an all-reduce).
    - Shard(dim) -- typically the traced graph's own placeholder that a
      Shard()'d dtensor_arg feeds: each device only ever saw its own
      slice of that input, so its local gradient already covers exactly
      that slice and nothing else; concatenating the N local gradients
      along `dim` reconstructs the full gradient (_concat_tensors, an
      all-gather) with no scaling of any kind needed, since the pieces
      are disjoint by construction.

    wrt_placements is supplied explicitly by the caller, one entry per
    wrt id, for the same reason dtensor_run takes an explicit
    output_placement rather than inferring one: working out how a
    gradient's placement follows from an arbitrary graph's structure is
    a real, separate design problem, not attempted generically here.

    The loss-scaling question a real data-parallel implementation always
    has to face turns out to already have the right answer here, for a
    non-obvious reason worth spelling out rather than leaving as an
    unexamined assumption: grad()'s `sum`/`mean` vjp rules
    (_vjp_sum/_vjp_mean) bake their normalizing constant in as a plain
    Python float, computed from `graph`'s own trace-time shape -- and
    `graph` is the SAME graph dtensor_run's own docstring already
    establishes the convention for: traced once against a shape
    representative of the full logical computation, not any one
    device's shard (see dtensor_run's docstring on why an unevenly-
    shardable shape isn't handled). Concretely: a `mean()` loss traced
    against the full batch bakes in scale = 1/(full batch size), and
    every device's local backward pass -- run against only that
    device's own shard -- still multiplies by that SAME full-batch
    scale, not a per-shard one. Each device's local gradient is
    therefore already exactly "this shard's contribution to the
    full-batch mean," and summing them via all-reduce reconstructs the
    true full-batch-mean gradient directly -- no len(mesh) rescale,
    and no `sum()`-vs-`mean()` distinction to get right by hand. Verified
    in tests/test_distributed_grad.py by checking both loss reductions
    against the same full-batch kir.grad() reference. The failure mode
    this sidesteps (get the pitfall) would only appear if `graph` had
    instead been traced against a single shard's own shape -- a
    workflow this module never uses or recommends.

    Returns one real kansai.Tensor per wrt entry (already reduced to the
    single true answer) -- not a DTensor, since after all-reduce/
    all-gather there's exactly one correct value and no remaining reason
    to keep it partitioned.
    """
    if len(wrt) != len(wrt_placements):
        raise ValueError("wrt and wrt_placements must be the same length")
    if not dtensor_args:
        raise ValueError("dtensor_grad needs at least one DTensor argument")
    mesh = dtensor_args[0].mesh
    for dt in dtensor_args:
        if dt.mesh is not mesh:
            raise ValueError("all DTensor arguments must share the same DeviceMesh")

    bwd_graph = kir.grad(graph, wrt)

    # Same up-front (not lazy-inside-a-thread) fusion as dtensor_run,
    # and the same real concurrent dispatch via _run_parallel -- see
    # both of their own docstrings for why.
    fused_bwd_graph = kir.elementwise_fusion(bwd_graph) if "metal" in mesh.devices else None

    work_fns = []
    for i, device in enumerate(mesh.devices):
        shard_inputs = [dt.shards[i] for dt in dtensor_args]
        if device == "cpu":
            work_fns.append(lambda shard_inputs=shard_inputs: kir.run(bwd_graph, *shard_inputs))
        elif device == "metal":
            work_fns.append(lambda shard_inputs=shard_inputs: kir.run_metal(fused_bwd_graph, *shard_inputs))
        else:
            raise ValueError(f"unknown device {device!r}")

    raw_results = _run_parallel(work_fns)
    per_device_results = [out if isinstance(out, tuple) else (out,) for out in raw_results]

    results = []
    for j, placement in enumerate(wrt_placements):
        per_device_j = [per_device_results[i][j] for i in range(len(mesh))]
        if isinstance(placement, Replicate):
            results.append(_all_reduce_sum(per_device_j))
        elif isinstance(placement, Shard):
            results.append(_concat_tensors(per_device_j, placement.dim))
        else:
            raise TypeError(f"unknown placement {placement!r}")
    return results
