# Kansai

A deep learning framework, in progress. Milestone 0 (Phase 1) is done: a
CPU-only tensor engine with working reverse-mode autograd, proven by
training a 2-layer MLP on XOR. Phase 2 (KIR) is done -- see below. A real
Metal (Phase 3) backend now also exists, on the same terms: real GPU
kernels, tested, benchmarked honestly.

## What's here

- `core/` — `Tensor`, `Storage`, `StoragePool` (the memory planner's
  pooled allocator), and the autograd engine (`GradNode` +
  `Tensor::backward()`, a tape-based reverse-mode implementation)
- `backend/cpu/` — raw float-buffer kernels (Accelerate-backed `matmul`
  on macOS, portable triple-loop fallback elsewhere)
- `backend/metal/` — a real Metal compute backend: MSL shaders compiled
  at runtime (`MTLDevice::newLibraryWithSource`), dispatched through
  `MTLComputeCommandEncoder`, for `matmul`, `bias_relu` (fused), and
  `add_bias`
- `python/` — nanobind bindings (`_core`) plus `kansai.nn` / `kansai.optim`
- `python/kansai/kir.py` — KIR prototype: `Graph`/`Node` schema, an
  operator-overload tracer (`kir.trace`, `kir.jit`), a reference
  interpreter (`kir.run`), dead code elimination
  (`kir.dead_code_elimination`), pattern-matched elementwise fusion
  (`kir.elementwise_fusion`, `kir.run_fused`), a memory planner
  (`kir.plan_memory`, `kir.run_planned`), and source-transform autograd
  (`kir.grad`, `kir.find_constant`)
- `tests/test_xor.py` — the Milestone 0 acceptance test
- `tests/test_kir_trace.py` — the Phase 2 tracing smoke test: trace/replay
  parity with eager, DCE correctness (including on an unused
  placeholder), and a full XOR training loop run entirely through the
  jit'd forward pass
- `tests/test_kir_fusion.py` — the Phase 2 fusion smoke test:
  correctness on both known fused patterns plus a timing benchmark
  proving the fused path is actually faster, not just numerically equal
- `tests/test_grad_check.py` — numerical gradient checking (central
  differences) for matmul and the Linear+ReLU forward shape, against
  first principles rather than "training still converges"
- `tests/test_kir_memory_planner.py` — plan correctness (no
  overlapping-liveness nodes share a slot), safety across many pooled
  runs with fresh random inputs, the pool actually converging instead of
  growing, the forward-only safety guard, and a timing benchmark at a
  scale where pooling is measured to actually win (see below for where
  it doesn't)
- `tests/test_kir_grad.py` — source-transform autograd: numerical
  gradient checks against central differences with the eager engine
  never involved, an exact cross-check against the eager engine as an
  independent second implementation, and a full XOR training loop where
  `.backward()` is never called at all
- `tests/test_metal_backend.py` — Phase 3: kernel correctness against
  CPU, `run_metal` correctness end-to-end on the fused Linear->ReLU->
  Linear graph, a size-sweep benchmark against Accelerate (see below for
  the honest result), and the batched-elementwise-chain benchmark on a
  real (not synthetic) chained graph

## Phase 2 status

Built: the KIR node/graph schema, a tracer (`kir.trace`), a naive
interpreter (`kir.run`), `kir.jit` (shape-specialized trace + cache +
replay), and dead code elimination as the first optimization pass. The
interpreter dispatches back to the real, autograd-tracked `Tensor` ops in
traced order, so `backward()` and the optimizer work through a jit'd
forward pass exactly as they do in eager mode — verified by training XOR
to convergence through `kir.jit(model.__call__)` instead of calling
`model(x)` directly.

This is a Python-side prototype of the schema, not the production IR —
per the project plan, KIR is meant to live in C++ as the real contract
between the frontend and every backend. Prototyping the tracing and pass
semantics here first, cheaply, is what de-risks that C++ implementation.

Elementwise fusion is also built (`kir.elementwise_fusion` +
`kir.run_fused`): it recognizes two chains -- `bias_add -> relu` (every
Linear layer's forward epilogue) and `sub -> mul(self)` (the diff*diff
core of MSE loss) -- and replaces each match with a single node that
runs a dedicated, straight-line C++ kernel in one pass instead of two or
three. It's a small, explicit lookup table of known patterns, not a
general "fuse any elementwise chain" compiler, and that was a real
finding, not a design choice made up front: a first version *was*
general, compiling arbitrary chains to a tiny per-element bytecode
interpreted by a generic stack machine. It was correct but measured
~7x *slower* than running the ops unfused, because interpreting a
bytecode per element can't be auto-vectorized, and for ops this cheap
(a single flop each) that dispatch overhead swamps the memory bandwidth
it was supposed to save. The pattern-matched version -- each recognized
chain gets its own plain, vectorizable loop, same style as the unfused
kernels -- measures a genuine ~2.2x speedup on a 2M-element benchmark
(`tests/test_kir_fusion.py`). Forward-only, deliberately: a fused
node's output carries no `grad_node`, so it's for inference/benchmarking
paths, not for dropping into a training loop as-is.

Layout optimization is also built, in a scoped-down form worth being
explicit about: the roadmap's original "layout optimizer" bullet meant
NCHW ↔ NHWC, which has no material to work with yet -- Kansai has no
`Conv2d` or 4D tensor at all (it was listed in Phase 1's roadmap but
never actually built in Milestone 0). What *is* real and load-bearing
today: `matmul`'s backward pass used to materialize a full transposed
copy of one operand (`transpose2d`) before its two gradient matmuls --
an extra allocation and a full read+write pass over memory for
something BLAS already does for free via its transpose flag. Replaced
with `matmul_nt`/`matmul_tn` (backend/cpu/Ops.hpp), which read the
original buffer directly (`CblasTrans`, or the portable fallback's
swapped indexing) -- same underlying principle as NCHW/NHWC layout
optimization (avoid physically moving data; choose how it's *read*
instead), just applied to where it actually matters in this codebase.
Measured ~7x faster for a single grad_a computation at a
Linear(512→512)-at-batch-256 size (a standalone benchmark, since the
old code path no longer exists to compare against in-repo). Correctness
is verified by `test_grad_check.py`'s numerical gradient check, which
is a stronger guarantee than comparing against the old implementation
would have been -- it validates the new kernels from first principles.

The memory planner is also built, and it's the one optimization this
session where the honest story is genuinely mixed rather than a clean
win. `kir.plan_memory(graph)` does real liveness analysis (each node's
last-use index) plus a greedy slot assignment (same family as
linear-scan register allocation) -- on the traced Linear->ReLU->Linear
forward, 5 nodes need a buffer but only 1 slot is ever concurrently
live, since it's a strict chain with no branching. `kir.run_planned`
executes a graph pooling every intermediate through a `StoragePool`
(backend: a first-fit free list) instead of malloc'ing fresh per node,
releasing each buffer the instant its last consumer (per the plan) has
run. Every op reaches the pool completely transparently -- the pool
hooks into `Tensor::zeros()`, the one place every op already builds its
output, so `add`/`matmul`/`relu`/etc needed zero changes.

Measured, not assumed: at the tiny XOR scale, pooling is a net *loss*
(~0.8-0.95x) -- the Python-level release bookkeeping plus its handful of
extra nanobind calls per run costs more than the malloc/free it avoids
for a few dozen floats. It crosses over, measured, around feature-dim
~1000 at batch 128: large enough that a fresh buffer's page faults
(touching never-before-mapped memory) cost more than this function's
own fixed per-call overhead, so reusing an already-resident buffer wins
outright -- 1.25-1.28x measured at dim 1024-2048. Same lesson as
elementwise fusion's first (bytecode-interpreter) attempt, applied
without needing to relearn it: measure the actual crossover, because
"avoids work" doesn't automatically mean "faster" once there's fixed
overhead on the path doing the avoiding.

Forward-only and enforced, not just documented: `run_planned` asserts
every input has `requires_grad=False`, refusing to run rather than risk
a pooled buffer getting reused while a `GradNode` still expects to read
it during a later `.backward()` -- silent memory corruption is a far
worse failure mode than a wrong forward value, so this is a hard check.
The graph's own output is deliberately never auto-released (the
function can't know how long the caller wants to keep it); the caller
calls `core.release_to_pool(pool, result)` once done with it, same as
any pooled allocator -- forgetting to do so was the first bug this pass
actually tripped over during testing (the output alone re-mallocing
every call, since nothing told the pool it was free).

Source-transform autograd is also built now: `kir.grad(graph, wrt)`
returns a *new* Graph computing gradients, not a tape replayed at
runtime -- the last of the project's original design decisions that was
still open. Every op in the forward graph is re-embedded into the new
graph first (so vjp rules have primal values available -- relu's needs
its input's sign, mul's needs the other operand), then walked in
reverse applying one vjp rule per op, accumulating cotangents into any
node with more than one consumer -- the same algorithm
`Tensor::backward()` already runs over the eager tape, just building
graph nodes here instead of executing real ops. `kir.find_constant`
locates the node holding a specific parameter Tensor, for building the
`wrt` list. A handful of ops needed exposing as first-class,
forward-only primitives to make this possible: `matmul_nt`/`matmul_tn`
(the same transpose-avoiding kernels the layout optimization built, now
also reachable from a *graph*, not just Tensor::matmul's own C++
backward closure), `relu_backward`, `sum_axis0` (bias-add's vjp), and
`broadcast_scalar` (sum/mean's vjp -- spreading a scalar cotangent back
out to a shape). A graph with more than one requested gradient returns
them packed into a `"tuple"` node, unpacked into a real Python tuple by
`run()`/`run_fused()`'s dispatch loops.

Verified three ways, each a stronger claim than the last:
numerically, against central differences, with the eager engine never
invoked anywhere in the check (`kir.trace` -> `kir.grad` -> `kir.run`,
start to finish); exactly, cross-checked against the eager engine as an
independent second implementation of the same math (both came back
identical to the bit -- unsurprising once you notice both paths
bottom out in the very same `matmul_nt`/`matmul_tn`/etc. kernels, but
worth having as a check that isn't just "close enough"); and
practically, training XOR to convergence with `.backward()` never
called anywhere in the loop -- gradients come entirely from running the
graph `kir.grad()` built once, ahead of time.

The honest cost, measured rather than left implicit: a training step
through `kir.grad()` takes about 1.6x as long as eager `backward()`,
because the backward graph recomputes the *entire* forward pass
internally (vjp rules need primal values, and this doesn't yet share
them with a separately-run forward graph) on top of the general
overhead of a Python-level graph interpreter versus calling eager
`Tensor` methods directly. A joint forward+backward graph sharing primal
computation would remove the doubled forward pass; not built here.

Must run on an *unfused*, *unplanned* graph, in that order deliberately:
apply `elementwise_fusion`/`plan_memory` to `grad()`'s result afterward
if wanted, never before, since the vjp rules are written against the
unfused op vocabulary. And a `grad()`-produced (multi-output, `"tuple"`)
graph doesn't work with `run_planned` yet -- see that function's own
docstring for the specific, understood hazard (a returned gradient's
buffer could be handed to a later node before the caller reads it) and
why it fails loudly (`KeyError`) rather than unsafely.

Not yet built: an actual layout optimizer once there's a layout-
sensitive op (Conv2d) for one to work on, and differentiating through
the backward graph itself (higher-order gradients).

Known limitations:

- `kir.jit` bakes any non-traced operand (a Module's weight/bias) into
  the graph as a constant node holding a *reference* to that Tensor.
  This stays correct across a training loop only because `SGD.step()`
  mutates parameters in place — a functional/immutable parameter-update
  scheme would break it silently. Solving that properly (weights as
  their own kind of graph input, not a baked-in constant) is later
  Phase 2 work.
- `Linear`'s weight init draws from `random_device` unless given an
  explicit `seed`. With plain SGD and ReLU, an unlucky init can push
  every hidden unit permanently negative within the first ~20 steps
  ("dying ReLU": all-zero gradient forever after, loss stuck predicting
  the batch mean). A 20-trial sweep at `lr=0.5` failed on 10 of them;
  `lr=0.1` failed on 0/20 over 1000 steps but still 1/50 -- real, if
  rare. `tests/test_xor.py` and `test_kir_trace.py` pass explicit,
  swept-and-verified seeds for exactly this reason: a smoke test that
  passes or fails by the luck of an unseeded RNG isn't actually testing
  anything reliably.
- Numerically gradient-checking a ReLU network is inherently fragile
  right at a near-zero pre-activation: perturbing by +/-EPS can flip the
  ReLU gate on only one side of the central difference, producing a
  large discrepancy that reflects the check, not the engine. Found this
  directly: `test_grad_check.py` originally shared one `random.seed(0)`
  across two unrelated test blocks, and adding the matmul check before
  the Linear+ReLU one silently shifted the second block's values onto
  exactly this boundary with zero changes to that block's own code.
  Fixed by giving each block its own `random.Random(seed)` instance —
  no shared global RNG state across unrelated sections — plus an
  explicit assertion that no pre-activation sits within 0.05 of zero,
  so a future coincidence like this fails with a clear message instead
  of a confusing gradient mismatch.

## Phase 3 status: a real Metal backend

`backend/metal/` runs actual GPU compute on this machine's own GPU --
three MSL shaders (`matmul`, `bias_relu`, `add_bias`), compiled at
runtime via `MTLDevice::newLibraryWithSource` (this machine has only
Command Line Tools, not full Xcode, so the offline `metal`/`metallib`
compilers aren't available -- runtime compilation doesn't need them),
dispatched through `MTLComputeCommandEncoder`. `kir.run_metal(graph,
*args)` takes an *already-fused* graph (`elementwise_fusion`'s output)
and dispatches `matmul` and `fused_bias_relu` to Metal, a plain
bias-broadcast `add` (a layer's unactivated final output) to
`metal_add_bias`, and everything else (the loss computation -- tiny,
and not what this demonstrates) to CPU. Correctness is verified at the
kernel level and end-to-end on the real Linear->ReLU->Linear graph,
matching CPU to within expected floating-point reduction-order
differences (GPU and CPU sum in different orders; exact bit-for-bit
agreement was never the right bar).

The honest performance result: **Metal loses to Accelerate at every
size tested**, from 64x64 up to 2048x2048 for matmul (0.00x-0.19x) and
from 0.1M to 16.8M elements for a single fused bias+relu kernel call
(0.03x-0.67x) -- though the gap consistently narrows as size grows in
both cases, which is the interesting part. Two real, separate causes,
not one:

1. **Accelerate is exceptional on Apple Silicon.** It uses the AMX
   matrix coprocessor -- a specialized hardware matrix-multiply unit --
   not just NEON/AVX-style vectorization. Beating it needs a seriously
   optimized GPU kernel (shared-memory tiling, register blocking) or
   Apple's own MPSGraph, which likely exploits its own specialized
   hardware paths. The naive kernel here (one GPU thread per output
   element, no tiling at all) was never going to be competitive on raw
   matmul FLOPs -- proving the pipeline and the correctness was the
   actual goal of writing it by hand instead of reaching for MPSGraph
   immediately. Still true, still the single biggest lever left on the
   matmul path -- not addressed by the batching below, which is about
   the elementwise kernels specifically.
2. **The original dispatch design paid real, avoidable overhead on
   every call**: `newBufferWithBytes` copies host memory into a new
   Metal buffer for every argument, and each op got its own command
   buffer plus a blocking `waitUntilCompleted` -- no overlap, no
   batching multiple ops into one command buffer before synchronizing
   once. For the elementwise kernels (bandwidth-bound, no tiling needed
   to be competitive in principle) this was very likely the dominant
   cost, not the compute itself -- which the next section fixes.

### Batching the elementwise kernels into one command buffer

Fixed: `metal::run_elementwise_chain` (backend/metal) encodes a whole
sequence of `bias_relu`/`add_bias` steps into ONE command buffer with
ONE `waitUntilCompleted`, each step's output staying resident on the GPU
and feeding directly into the next step's input -- a new encoder per
step (ended before the next begins), which is the standard way to chain
several dispatches without an explicit fence, since Metal's automatic
hazard tracking makes each step's writes visible to the next step's
reads within one command buffer. Only the first upload and the final
download ever touch host memory, however long the chain.

Measured on an isolated synthetic chain (N steps of alternating
bias_relu/add_bias, nothing else involved): **4-10x faster** than
calling the equivalent stepwise functions N times, depending on batch
size and chain length -- confirming the per-call round trip really was
the dominant cost, not the compute.

Wiring this into `kir.run_metal` needed a second, independent fix, not
just calling the new function: `elementwise_fusion`'s greedy single-use
grouping absorbs a layer's `add,relu` pair *and* the next layer's
unactivated `add` into one three-member group whenever nothing else
consumes the first layer's output, and the fusion pass used to require
a group to match one whole known pattern (`("add","relu")` or
`("sub","mul")`) -- a 3-member group matched neither, so *nothing* in
it got fused at all, silently, for any graph shaped like two chained
layers. Confirmed this directly on a real traced graph before fixing
it, not assumed. Fixed by having `elementwise_fusion` greedily segment
a group into a *sequence* of known 2-node patterns (`_segment_group`)
instead of requiring the whole group to match one shape -- a group
longer than a single recognized pair now fuses each recognizable pair
inside it, chained to whatever's on either side, rather than being
rejected wholesale. `run_metal` then batches any run of consecutive
`fused_bias_relu`/bias-broadcast-`add` nodes it finds (there can be
more than one such run per graph, separated by matmuls, each batched
independently) into a single `metal_elementwise_chain` call.

Measured on the real, now-correctly-fused graph (one matmul feeding a
fused bias+relu immediately followed by another layer's bias-add, batch
128, dim 512): **1.22x** -- smaller than the isolated benchmark's
4-10x, because savings scale with how long the elementwise-only run is,
and this graph has only a two-step run bookended by one matmul (whose
own cost dominates and isn't touched by this fix). A deeper network
with more consecutive elementwise steps between matmuls -- or fusing
matmul itself into the same command buffer -- would show a larger
share of the isolated benchmark's win; not attempted here.

### Tiling the matmul kernel

Fixed, partially: the naive kernel (one GPU thread per output element,
reading full rows of A and columns of B straight from device memory
every time) meant every thread in a threadgroup was independently
re-fetching the *same* data its neighbors were already fetching -- no
reuse at all, purely global-memory-bandwidth bound. `matmul_kernel` now
stages one 16x16 tile of A and one of B into threadgroup (on-chip
shared) memory per step, synchronizes once via `threadgroup_barrier` so
every thread has finished writing before any thread reads, then has all
256 threads in the group reuse those two tiles for 256 multiply-adds
each before moving to the next tile along K -- cutting global memory
traffic by roughly 16x compared to the naive version. Boundary tiles (M,
K, or N not a multiple of 16) are handled by zero-padding an
out-of-range read and masking an out-of-range write, verified directly
against CPU at several non-tile-aligned sizes (17x15x19, 100x50x77,
even 1x1x1), not just the round numbers.

Dispatch had to change alongside the kernel, not just the shader source:
`dispatchThreadgroups:threadsPerThreadgroup:` (a fixed 16x16 per group)
replaced `dispatchThreads:`, deliberately -- the tiled kernel needs
*every* threadgroup to be the full 16x16 even at the M/N boundary, since
every thread must participate in loading the shared tile (only its own
output write is masked); `dispatchThreads`'s non-uniform threadgroup
sizing at boundaries would have handed some boundary groups fewer
threads than 256, leaving part of the shared tile never written by
anyone.

The shader source itself is generated via `[NSString stringWithFormat:]`
now rather than a bare string literal, so the tile size is one real
number substituted into the MSL text rather than a C preprocessor
`#define` that a raw string literal would never have expanded in the
first place (the offline symptom: Metal's compiler would have seen the
literal text `KANSAI_MATMUL_TILE` as an undefined identifier) --
including remembering to escape the two literal `%` (modulo) operators
already in the bias kernels as `%%`, since `stringWithFormat:` treats
every unescaped `%` as its own format specifier.

Measured, and this is the "partially" in "fixed, partially": tiling
gets Metal to **0.22x** at 2048x2048 (up from 0.19x untiled) -- roughly
15-27% faster than the naive kernel across the sizes tested, but still
4-5x *slower* than Accelerate at every size, not competitive. Single-
level tiling with one output element per thread closes only part of the
gap to a specialized, AMX-backed CPU BLAS implementation; the standard
next steps (register blocking -- each thread computing a small tile of
outputs instead of one, to amortize the shared-memory load over more
work; double-buffering tile loads against compute; a larger tile size
where occupancy allows it) would close more of it, and MPSGraph would
likely close the rest by reaching the same specialized hardware paths
Accelerate does. None of that is done here -- this is one real,
measured step on the path, not the destination.

Still legitimate, understood next steps beyond that: `newBufferWithBytesNoCopy`
over page-aligned host allocations to remove the upload copy entirely
(Apple Silicon's unified memory makes this possible in principle).
Given CUDA is physically impossible on this machine (no NVIDIA GPU
exists to target) and Vulkan would mean testing against the very same
GPU through an extra translation layer (MoltenVK), Metal was the only
backend that could be verified end-to-end on real hardware in this pass.

## Build

Requires CMake ≥ 3.18, a C++17 compiler, Python ≥ 3.9, and `nanobind`
installed in that Python's environment:

```bash
python3 -m pip install --user nanobind
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The compiled extension lands directly in `python/kansai/`, so there's no
install step.

## Run

```bash
python3 tests/test_xor.py
```

## Design notes

- Only `float32` is implemented; `DType` is an enum specifically so more
  dtypes can be added later without touching call sites.
- `add`/`matmul`/etc. are autograd-aware: they build a `GradNode` that
  captures its parent tensors and a backward closure. `backward()` does a
  DFS topo-sort over that graph and applies each closure in reverse.
- Parameter updates (`Tensor::add_`) bypass the graph entirely — they're
  not part of the forward computation, so they shouldn't be tracked.
- `matmul`'s backward pass needs transposes; those are computed as plain
  buffers inside the backward closure, not as their own differentiable op
  (no second-order gradients in Milestone 0).
