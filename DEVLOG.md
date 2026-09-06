# Kansai devlog

The engineering diary: every design decision, every benchmark actually
run, every bug found and how, in the order it happened. See
[README.md](README.md) for the project overview, build instructions, and
current status at a glance -- this file is the detailed record behind it.

Milestone 0 (Phase 1) is done: a CPU-only tensor engine with working
reverse-mode autograd, proven by training a 2-layer MLP on XOR. Phase 2
(KIR) is done -- see below. A real Metal (Phase 3) backend now also
exists, on the same terms: real GPU kernels, tested, benchmarked
honestly.

## What's here

- `core/` — `Tensor`, `Storage`, `StoragePool` (the memory planner's
  pooled allocator), and the autograd engine (`GradNode` +
  `Tensor::backward()`, a tape-based reverse-mode implementation)
- `backend/cpu/` — raw float-buffer kernels (Accelerate-backed `matmul`
  on macOS, portable triple-loop fallback elsewhere)
- `backend/metal/` — a real Metal compute backend: a hand-written,
  16x16-tiled MSL `matmul` (compiled at runtime via
  `MTLDevice::newLibraryWithSource`, dispatched through
  `MTLComputeCommandEncoder`) plus `bias_relu` (fused) and `add_bias`;
  `matmul_mps` (Apple's `MPSMatrixMultiplication`) is what actually
  reaches parity with Accelerate and is `run_metal`'s default matmul --
  see the Phase 3 section for the honest comparison between all three
- `python/` — nanobind bindings (`_core`) plus `kansai.nn` (now including
  `Conv2d`) / `kansai.optim`
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
- `tests/test_conv2d.py` — Conv2d: forward checked against an
  independent direct/naive nested-loop reference (not just a
  self-consistency check), backward checked numerically, a tiny conv
  net trained by ordinary SGD to a real, checkable near-zero-loss target
  (not a threshold picked by guessing -- see the file for why an
  earlier classification-shaped target had an unreachable ~0.365
  theoretical floor), and KIR trace/run/run_planned integration

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
MSL shaders (`matmul`, `bias_relu`, `add_bias`, and more added as the
backend's op coverage grew, see below), compiled at runtime via
`MTLDevice::newLibraryWithSource` (this machine has only Command Line
Tools, not full Xcode, so the offline `metal`/`metallib` compilers
aren't available -- runtime compilation doesn't need them), dispatched
through `MTLComputeCommandEncoder`. `kir.run_metal(graph, *args)` takes
an *already-fused* graph (`elementwise_fusion`'s output) and dispatches
`matmul` and `fused_bias_relu` to Metal, a plain bias-broadcast `add`
(a layer's unactivated final output) to `metal_add_bias`, and
everything else (the loss computation -- tiny, and not what this
demonstrates) to CPU. Correctness is verified at the kernel level and
end-to-end on the real Linear->ReLU->Linear graph, matching CPU to
within expected floating-point reduction-order differences (GPU and CPU
sum in different orders; exact bit-for-bit agreement was never the
right bar).

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

Measured: tiling gets Metal to **0.22x** at 2048x2048 (up from 0.19x
untiled) -- roughly 15-27% faster than the naive kernel across the
sizes tested, but still 4-5x *slower* than Accelerate everywhere, not
competitive. Single-level tiling with one output element per thread
closes only part of the gap to a specialized, AMX-backed CPU BLAS
implementation. The next section is what actually closes it.

### Reaching parity: MPSMatrixMultiplication

`metal::matmul_mps` calls Apple's own `MPSMatrixMultiplication`
(Metal Performance Shaders) instead of the hand-written kernel --
still genuinely "the Metal backend" (MPS dispatches as Metal compute
through the same command-buffer machinery this project already uses),
just Apple's professionally-tuned GEMM instead of reinventing one by
hand. This was always the documented next step (see "Tiling the matmul
kernel" above) once hand-tiling alone proved insufficient, not a
change of plan.

Measured, warmed up first (see below for why that matters): **MPS
reaches parity with Accelerate around 2048x2048 (0.85x) and wins outright
at 4096x4096 (1.12x) and beyond (1.23x at 8192x8192)** -- a real
crossover, not a rounding-error win. Head to head against the hand-tiled
kernel, MPS wins everywhere measured except a wash at the smallest size
(64x64, 0.99x) -- up to **4.1x faster** at 4096x4096 -- so it replaced
the hand-tiled kernel as `run_metal`'s default matmul dispatch; the
hand-tiled kernel stays available under its own name (`metal_matmul`,
`core.metal_matmul`) as the "written by hand" reference and for direct
comparison, not deleted.

The honest caveat, not swept under the rug: this crossover is
size-shaped, not universal. Large, roughly square matrices are where MPS
wins -- the realistic shape a Linear layer's forward pass actually
produces (a *small* batch dimension against *large* feature dimensions,
e.g. 128x4096 @ 4096x4096) still loses to Accelerate, **0.35x**,
because a thin M dimension means less total work to amortize the fixed
per-call dispatch overhead (buffer creation, command buffer, blocking
wait) against, regardless of how large K and N are. `run_metal` uses
MPS unconditionally rather than switching between kernels by shape,
since it's still a strict upgrade over the hand-tiled kernel at every
size measured, including this one (1.54x faster than hand-tiled at this
exact shape) -- "competitive with Accelerate" and "the best of the
options this backend has" are different claims, and this section is
honest about which one is true where.

Caught before it produced a misleading number: MPS (like most libraries
of its kind) pays a one-time kernel-selection/compilation cost on its
*first* call at a new problem shape. An early, un-warmed benchmark run
produced a 1024x1024 result *slower* than both its 512x512 and 2048x2048
neighbors -- caught by noticing the non-monotonic shape, not by
assuming the first number was right. Every benchmark here now makes one
untimed warm-up call at each exact shape before starting the timed loop,
standard practice for any kernel library with first-use compilation,
applied because it was needed, not on principle alone.

Register blocking, double-buffering, or a larger tile in the hand-tiled
kernel remain real options for anyone who wants to keep pushing the
hand-written path specifically rather than leaning on MPS. Given CUDA
is physically impossible on this machine (no NVIDIA GPU exists to
target) and Vulkan would mean testing against the very same GPU through
an extra translation layer (MoltenVK), Metal was the only backend that
could be verified end-to-end on real hardware in this pass.

### Eliminating the copy: `newBufferWithBytesNoCopy`

The remaining overhead in every Metal call up to this point was
architectural, not computational: `newBufferWithBytes` copies host
memory into a *separate* Metal-owned buffer on every single call, and
the result gets copied *back* to host memory (`memcpy` from
`buf_out.contents`) after every dispatch too -- on Apple Silicon, where
CPU and GPU already share the same physical RAM, both copies are pure
waste, an artifact of the implementation rather than anything the
hardware actually requires.

`newBufferWithBytesNoCopy:length:options:deallocator:` wraps existing
host memory directly, with two hard requirements: the pointer must be
page-aligned, and the caller keeps owning the memory (`deallocator:nil`
here, since a `kan::Tensor`'s `Storage` already manages that lifetime;
letting Metal *also* try to free it would double-free). `Storage`
(`core/Storage.cpp`) didn't meet the first requirement before -- plain
`malloc` on this platform aligns to 16 bytes, not the page size -- so
its allocator now goes through `posix_memalign`, rounding every request
up to at least one full page. That's a real, if small in absolute
terms, cost: every tensor now has a floor of one page (16KB on Apple
Silicon) regardless of how few bytes it logically needs. `nbytes()` was
changed to report that actual (rounded-up) size rather than the
originally requested one -- the only other consumer, `StoragePool`'s
free-list size check, only becomes *more* permissive by seeing the true
figure, so nothing broke there.

With every `Storage` guaranteed page-aligned, every Metal function
(`matmul`, `matmul_mps`, `bias_relu`, `add_bias`,
`run_elementwise_chain`) now wraps its input *and output* buffers
directly via a shared `wrap_no_copy` helper instead of allocating and
copying. The output side removes the trailing `memcpy` entirely -- the
GPU writes straight into the `Tensor`'s own memory, and
`waitUntilCompleted` is exactly the synchronization point that makes
those writes safe to read from the CPU immediately after. In
`run_elementwise_chain` specifically, only the *first* input and the
*last* step's output touch NoCopy wrapping; the intermediate buffers
between steps are pure GPU scratch with no corresponding host tensor,
so there's nothing for them to wrap.

Measured before touching any Metal code: applying only the page-aligned
`Storage` change and rerunning the existing suite caught a real, if
minor, thing on its own -- a benchmark assertion (`batching should be
faster on the real graph`) that had been sitting right at the noise
floor started failing intermittently (confirmed by rerunning the exact
same comparison three times with no code change in between: 0.90x,
1.01x, 1.07x). Not a regression from this work, but a pre-existing
fragility this work's rebuild happened to surface; fixed by taking the
median of five repeated trials instead of one, the standard fix for a
benchmark this close to sub-millisecond noise -- not by loosening the
threshold.

Measured after NoCopy: the whole matmul comparison moved, not just the
weak point.

| Shape | Before (copy) | After (NoCopy) |
|---|---|---|
| 512x512 | 0.21-0.40x | 0.38-0.40x |
| 2048x2048 | 0.76-0.85x | **1.08-1.60x** |
| 4096x4096 | 1.09-1.15x | **1.50-1.57x** |
| 128x4096 @ 4096x4096 (realistic layer) | **0.32-0.35x** | **1.28-1.31x** |

That last row is the one that matters most: the realistic thin-batch
shape a Linear layer's forward pass actually produces was the backend's
one clear weak point two commits ago, losing to Accelerate by 3x. It
now *wins*. That's exactly consistent with the diagnosis at the time --
a thin batch dimension means less compute to amortize *fixed* dispatch
overhead against, and this change is precisely what removed that fixed
cost (two copies per call, gone) rather than adding more compute
throughput. The elementwise batching benchmark moved too (1.27x on this
run, consistent with the 1.1-1.3x range measured previously) --
expected, since eliminating the copies helps every call along that path
equally, not just matmul's.

Not attempted here: extending NoCopy (or the page-alignment it depends
on) to any future backend that doesn't share Metal's unified-memory
assumption -- a discrete GPU with real host/device memory separation
would need actual transfers regardless of alignment, and this
optimization is specific to Apple Silicon's architecture, not a general
technique that would carry over unmodified to, say, a hypothetical CUDA
backend.

## Conv2d

`Tensor::conv2d` (NCHW, `(N,Cin,H,W)` input against `(Cout,Cin,kH,kW)`
weight) is real, not a stub: forward is im2col (unfold each batch
item's input into a `(Cin*kH*kW, Hout*Wout)` patch matrix) followed by
one `matmul` call per batch item -- the same Accelerate-backed kernel
every other op in this codebase already uses, rather than a
hand-written convolution inner loop. Backward reuses the same
transpose-avoiding `matmul_nt`/`matmul_tn` the layout-optimization work
built for `Tensor::matmul`'s own backward, plus `col2im` (im2col's
inverse: a scatter-*add*, since overlapping patches at stride < kernel
size means multiple output positions contribute to the same input
pixel) for the gradient wrt the input.

Correctness is checked two genuinely independent ways on purpose, not
one: forward against a direct, textbook nested-loop convolution with no
im2col or matmul anywhere in it (a self-consistency check like
gradient-checking the *same* forward implementation would never catch
an indexing bug -- both sides would reflect the identical mistake), then
backward against central differences. A tiny `nn.Conv2d` then trains by
ordinary SGD on a task with a *known, exactly representable* target (a
3x3 filter regressing each non-overlapping patch to its own pixel sum,
representable exactly by an all-ones kernel and zero bias) rather than a
threshold picked by guessing: it converges to essentially zero loss and
the learned weight comes out as `[1,1,1,1,1,1,1,1,1]`, exactly the
kernel that reproduces a sum. An earlier version of that same sanity
check used a sign-classification target instead, whose achievable MSE
for any linear (activation-free) model turns out to be ~0.365 by a
symmetry argument -- an assertion threshold below that would have failed
regardless of whether conv2d's gradients were correct, which is exactly
the kind of test that looks like it's checking correctness while
actually just checking whether the target was reachable at all.

KIR integration: `kir.trace` records `conv2d` as a first-class node, and
`kir.run`/`kir.run_planned`/`kir.run_metal` all dispatch it correctly
(verified against eager -- see below for the `run_metal` path
specifically). Not wired into `elementwise_fusion`: no known fused
pattern involves conv2d, so there's nothing to gain from pretending
otherwise. It simply has no case for `"conv2d"`, so a graph containing
one raises a clear `KeyError` there rather than silently mishandling it.

NCHW only, deliberately: no layout optimizer exists yet to choose
between NCHW and NHWC, so there was nothing to gain from supporting
both from day one. Conv2d existing now is what actually gives that
optimizer something to work on -- building it was the whole point of
the "an actual layout optimizer once there's a layout-sensitive op"
line in this README's own earlier Phase 2 section. Not attempted here;
a natural next step whenever it's worth picking up.

## Phase 3 complete: full op coverage and Metal Conv2d

Two gaps remained in `run_metal` after the NoCopy work above: it fell
back to CPU for `sub`/`mul`/`sum`/`mean`/`relu` (the loss computation,
mostly -- `sub -> mul -> mean`, or the `fused_sub_square` shape of it,
plus a solo `relu` wherever one isn't part of a `fused_bias_relu`
pair), and Conv2d had no Metal path at all -- `kir.run_metal` would
`KeyError` outright on a `"conv2d"` node. Both are now closed.

**Elementwise/reduction coverage:** new MSL kernels for `add`, `sub`,
`mul`, `relu` (plain same-shape ops, no broadcast -- the bias-broadcast
cases were already covered by `bias_relu`/`add_bias`), `fused_sub_square`
(the Metal-side twin of the CPU backend's own fused sub-then-square),
and `reduce_sum` (`sum`/`mean` share one kernel via a `scale` parameter
-- 1.0 for sum, `1/n` for mean). The reduction kernel does a grid-stride
accumulation into 256 partial sums followed by a threadgroup-memory tree
reduction; verified correct specifically at n = 255, 256, and 257 to
catch an off-by-one at that fixed 256-thread boundary, with an
n-scaled tolerance for the comparison against CPU (`1e-6 * n`) since
floating-point reduction-order error grows with n by construction, not
by bug. With this, a complete `matmul -> fused_bias_relu -> matmul ->
add -> fused_sub_square -> mean` forward+loss graph now runs
end-to-end on Metal with zero CPU fallback, verified to match
`kir.run()` bit-for-bit on a real run (`cpu=0.358105, metal=0.358105`
-- exact match here, rather than the reduction-order tolerance
elsewhere, because both sides happened to reduce in the same order for
this particular graph shape).

**Metal Conv2d (`metal_conv2d`):** the same im2col+matmul structure as
`Tensor::conv2d` itself, not a separate design -- im2col stays on CPU
(a pure memory-layout unfold, not FLOP-heavy, so there was nothing to
gain from a GPU version of it) and the actual GEMM per batch item goes
through `metal_matmul_mps`, followed by one Metal kernel
(`add_bias_nchw`) for the per-output-channel bias broadcast. That
broadcast needed its own kernel rather than reusing `add_bias`:
`add_bias` broadcasts a bias across the *last* dimension of a 2D
`(batch, features)` tensor, whereas Conv2d's bias broadcasts across the
*channel* dimension of a 4D `(N, C, H, W)` tensor flattened to `(N, C,
HW)` -- a different indexing pattern, not an optional generalization of
the same one.

One page-alignment subtlety caught before it became a runtime bug: the
im2col scratch buffer can't be a plain `std::vector<float>`, because
`metal_matmul_mps` NoCopy-wraps every buffer it's handed (see the
NoCopy section above), which requires page-aligned memory that
`std::vector`'s allocator doesn't guarantee. Using `Tensor::zeros` for
the scratch buffer instead sidesteps the problem entirely -- it's
already page-aligned via `Storage`'s own `posix_memalign`-based
allocator, the same one every other tensor in this codebase goes
through, so the fix cost nothing beyond reusing what already existed.

Correctness: checked directly (`metal_conv2d` vs eager CPU `conv2d`)
across six shapes spanning stride/padding/channel-count combinations,
max error 7.6e-6; and through `kir.run_metal`'s new `"conv2d"` dispatch
case on a traced graph, matching eager to 2.4e-7. No dedicated
register-blocked or tiled Metal convolution kernel exists -- im2col
reduces the whole problem to the same `matmul_mps` call path already
proven correct and reasonably fast, which was the pragmatic choice here
just as it was for the CPU backend; a hand-optimized direct-convolution
kernel remains a possible future lever, not attempted.

With both gaps closed, Phase 3 is complete: every op this codebase's
KIR vocabulary has, Conv2d included, now has a working, tested Metal
dispatch path with no silent CPU fallback inside `run_metal`.

## Phase 4, started: DeviceMesh / DTensor

Deferred for most of this project's history on purpose: DeviceMesh only
means something once there's more than one real backend to shard
across, and until Metal reached genuine parity with Accelerate, this
would have been a data model with nothing real underneath it. It has
something real underneath it now.

What "device" actually means here needed pinning down before writing
any code, because the obvious mental model (a mesh spans separate
memory spaces, sharding moves bytes between them) is simply false on
this hardware. Every `kansai.Tensor` already lives in one page-aligned,
CPU-resident allocation, and Metal's own NoCopy path (the previous
entry) computes directly on that same memory rather than moving
anything anywhere. So `DeviceMesh(["cpu", "metal"])`'s two entries don't
name physical locations -- they name which backend's *interpreter*
processes a given shard's graph: `kir.run` for `"cpu"`,
`kir.run_metal` for `"metal"`. Splitting and gathering
(`python/kansai/distributed.py`'s `_split_tensor`/`_concat_tensors`) go
through `tolist()`/`from_flat()` precisely because there's no native
slice/concat kernel yet, not because there's a network to simulate --
the same "prototype the semantics in Python first" approach `kir.py`
itself took for the IR before any of this had a C++ implementation.

`DTensor.from_tensor(tensor, mesh, placement)` splits (`Shard(dim)`) or
replicates (`Replicate()`) a tensor across the mesh; `dtensor_run(graph,
output_placement, *dtensor_args)` runs a `kir.trace`d graph once per
mesh device, feeding each device its own shard and dispatching to that
device's real backend -- fusing the graph first for the `"metal"` shard
specifically, matching `run_metal`'s own existing contract, since
nothing about DTensor changes what that interpreter requires. Verified
against the same bar as everything else in this project: gather a batch
sharded across `["cpu", "metal"]` through a Linear+ReLU forward (weights
replicated, batch split -- the actual shape of data-parallel training)
and it matches running the identical graph unsharded through `kir.run`
exactly, and a fully `Replicate()`'d run produces bit-identical results
on both the CPU shard and the Metal shard independently. The split/
concat mechanics are checked separately too, including a genuinely
uneven split (7 rows across 3 pieces -> 3, 2, 2, not just the
evenly-divisible case that's easy to get right by accident) and a split
along a non-leading dimension, to exercise the outer/inner stride
bookkeeping rather than only the simplest case.

Two limitations stated up front rather than discovered later:

- **`dtensor_run` itself is forward-only**, same as fusion/pooling/
  `run_metal` before it -- no `grad_node` is attached to anything it
  computes. Distributed *backward* is handled separately, by
  `dtensor_grad` -- see the next section for why that needed its own
  design rather than being a small addition to `dtensor_run`.
- **No real concurrency.** `dtensor_run` dispatches to each device in a
  plain Python loop, one after another. Nothing in this codebase's
  nanobind bindings releases the GIL, so even calling the CPU and Metal
  paths from separate Python threads wouldn't overlap today -- true
  concurrent dispatch would mean releasing the GIL specifically around
  the blocking Metal calls (which already do their own synchronous
  `waitUntilCompleted`), a real, separate piece of work, not something
  this module gets for free by existing.

Also stated as a real, current constraint rather than glossed over:
`dtensor_run` traces its graph once, against one representative shard
shape, so every argument's shards need to match that shape -- true
whenever the sharded dimension divides evenly across the mesh, silently
*not* handled otherwise (tracing a separate graph per differently-shaped
shard is unattempted future work). The test suite's own batch size (8,
split 4+4 across two devices) was chosen specifically to satisfy this,
not by accident.

## Distributed gradients

`dtensor_grad(graph, wrt, wrt_placements, *dtensor_args)` closes the gap
the previous section left open on purpose: distributed *backward*, with
auto-inserted collectives, on the same two real backends `dtensor_run`
already proves forward. Design: build `kir.grad(graph, wrt)` once (an
*unfused* graph, per `kir.grad`'s own contract), run it per mesh device
against that device's own shard exactly the way `dtensor_run` runs the
forward graph, then combine each `wrt` entry's per-device *local*
gradient into the one true answer via the collective its own placement
calls for -- all-reduce (sum) for `Replicate()`, all-gather (concat) for
`Shard(dim)`. This needed adding `"tuple"` and `"broadcast_scalar"`
handling to `run_metal` first: a multi-`wrt` `kir.grad()` graph ends in a
`"tuple"` node, and `sum`/`mean`'s vjp rules emit `"broadcast_scalar"`,
neither of which `run_metal` had ever needed a case for before nothing
had run a backward graph through it. Without that fix `run_metal` would
`KeyError` outright on any distributed backward pass; `matmul_nt`,
`matmul_tn`, `relu_backward`, and `sum_axis0` (the rest of `grad()`'s
backward-only vocabulary) still have no dedicated Metal kernel and fall
through to the existing CPU `_OP_TABLE`, exactly the same fallback path
every other unrecognized op already used -- writing Metal kernels for
those is real, unattempted future work, not a gap this quietly hides.

Correctness bar: the gradient computed from N devices each seeing 1/N of
a batch, combined by these collectives, must equal the gradient computed
from one device seeing the whole batch at once -- the standard
data-parallel-training claim, checked directly against `kir.grad()` run
on the unsharded graph (not re-proving `kir.grad()` vs eager agreement,
which `test_kir_grad.py` already covers on its own). Verified on a
Linear+ReLU model, batch `Shard(0)`'d across `["cpu", "metal"]` with
weight/bias `Replicate()`'d (baked into the graph as `constant` nodes,
found via `find_constant()` on the original tensor objects -- the
ordinary data-parallel shape), for both a `wrt` list of three entries
(weight, bias, and the sharded input itself, to exercise both
collectives together) and a single-entry `wrt` (`kir.grad()` returns a
plain graph rather than a `"tuple"`-rooted one in that case, a genuinely
different code path worth its own check) -- all matching the full-batch
reference to within floating-point noise.

The loss-scaling question every real data-parallel implementation has to
face -- does summing per-device gradients silently need a `1/len(mesh)`
correction when the loss is `mean()`-reduced rather than `sum()`-reduced
-- got checked empirically rather than assumed either way, and the
answer turned out more interesting than either guess: no correction is
needed, for a reason specific to how this codebase's vjp rules work.
`_vjp_mean` bakes its normalizing constant in as a plain Python float
computed from `graph`'s own trace-time shape, and `graph` here is the
same graph whose tracing convention `dtensor_run` already established --
traced once against the full logical batch, never a single shard's
shape. So a `mean()` loss traced against a full 8-row batch bakes in
`scale = 1/8` regardless of which device later runs the backward graph,
and every device's local gradient -- computed against only its own
4-row shard -- already carries that full-batch `1/8`, not a per-shard
`1/4`. Summing two such quarters-of-the-truth back together lands
exactly on the true full-batch gradient with nothing left to correct.
An earlier draft of this section assumed the opposite (that summing a
`mean()`-based per-shard gradient would over-count by exactly
`len(mesh)`, the textbook data-parallel gotcha) and wrote a test
specifically to demonstrate that factor -- the test's own measured
ratio came back `1.0000`, not `2.0000`, which is what actually caught
the wrong assumption before it shipped as a false caveat in the
docstring. `tests/test_distributed_grad.py` keeps the corrected,
verified version of that check: both a `sum()`-loss and a `mean()`-loss
graph, run distributed, matching their own full-batch `kir.grad()`
reference exactly.

## Quantization

`python/kansai/quantize.py`: post-training int8 weight quantization,
inference-shaped and deliberately standalone -- the same "prototype the
semantics before committing to a real kernel" approach this project
already took twice (`kir.py`'s IR before any C++ backend existed;
`distributed.py`'s `_split_tensor`/`_concat_tensors` before a native
slice/concat kernel). Nothing in `Tensor`/`Storage`/`DType` changes:
`core/include/kansai/DType.hpp` still only ever declares `Float32`, and
`Tensor::data_ptr()` is unconditionally `float*` at hundreds of call
sites across this codebase -- retrofitting a second storage width into
that class would be invasive surgery on working code for a feature
nothing else yet depends on. A `QTensor` is instead a small, separate
Python type: an int8 payload (a plain list -- there's no int8 `Storage`
to hold it in) plus one float `scale`, converted back to an ordinary
`core.Tensor` (`dequantize()`) at the one point it actually needs to
enter a real kernel.

Symmetric, per-tensor quantization, not asymmetric/affine: `q =
round(x / scale)`, `scale = max(|x|) / 127`, no zero-point. The right
fit for what this actually quantizes -- trained weights, roughly
zero-centered by construction, both before training (sampled from a
zero-mean Gaussian) and typically after -- and the simpler of the two
standard schemes, the same "start simple, prove correctness first"
choice this project made for Metal's own first matmul kernel before
MPS. Asymmetric quantization would matter for post-ReLU activations
(all ≥ 0, wasting half of symmetric int8's range) -- not attempted,
since nothing here quantizes activations.

`qlinear(x, qweight, bias)` dequantizes the weight back to float32 and
runs the ordinary Accelerate-backed `matmul`+`add` underneath --
proving quantization's numerical correctness honestly rather than
pretending there's a real int8 GEMM kernel underneath (there isn't).
The claim this actually supports is memory footprint, not FLOPs: a
`QTensor` measures at exactly 4.00x smaller than the equivalent float32
tensor (1 byte/element vs 4, checked directly, not assumed from the
byte-width arithmetic alone), with zero speed claim attached -- a true
int8 GEMM kernel (feeding int8 operands directly into a hardware
int8 dot-product path, skipping the dequantize-then-float32-matmul
round trip entirely) is real, substantial, unattempted future work,
the same honest gap this project already left open for a Metal
convolution kernel and for concurrent multi-device dispatch.

Correctness, checked three separate ways: round-trip quantize/
dequantize error is bounded by the known quantization step (`scale/2`,
not just "looks close"), checked at 1000 random values, plus an exact
check that `0.0` quantizes to `q=0` precisely (what makes the
zero-point-free scheme valid at all); the 4x memory ratio is measured
directly rather than assumed from the byte-width math; and `QLinear`
(built from an already-trained `nn.Linear`, quantizing its weight once
at construction -- bias stays float32, since it's a tiny vector nowhere
near where the memory win matters, and keeping it exact avoids stacking
a second error source on the weight's own) is dropped into the exact
XOR model and training run `test_xor.py` already uses, and its
quantized predictions still classify XOR correctly (`[0, 1, 1, 0]`),
with a measured max prediction error of ~0.007 against the float32
model's own output -- the same "does it still actually work" bar
`test_xor.py`'s and `test_conv2d.py`'s own end-to-end sanity checks
already hold themselves to, not a new one invented for this feature.

Not attempted, stated up front: per-channel scales (one scale per
output channel/feature instead of one for the whole tensor -- a real
refinement that shrinks error further, not a prerequisite for this to
be useful); activation quantization; quantization-aware training
(continuing to train through int8 weights, rather than quantizing only
after training finishes); and, as above, an actual int8 GEMM kernel.

## Concurrent dispatch

Every earlier section of this Phase 4 work stated the same limitation:
`dtensor_run`/`dtensor_grad` dispatch to each mesh device in a plain
Python loop, one after another, because nanobind didn't release the
GIL around any binding in this codebase -- so even driving "cpu" and
"metal" from separate Python threads wouldn't have overlapped their
execution; the GIL would serialize them exactly as if they were one
loop. Closed now, on both ends of that gap.

`python/bindings.cpp`: every compute-heavy binding the "cpu" and
"metal" dispatch paths actually call -- the plain `Tensor` methods
(`add`, `sub`, `mul`, `matmul`, `relu`, `sum`, `mean`, `conv2d`), the
backward-only ops `kir.grad`'s vjp rules emit (`matmul_nt`, `matmul_tn`,
`relu_backward`, `sum_axis0`, `broadcast_scalar`), and every `metal_*`
function -- now carries `nb::call_guard<nb::gil_scoped_release>()`.
Each one is pure C++ number-crunching on the buffers it's handed (an
Accelerate call, a hand-written loop, or a Metal dispatch's own
synchronous `waitUntilCompleted`), touching no Python object once
inside, so releasing the GIL for that duration is safe -- and it's
exactly what lets another Python thread's own such call proceed
concurrently instead of waiting on the GIL. Checked before relying on
it, not assumed: Metal's command queue is documented thread-safe for
concurrent command-buffer creation from multiple threads (Apple's own
Metal Best Practices Guide), `state()`'s (`backend/metal/MetalOps.mm`)
one-time lazy initialization is a function-local `static`, which
C++11 already guarantees is thread-safe against concurrent first
callers, and the CPU backend's kernels touch only the buffers passed
to them -- no shared mutable state for two threads to race on. The one
real exception, stated rather than glossed over: `kir.run_planned`'s
`StoragePool` is a genuinely shared, non-thread-safe free-list;
`dtensor_run`/`dtensor_grad` never use pooling, and combining pooled
execution with concurrent dispatch is unverified, unattempted future
work, not silently assumed safe by this change.

`python/kansai/distributed.py`: a new `_run_parallel(work_fns)` runs
each device's work on its own `threading.Thread` and returns results in
device order (re-raising any thread's exception from the main thread
afterward, rather than losing it the way an uncaught exception in a
`threading.Thread` target normally would). `dtensor_run` and
`dtensor_grad` both build their per-device closures up front, fuse the
graph for the "metal" device up front too (not lazily inside a thread,
where two "metal" entries in the same mesh racing the same
`is None` check would be exactly the kind of unsynchronized-shared-
state bug the rest of this change explicitly doesn't extend to), then
dispatch through `_run_parallel` instead of a plain `for` loop.

Measured, not just argued: `tests/test_distributed_concurrency.py`
times a 4096×4096 matmul, split 2048+2048 rows across `["cpu",
"metal"]`, three ways -- "cpu" alone, "metal" alone, and through
`dtensor_run`'s real concurrent dispatch -- and checks the concurrent
wall time against the naive (serial) sum of the other two. First
attempt used a smaller (2048×2048, split 1024+1024) shape, where each
call lands in the single-digit milliseconds; three back-to-back runs
of that version came back 1.49x, then failed at 1.15x, then failed at
0.90x (concurrent dispatch measured *slower* than serial) -- at that
scale, thread creation and OS scheduling overhead is a large enough
fraction of the total time to swamp the actual concurrency signal, the
same kind of noise-floor problem the elementwise-batching benchmark hit
earlier in this project, fixed the same way: not by loosening the
assertion, but by sizing the workload so real compute dominates over
fixed overhead. At 4096×4096 (tens of milliseconds per call), seven
repeated trials came back 27.12-27.35ms for the concurrent path against
a 41.5-41.8ms naive serial sum -- three independent full runs of the
test measured 1.52x, 1.53x, 1.53x, consistent to within noise. The
concurrent time (~27ms) landing close to "cpu alone"'s own solo time
(~25.8ms) rather than partway between it and the naive sum is the
clearest sign of what's actually happening: "metal"'s ~15.7ms leg is
almost entirely hidden inside "cpu"'s own leg's duration, close to the
best case two genuinely concurrent, unequal-duration tasks can achieve.

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
