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

## Serialization

`python/kansai/serialize.py`: `save(module, path)` / `load(module,
path)`, closing the last item on this project's original Phase 4
roadmap. Scope drawn deliberately narrow, the same way quantization's
scope was: parameter *values* only, not architecture. Reconstructing a
model's hyperparameters (layer sizes, how a Sequential's children
compose) from a saved file is a real, separate, harder problem needing
a stable schema for describing structure itself, not just numbers -- so
the contract matches PyTorch's own `state_dict`/`load_state_dict` split:
the caller constructs the model first (with constructor arguments a
training script already has), then `load()` fills in the numbers. The
other real option here -- serializing a traced KIR `Graph`'s own
structure, nodes/edges/attrs, for a jit'd model -- is harder still (a
stable node-id/op-name schema) and unattempted, stated as such rather
than silently folded into this.

Getting from `Module` to a named, loadable set of tensors needed one
small prerequisite: `parameters()` only ever returned a flat, order-
dependent list, with no name attached to tell a saved tensor apart from
which attribute it came from. `Module.named_parameters()` (new) does
the identical recursion `parameters()` already did over `vars(self)`,
just keeping the dotted attribute path instead of discarding it --
`"layers.0.weight"` for a `Sequential`'s first sub-layer's weight, the
same shape of name PyTorch's own `named_parameters()` produces for the
same reason. `parameters()` itself is now defined in terms of it
(`[t for _, t in self.named_parameters()]`) rather than duplicating the
recursion -- a small, genuine simplification, not scope creep.

File format, invented for this project rather than reused wholesale,
but structurally close to a well-known one on purpose: a length-
prefixed JSON header (`{name: {shape, dtype, offset, nbytes}}`)
followed by one flat data blob, the same two-part shape HuggingFace's
`safetensors` uses -- NOT claimed binary-compatible with it, since this
omits things the real spec requires (8-byte offset alignment, an
`__metadata__` key) that nothing here needs. Deliberately not pickle,
PyTorch's own historical default and the reason loading an arbitrary
`.pt` file off the internet is a real, well-known security concern
(unpickling can execute arbitrary code as a side effect of reconstructing
an object graph). `json.loads` parses data, never executes it, and the
tensor payload is unpacked as raw floats via Python's `array` module,
not deserialized as objects -- the worst a corrupted or adversarial file
can do here is fail a loud, specific check, never run code.

Loading a value back into an already-constructed `Module` needed a
second small design decision: `Tensor` exposes no generic in-place
"overwrite my own data" operation (only `add_`, not a natural fit for
"replace this parameter's value" without an awkward subtract-then-add),
so `load()` reassigns the attribute directly (`setattr`) rather than
mutating the existing `Tensor` object -- walking the dotted name back
down through the module tree (`_set_by_path`, the exact inverse of how
`named_parameters()` built that name) to find where. Correct, but with
a real, stated consequence worth surfacing rather than discovering by
surprise: an optimizer built from `model.parameters()` *before* a
`load()` call still holds the OLD `Tensor` objects by identity in its
own params list, so it would keep updating parameters the model's
forward pass no longer uses after the load replaced them. `load()`
before constructing an optimizer, not after -- the same ordering
PyTorch's own `load_state_dict` requires for the identical reason.

Checked in `tests/test_serialize.py`: a round trip through
`save()`/`load()` reproduces a trained model's parameters bit-for-bit
(float32 through raw bytes and back is lossless, not approximate -- any
deviation at all would be a real bug) and its forward-pass output
exactly, for both the trained XOR `Sequential` model and a `Conv2d`
(confirming this isn't special-cased to one layer type); four distinct
failure modes -- a shape mismatch, a checkpoint missing a parameter the
model expects, a checkpoint carrying one the model doesn't, and
corrupted magic bytes -- are each rejected with a specific, useful error
rather than a silent wrong load or a crash three layers down inside
`json`/`array`; and the file's actual byte size is checked against what
the header independently predicts, not just trusted to be right because
nothing crashed.

Not attempted, stated up front: architecture/hyperparameter
serialization and KIR `Graph` serialization (both above), any form of
versioned migration between checkpoint format revisions (there's only
ever been the one, `"KAN1"`), and a C++ bulk reader/writer -- this goes
through `Tensor.tolist()`/`core.from_flat()` the same "prototype in
Python first" way `quantize.py`'s (de)quantization already does, fine at
this project's own model sizes, a real un-optimized cost (one Python
float object per element, both directions) at a size large enough for
that to matter. (`distributed.py`'s split/concat used to be in this same
category -- see the next section for why that's no longer true.)

## General tensor ops: reshape, transpose, slice, cat

Every op this codebase had before now computed *values*; none of them
could reorganize a tensor's own *shape*. That gap was real and repeatedly
self-documented, not discovered late: `distributed.py`'s own
`_split_tensor`/`_concat_tensors` went through `tolist()`/`from_flat()`
specifically because "Kansai has no slice op yet," stated in that
function's own docstring three separate times across this project's
history. Closed now with four new ops, all added at every layer this
codebase's other ops already go through -- `backend/cpu` kernels,
`Tensor` methods with real `GradNode` backward functions, nanobind
bindings (GIL-released, like every other compute-heavy one), KIR tracing
(`TraceValue` methods plus a module-level `kir.cat`), all four
interpreters (`run`, `run_fused`, `run_planned`, `run_metal`), and
`kir.grad`'s vjp rules -- not a special, second-class op category bolted
on beside the real ones.

**`reshape`** is a pure reinterpretation: row-major reshape never
reorders a single byte, it only moves where the shape boundaries fall
across the same flat sequence, so the "kernel" is literally `cpu::copy`
(a `memcpy`) into a freshly shaped buffer. Its own backward is exactly
itself, run again with the original shape.

**`transpose(dim0, dim1)`** is the opposite case: a genuine data
permutation, not a metadata change, because this codebase has no stride
concept at all -- every `Tensor` is always fully packed row-major, so
swapping any two axes (not just the trailing two of a 2D matrix)
requires actually moving every element. The new `cpu::transpose` kernel
computes this generically for any rank: decompose each flat input index
into N-D coordinates via that shape's own row-major strides, swap the
two coordinates at `dim0`/`dim1`, recompute the output's flat index from
*its* strides (the same shape with those two sizes swapped), write.
O(n) with one coordinate decomposition per element -- correct and
general, not vectorized or blocked; a real, unattempted lever if this
ever shows up as a bottleneck. Verified specifically at a 3D transpose
across *non-adjacent* axes (`dim0=0, dim1=2` on a `(2,3,4)` tensor), not
just the easy trailing-two-axes case a naive implementation could get
right by accident. Backward: swapping the same two axes is its own
inverse, so transpose's own backward is transpose again, same `dim0`/
`dim1`, applied to the cotangent.

**`slice(dim, start, stop)`** extracts a contiguous sub-range along one
axis -- the exact outer/inner/dim_size block-copy math
`distributed.py`'s own `_split_tensor` already had in Python, ported to
C++ (`cpu::slice`) for native speed. Its backward needed a real design
choice: rather than inventing a new backward-only "unslice" IR
primitive, it's built from existing ops -- `cat` the incoming gradient
back together with zero `constant` nodes padding out whatever `slice`
dropped on either side (skipping a padding piece entirely wherever it
would be zero-sized, i.e. `start == 0` or `stop` reaches the dimension's
full size). All three shapes of that logic -- padding on both sides,
one side only, or neither -- are exercised directly in
`tests/test_shape_ops.py`, not just the common middle case.

**`cat(tensors, dim)`** is the mirror image of `slice`, and reuses the
same primitive in the opposite direction: `cpu::scatter_range` (shared
with `slice`'s own backward) writes each input into its own disjoint
offset of the output -- disjoint by construction, so this is always a
plain write, never an accumulate. `cat`'s own backward is, in turn,
built from `slice`: each input's gradient is exactly the range of the
cotangent at the offset that input was originally written to. Three ops
(`transpose`'s self-inverse, `slice`↔`cat`'s mutual inverse via a shared
`scatter_range` kernel) covering four gradient rules, not four
independent backward kernels -- the same "reuse a forward-shaped op as
its own vjp building block" instinct `Conv2d`'s backward already applied
(`matmul_nt`/`matmul_tn` instead of a dedicated `conv2d_backward`).

All four **copy**, none of them **view**: no `Tensor` shares another's
`Storage`. A real zero-copy view -- meaningful for `reshape` especially,
since its data literally doesn't move -- would need `StoragePool`'s
pooling to become aware of aliased buffers (two `Tensor`s sharing one
`Storage` with different lifetimes); today the pool's release logic
assumes exclusive ownership tied to one node's liveness, and handing a
view's buffer back to the free list while another `Tensor` still aliases
it would be a silent correctness bug, not a missing optimization. Real,
unattempted future work, not assumed safe.

No Metal kernels for any of the four yet, on purpose, matching Conv2d's
own history (CPU first, Metal added later in a separate pass): all four
fall back to the CPU implementation inside `run_metal`, the same honest
fallback path `matmul_nt`/`matmul_tn`/`relu_backward`/`sum_axis0`/
`broadcast_scalar` already use there for the exact same reason. Verified
end-to-end anyway, on both backends: `tests/test_shape_ops.py` runs
every op's full stack -- forward value, eager backward against central
differences (the same gold-standard bar `test_grad_check.py` holds every
other op to), and the complete KIR path (`trace` → all four interpreters
→ `kir.grad`) checked against eager -- with Metal-path checks included
wherever `core.metal_available()`.

**A concrete proof this was worth building, not just an abstract
capability**: `distributed.py`'s `_split_tensor`/`_concat_tensors` are
now three lines apiece (`tensor.slice(dim, offset, offset + size)` in a
loop; `core.cat(tensors, dim)`), replacing roughly sixty lines of
manual outer/inner/stride bookkeeping over Python lists -- the exact
gap this module's own docstring called out as a known, stated
limitation across three separate places in this project's history,
closed by the first real consumer of the new ops rather than staying a
demo capability with nothing depending on it. Every distributed test
(`test_distributed.py`, `test_distributed_grad.py`,
`test_distributed_concurrency.py`) still passes unchanged against this
refactor, including the concurrency benchmark's own ~1.53x measurement
-- confirming the native ops didn't just work in isolation, they slot
into an existing, real caller with no behavior change.

Not attempted in this pass, stated up front at the time: general
NumPy-style broadcasting for `add`/`sub`/`mul` (still exactly one
hardcoded pattern, the `(batch, features) + (features,)` bias case) --
a separate, also-substantial piece of work touching the elementwise
kernels and their vjp rules rather than shape/layout, deliberately
scoped out rather than rushed alongside this. Done in the very next
piece of work -- see the next section.

## Broadcasting

`add`, `sub`, `mul` now accept any right-aligned, NumPy-compatible pair
of shapes -- before this, `add` supported exactly one broadcast shape
(the `(batch, features) + (features,)` bias case, hardcoded), and
`sub`/`mul` supported none at all: any other shape mismatch was a hard
error. The design question that mattered most here wasn't "how to
broadcast" (the standard right-align-and-stretch-size-1-dims rule,
same as NumPy/PyTorch), it was "how to add it without breaking or
slowing down the fast paths that already depend on the OLD, narrower
bias-broadcast shape check" -- `elementwise_fusion`'s pattern match into
`fused_bias_relu`, and `run_metal`'s own batched dispatch into
`metal_add_bias`/`metal_elementwise_chain`, both real, measured,
already-shipped performance wins that specifically recognize that one
2D shape.

The answer: three paths in `Tensor::add`, fastest-checked first --
the fixed bias case (`cpu::add_bias_broadcast`, its own dedicated
kernel, completely unchanged), an exact shape match (`cpu::add`, also
unchanged), and, new, a general fallback (`cpu::add_broadcast`) taken
only when neither of the first two applies. `sub`/`mul` get two paths
each (exact match, then the new general fallback -- they never had a
bias-specific case to preserve). The general kernels use the standard
stride-0 broadcast trick: a `broadcast_strides` helper computes, per
operand, a 0 stride for any axis that operand doesn't have or holds as
size 1 while the output is bigger there, so a single generic N-D
coordinate-decomposition loop (the same shape as `transpose`'s own, see
the previous section) reads the *same* source element for every output
position along a broadcast axis instead of materializing a larger
buffer.

Backward needed a genuinely new general primitive:
`reduce_to_shape(grad, target_shape)` sums a gradient back down to a
smaller shape, summing over every axis the smaller shape doesn't have
at all or holds as size 1 -- the exact inverse of how that shape
broadcast up to begin with, and a strict generalization of the old
`sum_axis0` (a fixed "sum over axis 0 of a 2D tensor" case; verified to
produce bit-identical results to it on the bias shape, and kept
alongside it rather than replacing it, since nothing needed it to
change). `mul`'s broadcast backward composes two primitives rather than
needing a third: `d/da(a*b) = grad_output * b` computed at the *full*
output shape first (via `mul_broadcast` again -- broadcasting `b` up
exactly the way the forward pass itself did), then `reduce_to_shape`
brings it back down to `a`'s own shape. Exposed as first-class KIR ops
too (`reduce_to_shape` joins `sum_axis0`/`matmul_nt`/`matmul_tn` in
`GradOps`, dispatched in `run`/`run_fused`/`run_metal` -- not
`run_planned`, which already excludes `grad()`-produced graphs entirely,
same as `tuple`/`broadcast_scalar`), so `kir.grad`'s `_vjp_add`/
`_vjp_sub`/`_vjp_mul` could be *simplified*, not just extended: each
now just compares an operand's own shape against the output's, calling
`reduce_to_shape` only when they differ, uniformly across the bias
case, the general case, and plain same-shape ops -- no separate
bias-specific branch left in any of the three.

A real bug surfaced and fixed during this work, not discovered later
by a user: `run_metal`'s `_metal_elementwise_kind` decided whether an
`"add"` node was "the bias case" by checking only *whether the two
input shapes differed at all* -- true for the bias case, but now also
true for a general broadcast like `(3,1) + (1,4)`, which would have
been silently misrouted through `metal_add_bias`'s kernel (built for a
completely different indexing scheme) rather than actually erroring or
computing the right answer. Caught by `tests/test_broadcasting.py`
itself failing on first run, not by inspection -- the fix tightens that
check to the exact same shape predicate `elementwise_fusion`'s own
(already-correct) `_fusable` uses, and a matching fix in `run_metal`'s
plain `add`/`sub`/`mul` dispatch (fall back to the CPU eager op for any
shape `metal_add`/`metal_sub`/`metal_mul` — none of which have a
broadcasting Metal kernel yet — can't handle directly).

Verified at the same bar as the ops in the previous section: forward
values (including a rank-mismatch case, `(2,3,4) + (4,)`, that the OLD
bias check's fixed "a must be exactly 2D" requirement could never have
accepted), eager backward against central differences for a genuine
2-way broadcast where *neither* operand already has the output's shape
(a case the old bias logic never exercised, since one side, the batch
dimension, always already matched), the full KIR path, and -- the
regression check that actually caught the `_metal_elementwise_kind` bug
above -- confirmation that the bias-broadcast case still takes the
exact same fused/Metal fast paths it always did, not just that it still
computes the right numbers.

## sqrt, reciprocal, div, and Adam

Kansai had no way to take a square root or divide two tensors at all --
not a gap anyone had hit yet, because nothing before this needed either
one: SGD's entire update is one `add_` call. Adam's update rule needs
both (`sqrt` for the second-moment normalizer, division to apply it),
so implementing Adam meant adding real elementwise ops first, not
Adam-specific shortcuts.

`sqrt`/`reciprocal` are ordinary new elementwise ops, following the
exact same recipe as every op before them: `backend/cpu` kernels
(`sqrt_fwd`/`sqrt_bwd`, `reciprocal_fwd`/`reciprocal_bwd` -- each
backward kernel takes the forward op's own *output*, not its input,
since `d/dx sqrt(x) = 0.5/sqrt(x) = 0.5/out` and `d/dx(1/x) = -1/x^2 =
-out^2` -- reusing `out` avoids recomputing the sqrt/reciprocal a second
time), `Tensor` methods with real `GradNode` backward closures, GIL-
released nanobind bindings, `TraceValue` methods, an `_OP_TABLE` entry
each, and `kir.grad` vjp rules. Unlike `reshape`/`transpose`/`slice`/
`cat`, neither needs any attrs (a shape, a pair of dims, a range) --
just one Tensor in, one Tensor out -- so neither needed special-casing
in any interpreter's dispatch loop at all: `_OP_TABLE` alone is enough,
and `run_metal` picks them up through its own existing CPU-fallback path
for free, the same one `matmul_nt`/`matmul_tn`/`relu_backward`/
`sum_axis0`/`broadcast_scalar` already use.

`div` isn't its own primitive at all: `Tensor::div(other)` is exactly
`this->mul(other.reciprocal())`, both at the eager level and in
`TraceValue`'s own tracing -- so a traced `.div(...)` call records a
`"reciprocal"` node followed by a `"mul"` node, never a `"div"` node,
and needs no dedicated backward rule or interpreter dispatch case of
its own: `mul`'s and `reciprocal`'s own already-correct chain rules
compose into the right answer automatically. A real, honest cost (two
elementwise passes -- reciprocal then multiply -- instead of one fused
division kernel) for skipping an entire new op category; a dedicated
`div` kernel is real, unattempted future work if this ever shows up as
a bottleneck.

**Adam** (`python/kansai/optim.py`): the standard per-parameter
first/second-moment running-average optimizer (Kingma & Ba, 2014) --
what essentially every real training recipe reaches for by default,
where plain SGD (the only optimizer this project had before) needs a
hand-tuned schedule and often momentum on top to converge at a
comparable rate. No weight decay -- this is the original paper's
algorithm, not AdamW's decoupled-decay variant, which stays a real,
separate, unattempted addition rather than an undocumented option
folded into the same class (PyTorch ships them as two distinct classes
for exactly this reason: silently changing what "Adam" computes by
adding a decay term would be a correctness surprise, not a
convenience).

Implemented entirely in Python over existing Tensor ops -- `mul`,
`add`, `sub`, `sqrt`, `div`, and general broadcasting for the scalar
hyperparameters (`beta1`, `1-beta2`, `eps`, and the folded
learning-rate/bias-correction terms, each built once per `step()` call
as a shape-`[1]` tensor that broadcasts against any parameter's own
shape) -- rather than a dedicated C++ optimizer kernel. The same
"prototype in Python first" tradeoff `distributed.py`'s split/concat
and `quantize.py`'s (de)quantization already made, and only possible at
all now that `sqrt`/`div` exist: several separate elementwise passes
per parameter per step, each its own allocation, instead of one fused
kernel -- correct and clear before fast, the same call this project's
history has made every time the two traded off against each other.

Verified two genuinely independent ways: step-by-step against a
from-scratch Adam re-implementation in plain Python (no `kansai.Tensor`
anywhere in it, so it can't share a bug with the implementation under
test), across five steps with a fixed, known gradient sequence -- long
enough that the running averages' accumulation over time and the bias
correction terms are both actually exercised, not just a first step
where `m`/`v` start at zero and a subtly wrong implementation might
still happen to agree -- matching to float32 precision every step; and
practically, training the exact same XOR model `test_xor.py` trains
with SGD, converging to the same near-zero loss bar with Adam instead,
confirming this is a genuine drop-in optimizer on a real model, not
just a formula that matches in isolation.

## More activations, softmax, cross_entropy

`relu` was this project's only activation until now. Added: `tanh`,
`sigmoid`, `gelu` (each an ordinary new elementwise op, backend kernel +
`Tensor` method + `GradNode` backward + KIR integration, same recipe
`sqrt`/`reciprocal` already established), `leaky_relu` (the first
activation that takes a real parameter -- `negative_slope` -- so it
needed the attrs-based interpreter dispatch `reshape`/`transpose`/
`conv2d` already established, not the simpler no-attrs `_OP_TABLE`-only
path `tanh`/`sigmoid`/`gelu` get away with), and `sum(dim)`/`mean(dim)`/
`max(dim)` -- reduction along ONE axis, the real new capability this
section actually needed, since `sum()`/`mean()` before this only ever
reduced to a full scalar.

`gelu` is the *exact* formulation (`x * Phi(x)`, `Phi` the standard
normal CDF, via C++11's `std::erf`), not the tanh-based approximation
some frameworks default to -- there was nothing to gain from
approximating when the exact form is a one-line standard-library call.
Its backward needs the ORIGINAL input, not just the output (unlike
`tanh`/`sigmoid`, whose derivatives are cheaply expressible in terms of
their own output alone), and isn't cheaply composable from other
existing ops either, so it gets a dedicated `gelu_backward` op -- the
same shape `relu_backward` already has, for the identical reason.
`leaky_relu_backward` follows the same pattern.

`sum(dim, keepdim)`'s forward turned out to need no new kernel at all:
it's exactly `reduce_to_shape` (built for general broadcasting's own
backward, previous section) called with a target shape equal to the
input's own shape but with `dim`'s extent set to 1 -- summing "down to
a shape with one axis collapsed" is precisely what that kernel already
does. Its backward *does* need something new -- `broadcast_to_shape`,
the direct inverse, spreading a reduced-shape cotangent back out along
the axis that got summed over, sharing the same `broadcast_strides`
helper `reduce_to_shape` itself uses internally. `mean(dim)` needed
nothing new at all: it's `sum(dim)` scaled by a shape-`[1]` broadcast
constant (the same scalar-broadcast idiom Adam's own hyperparameters
use), so its gradient falls out of `sum(dim)`'s and the broadcasting
`mul`'s own chain rules automatically.

`max(dim)` is forward-only, **on purpose, not by omission** -- this is
the one design choice in this section worth dwelling on. Max's own true
gradient is an argmax-scatter (1 at the winning position, 0 elsewhere),
but nothing here needs it: softmax's numerical-stability max-
subtraction trick is mathematically constant-shift-invariant --
`softmax(x) == softmax(x - c)` for *any* constant `c`, gradient
included -- so `max(x)`'s own gradient is providably irrelevant to
softmax's true gradient, and every real framework detaches it from the
graph for exactly this reason, not merely as a convenience. Enforced at
both levels: the eager `Tensor::max(dim)` never attaches a `GradNode`
regardless of the input's `requires_grad`, and `kir.grad`'s own
`_vjp_max_dim` explicitly returns a zero rather than raising or being
left out of `_VJP_RULES` entirely -- a graph that happens to ask for
this node's gradient gets a defined, correct (zero) answer instead of a
`KeyError`.

**`softmax(dim)`** and **`cross_entropy(logits, targets)`** are pure
compositions -- `max(dim)` → `sub` → `exp` → `sum(dim)` → `div` for
softmax, adding `log` for cross-entropy's log-sum-exp -- with no
dedicated kernel, `GradNode`, KIR op, or vjp rule of their own at all;
every op they're built from already has one. `cross_entropy` takes
ONE-HOT targets, not a class-index vector -- Kansai has no integer
gather/indexing op yet, so one-hot is what makes this expressible from
existing ops at all (converting a class-index label vector to one-hot
is the caller's own job, a plain Python loop, not something this
needed a kernel for). Computed via the log-sum-exp identity
(`logsumexp(logits) - sum(logits * targets)`), **never**
`softmax(logits).log()` -- the textbook-unstable way to compute this,
since softmax can legitimately underflow to exactly `0.0` in float32
before `log` ever sees it, producing `-inf` and then `NaN` once
multiplied by zero for a masked-out class. Verified directly: softmax
on logits as large as `[1000, 1001, 1002]` stays finite and still sums
to 1 (a naive `exp()` on those values would overflow float32 outright).

Verified at the same three levels the rest of this project's ops are
held to, plus one more specific to this section: `cross_entropy`
checked against an independent from-scratch Python re-implementation of
log-sum-exp (not Kansai's own `softmax().log()` composed a second time,
which could share a bug with the real implementation), and a genuine
practical test -- a 3-class classifier (three separated 2D Gaussian
blobs), trained with `Linear → ReLU → Linear → cross_entropy` and
`Adam`, reaching 100% classification accuracy. The first classification
task (as opposed to regression, XOR included) this project has ever
trained end to end.

## LayerNorm, BatchNorm1d

Both normalize (subtract a mean, divide by a standard deviation, then
apply a learnable per-feature scale and shift) -- they differ only in
*which axis* the mean/variance are computed over, and that difference
turned out to matter for how cheaply each could be built.

**`LayerNorm`** normalizes over the last axis only -- `(..., num_features)`
for any leading shape, the common case every transformer's own
LayerNorm actually uses (the embedding dimension alone, not several
axes jointly). It's a pure composition of ops that already existed
before this section started (`mean(dim)`, `sub`, `mul`, `div`, `sqrt`,
`add`, all already broadcasting-aware): no new kernel, `GradNode`, KIR
op, or vjp rule anywhere. That composability paid off immediately --
`LayerNorm` traces, fuses, dispatches to Metal, and differentiates
through `kir.grad` exactly like any other op, verified the same way
softmax/cross_entropy were, with no extra plumbing required to make any
of that true.

**`BatchNorm1d`** normalizes over the batch axis instead, per feature --
`(batch, num_features)` input only; `BatchNorm2d` for conv activations
(normalizing per-*channel* across `N`, `H`, and `W` jointly) is real,
unattempted future work, needing a multi-axis reduction `mean(dim)`
doesn't do in one call today (a transpose+reshape detour around it is
possible, just not built here). Two things make it a genuinely
different, harder problem than `LayerNorm`, not just "the same idea
with `dim=0`":

1. **Train/eval mode is real, load-bearing state**, not a convenience
   flag. Training mode normalizes by the *current batch's* own mean/
   variance; eval mode normalizes by a *running* mean/variance
   accumulated via an exponential moving average across every training
   batch seen so far -- because a single example at inference time has
   no batch statistics of its own to normalize by. This needed real
   infrastructure that didn't exist before: `Module.train(mode)`/
   `Module.eval()`, recursing into every reachable sub-`Module` the same
   way `named_parameters()` already does, with a class-level `training =
   True` default so it works without any subclass needing its own
   `__init__` to set it (none of them call `super().__init__()` today).
2. **The running-stats update has to NOT be part of the autograd
   graph**, or every training step would grow it indefinitely. Kansai
   has no `detach()` to cut a value out of an active graph, so this
   goes through `tolist()`/`from_flat()` for exactly that value --
   round-tripping through plain Python floats is what actually breaks
   the graph, not an incidental implementation detail. The differentiable
   path (this batch's own normalization, needed for `x`'s and `weight`/
   `bias`'s gradients) and the non-differentiable path (folding this
   batch's statistics into the running buffers) are genuinely separate
   pieces of the same `forward()` call, computed from the same `mean`/
   `var` tensors but consumed completely differently.
   `running_mean`/`running_var` are buffers, not parameters -- plain
   `requires_grad=False` attributes, invisible to `named_parameters()`
   on purpose, so no gradient ever reaches them and no optimizer ever
   steps them. The variance folded into `running_var` gets the standard
   unbiased (`n/(n-1)`) correction; the variance actually used to
   normalize *this* batch stays biased (divide by `n`) -- two different
   numbers from the same computation, matching the convention every
   real BatchNorm implementation uses, not a simplification.

One direct, honest consequence of point 2: `BatchNorm1d` in training
mode is **not** `kir.trace()`-able. Tracing calls `.tolist()` on a
`TraceValue`, which carries no real data (shape/dtype only, by design --
that's the entire point of tracing), so it fails outright with an
`AttributeError` rather than silently tracing something wrong. Checked
directly, not just asserted: `kir.trace()` on a graph containing a
training-mode `BatchNorm1d` raises exactly that error. `LayerNorm` has
no such restriction, and its own KIR-path test (`trace` → all
interpreters → `kir.grad`) proves it.

Verified: forward values for both against a from-scratch manual
normalization (not Kansai's own `mean()`/`var` composed a second time),
backward against central differences for both (`BatchNorm1d`'s check
uses a *fresh* instance per finite-difference evaluation, since reusing
one would let one evaluation's running-stats update -- a real,
intentional side effect of a training-mode forward pass -- contaminate
the next and corrupt the estimate itself, not just be untidy);
`running_mean`/`running_var` actually change after a training-mode
forward pass, and eval mode measurably normalizes by those running
statistics rather than by a fresh batch's own (checked by feeding a
single wildly-out-of-distribution example after training on unrelated
data and confirming the output is *not* trivially near zero, which
normalizing by that example's own -- degenerate, single-point --
statistics would produce); `Module.train()`/`eval()` recursing correctly
through a `Sequential`; and, practically, the same 3-class Gaussian-blob
classifier from the previous section trained twice more -- once with a
`LayerNorm` layer, once with a `BatchNorm1d` layer -- both reaching 100%
accuracy, confirming each is a genuine working layer in a real training
loop, not just forward-correct in isolation.

## A real end-to-end benchmark: MNIST

Every check in this project up to this point trained on XOR, a tiny
synthetic conv net, or a hand-generated 2D Gaussian-blob classifier --
each real in its own way (genuine gradients, genuine convergence), but
each also small and synthetic enough that "the framework can train a
real dataset end to end" was still an inference from smaller pieces,
not something directly demonstrated. `examples/mnist/` closes that gap:
real MNIST (60,000 training images, 10,000 held-out test images, actual
handwritten digits 0-9, downloaded from the same mirror torchvision
itself uses -- the original yann.lecun.com host has been unreliable for
years), a real mini-batch training loop, and a real test-set accuracy
number, using the same toolchain the rest of this devlog already proved
piece by piece: `Linear`, `BatchNorm1d`, `ReLU`, `cross_entropy`, `Adam`.

Deliberately kept separate from `tests/`: this needs network access (to
download MNIST on first run) and takes tens of seconds, both properties
the actual test suite -- fast, deterministic, no external dependencies
-- is built around avoiding. `examples/mnist/download_mnist.py` parses
the standard IDX ubyte format directly (stdlib `struct`/`gzip`/`array`
only, no `numpy`, matching this project's own "built from scratch"
identity even in an example script that isn't part of the framework
itself) and caches the downloaded files under `examples/mnist/data/`
(gitignored, so the ~11MB compressed dataset never enters git history).
One real, stated simplification in `train_mnist.py`: the training set is
shuffled once before training starts, not re-shuffled every epoch --
Kansai has no gather/index-select op, so a genuine per-epoch reshuffle
would mean rebuilding the full flat 60000x784 dataset every epoch
instead of once; MNIST's own canonical file order isn't sorted by class
already, so sequential mini-batches after the one shuffle still see a
reasonable mix of digits each batch, a defensible middle ground for a
benchmark script, not silently passed off as full per-epoch shuffling.

Model: `Linear(784→256) → BatchNorm1d → ReLU → Linear(256→64) → ReLU →
Linear(64→10)`, trained with `Adam` (`lr=1e-3`) and `cross_entropy`,
batch size 128, 15 epochs. Measured, one real run on this machine:
**97.58% test accuracy** (10,000 held-out images the model never
trained on) in **31.7 seconds** total (≈2.1s/epoch, 468 batches/epoch)
-- a perfectly ordinary result for a small MLP on MNIST (nowhere near
the ~99.7% a tuned CNN reaches, and not trying to be -- the point of
this benchmark is proving the training loop is real end to end, not
chasing a leaderboard number), reached with zero hyperparameter search.
Test accuracy is evaluated with `model.eval()` (so `BatchNorm1d`
normalizes by its running statistics, not the test batch's own -- see
the previous section for why that distinction is load-bearing here, not
cosmetic) against the full 10,000-image test set, batched only to keep
any one matmul at a modest size, not because a single 10000×784 matmul
would actually trouble Accelerate.

## detach()

`BatchNorm1d`'s running-stats update (previous section) went through
`tolist()`/`from_flat()` specifically because Kansai had no way to cut a
value out of an active autograd graph -- the batch mean/variance it
needs to fold into `running_mean`/`running_var` are differentiable
(needed for `x`'s own gradient), and folding them in *without* first
breaking that connection would grow the graph across every training
step. `Tensor::detach()` closes that gap directly: a new `Tensor`
sharing the original's `Storage` (a real view -- O(1), no data copy,
the same refcounted sharing `Storage` was already built for) but with
`requires_grad=false` and no `grad_node`. `BatchNorm1d` now uses it
directly (`mean.detach()`, real `Tensor` ops to fold into the running
buffers) instead of round-tripping through Python floats -- cleaner and
faster, with the exact same numerical result (checked: the normalization
test suite's own numbers are unchanged before and after this refactor).

Eager-only, on purpose: no `TraceValue.detach()`, no KIR op. The reason
isn't laziness -- it's that `detach()` wouldn't actually fix
`BatchNorm1d`'s own `kir.trace()`-ability even if it existed there too.
The real obstacle is that updating `self.running_mean`/`self.running_var`
is a Python-level attribute *reassignment*, a side effect a traced graph
has no way to express regardless of whether the value feeding it is
detached -- so extending tracing support to `detach()` would still leave
`BatchNorm1d` untraceable in training mode, for a different, deeper
reason already documented in the previous section. Not attempted where
it wouldn't actually solve the stated problem.

Verified beyond the BatchNorm1d refactor itself: `detach()` preserves
values exactly; it's a genuine *view*, not a copy (checked directly --
mutating through a detached handle via `add_` is visible on the
original, since they share `Storage`); a computation built entirely
from detached tensors never requires grad; and, the sharpest check,
detaching *mid-graph* cuts gradient flow through exactly that one path
while leaving any other path to the same leaf tensor intact -- `loss =
p.mul(q).detach().mul(q).sum()` correctly gives `q` a gradient (through
its own undetached path) while `p` gets none at all (its only path to
`loss` runs through the detached tensor).

## AvgPool2d, MaxPool2d, Dropout

Conv2d existed with nothing to reduce its own spatial output beyond
`stride`, and nothing in this codebase could regularize a layer's
activations at all -- three real gaps for anything CNN-shaped, closed
together since two of the three turned out to need no new C++ at all.

**`AvgPool2d`** is a pure composition, the same payoff `LayerNorm` and
`softmax`/`cross_entropy` already got from being one: `(N, C, H, W)`
reshaped to `(N, C, H/k, k, W/k, k)` -- a real row-major reshape, not a
relayout, verified against a hand-computed 4×4 example specifically to
confirm decomposing `H` this way lands `kh` exactly inside one pooling
window before relying on it -- then `mean(dim)` over the two `k`-sized
axes, one at a time (mean over a 2D window separates into two
sequential 1D means exactly, not an approximation). No new kernel,
`GradNode`, or KIR work; it traces, fuses, and differentiates through
`kir.grad` for free. Scoped to the exact-tiling case
(`stride == kernel_size`, `H`/`W` divisible by it) on purpose -- the
common case, and the one the reshape trick can express at all.

**`MaxPool2d`** could not take the same shortcut, and almost did by
mistake -- this section's one real design trap, caught before it
shipped rather than after. `max(dim)` already existed (built for
softmax's numerical-stability trick, see its own section above) and is
*deliberately* non-differentiable: softmax's max-subtraction is
provably gradient-irrelevant, so `max(dim)` never attaches a
`GradNode`, by design. `MaxPool2d` needs the OPPOSITE: a real,
argmax-routed gradient -- the whole reason a pooling layer's max is
usually worth taking at all is that it stays part of a trainable
network. Reusing `max(dim)` here would have silently given every model
using `MaxPool2d` a zero gradient at that layer, a correctness bug that
would only show up as "this network doesn't learn," not a crash. Built
instead as its own dedicated kernel pair (`maxpool2d_fwd`/`_bwd`):
forward records, per output position, the flat index of the winning
input element (ties broken toward the first-encountered max, the
standard convention); backward scatters the incoming gradient to
exactly that recorded position and nowhere else in the window --
checked directly, not just inferred from a passing central-difference
check: of a 4×4 input pooled 2×2, exactly 4 of 16 gradient entries come
back nonzero, one per window. Eager-only -- no `TraceValue.max_pool2d`
exists, so `kir.trace()` on a graph using it fails immediately with a
clear `AttributeError` rather than silently doing something wrong;
extending KIR support is real, unattempted future work, not something
this section needed to unblock the eager training path (backend/cpu's
`stride` support is already general, independent of `kernel_size`, so
overlapping windows work too, not just the exact-tiling case).

**`Dropout`** needed no new kernel or `GradNode` either: inverted
dropout (survivors scaled by `1/(1-p)`, so eval mode -- see
`Module.train()`/`eval()` -- is a plain identity, not a separate
rescale) is exactly `x.mul(mask)` for a random `{0, 1/(1-p)}` mask built
as an ordinary `requires_grad=False` `Tensor` -- `mul`'s own already-
correct backward routes the gradient through that same mask for free,
which *is* dropout's true gradient. Mask generation goes through
stdlib `random`, not a kernel -- fine at this project's own scale, a
real, stated cost (one Python-level draw per element, every training-
mode forward call) at a size large enough for that to matter, a real
C++ RNG-based kernel being unattempted future work. `kir.trace()`-able,
with a real caveat stated plainly rather than glossed over: tracing
bakes ONE fixed mask into the graph as a constant, correct for a single
traced run but not for a cached graph re-run multiple times (each call
would reuse that same mask, silently defeating dropout's own point) --
not a concern for this project's own eager training loops, which call
`forward()` fresh (and so draw a fresh mask) every time.

Verified: forward values for both pooling layers against hand-computed
references (including `MaxPool2d` with overlapping windows,
`stride < kernel_size`, not just the exact-tiling case `AvgPool2d` is
scoped to); backward for all three against central differences;
`AvgPool2d`'s full KIR path; `MaxPool2d`'s and `BatchNorm1d`-style
eager-only limitation confirmed to fail cleanly, not silently;
`Dropout`'s zero-fraction and survivor-scaling checked statistically
over 2000 elements (close to, not exactly, the expected ratios -- a
statistical check, correctly not held to exact-match tolerance); and,
practically, a real small CNN (`Conv2d → ReLU → MaxPool2d → Linear →
cross_entropy`) trained on a synthetic 3-class image task, reaching
100% accuracy -- proving `MaxPool2d`'s gradient composes correctly
through a real `Conv2d` backward, not just in isolation.

## Batched matmul

`matmul` required both operands to be exactly 2D -- the single biggest
concrete gap standing between this project and multi-head attention,
which fundamentally needs `(batch, heads, seq, d_k) @ (batch, heads,
d_k, seq)`. Closed by treating every dimension except the trailing two
as a NumPy-style broadcastable "batch" shape -- the exact same right-
aligned rule `add`/`sub`/`mul`'s own general broadcasting already
established two sections back, just applied to the dims *before* the
actual `M`/`K`/`N` matrix contraction instead of to the whole shape. A
rank-2 operand (the ordinary case) has an empty, rank-0 batch shape,
which broadcasts against any other batch shape by reading its one
matrix repeatedly -- "a shared weight matrix applied across an entire
batch" falls out of the general rule for free, not as a special case.

Reuses everything broadcasting already built rather than duplicating
it: `broadcast_strides` (the stride-0-for-a-broadcast-axis trick,
`backend/cpu`) drives three new batch-aware wrappers
(`batched_matmul`/`_nt`/`_tn`), each just an outer loop over every
broadcast batch index calling the EXISTING 2D `matmul`/`matmul_nt`/
`matmul_tn` once per item -- no new GEMM logic, only broadcast-aware
indexing around the same Accelerate-backed kernel every other op
already uses. Backward computes `grad_a`/`grad_b` at the *full*
broadcast batch shape first (via the same batched `_nt`/`_tn` calls),
then reduces down to whichever operand's own batch shape was smaller
via `reduce_to_shape` -- the identical "compute broadcast, then reduce"
pattern general `add`/`sub`/`mul` broadcasting's own backward already
uses, not a new one invented for this. The exact 2D+2D case keeps its
own original code path completely unchanged -- no batch-broadcast
bookkeeping, no behavior change, checked directly as a regression.

A real bug surfaced and fixed during this work, the same way the
`_metal_elementwise_kind` broadcasting bug was caught during the
broadcasting section: `run_metal`'s own `"matmul"` dispatch called
`metal_matmul_mps` (Metal's own GEMM, 2D-only) *unconditionally* --
correct before batched matmul existed, a guaranteed crash afterward on
anything with a batch dimension. Caught before it shipped by the test
suite's own `run_metal` check on a 4D input, not discovered by a user
later; fixed with the same honest CPU fallback `reshape`/`transpose`/
`leaky_relu`/etc. already use in `run_metal` for ops without a Metal
kernel of their own.

One real, stated gap this does NOT close, on purpose: `kir.grad`'s own
`_vjp_matmul` still emits `"matmul_nt"`/`"matmul_tn"` nodes, and
`core.matmul_nt`/`matmul_tn` (the `GradOps`-level functions those nodes
call) are themselves still 2D-only -- extending them to batched form
would need their own `batched_matmul_nt`/`_tn`-equivalent exposed all
the way through `GradOps`/bindings/KIR, real, separate work not
attempted here. This is the *same* pre-existing gap `conv2d` already
had with `kir.grad` (never fixed either, stated honestly at the time),
not a new one introduced by this section -- eager `.backward()`
differentiates a batched matmul correctly right now (verified
extensively below); `kir.grad` on a graph containing one fails with a
clear error instead of silently computing something wrong.

Verified: forward values against an independent from-scratch nested-
loop matmul (not Kansai's own matmul called a different way, which
could share a bug with the implementation under test) for the matching-
batch case, the real 4D attention shape specifically, and both
broadcast directions (a shared 2D weight across a batch, and a batch
dimension of size 1 stretching to match the other operand's); backward
against central differences for both the matching-batch and the
broadcast case; the full KIR forward path (`trace` → `run`/`run_fused`/
`run_metal`, the last of which is exactly where the bug above was
caught); and `kir.grad`'s documented gap confirmed to fail cleanly.

## MultiHeadAttention

The capstone this whole recent stretch of sections was actually
building toward: standard scaled dot-product multi-head attention --
`softmax(Q W_q (K W_k)^T / sqrt(d_k)) (V W_v) W_o`, split across
`num_heads` independent heads -- general cross-attention (`query`,
`key`, `value` can be different tensors with different sequence
lengths; ordinary self-attention is just `mha(x, x, x)`).

Only reachable now because two things landed together in this same
session: batched matmul (previous section) and `softmax(dim)` (several
sections back). `Q`/`K`/`V` here are genuinely 4D --
`(batch, num_heads, seq, d_k)` -- and every matmul inside (`Q @ K^T`,
`weights @ V`) is a REAL batched matmul over `(batch, num_heads)`, not
a Python-level loop calling 2D matmul once per head. Splitting and
merging heads is `reshape` + `transpose` (already existing, already
correct for this exact use before today); the `1/sqrt(d_k)` scale
reuses the same shape-`[1]`-broadcast-constant idiom `Adam`/
`BatchNorm1d` already established. The entire class is a pure
composition -- no new kernel, `GradNode`, or KIR work of its own,
compounding the same payoff `LayerNorm`/`softmax`/`AvgPool2d` each got
individually, now all at once in one real, useful layer.

The optional `mask` is ADDITIVE (broadcast-added to the raw scores
before `softmax`), not a boolean selection -- a real design constraint,
not a stylistic choice: Kansai has no `where`/comparison-op/boolean-
masking primitive yet (a real, stated gap in this project's own feature
inventory), so an additive mask is what makes masking possible AT ALL
right now, using only `add`, which already exists and is already
broadcasting-aware. Using an actual `-inf` for masked positions was
deliberately avoided in favor of a large negative finite number -- the
textbook `-inf` choice produces `NaN` the instant every score in a row
is masked (`softmax`'s numerator becomes `exp(-inf - (-inf))`, an
indeterminate `0/0` after the max-subtraction step), a real numerical
trap most naive attention implementations don't think to check for
until it actually happens.

Eager-only in practice, though not by an enforced restriction the way
`BatchNorm1d`'s training path or `MaxPool2d` are: nothing here refuses
to trace, but `kir.grad`'s own matmul vjp rule is 2D-only (this
project's own stated, pre-existing gap, same as `conv2d`'s), so
differentiating a graph built from this via `kir.grad` would hit that
limitation. Eager `.backward()` is unaffected, and is what every check
below actually verifies.

Verified: output shapes for self-attention and cross-attention
(different query/key sequence lengths, confirming this isn't secretly
self-attention-only); forward values against a from-scratch single-head
(`num_heads=1`) attention implementation in plain Python -- not
Kansai's own ops called a different way, which could share a bug with
the real implementation; backward against central differences,
checked for the input AND confirmed that all four projection weights
(`w_q`/`w_k`/`w_v`/`w_o`) actually receive a gradient, not just the
easiest one to check; an additive causal mask confirmed to zero
attention to every future position while each row's weights still sum
to exactly 1 (not inferred from "the loss looks reasonable" -- read
directly off the attention weights themselves); and, practically, a
small attention-based sequence classifier (`MultiHeadAttention` →
mean-pool → `Linear` → `cross_entropy`) trained on a synthetic task --
each sequence carries a class-specific "marker" vector planted at a
RANDOM position among distractors, so the model has to find it
regardless of where it lands -- reaching 100% accuracy.

## index_select and Embedding

Kansai had no way to pick tensor rows (or any-dim slices) out by an
arbitrary list of positions -- every existing op picks a tensor apart
by shape (`reshape`/`transpose`/`slice`/`cat`) or by value (`add`/
`mul`/...), never by a caller-supplied, possibly-repeating list of
indices. That's exactly what a token-embedding lookup table needs, so
it's the concrete gap that had to close first: `Embedding.forward` IS
`index_select` plus a `reshape`.

`indices` is a plain `std::vector<int64_t>` / Python `list[int]`, not a
`core.Tensor` -- a deliberate design choice, not an oversight. Two
reasons converge on it: Kansai's `DType` enum is `float32`-only (there
is no integer dtype to hold indices in even if this wanted to be a
Tensor), and an index into a lookup table isn't a differentiable
quantity in the first place -- it's discrete, wouldn't have a sensible
gradient, and `slice`'s own `dim`/`start`/`stop` already established
the precedent of plain-int-not-Tensor parameters for exactly this kind
of non-differentiable shape/position argument.

New kernels, one pair: `cpu::index_select` (an `outer`/`dim_size`/
`inner` decomposition around `dim`, a `memcpy` per selected slice --
the same three-way split `reduce_to_shape`/`broadcast_to_shape`
already use for N-D axis-aware iteration) and `cpu::index_select_bwd`,
its exact inverse -- except an ACCUMULATE (`+=`), not a plain
overwrite, because `indices` can repeat. That accumulation is the one
real subtlety here: selecting row 2 twice must give row 2 twice the
gradient, not the same as selecting it once, or a token appearing
twice in one training batch would silently only get credit for one of
its two occurrences. Not composable from `slice`/`cat` at all -- both
assume disjoint, contiguous ranges, and a scatter-add over a
repeating index list isn't expressible that way -- so, like `gelu`/
`leaky_relu`'s backward, it gets its own dedicated `GradOps`-exposed
op (`index_select_backward`) rather than being built from existing
pieces.

KIR integration follows the same `attrs`-based pattern every non-
Tensor-parameterized op needs: `TraceValue.index_select` validates
`dim`/`indices` up front (the same bounds check the eager path makes,
so a bad index fails at trace time, not silently downstream) and emits
an `index_select` node carrying `dim`/`indices` as attrs; `run`/
`run_fused`/`run_metal`/`run_planned` each got a dispatch case (three
near-identical, one with `run_planned`'s own separate `elif`-chain
convention, same as every other attrs-based op before it); and
`_vjp_index_select` builds a backward-graph node out of the new
`index_select_backward` op, itself needing its own interpreter
dispatch in `run`/`run_fused`/`run_metal` (not `run_planned`, which
never sees `kir.grad`-produced graphs). Checked at the full usual bar,
INCLUDING through the traced backward graph, not just eager: forward
values (row selection with repeats, column selection), eager backward
against central differences with the repeated-index accumulation
explicitly asserted (not just "central diff matches" -- the literal
expected per-row gradient), all four interpreters, `kir.grad` matching
eager exactly, and `kir.grad`'s output run through `run_metal` too
(confirming the accumulation survives Metal dispatch, not only the
CPU path) -- plus out-of-range rejection.

`nn.Embedding(vocab_size, embed_dim, seed)` is then almost nothing on
top: a `(vocab_size, embed_dim)` weight (same `1/sqrt(fan_in)` init
scale `Linear`/`Conv2d` already use), and `forward(token_ids)` that
flattens a plain (possibly nested) Python list of ints via a small
`_flatten_ids` helper (mirroring `kansai.__init__`'s own `_flatten`,
just producing `int`s instead of `float`s and keeping the nesting
shape around), calls `weight.index_select(0, flat_ids)`, and
`reshape`s the result back to the input's own nesting shape with
`embed_dim` appended -- a flat `(seq_len,)` id list gives `(seq_len,
embed_dim)`, a `(batch, seq_len)` nested list gives `(batch, seq_len,
embed_dim)`. Zero new kernels, zero new `GradNode` logic -- the same
"reuse composition" payoff `LayerNorm`/`AvgPool2d`/`softmax` each got
individually.

Verified: forward values against a manual per-row lookup for both a
flat and a `(batch, seq_len)`-nested id list; backward gradient
ACCUMULATION when the same token id repeats within one batch (id 3
used twice among four positions correctly gets gradient `2.0` in every
component, not `1.0`, with every unused row's gradient confirmed
exactly zero) plus the same check against central differences taken
directly on the weight tensor (token ids themselves can't be
perturbed, being non-differentiable, so the numerical check has to go
through the weight instead); and, practically, a small "does this
sequence contain the marker token" binary classifier (`Embedding` →
mean-pool over the sequence → `Linear` → `cross_entropy`) trained with
`Adam` on synthetic sequences -- a marker token planted at a random
position in half the sequences, pure distractor tokens filling the
rest, forcing the model to actually learn individual token identity
through the embedding table rather than any positional or count-based
shortcut -- reaching 100% test accuracy.

## Comparison ops and where

`MultiHeadAttention`'s own `mask` argument had to be strictly
additive, a real, explicitly stated limitation at the time it shipped:
Kansai had no `where`/comparison-op/boolean-masking primitive at all,
so an additive mask was the only way to make masking possible. That
gap closes here.

Three new comparison kernels -- `gt`/`lt`/`eq` -- slot into
`broadcast_binary`'s existing `BinOp` switch (`backend/cpu/src/
Ops.cpp`) right alongside `Add`/`Sub`/`Mul`, reusing the exact same
general-broadcasting machinery (`broadcast_strides`, the stride-0-for-
a-broadcast-axis trick) `add`/`sub`/`mul`'s own broadcast kernels
already established -- no new broadcasting logic, just three new
per-element comparisons (`1.0f`/`0.0f`) dropped into the same switch
statement. `Tensor::gt/lt/eq` follow the exact-shape-fast-path-vs-
general-broadcast shape every other binary op uses, with one
deliberate, permanent difference: they never attach a `GradNode`, even
when an operand requires grad. A comparison is a step function of its
inputs -- its true gradient is zero (or undefined right at the
boundary) everywhere, not "whatever the usual chain rule would give
if this were differentiable," so it's correct to never wire a
backward path at all, not an oversight to fix later.

KIR integration follows the by-now-established binary-op pattern
(`TraceValue.gt/lt/eq`, `_OP_TABLE` entries) -- but with a twist
worth noting: none of the four interpreters needed a single new
dispatch line. `gt`/`lt`/`eq` take no attrs (same as `add`/`sub`/
`mul`), so `run`/`run_fused`/`run_planned`'s shared `_OP_TABLE`
fallback already covers them for free, and `run_metal` (which has no
Metal comparison kernel) falls through to that exact same `_OP_TABLE`
CPU path automatically too -- the same honest fallback `reshape`/
`transpose`/`slice`/`cat` already take, just without even needing the
explicit `if node.op == ...` line those needed (they carry attrs;
comparisons don't). `_vjp_compare` gives `kir.grad` a defined answer
rather than a `KeyError` if a differentiated graph happens to touch a
comparison node -- an explicit zero for BOTH operands, the identical
"deliberate zero" pattern `_vjp_max_dim` already established for
`max(dim)`, not a new kind of special case.

`where(cond, a, b)` is the real payoff, and needed literally nothing
new: it's `b + cond * (a - b)`, an algebraic rearrangement of the
more obvious `cond*a + (1-cond)*b` chosen specifically to avoid
needing a `ones_like(cond)` -- which would have meant either a new
kernel or new KIR plumbing. Written that way, `where` only calls
`sub`/`mul`/`add`, which already exist, identically, on both eager
`Tensor` and traced `TraceValue` -- so `kansai.where`, one plain
Python function living in `kansai/__init__.py` (not a C++ method, not
a KIR node type at all), works unmodified whether it's called on real
Tensors or inside a `kir.trace`'d graph. Gradient flows into `a`/`b`
through their ordinary `sub`/`mul`/`add` vjps with zero special-
casing needed for `where` itself; gradient into `cond` is exactly
zero, inherited automatically from `gt`/`lt`/`eq`'s own vjp.

Verified: `gt`/`lt`/`eq` forward at both exact-matching and general-
broadcast shapes, and the non-differentiability itself (`requires_grad`
confirmed to stay `False` on a comparison's output even when its input
requires grad) plus rejection of a genuinely non-broadcastable shape
pair; `where`'s forward against a manual selection and backward
confirming gradient lands on `a` at exactly the positions its branch
was taken, on `b` at exactly the complementary positions, and NEVER on
`cond`; the combined `gt`+`where` pattern through the full KIR path --
`trace`, all four interpreters, and `kir.grad` -- confirming that same
gradient routing (zero into the compared values, correct into whichever
branch) survives tracing exactly, not just eager; and, practically,
fitting a genuinely piecewise-linear function (slope `2` for `x>0`,
slope `-1` for `x<=0`) by training `where(x>0, w_pos*x, w_neg*x)` with
`Adam` -- a task that only converges to the two correct slopes if
gradient is actually routed through whichever branch each individual
example took, and it does, landing on `w_pos=2.0000`, `w_neg=-1.0000`
to four decimal places.

## AdamW

`Adam`'s own docstring had said, since the day it shipped, that it
deliberately didn't implement weight decay -- a real, separate,
explicitly-flagged gap, not an oversight. Closed here as its own
class, `AdamW(Adam)`, not a flag on `Adam` itself, for the same reason
PyTorch keeps them separate: silently changing what "Adam" computes by
adding an undocumented decay term would be a correctness surprise for
anyone already relying on it, not a convenience.

Decoupled decay (Loshchilov & Hutter, 2019), not the more naive L2-
regularization approach: L2 would fold `weight_decay * p` into the
gradient itself before Adam's own moment estimates ever see it, so it
gets divided by `sqrt(v_hat)` along with the real gradient -- a
parameter with a large gradient ends up decayed LESS, backwards from
what "weight decay" is supposed to mean. Decoupled decay shrinks the
parameter directly instead, `p *= (1 - lr * weight_decay)`, entirely
outside the moment machinery -- same order of operations PyTorch's own
`AdamW` uses (decay against the pre-step parameter, then the ordinary
Adam update on the now-decayed value).

Implementation is almost nothing on top of `Adam`: subclasses it
rather than duplicating the moment bookkeeping, and the one new line
is `p.add_(p, alpha=-lr*weight_decay)` before calling `super().step()`
-- exactly `p *= (1 - lr*weight_decay)`, expressed through the
existing `axpy_`-backed `add_` (safe to alias a tensor against itself:
`axpy_` is a plain elementwise loop, `p[i] += alpha*p[i]`, no cross-
index dependency). Zero new kernels, same "reuse what already exists"
call every optimizer and composed layer in this project has made.

Verified: with `weight_decay=0`, `AdamW.step()` produces the IDENTICAL
parameter trajectory to plain `Adam.step()` given the same gradient
sequence, step by step -- confirming the subclass changes nothing
about the inherited update at the boundary condition where it
shouldn't; the decay term isolated on its own, using a gradient that's
a genuine Tensor (not `None` -- decay only applies when `p.grad` is
set) but analytically exactly zero everywhere, so `Adam`'s own update
contributes nothing (`m`/`v` stay at zero, `0/eps = 0`) and decay is
provably the ONLY thing moving the parameter -- checked against the
closed form `p₀ · (1 − lr·weight_decay)^steps` after 10 steps, and
confirmed to leave the parameter completely unchanged when
`weight_decay=0`; and, practically, `AdamW` trains the same XOR model
`test_adam.py` trains with plain `Adam`, to the identical convergence
bar.

## Packaging and CI

Two real, previously-confirmed-absent gaps: nothing made Kansai
`pip install`-able, and nothing ran the test suite automatically. Both
close here, and neither was left as "config that looks plausible" --
each was actually run and its real output checked, the same bar every
feature above is held to.

`pyproject.toml` uses `scikit-build-core` (nanobind's own recommended
build backend for exactly this CMake+nanobind combination) rather than
a hand-rolled `setup.py`. The existing `python/CMakeLists.txt` already
had a `set_target_properties(... LIBRARY_OUTPUT_DIRECTORY
".../python/kansai")` line that drops the compiled extension straight
into the source tree for the no-install local-dev workflow the
Quickstart already documented -- left completely unchanged, since
that's a real, working path people already rely on. Packaging needed
one new, additive line instead: `install(TARGETS _core LIBRARY
DESTINATION kansai)`, which only fires when `cmake --install` actually
runs (i.e. only inside `scikit-build-core`'s own wheel-build step, a
plain `cmake --build build` never touches it) -- so the two paths
coexist without either one changing the other's behavior.

Actually verified rather than assumed correct: built a real wheel
(`python3 -m build --wheel`) in a from-scratch isolated venv, installed
it into a SEPARATE clean venv with no relationship to the source tree
at all, then ran `import kansai` and a full XOR training loop
(`Sequential` → `AdamW` → convergence) from `/tmp`, nowhere near the
repository -- confirming the wheel is real and self-contained, not
just "the build didn't error." It converged to loss `5.1e-15` and
`core.metal_available()` reported `True` even from the installed
wheel, off the source tree entirely.

CI is one GitHub Actions workflow (`.github/workflows/tests.yml`) with
two jobs, both on `macos-14` -- GitHub's Apple Silicon (M1) runner, the
ONLY target this project has ever built or run on; a Linux or Intel
runner would be testing a platform nothing here has ever been verified
against, so it isn't used. The `test` job configures and builds via
plain CMake, then runs every `tests/test_*.py` file exactly as a
developer would locally. The `build-wheel` job repeats the exact
manual verification above, automatically, on every push: build the
wheel, `pip install` it, import it from outside the checkout, and
train XOR to convergence -- so a packaging regression (a missing
`install()` rule, a wrong `wheel.packages` path, anything that would
make the *published* package broken even while the source-tree dev
workflow kept working) gets caught by CI itself, not discovered by the
first person who actually tries to `pip install` it. Every existing
test file already gates its own Metal-path assertions behind
`if core.metal_available()` (established well before this session,
back when Metal support first landed), so the suite stays meaningful
whether or not the CI runner's sandboxed GPU access reports available
-- no CI-specific skip logic had to be added anywhere for this to
work correctly.

## Gradient clipping and LR schedulers

The first slice of a broader push: close the remaining gaps standing
between this project and training something real (a small transformer
on actual text, not synthetic data) rather than continuing to add
isolated ops. Gradient clipping and learning-rate schedules are the
two training-loop utilities every serious setup reaches for and this
project had neither of -- clipping in particular matters specifically
because the next real target is attention-based training, a genuinely
common source of the occasional huge-gradient batch that can wreck
Adam's moment estimates for the rest of a run if nothing bounds it.

`clip_grad_norm_(params, max_norm)` computes ONE global L2 norm across
every parameter's gradient combined, not each parameter clipped to its
own norm independently -- clipping per-parameter would change the
relative scale between parameters and distort the update direction,
not just its magnitude, which defeats the point. If the combined norm
exceeds `max_norm`, every gradient is scaled down by the identical
factor so the combined norm becomes exactly `max_norm`; left completely
untouched otherwise. No new kernel, and no new plumbing to expose a
`.grad` setter either (there isn't one, deliberately -- `.grad` is
read-only, populated only by `backward()`): the existing `axpy_`-backed
`add_`, aliased against its own tensor (`g.add_(g, alpha=clip_coef -
1.0)`, computing exactly `g *= clip_coef`), is the same self-aliasing
trick AdamW's own decoupled decay already established, applied here to
`.grad` instead of the parameter itself.

`StepLR` and `CosineAnnealingLR` both just read and write
`optimizer.lr` directly -- every optimizer class (`SGD`/`Adam`/
`AdamW`) already re-reads `self.lr` fresh inside its own `step()`
rather than caching it once at construction, so external mutation
between steps was ALREADY the mechanism a schedule needs; neither
scheduler required touching the optimizer classes at all.
`CosineAnnealingLR` follows the standard half-cosine SGDR schedule
(Loshchilov & Hutter -- the same authors as `AdamW`), the pairing most
commonly used with transformer training in practice, and clamps its
own progress at `T_max` so calling `step()` past the schedule's end
pins `lr` at `eta_min` rather than the cosine argument overshooting
past π and the rate climbing back up.

Verified: `clip_grad_norm_`'s returned norm against a hand-computed
value; that clipping actually rescales a gradient to land at EXACTLY
`max_norm` (checked by recomputing the norm after clipping, not just
trusting the formula) while preserving its direction exactly; that a
norm already within bounds is left completely untouched; that the
norm combines correctly across MULTIPLE parameters at once, not
clipped one at a time (confirmed with a two-parameter case where only
their combined vector has the expected norm); and the all-`None`-grad
case returns `0` rather than erroring. `StepLR` against the exact
expected step sequence across several decay boundaries.
`CosineAnnealingLR` against the closed-form formula at every point
across a full schedule, including both endpoints (`base_lr` before any
`step()`, exactly `eta_min` at `T_max`) and the post-`T_max` pin.
Practically: training a model with `Adam` + `clip_grad_norm_` +
`CosineAnnealingLR` together, with hyperparameters chosen so clipping
is CONFIRMED to actually engage on the very first step (initial
gradient norm 2.95 against `max_norm=0.5`, not a silent no-op) and
`lr` is confirmed to have actually reached `eta_min` by the end of
training, while the model still reaches loss `~0` -- proving the three
pieces compose correctly together, not just individually.

## Dataset and DataLoader

Every training loop in this project up to now -- `test_embedding.py`'s
`MarkerClassifier`, `test_optim_utils.py`'s practical check,
`examples/mnist/train_mnist.py` -- had been writing the same
"shuffle a list of indices, slice it into chunks" boilerplate by hand.
`python/kansai/data.py` is that pattern extracted once, and closes it
for real training loops going forward, the next real target being
attention-based training on actual text.

Map-style only: a `Dataset` is anything with `__len__`/`__getitem__`
(the base class exists to document intent and give a clear
`NotImplementedError` rather than a confusing `TypeError` -- `DataLoader`
itself only ever calls `len(dataset)`/`dataset[i]`, so a plain list
already satisfies the whole interface, no subclassing required).
Deliberately no multi-process worker pool, no pinned memory, no custom
samplers -- Kansai has no threading/IPC infrastructure that would make
out-of-process workers meaningful (`distributed.py`'s own concurrent
dispatch is real threads across `DeviceMesh` devices, a different
problem), and nothing in this project's own training loops has ever
been slow enough for data loading to be the bottleneck. Adding that
plumbing now would be exactly the premature generality this project's
own conventions avoid.

`_default_collate` stops at "grouped into per-field Python lists," not
built into a `Tensor` -- a batch of `(x, y)` tuples becomes
`(list_of_x, list_of_y)` via `zip(*items)`, the exact shape every
training loop above already builds by hand. Deliberately NOT forced
further into a `core.Tensor`: different model inputs need different
shapes from the identical loader (a flat `Tensor` for a `Linear`
layer's features, a plain nested list of ints for `nn.Embedding`'s
token ids -- see `Tensor::index_select`'s own doc comment for why
indices are plain ints, never a `Tensor`), so committing to one
Tensor-construction convention inside the loader would be wrong for
half of what it needs to feed.

`DataLoader(dataset, batch_size, shuffle, seed, drop_last)` reshuffles
on every fresh `__iter__()` call (matching PyTorch's own per-epoch
reshuffle semantics: two consecutive `for batch in loader:` loops over
the same instance get two different orderings, both covering every
example) using its OWN `random.Random(seed)` instance, not the global
`random` module -- so a fixed seed makes a run reproducible completely
independent of whatever else in the process has called `random.*`.

Verified: `shuffle=False` batch order matches manual slicing exactly
(the baseline every other test's own hand-rolled batching loop already
implicitly trusted); `shuffle=True` reproducibility across two
independently constructed loaders sharing a seed, confirmed
independent of the global `random` module's state by deliberately
perturbing it in between and getting the identical order anyway;
every example visited exactly once per epoch (a set-equality check
against `range(n)`, not just "output looks shuffled"); successive
epochs from the SAME loader instance reshuffle rather than repeat;
`drop_last`'s effect on both the trailing partial batch and `__len__()`;
`_default_collate`'s tuple-transposition and plain-value cases; and,
practically, retraining `test_embedding.py`'s own marker-token
classifier through a real `Dataset`/`DataLoader` instead of its
original hand-rolled batching, to the identical 100% test-accuracy
bar -- a genuine drop-in replacement for a real `Embedding`-based
model, not just correct in isolation.

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
