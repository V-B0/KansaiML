# Kansai

A deep learning framework, in progress. Milestone 0 (Phase 1) is done: a
CPU-only tensor engine with working reverse-mode autograd, proven by
training a 2-layer MLP on XOR. Phase 2 (KIR) is now underway — see below.

## What's here

- `core/` — `Tensor`, `Storage`, `StoragePool` (the memory planner's
  pooled allocator), and the autograd engine (`GradNode` +
  `Tensor::backward()`, a tape-based reverse-mode implementation)
- `backend/cpu/` — raw float-buffer kernels (Accelerate-backed `matmul`
  on macOS, portable triple-loop fallback elsewhere)
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
