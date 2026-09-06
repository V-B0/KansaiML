<p align="center">
  <img src="KansaiML.png" width="96" alt="Kansai">
</p>

<h1 align="center">Kansai</h1>

<p align="center">
  A deep learning framework built without the compromises PyTorch made<br>
  when it still had to ship in 2016.
</p>

<p align="center">
  <a href="https://github.com/V-B0/KansaiML/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/V-B0/KansaiML/actions/workflows/tests.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-active%20development-orange.svg">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey.svg">
</p>

---

Kansai treats compilation, hardware parity, and distributed training as
foundations, not extensions bolted on after the fact. The core idea:
Python traces into **KIR** (Kansai Intermediate Representation), an
optimizer runs over that graph once, and any backend can consume the
result — CPU today, a real Metal GPU backend already, more later. Every
op the same graph produces is fair game for fusion, memory planning, or
a completely different backend, without touching the code that wrote it.

This is a from-scratch systems project, built and measured in the open.
Every performance claim below was actually run, not assumed — where a
technique lost to something simpler, that's written down too. The
[**devlog**](DEVLOG.md) has the full, unfiltered record: every
benchmark, every bug, every dead end, in the order it happened.

## Status

| Phase | What | State |
|---|---|---|
| 1 — Foundation | Tensor/autograd core, CPU backend, `nn`/`optim` | ✅ done |
| 2 — KIR | Tracing, fusion, memory pooling, source-transform autograd | ✅ done |
| 3 — GPU backend | Real Metal compute (tiled kernel + MPS, full op + Conv2d coverage) | ✅ done |
| 4 — Distributed | `DeviceMesh`/`DTensor`, distributed gradients, concurrent dispatch, int8 quantization, serialization | ✅ done |
| — Core ops | `reshape`/`transpose`/`slice`/`cat`, general broadcasting, `sqrt`/`reciprocal`/`div`, `Adam` | ✅ done |
| — Activations/losses | `tanh`/`sigmoid`/`gelu`/`leaky_relu`, `sum`/`mean`/`max(dim)`, `softmax`, `cross_entropy` | ✅ done |
| — Normalization | `LayerNorm`, `BatchNorm1d`, `Module.train()`/`eval()` | ✅ done |
| — Real benchmark | MNIST end to end: **97.58% test accuracy**, 31.7s | ✅ done |
| — `detach()` | Cut a value from the autograd graph — a real view, O(1) | ✅ done |
| — Pooling/regularization | `AvgPool2d`, `MaxPool2d` (real argmax gradient), `Dropout` | ✅ done |
| — Batched matmul | Any rank ≥ 2, NumPy-style batch-dim broadcasting | ✅ done |
| — MultiHeadAttention | Real scaled dot-product attention, cross-attention, masking | ✅ done |
| — Indexing/`Embedding` | `index_select` (repeat-accumulating backward), token-lookup `nn.Embedding` | ✅ done |
| — Comparisons/`where` | `gt`/`lt`/`eq` (deliberately non-differentiable), boolean-style `where` | ✅ done |
| — `AdamW` | Decoupled weight decay (Loshchilov & Hutter, 2019), not L2 | ✅ done |
| — Packaging/CI | `pip install`-able (`scikit-build-core`), GitHub Actions on Apple Silicon | ✅ done |
| — Training utilities | `clip_grad_norm_`, `StepLR`, `CosineAnnealingLR` | ✅ done |
| — `Dataset`/`DataLoader` | Map-style, per-epoch reshuffle, replaces every hand-rolled batching loop | ✅ done |
| — `kir.grad` batched matmul | Closed the 2D-only gap — `MultiHeadAttention` now differentiable via `kir.grad`, not just eager | ✅ done |
| — `TransformerBlock` | Pre-LN attention + FFN, both residuals proven structurally, causal masking verified | ✅ done |
| — Memory leak fix | Permanent `shared_ptr` cycle in exp/sqrt/reciprocal/tanh/sigmoid, found training a real transformer | ✅ fixed |
| — tinyshakespeare capstone | Real transformer on real text, 2,000 steps, perplexity 10.5→8.28 | ✅ done |
| — Scaled-up run (7.6x) | 2.1M params, 2,000 steps, perplexity → 6.27, clean under real sustained load | ✅ done |
| — `no_grad()` | Reentrant context manager, skips graph-building for inference/validation | ✅ done |
| — `kir.grad` conv2d | Closed the last of the two 2D/gap-scope holes — matmul was the other | ✅ done |
| — SGD momentum | Heavy-ball + Nesterov, `buf` initialized to `grad` on the first step | ✅ done |
| — `BatchNorm2d` | Per-channel over N,H,W jointly — transpose+reshape onto `BatchNorm1d`, zero new kernels | ✅ done |
| — Gradient checkpointing | `kansai.checkpoint()` — recompute instead of store, multiple x less peak memory on a deep stack | ✅ done |
| — `LinearWarmup` | Composes with any scheduler by construction order — the standard warmup+decay pairing | ✅ done |
| — `Conv1d` | Sequence data (audio, time-series) — reuses `Conv2d`'s kernel entirely via reshape | ✅ done |

**Verified, not asserted:** a two-layer MLP trains XOR to convergence
through three independent execution paths (eager autograd, a jit'd KIR
graph, and source-transform `kir.grad`, cross-checked to agree exactly);
elementwise fusion measures 2.2x; the memory pool measures 1.25x at
realistic scale (and an honest *loss* at tiny scale — see the devlog);
Metal's `MPSMatrixMultiplication` path, paired with zero-copy buffers
(`newBufferWithBytesNoCopy` over page-aligned tensor storage), beats
Accelerate from 2048² up (1.5-1.6x at 4096²) *and* at the thin-batch
shape a real Linear layer's forward pass actually produces (1.3x at
128×4096 @ 4096×4096 — a shape that lost 3x to Accelerate before the
zero-copy work); every op in the KIR vocabulary, `Conv2d` included, now
has a working Metal dispatch path (`kir.run_metal` has zero CPU
fallback left), verified against CPU on a full forward+loss graph and,
for `Conv2d` specifically, against an independent nested-loop reference
implementation with error on the order of 1e-6; `Conv2d`'s backward is
checked against numerical differentiation; a batch sharded across
`DeviceMesh(["cpu", "metal"])` runs each shard through that device's
real interpreter and gathers back to exactly the unsharded result;
`dtensor_grad`'s auto-inserted collectives (all-reduce for a
`Replicate()`'d gradient, all-gather for a `Shard()`'d one) reproduce
the exact full-batch `kir.grad()` gradient from two devices each seeing
half the batch, for both a `sum()`- and a `mean()`-reduced loss — the
data-parallel-training correctness bar, not a smaller stand-in for it;
post-training int8 weight quantization (`QTensor`/`QLinear`) measures
an exact 4.00x memory reduction (1 byte/element vs 4), with round-trip
error bounded by the known quantization step and a quantized XOR model
still classifying every input correctly — a real, honest memory-only
win (no int8 GEMM kernel exists yet, so there's no FLOPs claim attached);
`dtensor_run`/`dtensor_grad` now dispatch every mesh device on its own
thread with the GIL released around each device's actual compute
(`nb::call_guard<nb::gil_scoped_release>()` on every hot binding), and
a 4096×4096 matmul split across `["cpu", "metal"]` measures a real,
repeatable 1.52-1.53x speedup over running the same two device-legs one
after another — genuine overlap, not just extra threads; `save`/`load`
round-trips a trained model's parameters bit-for-bit (not approximately)
through a pickle-free file format, for both a `Sequential` and a
`Conv2d` model, with four distinct failure modes (shape mismatch,
missing/unexpected parameters, a corrupted file) each rejected with a
specific error rather than a silent wrong load; the new `reshape`/
`transpose`/`slice`/`cat` ops (full autograd — eager and `kir.grad` —
plus all four KIR interpreters) are checked against central-difference
gradients and, for `transpose`, at a 3D permutation across non-adjacent
axes rather than only the easy case; `distributed.py`'s own split/concat
are three lines apiece on top of these now, replacing roughly sixty
lines of manual Python stride bookkeeping that used to exist specifically
because Kansai had no slice/cat kernel — a real internal caller adopting
the new ops on day one, not a capability sitting unused; `add`/`sub`/`mul`
now accept any NumPy-compatible broadcast shape (previously one hardcoded
pattern for `add`, none at all for `sub`/`mul`), with the pre-existing
bias-broadcast fast paths (`fused_bias_relu`, Metal's batched dispatch)
confirmed to still take the exact same route they always did, not just
compute the same numbers — a real routing bug this work surfaced and
fixed (`run_metal` was one shape-equality check away from silently
sending a general broadcast through a Metal kernel built for a different
shape entirely), caught by the test suite failing on first run; `Adam`
(the standard first/second-moment optimizer, implemented entirely from
`sqrt`/`div`/broadcasting over existing Tensor ops rather than a
dedicated kernel) matches a from-scratch independent Python
re-implementation to float32 precision across five steps of a known
gradient sequence, and trains the same XOR model `test_xor.py` trains
with SGD to the same near-zero loss; `softmax` stays finite and still
sums to 1 on logits as large as `[1000, 1001, 1002]` (a naive `exp()`
on those would overflow float32 outright), `cross_entropy` matches an
independent from-scratch log-sum-exp implementation (not Kansai's own
`softmax().log()` composed a second time), and a 3-class classifier
trained end to end with `Linear → ReLU → Linear → cross_entropy` +
`Adam` reaches 100% accuracy — the first classification task (XOR is
regression-shaped) this project has ever trained; `LayerNorm` is a pure
composition of already-existing ops (traces, fuses, and runs on Metal
like any other), `BatchNorm1d` measurably normalizes by its own running
statistics in eval mode rather than a fresh batch's (checked by feeding
a wildly out-of-distribution example after training and confirming the
output isn't the near-zero a fresh batch's own trivial statistics would
give), and the same 3-class classifier reaches 100% accuracy again with
either layer dropped in; on real MNIST (60,000 training / 10,000
held-out test images, not a synthetic proxy), `Linear(784→256) →
BatchNorm1d → ReLU → Linear(256→64) → ReLU → Linear(64→10)` trained with
`Adam` reaches **97.58% test accuracy in 31.7s** (15 epochs, measured
one real run, zero hyperparameter search) — see
[`examples/mnist/`](examples/mnist/) to reproduce it; `MaxPool2d`'s
gradient is checked to land on exactly the winning position in each
window (4 of 16 entries nonzero for a 4×4 input pooled 2×2, not spread
across it — reusing the existing, deliberately non-differentiable
`max(dim)` here would have silently zeroed this layer's entire
gradient, a real trap caught before it shipped), and a real
`Conv2d → ReLU → MaxPool2d → Linear → cross_entropy` CNN reaches 100%
accuracy on a synthetic image task, proving that gradient composes
correctly through a real `Conv2d` backward, not just in isolation;
`matmul` now handles any rank ≥ 2 with NumPy-style batch-dim
broadcasting — `(batch, heads, seq, d_k) @ (batch, heads, d_k, seq)`,
the shape multi-head attention needs, checked against an independent
nested-loop reference, backward included. A real bug surfaced and
fixed along the way: `run_metal`'s own matmul dispatch called Metal's
2D-only GEMM kernel unconditionally, which would have crashed on
anything batched — caught by the test suite itself, not a user, fixed
with the same CPU-fallback pattern other Metal-less ops already use;
`MultiHeadAttention` (real scaled dot-product attention with genuinely
batched `Q@K^T`/`weights@V`, general cross-attention, and an additive
mask for causal/decoder-style use) matches a from-scratch single-head
Python reference exactly, a causal mask is checked to zero every
future-position weight while each row still sums to 1, and a small
attention-based sequence classifier reaches 100% accuracy on a
synthetic "find the marker at a random position" task. `index_select`
(indices are a plain `list[int]`, not a `Tensor` — Kansai has no
integer dtype, and an index isn't a differentiable quantity) is
checked to ACCUMULATE its gradient correctly when an index repeats
(selecting a row twice gives it exactly double the gradient, verified
component-by-component, not just "some gradient flows"), through both
eager backward and the full traced `kir.grad` path, including run
through `run_metal`; `nn.Embedding`, built as pure `index_select` +
`reshape` with zero new kernels, is checked the same way for repeated
*token ids* within a batch, and a small "does this sequence contain
the marker token" classifier (`Embedding` → mean-pool → `Linear` →
`cross_entropy`) reaches 100% test accuracy. `gt`/`lt`/`eq` close the
gap `MultiHeadAttention`'s own additive-only mask used to have — real
comparison ops, deliberately non-differentiable (confirmed:
`requires_grad` never turns on, even when the input does) — and
`where(cond, a, b)` is built as pure `sub`/`mul`/`add` composition
(`b + cond*(a-b)`, no new kernel or KIR node at all), checked to route
gradient into `a`/`b` at exactly the positions each branch was taken
and never into `cond`, through both eager and the full traced
`kir.grad` path; practically, training `where(x>0, w_pos·x, w_neg·x)`
with `Adam` on a genuinely piecewise-linear target recovers both true
slopes to four decimal places. `AdamW` (decoupled weight decay,
subclassing `Adam` rather than duplicating its moment bookkeeping) is
checked to match plain `Adam` exactly at `weight_decay=0`, and its
decay term isolated with an honest zero-gradient Tensor (so `Adam`'s
own update contributes nothing) matches the closed-form `p₀·(1 −
lr·weight_decay)^steps` after 10 steps. `clip_grad_norm_` (one global
L2 norm across every parameter's gradient combined, not clipped per-
parameter) is checked to rescale a gradient to exactly `max_norm`
while preserving its direction, leave an in-bounds gradient untouched,
and combine correctly across multiple parameters at once;
`CosineAnnealingLR` matches the closed-form SGDR schedule at every
point including both endpoints; and, practically, `Adam` +
`clip_grad_norm_` + `CosineAnnealingLR` trained together — with
clipping CONFIRMED to actually engage on the first step, not a silent
no-op — reach loss `~0`. `DataLoader` (map-style, per-epoch reshuffle
via its own `random.Random` instance, independent of global `random`
state — confirmed by perturbing global state in between and getting
the identical order anyway) is checked to match manual slicing exactly
when unshuffled, visit every example exactly once per epoch when
shuffled, and reshuffle rather than repeat across successive epochs;
practically, retraining `test_embedding.py`'s own marker-token
classifier through a real `Dataset`/`DataLoader` instead of its
original hand-rolled batching reaches the identical 100% test accuracy.
`kir.grad`'s own matmul vjp was 2D-only — a real, previously documented
gap that blocked differentiating a *traced* `MultiHeadAttention` forward
(every matmul inside it is genuinely batched), even though eager
`.backward()` was always fine. Closed with two new `batched_matmul_nt`/
`_tn` ops mirroring `Tensor::matmul`'s own eager batched backward, with
zero change to the original 2D-only path (checked directly as a
regression) — and, the actual payoff: tracing a full
`MultiHeadAttention.forward` and differentiating it via `kir.grad` now
matches eager `.backward()` to exact zero difference, including through
`run_metal`. `TransformerBlock` (pre-LN attention + FFN, both with
residual connections) is checked structurally, not just by "training
seems to work": zeroing exactly the two sublayers' own output
projections makes the block reduce to an EXACT identity (output equals
input to exact equality, gradient exactly all-ones), proving both
residuals are wired correctly; a causal mask is confirmed to make each
position's output genuinely independent of every later position's
input (checked to exact equality, not approximately); and, practically,
stacking two blocks into a tiny causal model trained to predict the
first token's identity at every later position — a task no per-position
network can solve without attention — reaches 100% accuracy.

None of the tests above run long enough to catch what training a real
transformer on real text for real did: a genuine, permanent C++-side
memory leak. `exp()`/`sqrt()`/`reciprocal()`/`tanh()`/`sigmoid()` each
captured their own OUTPUT tensor inside their own backward closure — a
textbook `shared_ptr` cycle (`TensorData → GradNode → captured Tensor
→ same TensorData`) invisible to Python's garbage collector, since
it's a pure C++ cycle, not a Python one. `softmax` composes `exp()`
internally and `Adam`/`AdamW` call `sqrt()` every step, so this had
been silently leaking during every training loop in this project's
history — just too slowly, in runs too short, for anything to notice
until a multi-thousand-step run actually got killed by it. Fixed by
capturing a disconnected value snapshot instead of the tensor itself;
verified with a dedicated regression test (4,000 iterations per op,
`ru_maxrss` growth asserted bounded) and, practically, the training
run below — which reliably crashed before this fix — now runs 2,000
steps to completion without incident. See
[`DEVLOG.md`](DEVLOG.md#a-real-permanent-memory-leak-found-by-actually-training-something)
for the full bisection.

**[`examples/tinyshakespeare/`](examples/tinyshakespeare/)** is the
real capstone: a small decoder-only, character-level transformer
(277,601 parameters — `Embedding` + learned positions → 3
`TransformerBlock`s → `LayerNorm` → `Linear`) trained with `AdamW` +
`clip_grad_norm_` + `CosineAnnealingLR` on the actual ~1.1MB "tiny
Shakespeare" corpus, not MNIST, not a synthetic task. One real
run: 2,000 steps in 562s on Apple Silicon CPU, training loss 3.12 →
2.01, held-out validation perplexity 10.5 → 8.28 (random-uniform
baseline: 65). Sampled generations are legibly English-adjacent — real
word shapes, plausible capitalization, character names in roughly the
right places — without being coherent, exactly what a 277K-parameter
character model trained for a couple thousand steps should honestly
produce.

A second run at 7.6x the scale (2,110,913 parameters — wider, deeper,
longer context, bigger batches — via
[`train_shakespeare_large.py`](examples/tinyshakespeare/train_shakespeare_large.py),
kept separate so the numbers above stay fast to reproduce) finished
clean: 2,000 steps in 4,015s on Apple Silicon CPU, held-out perplexity
falling to **6.27** — meaningfully better than the smaller run's 8.28,
the direction more capacity should move it, actually observed rather
than assumed. No crash and no memory growth across the full 67-minute
run, the leak fix above holding under real sustained load, not just a
short probe. Sample generations take a visible step up too —
recognizable Shakespeare character-name fragments emerging on their
own ("WARWICK:", "Second Messer:", never told to the model explicitly)
— while still, honestly, not coherent prose.

Before chasing GPU dispatch as the next lever, it was actually
measured: eager CPU vs. `kir.trace()` + `run_metal()` on the identical
`TransformerBlock` forward, batch sizes 48 through 512, showed no real
speedup at any of them (0.96x–1.00x, outputs matching to exact
equality) — Apple Silicon's Accelerate framework (AMX + NEON) is
already highly competitive with the GPU at these sizes, so eager mode
stays CPU-only for now rather than adding GPU-dispatch complexity a
real workload hasn't actually needed yet. `no_grad()` (a reentrant
context manager suppressing graph-building — `torch.no_grad()`'s
equivalent, previously missing entirely) closes a real gap the
capstone above ran into directly: `estimate_val_loss` and `generate`
both built full, immediately-discarded backward graphs on every call
before this existed. Checked to correctly restore prior state after
the block exits (including on exception, and correctly when nested)
while never changing the actual computed value. `conv2d` was the other
half of `kir.grad`'s scope gap alongside matmul — no vjp rule at all,
so tracing a conv2d forward and differentiating it via `kir.grad` was a
hard error, eager `.backward()` always fine. Closed with three
independent ops (the graph has no multi-output node, the same reason
matmul's own vjp is two ops rather than one), each checked directly
against `Tensor::conv2d`'s own eager backward to exact equality, and
the full traced path — including through `run_metal`, which has no
dedicated conv2d-backward kernel and falls back to CPU the same way
reshape/transpose already do — matching already central-difference-
verified eager gradients exactly. `SGD` gained momentum (classic
heavy-ball plus its Nesterov variant, `buf` initialized to `grad`
itself on the first step rather than zero, PyTorch's own convention) —
a real, previously-missing piece of the optimizer vocabulary, since
plain SGD alone needs an impractically small learning rate to converge
at reasonable speed on anything but a toy problem. Both variants
checked against a from-scratch pure-Python reference of the
recurrence, and, practically, on a genuinely ill-conditioned loss
surface (100x steeper along one axis than the other — plain gradient
descent is textbook-known to oscillate there) momentum reaches under a
third of plain SGD's loss at the same step budget, Nesterov beating
plain momentum again on top of that. `BatchNorm2d` closes the gap
`BatchNorm1d`'s own docstring had flagged since it shipped —
per-channel normalization over N, H, *and* W jointly for `Conv2d`'s
`(N,C,H,W)` activations, via a transpose+reshape onto `BatchNorm1d`'s
own unmodified `forward()` (zero new kernels). Checked against a
manual per-channel mean/variance and, for backward, a deliberately
non-degenerate position-weighted loss — `.sum()` or sum-of-squares of
any batch-normalized output is mathematically CONSTANT with respect to
the input (a real trap: both would make a genuinely broken
implementation and a correct one produce identical, uninformatively
near-zero results) — and, practically, a real `Conv2d → BatchNorm2d →
ReLU → Linear` network reaches 100% accuracy on a synthetic
classification task. `kansai.checkpoint(fn, *inputs)` (activation
checkpointing — recompute instead of store) closes the memory-scaling
story the leak fix opened: Kansai's eager autograd holds every
intermediate activation live per layer until `backward()` consumes it,
so a deep stack means one full activation set per layer, all
simultaneously resident. Checkpointing collapses a wrapped segment to
O(1) — keep only its input and output, recompute the rest from scratch
when backward actually needs it. Built on three new primitives (the
general explicit-seed `backward(grad_output)`, not just scalar-only
implicit-ones; `Tensor._set_requires_grad`; `core.attach_custom_grad`,
the first way Python-level code can attach a `GradNode` the way every
C++ op already does internally). Checked to match a non-checkpointed
call exactly (forward and backward, through both a single checkpoint
and a chain of 8), correctly omit gradient for a non-grad-requiring
input, stay completely flat over 3,000 iterations with `gc` disabled
(the same bar the leak investigation itself established), and —
the actual point — measurably use substantially less peak memory
checkpointing a real 40-layer stack than running it without.
`LinearWarmup` closes the other missing standard piece — ramps `lr`
linearly to its own target over `warmup_steps`, then hands off to any
wrapped scheduler (`CosineAnnealingLR`, say), landing EXACTLY on the
target at the boundary, not approximately. `Conv1d` (audio, time-series,
genomic sequences) reuses `Conv2d`'s entire kernel/backward/KIR support
via reshape (1D convolution is 2D with the height axis pinned at 1) —
the one real wrinkle being that binary ops auto-coerce a fresh constant
against a traced value but `cat` (used for manual length-axis padding)
can't, the first op in this codebase needing an explicit eager/traced
branch. Checked against a nested-loop reference, central differences,
the full KIR path (both the padding and no-padding branches, including
`run_metal`), and a practical spike-detection classifier reaching 100%
accuracy.

## Why

- **Compilation isn't optional.** Kansai is always building a graph
  KIR's optimizer can see — not an eager-by-default engine with a
  compiler bolted on as an afterthought.
- **One IR, many backends.** Adding hardware support means writing a
  new codegen pass against KIR, not touching the frontend or autograd.
  Metal already proves this: the same fused graph runs on Accelerate or
  on the GPU by swapping one interpreter.
- **Autograd is a source transformation.** `kir.grad(graph, wrt)`
  returns a *new graph* computing derivatives — not a tape replayed at
  runtime — so the backward pass is exactly as optimizable as the
  forward one.
- **Shapes and layout are first-class**, not something the runtime
  discovers by crashing.

## Quickstart

Requires CMake ≥ 3.18, a C++17 compiler, Python ≥ 3.9, and `nanobind`.
Apple Silicon is the only tested target (Accelerate for CPU, Metal +
MetalPerformanceShaders for GPU).

Either `pip install` it directly (via `pyproject.toml`'s
`scikit-build-core` + `nanobind` build backend — no manual CMake
invocation needed, and this is what [CI](.github/workflows/tests.yml)
itself builds and imports on every push to check the packaging stays
real, not just configured):

```bash
python3 -m pip install .
```

or build in place from the source tree, with no install step at all —
the compiled extension lands directly in `python/kansai/`:

```bash
python3 -m pip install --user nanobind
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Then, from the source tree (skip the `sys.path` line if you `pip
install`ed instead):

```python
import sys; sys.path.insert(0, "python")
import kansai
from kansai import nn, optim

X = kansai.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = kansai.tensor([[0], [1], [1], [0]])

model = nn.Sequential(nn.Linear(2, 8, seed=3), nn.ReLU(), nn.Linear(8, 1, seed=4))
opt = optim.SGD(model.parameters(), lr=0.1)

for step in range(500):
    pred = model(X)
    loss = pred.sub(Y).mul(pred.sub(Y)).mean()
    model.zero_grad()
    loss.backward()
    opt.step()

print(loss.tolist())  # -> ~0.0
```

Run the whole test suite (eight files, everything below is exercised):

```bash
for f in tests/test_*.py; do python3 "$f"; done
```

## Architecture

```
Python (Tensor, nn.Module, optim)
            │
            ▼
   KIR (trace → optimize → run)
   fusion · memory planning · autograd
            │
     ┌──────┴──────┐
     ▼             ▼
  CPU backend   Metal backend         DeviceMesh / DTensor
  (Accelerate)  (tiled kernel, MPS,   -- shards a graph's input across
                 batched dispatch)    both backends, dispatches each
                                      shard to its real interpreter
```

- `core/` — `Tensor`, `Storage`, the pooled allocator, and the
  tape-based eager autograd engine
- `backend/cpu/` — raw kernels (Accelerate-backed `matmul` on macOS,
  plus shape-generic `transpose`/`slice`/`scatter_range` for
  `reshape`/`transpose`/`slice`/`cat`)
- `backend/metal/` — real Metal compute: a hand-tiled kernel,
  `MPSMatrixMultiplication`, and batched elementwise dispatch, all
  zero-copy over page-aligned tensor storage
- `python/kansai/kir.py` — the IR: tracer, optimizer passes, and four
  interpreters (`run`, `run_fused`, `run_planned`, `run_metal`)
- `python/kansai/distributed.py` — `DeviceMesh`, `DTensor`,
  `Shard`/`Replicate` placements, `dtensor_run` (forward), and
  `dtensor_grad` (backward, with auto-inserted collectives) — both
  dispatch every mesh device on its own real, concurrently-overlapping
  thread
- `python/kansai/quantize.py` — post-training int8 weight quantization
  (`QTensor`, `QLinear`), standalone and inference-only
- `python/kansai/serialize.py` — `save`/`load` a `Module`'s parameters
  to/from disk, a pickle-free length-prefixed-JSON-header + flat-blob
  format
- `python/kansai/{nn,optim}.py` — `Linear`, `Conv1d`, `Conv2d`, `ReLU`, `Tanh`,
  `Sigmoid`, `GELU`, `LeakyReLU`, `Softmax`, `LayerNorm`, `BatchNorm1d`, `BatchNorm2d`,
  `AvgPool2d`, `MaxPool2d`, `Dropout`, `MultiHeadAttention`,
  `TransformerBlock`, `Embedding`,
  `Sequential`, `SGD`, `Adam`, `AdamW`, `clip_grad_norm_`, `StepLR`,
  `CosineAnnealingLR`, `LinearWarmup`
- `python/kansai/data.py` — `Dataset`, `DataLoader` (map-style,
  per-epoch reshuffle, no worker-process plumbing)
- `tests/` — every claim above, checked: numerical gradient checks,
  cross-checks between independent implementations, and benchmarks that
  assert real speedups rather than just printing numbers
- `examples/mnist/` — the real end-to-end MNIST benchmark above,
  kept separate from `tests/` (needs network access, takes tens of
  seconds) rather than part of the fast/deterministic suite
- `examples/tinyshakespeare/` — the transformer capstone above, same
  reason kept separate from `tests/` (network access, minutes not
  seconds)

## Documentation

- **[DEVLOG.md](DEVLOG.md)** — the complete engineering record: design
  rationale, every benchmark with its actual numbers, bugs found and
  how, and the honest limitations of everything listed above.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
