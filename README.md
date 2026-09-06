<p align="center">
  <img src="KansaiML.png" width="96" alt="Kansai">
</p>

<h1 align="center">Kansai</h1>

<p align="center">
  A deep learning framework where compilation and hardware parity are<br>
  the foundation, not features bolted on after the fact.
</p>

<p align="center">
  <a href="https://github.com/V-B0/KansaiML/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/V-B0/KansaiML/actions/workflows/tests.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-active%20development-orange.svg">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey.svg">
</p>

---

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

Run the whole test suite (fast, deterministic, no network access —
everything in the Status table below is exercised by it):

```bash
for f in tests/test_*.py; do python3 "$f"; done
```

## Status

| Phase | What | State |
|---|---|---|
| 1 — Foundation | Tensor/autograd core, CPU backend, `nn`/`optim` | ✅ done |
| 2 — KIR | Tracing, fusion, memory pooling, source-transform autograd | ✅ done |
| 3 — GPU backend | Real Metal compute (tiled kernel + MPS, full op coverage) | ✅ done |
| 4 — Distributed | `DeviceMesh`/`DTensor`, distributed gradients, concurrent dispatch, int8 quantization, serialization | ✅ done |
| — Core ops | `reshape`/`transpose`/`slice`/`cat`, general broadcasting, `sqrt`/`reciprocal`/`div`, `Adam` | ✅ done |
| — Activations/losses | `tanh`/`sigmoid`/`gelu`/`leaky_relu`, `sum`/`mean`/`max(dim)`, `softmax`, `cross_entropy` | ✅ done |
| — Normalization | `LayerNorm`, `BatchNorm1d`/`BatchNorm2d`, `Module.train()`/`eval()` | ✅ done |
| — Convolution | `Conv1d`, `Conv2d`, `AvgPool2d`, `MaxPool2d` (real argmax gradient), `Dropout` | ✅ done |
| — Attention | Batched `matmul` (any rank ≥ 2), `MultiHeadAttention`, `TransformerBlock` | ✅ done |
| — Sequence modeling | `index_select`, `Embedding`, comparison ops (`gt`/`lt`/`eq`), `where` | ✅ done |
| — Optimizers/schedules | `SGD` (+ momentum/Nesterov), `AdamW`, `clip_grad_norm_`, `StepLR`/`CosineAnnealingLR`/`LinearWarmup` | ✅ done |
| — Training ergonomics | `Dataset`/`DataLoader`, `no_grad()`, gradient checkpointing | ✅ done |
| — `kir.grad` full parity | Both scope gaps closed (batched matmul, `conv2d`) — matches eager exactly | ✅ done |
| — Packaging/CI | `pip install`-able (`scikit-build-core`), GitHub Actions on Apple Silicon | ✅ done |
| — Real benchmark: MNIST | End to end, **97.58% test accuracy**, 31.7s | ✅ done |
| — Real benchmark: tinyshakespeare | 277K params, perplexity 8.28; 2.1M params, perplexity 6.27 | ✅ done |
| — Memory leak, found and fixed | Permanent `shared_ptr` cycle in five core ops, found training a real transformer | ✅ fixed |

## Proven, not just claimed

A representative sample — the full account for every line here,
including the numbers, the bugs, and the dead ends, is in
[DEVLOG.md](DEVLOG.md).

- **Cross-validated by construction.** The same model trained through
  three independent execution paths — eager autograd, a jit'd KIR
  graph, and source-transform `kir.grad` — agrees to the same result.
  This isn't a one-off check; every op added since gets the same
  three-way (or more) treatment: forward against an independent
  reference, backward against central differences, and the full traced
  path, before it's considered done.
- **A real bug, found the hard way.** Training a transformer for
  thousands of steps (not the few hundred every earlier benchmark used)
  surfaced a permanent C++ reference-cycle leak in five core ops
  (`exp`/`sqrt`/`reciprocal`/`tanh`/`sigmoid`) that had been silently
  leaking through every training loop in this project's history —
  `softmax` uses `exp` internally, and `Adam`/`AdamW` call `sqrt` every
  step. Bisected down to the exact five ops, root-caused (a
  self-referential `GradNode` cycle invisible to Python's own garbage
  collector), fixed, and covered by a dedicated regression test.
  [Full account →](DEVLOG.md#a-real-permanent-memory-leak-found-by-actually-training-something)
- **Real end-to-end training, not synthetic proxies.** MNIST (60,000
  training images, not a synthetic stand-in): **97.58% test accuracy in
  31.7s**, 15 epochs, one real run, zero hyperparameter search — see
  [`examples/mnist/`](examples/mnist/). A small transformer trained on
  the actual "tiny Shakespeare" corpus, twice: 277,601 parameters
  (perplexity 10.5 → 8.28) and a 7.6x-larger 2,110,913-parameter run
  (perplexity → 6.27, confirming scale actually helps, and that the
  leak fix holds under real sustained load) — see
  [`examples/tinyshakespeare/`](examples/tinyshakespeare/).
- **Honest benchmarking, including the negative result.** Before
  investing in GPU dispatch for eager mode, it was actually measured:
  eager CPU vs. Metal on the same `TransformerBlock`, batch sizes 48
  through 512, showed no real speedup at any of them (0.96x–1.00x) —
  Apple Silicon's Accelerate framework is already highly competitive at
  these sizes, so that work wasn't done just because a GPU was
  available. Where Metal *does* win — `MPSMatrixMultiplication` with
  zero-copy buffers on real matmul shapes — it's 1.3–1.6x over
  Accelerate, also measured, not assumed.
- **Distributed correctness verified against ground truth, not just
  "runs without crashing."** `dtensor_grad`'s auto-inserted collectives
  (all-reduce, all-gather) reproduce the exact full-batch gradient from
  two devices each seeing half the batch. A matmul split across
  `["cpu", "metal"]`, each device dispatched on its own real thread
  with the GIL released, measures a repeatable 1.52–1.53x speedup from
  genuine overlap — not just extra threads doing nothing in parallel.
- **`kir.grad` at full parity with eager autograd.** Both scope gaps
  that used to exist (batched `matmul`, `conv2d` — tracing either and
  differentiating via `kir.grad` used to be a hard error, even though
  eager `.backward()` was always fine) are closed, each checked to
  match eager backward to exact equality, including run back through
  the Metal interpreter.
- **Gradient checkpointing that actually saves memory**, not just
  correct in theory: checkpointing a real 40-layer stack measures
  multiple times less peak memory than running the identical stack
  without it, verified via `ru_maxrss` deltas, not asserted from the
  algorithm's description.
- **Bit-exact serialization and honest quantization.** `save`/`load`
  round-trips a trained model's parameters bit-for-bit through a
  pickle-free format, with four distinct failure modes each rejected
  with a specific error rather than a silent wrong load. Post-training
  int8 weight quantization measures an exact 4.00x memory reduction —
  reported as a memory-only win, since no int8 GEMM kernel exists yet
  to also claim a FLOPs speedup.

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
- `python/kansai/{nn,optim}.py` — `Linear`, `Conv1d`, `Conv2d`, `ReLU`,
  `Tanh`, `Sigmoid`, `GELU`, `LeakyReLU`, `Softmax`, `LayerNorm`,
  `BatchNorm1d`, `BatchNorm2d`, `AvgPool2d`, `MaxPool2d`, `Dropout`,
  `MultiHeadAttention`, `TransformerBlock`, `Embedding`, `Sequential`,
  `SGD`, `Adam`, `AdamW`, `clip_grad_norm_`, `StepLR`,
  `CosineAnnealingLR`, `LinearWarmup`
- `python/kansai/data.py` — `Dataset`, `DataLoader` (map-style,
  per-epoch reshuffle, no worker-process plumbing)
- `python/kansai/__init__.py` — `kansai.no_grad()` and
  `kansai.checkpoint()` (activation checkpointing)
- `tests/` — every claim above, checked: numerical gradient checks,
  cross-checks between independent implementations, and benchmarks that
  assert real speedups rather than just printing numbers
- `examples/mnist/` — the real end-to-end MNIST benchmark above,
  kept separate from `tests/` (needs network access, takes tens of
  seconds) rather than part of the fast/deterministic suite
- `examples/tinyshakespeare/` — the transformer capstones above, same
  reason kept separate from `tests/` (network access, minutes not
  seconds)

## Documentation

- **[DEVLOG.md](DEVLOG.md)** — the complete engineering record: design
  rationale, every benchmark with its actual numbers, bugs found and
  how, and the honest limitations of everything listed above.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
