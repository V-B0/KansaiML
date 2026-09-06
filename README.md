<p align="center">
  <img src="KansaiML.png" width="96" alt="Kansai">
</p>

<h1 align="center">Kansai</h1>

<p align="center">
  A deep learning framework built without the compromises PyTorch made<br>
  when it still had to ship in 2016.
</p>

<p align="center">
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
regression-shaped) this project has ever trained.

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

```bash
python3 -m pip install --user nanobind
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The compiled extension lands directly in `python/kansai/`, so there's
no install step. Then:

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
- `python/kansai/{nn,optim}.py` — `Linear`, `Conv2d`, `ReLU`, `Tanh`,
  `Sigmoid`, `GELU`, `LeakyReLU`, `Softmax`, `Sequential`, `SGD`, `Adam`
- `tests/` — every claim above, checked: numerical gradient checks,
  cross-checks between independent implementations, and benchmarks that
  assert real speedups rather than just printing numbers

## Documentation

- **[DEVLOG.md](DEVLOG.md)** — the complete engineering record: design
  rationale, every benchmark with its actual numbers, bugs found and
  how, and the honest limitations of everything listed above.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
