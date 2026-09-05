#pragma once
#include <cstdint>

// Hand-specialized fused kernels for the two elementwise chains
// kansai.kir's fusion pass currently recognizes. Each is a plain,
// straight-line, auto-vectorizable loop that does the whole chain in
// one pass -- no intermediate buffer between the ops it fuses.
//
// An earlier version of this file was a generic per-element bytecode
// interpreter (load/dup/swap/add/sub/mul/relu, walked once per output
// index). It was correct but measured ~7x *slower* than running the
// unfused ops separately: the switch-based dispatch can't be
// auto-vectorized, so it loses more to scalar branch/dispatch overhead
// than it saves in memory traffic, for ops this cheap (a single flop
// per element). Real fusion compilers (XLA, TVM, Triton) generate or
// select specialized code per fusion pattern for exactly this reason --
// they don't interpret a bytecode in the hot loop. This is that lesson
// applied: a small, explicit set of known patterns, each its own
// vectorizable function, rather than one general (and slower) VM.
namespace kan::cpu {

// out[i] = relu(x[i] + bias[i % features]) -- the bias-add+ReLU
// epilogue of every Linear layer's forward pass.
void fused_bias_relu(const float* x, const float* bias, float* out, int64_t batch, int64_t features);

// out[i] = (a[i] - b[i])^2 -- the diff*diff core of MSE loss.
void fused_sub_square(const float* a, const float* b, float* out, int64_t n);

} // namespace kan::cpu
