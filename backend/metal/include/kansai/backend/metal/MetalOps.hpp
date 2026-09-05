#pragma once
#include <cstdint>

// A real Metal compute backend: MSL shaders compiled at runtime (via
// MTLDevice::newLibraryWithSource, a Metal.framework API -- this
// machine has only Command Line Tools, not full Xcode, so the offline
// `metal`/`metallib` compilers aren't available; runtime compilation
// doesn't need them), dispatched through MTLComputeCommandEncoder. This
// header is plain C++ so the rest of the codebase (nanobind bindings,
// core/) never needs to know Objective-C exists -- all of that lives
// in MetalOps.mm.
namespace kan::metal {

// False if no Metal device was found (e.g. running under a VM/CI image
// with no GPU) -- every other function throws if called while this is
// false, rather than dereferencing a null pipeline.
bool available();

// out (M,N) = a (M,K) @ b (K,N), row-major host buffers in, host buffer
// out. Naive: one GPU thread per output element, no shared-memory
// tiling -- see the module README for how this actually compares to
// Accelerate's CPU matmul (measured, not assumed).
void matmul(const float* a, const float* b, float* out, int64_t M, int64_t K, int64_t N);

// out[i] = relu(x[i] + bias[i % features]) -- one GPU kernel for the
// whole bias-add-then-relu chain, the Metal-side analogue of the CPU
// backend's fused_bias_relu.
void bias_relu(const float* x, const float* bias, float* out, int64_t batch, int64_t features);

// out[i] = x[i] + bias[i % features] -- the same broadcast add without
// the relu clamp, for a layer's final (unactivated) output.
void add_bias(const float* x, const float* bias, float* out, int64_t batch, int64_t features);

} // namespace kan::metal
