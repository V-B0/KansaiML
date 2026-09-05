#pragma once
#include <cstdint>
#include <vector>

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
// out. Hand-written: one GPU thread per output element, 16x16
// threadgroup-memory tiling (see the .mm file for how) -- see the
// project README for how this actually compares to Accelerate's CPU
// matmul (measured, not assumed): closer than the untiled version, but
// nowhere near competitive. Kept as the "written by hand" reference;
// matmul_mps below is the one that's actually fast.
void matmul(const float* a, const float* b, float* out, int64_t M, int64_t K, int64_t N);

// Same signature, Apple's own implementation (MPSMatrixMultiplication)
// instead of a hand-written kernel -- still genuinely "the Metal
// backend" (MPS runs as Metal compute dispatched through the same
// command-buffer machinery), just Apple's professionally-tuned GEMM
// instead of reinventing one by hand. See the README for how much
// closer to (or past) Accelerate this actually gets.
void matmul_mps(const float* a, const float* b, float* out, int64_t M, int64_t K, int64_t N);

// out[i] = relu(x[i] + bias[i % features]) -- one GPU kernel for the
// whole bias-add-then-relu chain, the Metal-side analogue of the CPU
// backend's fused_bias_relu.
void bias_relu(const float* x, const float* bias, float* out, int64_t batch, int64_t features);

// out[i] = x[i] + bias[i % features] -- the same broadcast add without
// the relu clamp, for a layer's final (unactivated) output.
void add_bias(const float* x, const float* bias, float* out, int64_t batch, int64_t features);

// One step of a batched elementwise chain (see run_elementwise_chain):
// which kernel to run, and its bias operand.
enum class ElemKernel { BiasRelu, AddBias };

struct ElemStep {
    ElemKernel kernel;
    const float* bias;
};

// Runs `steps` in sequence -- x0 -> steps[0] -> steps[1] -> ... -> out
// -- entirely within ONE command buffer and ONE waitUntilCompleted,
// each step's output staying resident on the GPU and feeding directly
// into the next step's input. Calling bias_relu()/add_bias() N times in
// a row instead pays N separate command-buffer round trips (encode,
// commit, block on waitUntilCompleted, copy the result back to host) --
// this pays for exactly one, no matter how many steps. Only the
// initial upload (x0, and each step's bias) and the final download
// (out) ever touch host memory.
void run_elementwise_chain(const float* x0, int64_t batch, int64_t features,
                            const std::vector<ElemStep>& steps, float* out);

} // namespace kan::metal
