#pragma once
#include <cstdint>

// Raw float-buffer kernels. No shape/autograd knowledge — the core layer
// is responsible for allocating outputs and wiring gradients; this layer
// only moves numbers.
namespace kan::cpu {

void fill(float* x, float value, int64_t n);
void copy(const float* src, float* dst, int64_t n);

void add(const float* a, const float* b, float* out, int64_t n);
void sub(const float* a, const float* b, float* out, int64_t n);
void mul(const float* a, const float* b, float* out, int64_t n);

// x: (batch, features) + bias: (features,) -> out: (batch, features)
void add_bias_broadcast(const float* x, const float* bias, float* out, int64_t batch, int64_t features);
// reduces grad_out: (batch, features) over the batch dim -> grad_bias: (features,)
void sum_over_batch(const float* grad_out, float* grad_bias, int64_t batch, int64_t features);

void relu_fwd(const float* x, float* out, int64_t n);
void relu_bwd(const float* x, const float* grad_out, float* grad_in, int64_t n);

// a: (M,K) row-major, b: (K,N) row-major, out: (M,N) row-major
void matmul(const float* a, const float* b, float* out, int64_t M, int64_t K, int64_t N);

// out (rows x cols) = A (rows x reduce) @ B (cols x reduce)^T
// A and B are both read in their own native row-major layout -- B's
// transpose is never materialized into a separate buffer; BLAS's
// transpose flag (or the fallback loop's swapped indexing) reads it
// directly. This is the one matmul's backward pass actually needs for
// grad_a = grad_output @ b^T.
void matmul_nt(const float* a, const float* b, float* out, int64_t rows, int64_t reduce, int64_t cols);

// out (rows x cols) = A (reduce x rows)^T @ B (reduce x cols)
// Same idea, transposing A instead of B -- what grad_b = a^T @
// grad_output needs.
void matmul_tn(const float* a, const float* b, float* out, int64_t reduce, int64_t rows, int64_t cols);

float reduce_sum(const float* x, int64_t n);

// out += alpha * x, in place. Not autograd-tracked — used by optimizers.
void axpy_(float* out, const float* x, float alpha, int64_t n);

} // namespace kan::cpu
