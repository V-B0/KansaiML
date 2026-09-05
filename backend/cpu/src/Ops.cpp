#include "kansai/backend/cpu/Ops.hpp"
#include <algorithm>
#include <cstring>

#ifdef KANSAI_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

namespace kan::cpu {

void fill(float* x, float value, int64_t n) {
    std::fill(x, x + n, value);
}

void copy(const float* src, float* dst, int64_t n) {
    std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
}

void add(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

void sub(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
}

void mul(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
}

void add_bias_broadcast(const float* x, const float* bias, float* out, int64_t batch, int64_t features) {
    for (int64_t i = 0; i < batch; ++i)
        for (int64_t j = 0; j < features; ++j)
            out[i * features + j] = x[i * features + j] + bias[j];
}

void sum_over_batch(const float* grad_out, float* grad_bias, int64_t batch, int64_t features) {
    std::fill(grad_bias, grad_bias + features, 0.0f);
    for (int64_t i = 0; i < batch; ++i)
        for (int64_t j = 0; j < features; ++j)
            grad_bias[j] += grad_out[i * features + j];
}

void relu_fwd(const float* x, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

void relu_bwd(const float* x, const float* grad_out, float* grad_in, int64_t n) {
    for (int64_t i = 0; i < n; ++i) grad_in[i] = x[i] > 0.0f ? grad_out[i] : 0.0f;
}

void matmul(const float* a, const float* b, float* out, int64_t M, int64_t K, int64_t N) {
#ifdef KANSAI_USE_ACCELERATE
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                1.0f, a, static_cast<int>(K), b, static_cast<int>(N),
                0.0f, out, static_cast<int>(N));
#else
    std::fill(out, out + M * N, 0.0f);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            float av = a[i * K + k];
            const float* brow = b + k * N;
            float* orow = out + i * N;
            for (int64_t j = 0; j < N; ++j) orow[j] += av * brow[j];
        }
    }
#endif
}

void matmul_nt(const float* a, const float* b, float* out, int64_t rows, int64_t reduce, int64_t cols) {
#ifdef KANSAI_USE_ACCELERATE
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(rows), static_cast<int>(cols), static_cast<int>(reduce),
                1.0f, a, static_cast<int>(reduce), b, static_cast<int>(reduce),
                0.0f, out, static_cast<int>(cols));
#else
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            float acc = 0.0f;
            for (int64_t r = 0; r < reduce; ++r) acc += a[i * reduce + r] * b[j * reduce + r];
            out[i * cols + j] = acc;
        }
    }
#endif
}

void matmul_tn(const float* a, const float* b, float* out, int64_t reduce, int64_t rows, int64_t cols) {
#ifdef KANSAI_USE_ACCELERATE
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                static_cast<int>(rows), static_cast<int>(cols), static_cast<int>(reduce),
                1.0f, a, static_cast<int>(rows), b, static_cast<int>(cols),
                0.0f, out, static_cast<int>(cols));
#else
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            float acc = 0.0f;
            for (int64_t r = 0; r < reduce; ++r) acc += a[r * rows + i] * b[r * cols + j];
            out[i * cols + j] = acc;
        }
    }
#endif
}

float reduce_sum(const float* x, int64_t n) {
    float acc = 0.0f;
    for (int64_t i = 0; i < n; ++i) acc += x[i];
    return acc;
}

void axpy_(float* out, const float* x, float alpha, int64_t n) {
    for (int64_t i = 0; i < n; ++i) out[i] += alpha * x[i];
}

} // namespace kan::cpu
