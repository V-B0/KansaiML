#include "kansai/backend/cpu/Ops.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

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

namespace {
enum class BinOp { Add, Sub, Mul };

// Broadcast strides for one operand (`shape`, rank `rank`) against the
// output (`out_shape`, rank `out_rank >= rank`), right-aligned: output
// axis i (0-indexed from the left) corresponds to this operand's own
// axis i - (out_rank - rank), if that's >= 0 -- else the operand has no
// such axis at all (an implicit leading size-1). The returned stride
// for an axis this operand broadcasts along (no such axis, or a size-1
// axis stretching to something bigger) is 0, so every output position
// along that axis reads the same single source element -- the standard
// stride-0 broadcast trick, avoiding materializing a larger buffer.
void broadcast_strides(const int64_t* shape, int64_t rank, int64_t out_rank, int64_t* strides_out) {
    std::vector<int64_t> own_strides(static_cast<size_t>(rank));
    if (rank > 0) {
        own_strides[static_cast<size_t>(rank - 1)] = 1;
        for (int64_t i = rank - 2; i >= 0; --i)
            own_strides[static_cast<size_t>(i)] = own_strides[static_cast<size_t>(i + 1)] * shape[i + 1];
    }
    int64_t offset = out_rank - rank;
    for (int64_t i = 0; i < out_rank; ++i) {
        int64_t j = i - offset;
        // j < 0: this operand has no such axis at all -- broadcast (stride 0).
        // shape[j] == 1: broadcasting a size-1 axis (stride 0 either way -- if
        // out_shape[i] is also 1 the decomposed coordinate there is always 0
        // regardless of stride, and if out_shape[i] is bigger, 0 is exactly
        // the "always read the one element" stride this needs).
        strides_out[i] = (j < 0 || shape[j] == 1) ? 0 : own_strides[static_cast<size_t>(j)];
    }
}

void broadcast_binary(const float* a, const int64_t* a_shape, int64_t a_rank,
                       const float* b, const int64_t* b_shape, int64_t b_rank,
                       const int64_t* out_shape, int64_t out_rank, float* out, BinOp op) {
    std::vector<int64_t> a_strides(static_cast<size_t>(out_rank));
    std::vector<int64_t> b_strides(static_cast<size_t>(out_rank));
    broadcast_strides(a_shape, a_rank, out_rank, a_strides.data());
    broadcast_strides(b_shape, b_rank, out_rank, b_strides.data());

    std::vector<int64_t> out_strides(static_cast<size_t>(out_rank));
    if (out_rank > 0) {
        out_strides[static_cast<size_t>(out_rank - 1)] = 1;
        for (int64_t i = out_rank - 2; i >= 0; --i)
            out_strides[static_cast<size_t>(i)] = out_strides[static_cast<size_t>(i + 1)] * out_shape[i + 1];
    }

    int64_t n = 1;
    for (int64_t i = 0; i < out_rank; ++i) n *= out_shape[i];

    std::vector<int64_t> coord(static_cast<size_t>(out_rank));
    for (int64_t idx = 0; idx < n; ++idx) {
        int64_t rem = idx;
        for (int64_t d = 0; d < out_rank; ++d) {
            coord[d] = rem / out_strides[d];
            rem %= out_strides[d];
        }
        int64_t aidx = 0, bidx = 0;
        for (int64_t d = 0; d < out_rank; ++d) {
            aidx += coord[d] * a_strides[d];
            bidx += coord[d] * b_strides[d];
        }
        float av = a[aidx], bv = b[bidx];
        switch (op) {
            case BinOp::Add: out[idx] = av + bv; break;
            case BinOp::Sub: out[idx] = av - bv; break;
            case BinOp::Mul: out[idx] = av * bv; break;
        }
    }
}
} // namespace

void add_broadcast(const float* a, const int64_t* a_shape, int64_t a_rank,
                    const float* b, const int64_t* b_shape, int64_t b_rank,
                    const int64_t* out_shape, int64_t out_rank, float* out) {
    broadcast_binary(a, a_shape, a_rank, b, b_shape, b_rank, out_shape, out_rank, out, BinOp::Add);
}

void sub_broadcast(const float* a, const int64_t* a_shape, int64_t a_rank,
                    const float* b, const int64_t* b_shape, int64_t b_rank,
                    const int64_t* out_shape, int64_t out_rank, float* out) {
    broadcast_binary(a, a_shape, a_rank, b, b_shape, b_rank, out_shape, out_rank, out, BinOp::Sub);
}

void mul_broadcast(const float* a, const int64_t* a_shape, int64_t a_rank,
                    const float* b, const int64_t* b_shape, int64_t b_rank,
                    const int64_t* out_shape, int64_t out_rank, float* out) {
    broadcast_binary(a, a_shape, a_rank, b, b_shape, b_rank, out_shape, out_rank, out, BinOp::Mul);
}

void reduce_to_shape(const float* grad, const int64_t* grad_shape, int64_t grad_rank,
                      const int64_t* target_shape, int64_t target_rank, float* out) {
    std::vector<int64_t> grad_strides(static_cast<size_t>(grad_rank));
    if (grad_rank > 0) {
        grad_strides[static_cast<size_t>(grad_rank - 1)] = 1;
        for (int64_t i = grad_rank - 2; i >= 0; --i)
            grad_strides[static_cast<size_t>(i)] = grad_strides[static_cast<size_t>(i + 1)] * grad_shape[i + 1];
    }
    std::vector<int64_t> target_strides(static_cast<size_t>(target_rank));
    if (target_rank > 0) {
        target_strides[static_cast<size_t>(target_rank - 1)] = 1;
        for (int64_t i = target_rank - 2; i >= 0; --i)
            target_strides[static_cast<size_t>(i)] = target_strides[static_cast<size_t>(i + 1)] * target_shape[i + 1];
    }

    int64_t offset = grad_rank - target_rank;
    int64_t n = 1;
    for (int64_t i = 0; i < grad_rank; ++i) n *= grad_shape[i];
    int64_t target_numel = 1;
    for (int64_t i = 0; i < target_rank; ++i) target_numel *= target_shape[i];
    std::fill(out, out + target_numel, 0.0f);

    std::vector<int64_t> coord(static_cast<size_t>(grad_rank));
    for (int64_t idx = 0; idx < n; ++idx) {
        int64_t rem = idx;
        for (int64_t d = 0; d < grad_rank; ++d) {
            coord[d] = rem / grad_strides[d];
            rem %= grad_strides[d];
        }
        int64_t tidx = 0;
        for (int64_t d = 0; d < grad_rank; ++d) {
            int64_t td = d - offset;
            if (td < 0) continue;  // this axis doesn't exist in target -- summed away entirely
            int64_t c = (target_shape[td] == 1) ? 0 : coord[d];
            tidx += c * target_strides[td];
        }
        out[tidx] += grad[idx];
    }
}

void relu_fwd(const float* x, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

void relu_bwd(const float* x, const float* grad_out, float* grad_in, int64_t n) {
    for (int64_t i = 0; i < n; ++i) grad_in[i] = x[i] > 0.0f ? grad_out[i] : 0.0f;
}

void sqrt_fwd(const float* x, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) out[i] = std::sqrt(x[i]);
}

void sqrt_bwd(const float* out, const float* grad_out, float* grad_in, int64_t n) {
    for (int64_t i = 0; i < n; ++i) grad_in[i] = 0.5f * grad_out[i] / out[i];
}

void reciprocal_fwd(const float* x, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) out[i] = 1.0f / x[i];
}

void reciprocal_bwd(const float* out, const float* grad_out, float* grad_in, int64_t n) {
    for (int64_t i = 0; i < n; ++i) grad_in[i] = -grad_out[i] * out[i] * out[i];
}

void exp_fwd(const float* x, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) out[i] = std::exp(x[i]);
}

void log_fwd(const float* x, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) out[i] = std::log(x[i]);
}

void tanh_fwd(const float* x, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) out[i] = std::tanh(x[i]);
}

void tanh_bwd(const float* out, const float* grad_out, float* grad_in, int64_t n) {
    for (int64_t i = 0; i < n; ++i) grad_in[i] = grad_out[i] * (1.0f - out[i] * out[i]);
}

void sigmoid_fwd(const float* x, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) out[i] = 1.0f / (1.0f + std::exp(-x[i]));
}

void sigmoid_bwd(const float* out, const float* grad_out, float* grad_in, int64_t n) {
    for (int64_t i = 0; i < n; ++i) grad_in[i] = grad_out[i] * out[i] * (1.0f - out[i]);
}

namespace {
constexpr float kInvSqrt2 = 0.7071067811865476f;       // 1/sqrt(2)
constexpr float kInvSqrt2Pi = 0.3989422804014327f;      // 1/sqrt(2*pi)
} // namespace

void gelu_fwd(const float* x, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        float xi = x[i];
        float cdf = 0.5f * (1.0f + std::erf(xi * kInvSqrt2));
        out[i] = xi * cdf;
    }
}

void gelu_bwd(const float* x, const float* grad_out, float* grad_in, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        float xi = x[i];
        float cdf = 0.5f * (1.0f + std::erf(xi * kInvSqrt2));
        float pdf = kInvSqrt2Pi * std::exp(-0.5f * xi * xi);
        grad_in[i] = grad_out[i] * (cdf + xi * pdf);
    }
}

void leaky_relu_fwd(const float* x, float* out, int64_t n, float negative_slope) {
    for (int64_t i = 0; i < n; ++i) out[i] = x[i] > 0.0f ? x[i] : negative_slope * x[i];
}

void leaky_relu_bwd(const float* x, const float* grad_out, float* grad_in, int64_t n, float negative_slope) {
    for (int64_t i = 0; i < n; ++i) grad_in[i] = x[i] > 0.0f ? grad_out[i] : negative_slope * grad_out[i];
}

void max_along_dim(const float* x, const int64_t* shape, int64_t ndim, int64_t dim, float* out) {
    int64_t outer = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    int64_t inner = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner *= shape[i];
    int64_t dim_size = shape[dim];

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t in = 0; in < inner; ++in) {
            float best = x[o * dim_size * inner + 0 * inner + in];
            for (int64_t d = 1; d < dim_size; ++d) {
                float v = x[o * dim_size * inner + d * inner + in];
                if (v > best) best = v;
            }
            out[o * inner + in] = best;
        }
    }
}

void broadcast_to_shape(const float* x, const int64_t* x_shape, int64_t x_rank,
                         const int64_t* target_shape, int64_t target_rank, float* out) {
    std::vector<int64_t> x_strides(static_cast<size_t>(target_rank));
    broadcast_strides(x_shape, x_rank, target_rank, x_strides.data());

    std::vector<int64_t> target_strides(static_cast<size_t>(target_rank));
    if (target_rank > 0) {
        target_strides[static_cast<size_t>(target_rank - 1)] = 1;
        for (int64_t i = target_rank - 2; i >= 0; --i)
            target_strides[static_cast<size_t>(i)] = target_strides[static_cast<size_t>(i + 1)] * target_shape[i + 1];
    }

    int64_t n = 1;
    for (int64_t i = 0; i < target_rank; ++i) n *= target_shape[i];

    std::vector<int64_t> coord(static_cast<size_t>(target_rank));
    for (int64_t idx = 0; idx < n; ++idx) {
        int64_t rem = idx;
        for (int64_t d = 0; d < target_rank; ++d) {
            coord[d] = rem / target_strides[d];
            rem %= target_strides[d];
        }
        int64_t xidx = 0;
        for (int64_t d = 0; d < target_rank; ++d) xidx += coord[d] * x_strides[d];
        out[idx] = x[xidx];
    }
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

namespace {
// Shared by batched_matmul/_nt/_tn below: for every index in the
// broadcast output batch shape, the element offset into a's own (and
// b's own) batch region -- reusing broadcast_strides (above), the same
// stride-0-for-a-broadcast-axis trick add/sub/mul's own broadcasting
// kernels already share. Computed once, up front, rather than
// re-decomposing the batch coordinate inside each of the three
// functions' own per-item loop.
struct BatchOffsets {
    std::vector<int64_t> a_offsets, b_offsets;
};

BatchOffsets compute_batch_offsets(const int64_t* a_batch_shape, int64_t a_batch_rank,
                                    const int64_t* b_batch_shape, int64_t b_batch_rank,
                                    const int64_t* out_batch_shape, int64_t out_batch_rank) {
    std::vector<int64_t> a_strides(static_cast<size_t>(out_batch_rank));
    std::vector<int64_t> b_strides(static_cast<size_t>(out_batch_rank));
    broadcast_strides(a_batch_shape, a_batch_rank, out_batch_rank, a_strides.data());
    broadcast_strides(b_batch_shape, b_batch_rank, out_batch_rank, b_strides.data());

    std::vector<int64_t> out_strides(static_cast<size_t>(out_batch_rank));
    if (out_batch_rank > 0) {
        out_strides[static_cast<size_t>(out_batch_rank - 1)] = 1;
        for (int64_t i = out_batch_rank - 2; i >= 0; --i)
            out_strides[static_cast<size_t>(i)] = out_strides[static_cast<size_t>(i + 1)] * out_batch_shape[i + 1];
    }

    int64_t num_batches = 1;
    for (int64_t i = 0; i < out_batch_rank; ++i) num_batches *= out_batch_shape[i];

    BatchOffsets result;
    result.a_offsets.resize(static_cast<size_t>(num_batches));
    result.b_offsets.resize(static_cast<size_t>(num_batches));
    std::vector<int64_t> coord(static_cast<size_t>(out_batch_rank));
    for (int64_t idx = 0; idx < num_batches; ++idx) {
        int64_t rem = idx;
        for (int64_t d = 0; d < out_batch_rank; ++d) {
            coord[d] = rem / out_strides[d];
            rem %= out_strides[d];
        }
        int64_t aoff = 0, boff = 0;
        for (int64_t d = 0; d < out_batch_rank; ++d) {
            aoff += coord[d] * a_strides[d];
            boff += coord[d] * b_strides[d];
        }
        result.a_offsets[static_cast<size_t>(idx)] = aoff;
        result.b_offsets[static_cast<size_t>(idx)] = boff;
    }
    return result;
}
} // namespace

void batched_matmul(const float* a, const int64_t* a_batch_shape, int64_t a_batch_rank,
                     const float* b, const int64_t* b_batch_shape, int64_t b_batch_rank,
                     const int64_t* out_batch_shape, int64_t out_batch_rank,
                     int64_t M, int64_t K, int64_t N, float* out) {
    auto offs = compute_batch_offsets(a_batch_shape, a_batch_rank, b_batch_shape, b_batch_rank,
                                       out_batch_shape, out_batch_rank);
    for (size_t idx = 0; idx < offs.a_offsets.size(); ++idx) {
        matmul(a + offs.a_offsets[idx] * M * K, b + offs.b_offsets[idx] * K * N,
               out + static_cast<int64_t>(idx) * M * N, M, K, N);
    }
}

void batched_matmul_nt(const float* a, const int64_t* a_batch_shape, int64_t a_batch_rank,
                        const float* b, const int64_t* b_batch_shape, int64_t b_batch_rank,
                        const int64_t* out_batch_shape, int64_t out_batch_rank,
                        int64_t rows, int64_t reduce, int64_t cols, float* out) {
    auto offs = compute_batch_offsets(a_batch_shape, a_batch_rank, b_batch_shape, b_batch_rank,
                                       out_batch_shape, out_batch_rank);
    for (size_t idx = 0; idx < offs.a_offsets.size(); ++idx) {
        matmul_nt(a + offs.a_offsets[idx] * rows * reduce, b + offs.b_offsets[idx] * cols * reduce,
                  out + static_cast<int64_t>(idx) * rows * cols, rows, reduce, cols);
    }
}

void batched_matmul_tn(const float* a, const int64_t* a_batch_shape, int64_t a_batch_rank,
                        const float* b, const int64_t* b_batch_shape, int64_t b_batch_rank,
                        const int64_t* out_batch_shape, int64_t out_batch_rank,
                        int64_t reduce, int64_t rows, int64_t cols, float* out) {
    auto offs = compute_batch_offsets(a_batch_shape, a_batch_rank, b_batch_shape, b_batch_rank,
                                       out_batch_shape, out_batch_rank);
    for (size_t idx = 0; idx < offs.a_offsets.size(); ++idx) {
        matmul_tn(a + offs.a_offsets[idx] * reduce * rows, b + offs.b_offsets[idx] * reduce * cols,
                  out + static_cast<int64_t>(idx) * rows * cols, reduce, rows, cols);
    }
}

void index_select(const float* x, const int64_t* shape, int64_t ndim, int64_t dim,
                   const int64_t* indices, int64_t num_indices, float* out) {
    int64_t outer = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    int64_t inner = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner *= shape[i];
    int64_t dim_size = shape[dim];

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t n = 0; n < num_indices; ++n) {
            const float* src = x + o * dim_size * inner + indices[n] * inner;
            float* dst = out + o * num_indices * inner + n * inner;
            std::memcpy(dst, src, static_cast<size_t>(inner) * sizeof(float));
        }
    }
}

void index_select_bwd(const float* grad_out, const int64_t* shape, int64_t ndim, int64_t dim,
                       const int64_t* indices, int64_t num_indices, float* grad_in) {
    int64_t outer = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    int64_t inner = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner *= shape[i];
    int64_t dim_size = shape[dim];

    std::fill(grad_in, grad_in + outer * dim_size * inner, 0.0f);
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t n = 0; n < num_indices; ++n) {
            const float* src = grad_out + o * num_indices * inner + n * inner;
            float* dst = grad_in + o * dim_size * inner + indices[n] * inner;
            for (int64_t j = 0; j < inner; ++j) dst[j] += src[j];
        }
    }
}

float reduce_sum(const float* x, int64_t n) {
    float acc = 0.0f;
    for (int64_t i = 0; i < n; ++i) acc += x[i];
    return acc;
}

void axpy_(float* out, const float* x, float alpha, int64_t n) {
    for (int64_t i = 0; i < n; ++i) out[i] += alpha * x[i];
}

void im2col(const float* x, float* col, int64_t C, int64_t H, int64_t W,
            int64_t kH, int64_t kW, int64_t stride, int64_t padding,
            int64_t Hout, int64_t Wout) {
    int64_t HWout = Hout * Wout;
    for (int64_t c = 0; c < C; ++c) {
        for (int64_t kh = 0; kh < kH; ++kh) {
            for (int64_t kw = 0; kw < kW; ++kw) {
                float* row = col + ((c * kH + kh) * kW + kw) * HWout;
                for (int64_t oh = 0; oh < Hout; ++oh) {
                    int64_t ih = oh * stride + kh - padding;
                    for (int64_t ow = 0; ow < Wout; ++ow) {
                        int64_t iw = ow * stride + kw - padding;
                        row[oh * Wout + ow] =
                            (ih >= 0 && ih < H && iw >= 0 && iw < W) ? x[(c * H + ih) * W + iw] : 0.0f;
                    }
                }
            }
        }
    }
}

void col2im(const float* dcol, float* dx, int64_t C, int64_t H, int64_t W,
            int64_t kH, int64_t kW, int64_t stride, int64_t padding,
            int64_t Hout, int64_t Wout) {
    std::fill(dx, dx + C * H * W, 0.0f);
    int64_t HWout = Hout * Wout;
    for (int64_t c = 0; c < C; ++c) {
        for (int64_t kh = 0; kh < kH; ++kh) {
            for (int64_t kw = 0; kw < kW; ++kw) {
                const float* row = dcol + ((c * kH + kh) * kW + kw) * HWout;
                for (int64_t oh = 0; oh < Hout; ++oh) {
                    int64_t ih = oh * stride + kh - padding;
                    for (int64_t ow = 0; ow < Wout; ++ow) {
                        int64_t iw = ow * stride + kw - padding;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            dx[(c * H + ih) * W + iw] += row[oh * Wout + ow];
                        }
                    }
                }
            }
        }
    }
}

void add_bias_nchw(const float* x, const float* bias, float* out, int64_t N, int64_t C, int64_t HW) {
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const float* xrow = x + (n * C + c) * HW;
            float* orow = out + (n * C + c) * HW;
            float bv = bias[c];
            for (int64_t i = 0; i < HW; ++i) orow[i] = xrow[i] + bv;
        }
    }
}

void sum_over_batch_and_spatial(const float* dy, float* db, int64_t N, int64_t C, int64_t HW) {
    std::fill(db, db + C, 0.0f);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const float* row = dy + (n * C + c) * HW;
            float s = 0.0f;
            for (int64_t i = 0; i < HW; ++i) s += row[i];
            db[c] += s;
        }
    }
}

void maxpool2d_fwd(const float* x, float* out, int64_t* argmax,
                    int64_t N, int64_t C, int64_t H, int64_t W,
                    int64_t kernel_size, int64_t stride,
                    int64_t Hout, int64_t Wout) {
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const float* xin = x + (n * C + c) * H * W;
            float* oout = out + (n * C + c) * Hout * Wout;
            int64_t* aout = argmax + (n * C + c) * Hout * Wout;
            for (int64_t oh = 0; oh < Hout; ++oh) {
                for (int64_t ow = 0; ow < Wout; ++ow) {
                    float best = -std::numeric_limits<float>::infinity();
                    int64_t best_idx = -1;
                    for (int64_t kh = 0; kh < kernel_size; ++kh) {
                        int64_t ih = oh * stride + kh;
                        for (int64_t kw = 0; kw < kernel_size; ++kw) {
                            int64_t iw = ow * stride + kw;
                            float v = xin[ih * W + iw];
                            if (v > best) {
                                best = v;
                                best_idx = ih * W + iw;
                            }
                        }
                    }
                    oout[oh * Wout + ow] = best;
                    aout[oh * Wout + ow] = best_idx;
                }
            }
        }
    }
}

void maxpool2d_bwd(const float* grad_out, const int64_t* argmax, float* grad_in,
                    int64_t N, int64_t C, int64_t H, int64_t W,
                    int64_t Hout, int64_t Wout) {
    std::fill(grad_in, grad_in + N * C * H * W, 0.0f);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const float* gout = grad_out + (n * C + c) * Hout * Wout;
            const int64_t* aout = argmax + (n * C + c) * Hout * Wout;
            float* gin = grad_in + (n * C + c) * H * W;
            for (int64_t i = 0; i < Hout * Wout; ++i) {
                gin[aout[i]] += gout[i];
            }
        }
    }
}

namespace {
// Row-major strides for `shape` (ndim entries): stride[ndim-1] = 1,
// stride[i] = stride[i+1] * shape[i+1]. Shared by transpose's own
// input- and output-side coordinate math.
void row_major_strides(const int64_t* shape, int64_t ndim, int64_t* strides) {
    strides[ndim - 1] = 1;
    for (int64_t i = ndim - 2; i >= 0; --i) strides[i] = strides[i + 1] * shape[i + 1];
}
} // namespace

void transpose(const float* x, const int64_t* shape, int64_t ndim,
               int64_t dim0, int64_t dim1, float* out) {
    std::vector<int64_t> strides(static_cast<size_t>(ndim));
    row_major_strides(shape, ndim, strides.data());

    std::vector<int64_t> out_shape(shape, shape + ndim);
    std::swap(out_shape[dim0], out_shape[dim1]);
    std::vector<int64_t> out_strides(static_cast<size_t>(ndim));
    row_major_strides(out_shape.data(), ndim, out_strides.data());

    int64_t n = 1;
    for (int64_t i = 0; i < ndim; ++i) n *= shape[i];

    std::vector<int64_t> coord(static_cast<size_t>(ndim));
    for (int64_t idx = 0; idx < n; ++idx) {
        int64_t rem = idx;
        for (int64_t d = 0; d < ndim; ++d) {
            coord[d] = rem / strides[d];
            rem %= strides[d];
        }
        std::swap(coord[dim0], coord[dim1]);
        int64_t oidx = 0;
        for (int64_t d = 0; d < ndim; ++d) oidx += coord[d] * out_strides[d];
        out[oidx] = x[idx];
    }
}

void slice(const float* x, const int64_t* shape, int64_t ndim,
           int64_t dim, int64_t start, int64_t stop, float* out) {
    int64_t outer = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    int64_t inner = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner *= shape[i];
    int64_t dim_size = shape[dim];
    int64_t size = stop - start;

    for (int64_t o = 0; o < outer; ++o) {
        const float* src = x + o * dim_size * inner + start * inner;
        float* dst = out + o * size * inner;
        std::memcpy(dst, src, static_cast<size_t>(size * inner) * sizeof(float));
    }
}

void scatter_range(const float* x, const int64_t* full_shape, int64_t ndim,
                    int64_t dim, int64_t start, int64_t stop, float* out) {
    int64_t outer = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= full_shape[i];
    int64_t inner = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner *= full_shape[i];
    int64_t dim_size = full_shape[dim];
    int64_t size = stop - start;

    for (int64_t o = 0; o < outer; ++o) {
        const float* src = x + o * size * inner;
        float* dst = out + o * dim_size * inner + start * inner;
        std::memcpy(dst, src, static_cast<size_t>(size * inner) * sizeof(float));
    }
}

} // namespace kan::cpu
