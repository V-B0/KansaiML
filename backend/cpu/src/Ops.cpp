#include "kansai/backend/cpu/Ops.hpp"
#include <algorithm>
#include <cstring>
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
