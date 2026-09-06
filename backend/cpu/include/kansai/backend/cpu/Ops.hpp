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

// General NumPy-style broadcasting for add/sub/mul: `a` (shape a_shape,
// rank a_rank) and `b` (shape b_shape, rank b_rank) combine into `out`
// (shape out_shape, rank out_rank -- already computed by the caller via
// right-aligned broadcasting rules, same as NumPy's). Unlike every
// other kernel above (and the fixed bias_broadcast case), the two
// operands can each independently be smaller than `out` along any
// dimension -- a size-1 dimension, or a dimension that doesn't exist at
// all on that operand -- broadcasts via a stride-0 read (every position
// along that axis reads the same single source element) rather than
// materializing a larger buffer. This is the fallback path taken only
// when the operands' shapes don't already match exactly (the fast
// same-shape kernels above stay the common case); a real, unattempted
// lever for later is specializing this generic per-element coordinate
// decomposition into 1D/2D-specific loops the way the fast paths already
// are for the exact-match case.
void add_broadcast(const float* a, const int64_t* a_shape, int64_t a_rank,
                    const float* b, const int64_t* b_shape, int64_t b_rank,
                    const int64_t* out_shape, int64_t out_rank, float* out);
void sub_broadcast(const float* a, const int64_t* a_shape, int64_t a_rank,
                    const float* b, const int64_t* b_shape, int64_t b_rank,
                    const int64_t* out_shape, int64_t out_rank, float* out);
void mul_broadcast(const float* a, const int64_t* a_shape, int64_t a_rank,
                    const float* b, const int64_t* b_shape, int64_t b_rank,
                    const int64_t* out_shape, int64_t out_rank, float* out);

// The general broadcasting backward primitive: sums `grad` (shape
// grad_shape, rank grad_rank) down to `target_shape` (rank
// target_rank <= grad_rank) -- the exact inverse of how a smaller
// operand broadcasts up to a larger output in add/sub/mul's forward
// pass above. Every axis `target_shape` doesn't have at all (a leading
// axis grad_shape carries but target_shape doesn't), or holds as size 1
// while grad_shape is bigger there, gets summed over. Generalizes the
// old add-specific sum_over_batch (a fixed 2D "sum over axis 0" case)
// to any rank and any combination of broadcast axes -- sum_over_batch
// itself is unchanged and still used for that one fixed shape, since
// nothing needed it to change.
void reduce_to_shape(const float* grad, const int64_t* grad_shape, int64_t grad_rank,
                      const int64_t* target_shape, int64_t target_rank, float* out);

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

// Unfolds a single (C,H,W) image into a (C*kH*kW, Hout*Wout) matrix of
// patches -- row (c*kH+kh)*kW+kw, column oh*Wout+ow holds
// x[c, oh*stride+kh-padding, ow*stride+kw-padding], or 0 where that
// falls outside the (zero-)padded image. This is conv2d's forward
// reduced to a single matmul: reshape the weight to (Cout, C*kH*kW) and
// multiply by this matrix to get the (Cout, Hout*Wout) output for one
// batch item -- reusing the same (Accelerate-backed) matmul kernel
// everything else in this codebase already uses, rather than a
// hand-written convolution inner loop.
void im2col(const float* x, float* col, int64_t C, int64_t H, int64_t W,
            int64_t kH, int64_t kW, int64_t stride, int64_t padding,
            int64_t Hout, int64_t Wout);

// The inverse of im2col: scatter-adds a (C*kH*kW, Hout*Wout) gradient
// matrix back into a (C,H,W) gradient image (multiple patch positions
// can overlap the same input pixel at stride < kernel size, so this
// accumulates, it doesn't just place values). Overwrites `dx` first
// (every element gets at least zero-initialized) then adds -- callers
// wanting to accumulate across multiple im2col/col2im pairs (e.g. across
// a batch) do that themselves, one call's dx at a time.
void col2im(const float* dcol, float* dx, int64_t C, int64_t H, int64_t W,
            int64_t kH, int64_t kW, int64_t stride, int64_t padding,
            int64_t Hout, int64_t Wout);

// out[n,c,i] = x[n,c,i] + bias[c] -- x/out shaped (N, C, HW) flattened,
// conv2d's per-output-channel bias broadcast over every spatial position.
void add_bias_nchw(const float* x, const float* bias, float* out,
                    int64_t N, int64_t C, int64_t HW);

// db[c] = sum over n, i of dy[n,c,i] -- conv2d bias's vjp; dy shaped
// (N, C, HW) flattened.
void sum_over_batch_and_spatial(const float* dy, float* db,
                                 int64_t N, int64_t C, int64_t HW);

// General shape ops -- unlike the fixed-rank kernels above (matmul's
// M/K/N, conv2d's im2col/col2im), these work on any rank via an
// explicit shape array, not a name per dimension. Still "no autograd
// knowledge, just moves numbers" -- shape is just a more general form
// of the same kind of dimension parameter matmul/conv2d already take.

// Permutes x (row-major, `ndim` dims described by `shape`) into `out`
// with axes dim0 and dim1 exchanged -- out's own shape is `shape` with
// those two entries swapped. A real data reorder (not a metadata-only
// view): this codebase has no stride concept, every Tensor is always
// fully packed row-major, so transposing anything but the last two
// axes of a 2D tensor genuinely moves every element. O(n) in the
// tensor's element count, with one coordinate decomposition per
// element -- not vectorized or blocked; a real, unattempted lever if
// this ever shows up as a bottleneck.
void transpose(const float* x, const int64_t* shape, int64_t ndim,
               int64_t dim0, int64_t dim1, float* out);

// Copies x's [start, stop) sub-range along `dim` into `out` (sized for
// the resulting shape: `shape` with dim's extent replaced by
// stop - start). The same outer/inner/dim_size block-copy shape
// python/kansai/distributed.py's _split_tensor already uses -- ported
// to C++ here for the same operation at native speed instead of going
// through tolist()/from_flat().
void slice(const float* x, const int64_t* shape, int64_t ndim,
           int64_t dim, int64_t start, int64_t stop, float* out);

// The mirror image of slice: writes `x` (shaped like a [start, stop)
// sub-range) into `out`'s corresponding range along `dim`, where `out`
// is already sized for `full_shape`. Two distinct callers, one
// primitive: slice's own backward (scatter a cotangent into an
// otherwise-zero gradient at the range that was actually read) and
// Tensor::cat's forward (write each input into its own disjoint slot
// of the concatenated output) are the same "place this sub-range at
// this offset" operation -- ranges never overlap in either use, so
// this is a plain write, never an accumulate.
void scatter_range(const float* x, const int64_t* full_shape, int64_t ndim,
                    int64_t dim, int64_t start, int64_t stop, float* out);

} // namespace kan::cpu
