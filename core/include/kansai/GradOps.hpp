#pragma once
#include "kansai/Tensor.hpp"

namespace kan {

// Forward-only helpers backing kansai.kir's source-transform autograd
// (grad()): each is the "vjp op" one forward primitive's backward pass
// needs, exposed as its own op so the *backward* graph can be built
// (and, later, optimized) out of the same kind of nodes as any other
// KIR graph, rather than needing bespoke non-graph handling. None of
// these attach a grad_node -- differentiating the backward graph itself
// (higher-order gradients) isn't implemented.

// grad_input = grad_output * (input > 0) -- relu's vjp. `input` is the
// ORIGINAL (pre-relu) tensor, not its output.
Tensor relu_backward(const Tensor& input, const Tensor& grad_output);

// grad_bias = sum(grad_output, axis=0) -- the vjp for the bias operand
// of a (batch, features) + (features,) broadcast add.
Tensor sum_axis0(const Tensor& grad_output);

// out = grad_output.item() * scale, broadcast to `shape` -- the vjp for
// sum() (scale=1) and mean() (scale=1/numel), both of which collapse a
// tensor to a single scalar cotangent that has to be spread back out to
// every position of the original input.
Tensor broadcast_scalar(const Tensor& grad_output, std::vector<int64_t> shape, float scale);

// out (rows x cols) = A (rows x reduce) @ B (cols x reduce)^T, and its
// mirror out = A (reduce x rows)^T @ B (reduce x cols) -- matmul's own
// two vjps, exposed as first-class ops so a backward graph can trace
// them. Same kernels Tensor::matmul's own eager backward closure already
// uses (core/src/Tensor.cpp) -- this just gives the *graph* a way to
// express the same computation instead of it living only inside a C++
// closure.
Tensor matmul_nt(const Tensor& a, const Tensor& b);
Tensor matmul_tn(const Tensor& a, const Tensor& b);

// Sums `grad` down to `target_shape` -- the general broadcasting vjp
// add/sub/mul's own eager backward_fn closures (core/src/Tensor.cpp)
// already use directly via cpu::reduce_to_shape; this is that same
// operation exposed as a first-class KIR op instead, for kir.grad's
// vjp rules to build a backward *graph* out of, the same reason
// sum_axis0/matmul_nt/matmul_tn exist as graph-visible ops above rather
// than staying C++-closure-only. sum_axis0 is the fixed 2D "sum over
// axis 0" special case of exactly this; this is its general N-D form.
Tensor reduce_to_shape(const Tensor& grad, std::vector<int64_t> target_shape);

} // namespace kan
