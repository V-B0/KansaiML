#pragma once
#include "kansai/Tensor.hpp"

namespace kan {

// Tensor-level wrappers around backend/metal's Metal compute kernels.
// Forward-only, same as the CPU fused_bias_relu/fused_sub_square: no
// grad_node is attached.
bool metal_available();
Tensor metal_matmul(const Tensor& a, const Tensor& b);
Tensor metal_bias_relu(const Tensor& x, const Tensor& bias);
Tensor metal_add_bias(const Tensor& x, const Tensor& bias);

} // namespace kan
