#pragma once
#include "kansai/Tensor.hpp"
#include <string>
#include <vector>

namespace kan {

// Tensor-level wrappers around backend/metal's Metal compute kernels.
// Forward-only, same as the CPU fused_bias_relu/fused_sub_square: no
// grad_node is attached.
bool metal_available();
Tensor metal_matmul(const Tensor& a, const Tensor& b);
Tensor metal_matmul_mps(const Tensor& a, const Tensor& b);
Tensor metal_bias_relu(const Tensor& x, const Tensor& bias);
Tensor metal_add_bias(const Tensor& x, const Tensor& bias);

// Runs a sequence of elementwise ops (each kind either "bias_relu" or
// "add_bias") as ONE Metal command buffer instead of one per step --
// see backend/metal's run_elementwise_chain for what that saves.
// kinds.size() must equal biases.size().
Tensor metal_elementwise_chain(const Tensor& x, const std::vector<std::string>& kinds,
                                const std::vector<Tensor>& biases);

} // namespace kan
