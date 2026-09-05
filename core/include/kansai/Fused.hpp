#pragma once
#include "kansai/Tensor.hpp"

namespace kan {

// Tensor-level wrappers around the pattern-matched fused kernels in
// backend/cpu/Fused.hpp. Forward-only: the result carries no grad_node
// -- fusing *through* backward() is future work (see kir.py).
Tensor fused_bias_relu(const Tensor& x, const Tensor& bias);
Tensor fused_sub_square(const Tensor& a, const Tensor& b);

} // namespace kan
