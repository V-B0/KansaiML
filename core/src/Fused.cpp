#include "kansai/Fused.hpp"
#include "kansai/backend/cpu/Fused.hpp"

namespace kan {

Tensor fused_bias_relu(const Tensor& x, const Tensor& bias) {
    Tensor out = Tensor::zeros(x.shape(), false);
    cpu::fused_bias_relu(x.data_ptr(), bias.data_ptr(), out.data_ptr(), x.shape()[0], x.shape()[1]);
    return out;
}

Tensor fused_sub_square(const Tensor& a, const Tensor& b) {
    Tensor out = Tensor::zeros(a.shape(), false);
    cpu::fused_sub_square(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel());
    return out;
}

} // namespace kan
