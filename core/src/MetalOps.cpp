#include "kansai/MetalOps.hpp"
#include "kansai/backend/metal/MetalOps.hpp"
#include <stdexcept>

namespace kan {

bool metal_available() { return metal::available(); }

Tensor metal_matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2 || a.shape()[1] != b.shape()[0])
        throw std::runtime_error("metal_matmul: incompatible shapes");
    int64_t M = a.shape()[0], K = a.shape()[1], N = b.shape()[1];
    Tensor out = Tensor::zeros({M, N}, false);
    metal::matmul(a.data_ptr(), b.data_ptr(), out.data_ptr(), M, K, N);
    return out;
}

Tensor metal_bias_relu(const Tensor& x, const Tensor& bias) {
    Tensor out = Tensor::zeros(x.shape(), false);
    metal::bias_relu(x.data_ptr(), bias.data_ptr(), out.data_ptr(), x.shape()[0], x.shape()[1]);
    return out;
}

Tensor metal_add_bias(const Tensor& x, const Tensor& bias) {
    Tensor out = Tensor::zeros(x.shape(), false);
    metal::add_bias(x.data_ptr(), bias.data_ptr(), out.data_ptr(), x.shape()[0], x.shape()[1]);
    return out;
}

} // namespace kan
