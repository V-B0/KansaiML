#include "kansai/GradOps.hpp"
#include "kansai/backend/cpu/Ops.hpp"
#include <stdexcept>

namespace kan {

Tensor relu_backward(const Tensor& input, const Tensor& grad_output) {
    Tensor out = Tensor::zeros(input.shape(), false);
    cpu::relu_bwd(input.data_ptr(), grad_output.data_ptr(), out.data_ptr(), input.numel());
    return out;
}

Tensor sum_axis0(const Tensor& grad_output) {
    if (grad_output.ndim() != 2)
        throw std::runtime_error("sum_axis0: expected a 2D tensor");
    int64_t batch = grad_output.shape()[0], features = grad_output.shape()[1];
    Tensor out = Tensor::zeros({features}, false);
    cpu::sum_over_batch(grad_output.data_ptr(), out.data_ptr(), batch, features);
    return out;
}

Tensor broadcast_scalar(const Tensor& grad_output, std::vector<int64_t> shape, float scale) {
    if (grad_output.numel() != 1)
        throw std::runtime_error("broadcast_scalar: expected a scalar (numel==1) tensor");
    Tensor out = Tensor::zeros(std::move(shape), false);
    cpu::fill(out.data_ptr(), grad_output.data_ptr()[0] * scale, out.numel());
    return out;
}

Tensor matmul_nt(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2 || a.shape()[1] != b.shape()[1])
        throw std::runtime_error("matmul_nt: incompatible shapes");
    int64_t rows = a.shape()[0], reduce = a.shape()[1], cols = b.shape()[0];
    Tensor out = Tensor::zeros({rows, cols}, false);
    cpu::matmul_nt(a.data_ptr(), b.data_ptr(), out.data_ptr(), rows, reduce, cols);
    return out;
}

Tensor matmul_tn(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2 || a.shape()[0] != b.shape()[0])
        throw std::runtime_error("matmul_tn: incompatible shapes");
    int64_t reduce = a.shape()[0], rows = a.shape()[1], cols = b.shape()[1];
    Tensor out = Tensor::zeros({rows, cols}, false);
    cpu::matmul_tn(a.data_ptr(), b.data_ptr(), out.data_ptr(), reduce, rows, cols);
    return out;
}

Tensor reduce_to_shape(const Tensor& grad, std::vector<int64_t> target_shape) {
    Tensor out = Tensor::zeros(target_shape, false);
    cpu::reduce_to_shape(grad.data_ptr(), grad.shape().data(), grad.ndim(),
                          target_shape.data(), static_cast<int64_t>(target_shape.size()), out.data_ptr());
    return out;
}

Tensor broadcast_to_shape(const Tensor& grad, std::vector<int64_t> target_shape) {
    Tensor out = Tensor::zeros(target_shape, false);
    cpu::broadcast_to_shape(grad.data_ptr(), grad.shape().data(), grad.ndim(),
                             target_shape.data(), static_cast<int64_t>(target_shape.size()), out.data_ptr());
    return out;
}

Tensor gelu_backward(const Tensor& input, const Tensor& grad_output) {
    Tensor out = Tensor::zeros(input.shape(), false);
    cpu::gelu_bwd(input.data_ptr(), grad_output.data_ptr(), out.data_ptr(), input.numel());
    return out;
}

Tensor leaky_relu_backward(const Tensor& input, const Tensor& grad_output, float negative_slope) {
    Tensor out = Tensor::zeros(input.shape(), false);
    cpu::leaky_relu_bwd(input.data_ptr(), grad_output.data_ptr(), out.data_ptr(), input.numel(), negative_slope);
    return out;
}

} // namespace kan
