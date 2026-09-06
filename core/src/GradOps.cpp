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

Tensor batched_matmul_nt(const Tensor& a, const Tensor& b) {
    if (a.ndim() < 2 || b.ndim() < 2)
        throw std::runtime_error("batched_matmul_nt: both operands must be at least 2D");
    int64_t rows = a.shape()[a.ndim() - 2], reduce_a = a.shape()[a.ndim() - 1];
    int64_t cols = b.shape()[b.ndim() - 2], reduce_b = b.shape()[b.ndim() - 1];
    if (reduce_a != reduce_b)
        throw std::runtime_error("batched_matmul_nt: inner dimensions don't match");

    std::vector<int64_t> a_batch(a.shape().begin(), a.shape().end() - 2);
    std::vector<int64_t> b_batch(b.shape().begin(), b.shape().end() - 2);
    std::vector<int64_t> out_batch = broadcast_shapes(a_batch, b_batch, "batched_matmul_nt");

    std::vector<int64_t> out_shape = out_batch;
    out_shape.push_back(rows);
    out_shape.push_back(cols);
    Tensor out = Tensor::zeros(out_shape, false);
    cpu::batched_matmul_nt(a.data_ptr(), a_batch.data(), static_cast<int64_t>(a_batch.size()), b.data_ptr(),
                            b_batch.data(), static_cast<int64_t>(b_batch.size()), out_batch.data(),
                            static_cast<int64_t>(out_batch.size()), rows, reduce_a, cols, out.data_ptr());
    return out;
}

Tensor batched_matmul_tn(const Tensor& a, const Tensor& b) {
    if (a.ndim() < 2 || b.ndim() < 2)
        throw std::runtime_error("batched_matmul_tn: both operands must be at least 2D");
    int64_t reduce_a = a.shape()[a.ndim() - 2], rows = a.shape()[a.ndim() - 1];
    int64_t reduce_b = b.shape()[b.ndim() - 2], cols = b.shape()[b.ndim() - 1];
    if (reduce_a != reduce_b)
        throw std::runtime_error("batched_matmul_tn: inner dimensions don't match");

    std::vector<int64_t> a_batch(a.shape().begin(), a.shape().end() - 2);
    std::vector<int64_t> b_batch(b.shape().begin(), b.shape().end() - 2);
    std::vector<int64_t> out_batch = broadcast_shapes(a_batch, b_batch, "batched_matmul_tn");

    std::vector<int64_t> out_shape = out_batch;
    out_shape.push_back(rows);
    out_shape.push_back(cols);
    Tensor out = Tensor::zeros(out_shape, false);
    cpu::batched_matmul_tn(a.data_ptr(), a_batch.data(), static_cast<int64_t>(a_batch.size()), b.data_ptr(),
                            b_batch.data(), static_cast<int64_t>(b_batch.size()), out_batch.data(),
                            static_cast<int64_t>(out_batch.size()), reduce_a, rows, cols, out.data_ptr());
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

Tensor index_select_backward(const Tensor& grad_output, int64_t dim, std::vector<int64_t> indices,
                              std::vector<int64_t> input_shape) {
    Tensor grad_input = Tensor::zeros(input_shape, false);
    cpu::index_select_bwd(grad_output.data_ptr(), input_shape.data(), static_cast<int64_t>(input_shape.size()), dim,
                           indices.data(), static_cast<int64_t>(indices.size()), grad_input.data_ptr());
    return grad_input;
}

Tensor conv2d_backward_bias(const Tensor& grad_output) {
    int64_t N = grad_output.shape()[0], Cout = grad_output.shape()[1];
    int64_t HWout = grad_output.shape()[2] * grad_output.shape()[3];
    Tensor grad_b = Tensor::zeros({Cout}, false);
    cpu::sum_over_batch_and_spatial(grad_output.data_ptr(), grad_b.data_ptr(), N, Cout, HWout);
    return grad_b;
}

Tensor conv2d_backward_weight(const Tensor& x, const Tensor& grad_output, std::vector<int64_t> weight_shape,
                               int64_t stride, int64_t padding) {
    int64_t N = x.shape()[0], Cin = x.shape()[1], H = x.shape()[2], W = x.shape()[3];
    int64_t Cout = weight_shape[0], kH = weight_shape[2], kW = weight_shape[3];
    int64_t Hout = grad_output.shape()[2], Wout = grad_output.shape()[3];
    int64_t HWout = Hout * Wout;
    int64_t colRows = Cin * kH * kW;

    Tensor grad_w = Tensor::zeros(weight_shape, false);
    std::vector<float> col(static_cast<size_t>(colRows * HWout));
    std::vector<float> grad_w_step(static_cast<size_t>(Cout * colRows));
    for (int64_t n = 0; n < N; ++n) {
        const float* xn = x.data_ptr() + n * Cin * H * W;
        const float* dyn = grad_output.data_ptr() + n * Cout * HWout;
        cpu::im2col(xn, col.data(), Cin, H, W, kH, kW, stride, padding, Hout, Wout);
        cpu::matmul_nt(dyn, col.data(), grad_w_step.data(), Cout, HWout, colRows);
        cpu::axpy_(grad_w.data_ptr(), grad_w_step.data(), 1.0f, Cout * colRows);
    }
    return grad_w;
}

Tensor conv2d_backward_input(const Tensor& weight, const Tensor& grad_output, std::vector<int64_t> x_shape,
                              int64_t stride, int64_t padding) {
    int64_t Cin = x_shape[1], H = x_shape[2], W = x_shape[3];
    int64_t Cout = weight.shape()[0], kH = weight.shape()[2], kW = weight.shape()[3];
    int64_t Hout = grad_output.shape()[2], Wout = grad_output.shape()[3];
    int64_t HWout = Hout * Wout;
    int64_t colRows = Cin * kH * kW;

    Tensor grad_x = Tensor::zeros(x_shape, false);
    std::vector<float> dcol(static_cast<size_t>(colRows * HWout));
    for (int64_t n = 0; n < x_shape[0]; ++n) {
        const float* dyn = grad_output.data_ptr() + n * Cout * HWout;
        float* dxn = grad_x.data_ptr() + n * Cin * H * W;
        cpu::matmul_tn(weight.data_ptr(), dyn, dcol.data(), Cout, colRows, HWout);
        cpu::col2im(dcol.data(), dxn, Cin, H, W, kH, kW, stride, padding, Hout, Wout);
    }
    return grad_x;
}

} // namespace kan
