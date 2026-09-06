#include "kansai/MetalOps.hpp"
#include "kansai/backend/cpu/Ops.hpp"
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

Tensor metal_matmul_mps(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2 || a.shape()[1] != b.shape()[0])
        throw std::runtime_error("metal_matmul_mps: incompatible shapes");
    int64_t M = a.shape()[0], K = a.shape()[1], N = b.shape()[1];
    Tensor out = Tensor::zeros({M, N}, false);
    metal::matmul_mps(a.data_ptr(), b.data_ptr(), out.data_ptr(), M, K, N);
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

Tensor metal_add(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) throw std::runtime_error("metal_add: shape mismatch");
    Tensor out = Tensor::zeros(a.shape(), false);
    metal::add(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel());
    return out;
}

Tensor metal_sub(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) throw std::runtime_error("metal_sub: shape mismatch");
    Tensor out = Tensor::zeros(a.shape(), false);
    metal::sub(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel());
    return out;
}

Tensor metal_mul(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) throw std::runtime_error("metal_mul: shape mismatch");
    Tensor out = Tensor::zeros(a.shape(), false);
    metal::mul(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel());
    return out;
}

Tensor metal_relu(const Tensor& x) {
    Tensor out = Tensor::zeros(x.shape(), false);
    metal::relu(x.data_ptr(), out.data_ptr(), x.numel());
    return out;
}

Tensor metal_fused_sub_square(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) throw std::runtime_error("metal_fused_sub_square: shape mismatch");
    Tensor out = Tensor::zeros(a.shape(), false);
    metal::fused_sub_square(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel());
    return out;
}

Tensor metal_sum(const Tensor& x) {
    Tensor out = Tensor::zeros({1}, false);
    metal::reduce_sum(x.data_ptr(), out.data_ptr(), x.numel(), 1.0f);
    return out;
}

Tensor metal_mean(const Tensor& x) {
    Tensor out = Tensor::zeros({1}, false);
    metal::reduce_sum(x.data_ptr(), out.data_ptr(), x.numel(), 1.0f / static_cast<float>(x.numel()));
    return out;
}

Tensor metal_elementwise_chain(const Tensor& x, const std::vector<std::string>& kinds,
                                const std::vector<Tensor>& biases) {
    if (kinds.size() != biases.size())
        throw std::runtime_error("metal_elementwise_chain: kinds and biases must be the same length");

    std::vector<metal::ElemStep> steps;
    steps.reserve(kinds.size());
    for (size_t i = 0; i < kinds.size(); ++i) {
        metal::ElemKernel k;
        if (kinds[i] == "bias_relu") k = metal::ElemKernel::BiasRelu;
        else if (kinds[i] == "add_bias") k = metal::ElemKernel::AddBias;
        else throw std::runtime_error("metal_elementwise_chain: unknown kind '" + kinds[i] + "'");
        steps.push_back({k, biases[i].data_ptr()});
    }

    Tensor out = Tensor::zeros(x.shape(), false);
    metal::run_elementwise_chain(x.data_ptr(), x.shape()[0], x.shape()[1], steps, out.data_ptr());
    return out;
}

Tensor metal_conv2d(const Tensor& x, const Tensor& weight, const Tensor& bias, int64_t stride, int64_t padding) {
    if (x.ndim() != 4) throw std::runtime_error("metal_conv2d: input must be 4D (N, Cin, H, W)");
    if (weight.ndim() != 4) throw std::runtime_error("metal_conv2d: weight must be 4D (Cout, Cin, kH, kW)");

    int64_t N = x.shape()[0], Cin = x.shape()[1], H = x.shape()[2], W = x.shape()[3];
    int64_t Cout = weight.shape()[0], Cin_w = weight.shape()[1], kH = weight.shape()[2], kW = weight.shape()[3];
    if (Cin != Cin_w) throw std::runtime_error("metal_conv2d: input and weight channel counts don't match");
    if (bias.numel() != Cout) throw std::runtime_error("metal_conv2d: bias must have Cout elements");

    int64_t Hout = (H + 2 * padding - kH) / stride + 1;
    int64_t Wout = (W + 2 * padding - kW) / stride + 1;
    if (Hout <= 0 || Wout <= 0)
        throw std::runtime_error("metal_conv2d: kernel/stride/padding produce a non-positive output size");
    int64_t HWout = Hout * Wout;
    int64_t colRows = Cin * kH * kW;

    Tensor out = Tensor::zeros({N, Cout, Hout, Wout}, false);

    // The im2col scratch buffer has to be a Tensor, not a plain
    // std::vector: metal_matmul_mps NoCopy-wraps every buffer it's
    // given, which requires page-aligned memory (see Storage.cpp) --
    // a std::vector's allocator gives no such guarantee, but
    // Tensor::zeros already does, for free, via the same Storage every
    // other tensor in this codebase goes through.
    Tensor col = Tensor::zeros({colRows, HWout}, false);

    for (int64_t n = 0; n < N; ++n) {
        const float* xn = x.data_ptr() + n * Cin * H * W;
        float* yn = out.data_ptr() + n * Cout * HWout;
        cpu::im2col(xn, col.data_ptr(), Cin, H, W, kH, kW, stride, padding, Hout, Wout);
        metal::matmul_mps(weight.data_ptr(), col.data_ptr(), yn, Cout, colRows, HWout);
    }
    metal::add_bias_nchw(out.data_ptr(), bias.data_ptr(), out.data_ptr(), N, Cout, HWout);
    return out;
}

} // namespace kan
