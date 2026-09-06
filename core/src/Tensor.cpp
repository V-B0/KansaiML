#include "kansai/Tensor.hpp"
#include "kansai/backend/cpu/Ops.hpp"
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace kan {

static int64_t numel_of(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

// Declared in Tensor.hpp (shared with GradOps.cpp -- see that
// declaration's own comment for why this moved out of being `static`
// here). NumPy-style right-aligned broadcasting: pads the shorter
// shape with implicit leading 1s, then each aligned pair of dims must
// either match or one of them must be 1 -- the standard rule, used
// here (rather than the fixed "(batch, features) + (features,)" bias
// case add() already special-cased before this existed) for any OTHER
// shape mismatch add/sub/mul now accept. Throws on an incompatible
// pair rather than silently picking one side, the same "fail loud"
// stance every other shape check in this file already takes.
std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& a, const std::vector<int64_t>& b,
                                       const char* op_name) {
    int64_t ra = static_cast<int64_t>(a.size()), rb = static_cast<int64_t>(b.size());
    int64_t out_rank = ra > rb ? ra : rb;
    std::vector<int64_t> out(static_cast<size_t>(out_rank));
    for (int64_t i = 0; i < out_rank; ++i) {
        int64_t ai = i - (out_rank - ra);
        int64_t bi = i - (out_rank - rb);
        int64_t av = (ai >= 0) ? a[ai] : 1;
        int64_t bv = (bi >= 0) ? b[bi] : 1;
        if (av != bv && av != 1 && bv != 1)
            throw std::runtime_error(std::string(op_name) + ": shapes are not broadcast-compatible");
        out[i] = av > bv ? av : bv;
    }
    return out;
}

int64_t Tensor::numel() const { return numel_of(impl_->shape); }

namespace {
thread_local StoragePool* g_active_pool = nullptr;
}

void set_active_pool(StoragePool* pool) { g_active_pool = pool; }

std::shared_ptr<GradNode> Tensor::grad_node() const { return impl_->grad_node; }
void Tensor::set_grad_node(std::shared_ptr<GradNode> node) { impl_->grad_node = std::move(node); }

// ---------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------

Tensor Tensor::zeros(std::vector<int64_t> shape, bool requires_grad) {
    auto data = std::make_shared<TensorData>();
    data->shape = std::move(shape);
    int64_t n = numel_of(data->shape);
    size_t nbytes = static_cast<size_t>(n) * sizeof(float);
    data->storage = g_active_pool ? g_active_pool->acquire(nbytes)
                                   : std::make_shared<Storage>(nbytes);
    cpu::fill(static_cast<float*>(data->storage->data()), 0.0f, n);
    data->requires_grad = requires_grad;
    return Tensor(data);
}

Tensor Tensor::ones(std::vector<int64_t> shape, bool requires_grad) {
    Tensor t = zeros(std::move(shape), requires_grad);
    cpu::fill(t.data_ptr(), 1.0f, t.numel());
    return t;
}

Tensor Tensor::randn(std::vector<int64_t> shape, float std, bool requires_grad, uint64_t seed) {
    Tensor t = zeros(std::move(shape), requires_grad);
    std::mt19937 rng(seed ? seed : std::random_device{}());
    std::normal_distribution<float> dist(0.0f, std);
    float* p = t.data_ptr();
    for (int64_t i = 0; i < t.numel(); ++i) p[i] = dist(rng);
    return t;
}

Tensor Tensor::from_flat(std::vector<float> values, std::vector<int64_t> shape, bool requires_grad) {
    Tensor t = zeros(std::move(shape), requires_grad);
    if (static_cast<int64_t>(values.size()) != t.numel())
        throw std::runtime_error("from_flat: data length does not match shape");
    cpu::copy(values.data(), t.data_ptr(), t.numel());
    return t;
}

Tensor Tensor::zeros_like(const Tensor& other) { return zeros(other.shape(), false); }
Tensor Tensor::ones_like(const Tensor& other) { return ones(other.shape(), false); }

// ---------------------------------------------------------------------
// Gradient bookkeeping
// ---------------------------------------------------------------------

std::optional<Tensor> Tensor::grad() const {
    if (!impl_->grad) return std::nullopt;
    return Tensor(impl_->grad);
}

void Tensor::zero_grad() { impl_->grad = nullptr; }

void Tensor::add_(const Tensor& other, float alpha) {
    if (other.numel() != numel())
        throw std::runtime_error("add_: size mismatch");
    cpu::axpy_(data_ptr(), other.data_ptr(), alpha, numel());
}

Tensor Tensor::detach() const {
    auto data = std::make_shared<TensorData>();
    data->storage = impl_->storage;
    data->shape = impl_->shape;
    data->dtype = impl_->dtype;
    data->requires_grad = false;
    // grad and grad_node both default-null (never copied) -- that's
    // exactly what "cut from the graph" means.
    return Tensor(data);
}

std::vector<float> Tensor::to_vector() const {
    std::vector<float> out(static_cast<size_t>(numel()));
    cpu::copy(data_ptr(), out.data(), numel());
    return out;
}

// ---------------------------------------------------------------------
// Ops — each computes the forward value with a raw cpu:: kernel, then (if
// any input requires grad) attaches a GradNode whose backward_fn is the
// op's vector-Jacobian product.
// ---------------------------------------------------------------------

Tensor Tensor::add(const Tensor& other) const {
    const Tensor& a = *this;
    const Tensor& b = other;

    // Three paths, fastest first: the fixed 2D bias case (its own
    // dedicated kernel, and what elementwise_fusion/the Metal backend's
    // fused_bias_relu/add_bias specifically pattern-match -- unchanged,
    // since real fused/GPU fast paths depend on this exact shape check
    // staying exactly what it was), an exact shape match (the other
    // pre-existing fast path), and -- new -- any other broadcast-
    // compatible shape pair, via the general kernel. See
    // broadcast_shapes's own comment for the rule; it throws if the
    // shapes don't broadcast at all.
    bool bias_broadcast = (b.ndim() == 1 && a.ndim() == 2 && a.shape()[1] == b.shape()[0]);
    bool exact = (a.shape() == b.shape());
    bool general_broadcast = !bias_broadcast && !exact;

    std::vector<int64_t> out_shape = general_broadcast ? broadcast_shapes(a.shape(), b.shape(), "add") : a.shape();

    Tensor out = Tensor::zeros(out_shape, false);
    if (bias_broadcast)
        cpu::add_bias_broadcast(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.shape()[0], a.shape()[1]);
    else if (exact)
        cpu::add(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel());
    else
        cpu::add_broadcast(a.data_ptr(), a.shape().data(), a.ndim(), b.data_ptr(), b.shape().data(), b.ndim(),
                            out_shape.data(), static_cast<int64_t>(out_shape.size()), out.data_ptr());

    if (a.requires_grad() || b.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "add";
        node->inputs = {a, b};
        auto a_shape = a.shape();
        auto b_shape = b.shape();
        node->backward_fn = [bias_broadcast, general_broadcast, a_shape, b_shape](
                                 const Tensor& grad_output) -> std::vector<Tensor> {
            if (bias_broadcast) {
                Tensor grad_b = Tensor::zeros({grad_output.shape()[1]}, false);
                cpu::sum_over_batch(grad_output.data_ptr(), grad_b.data_ptr(),
                                     grad_output.shape()[0], grad_output.shape()[1]);
                return {grad_output, grad_b};
            }
            if (!general_broadcast) return {grad_output, grad_output};

            // General case: whichever operand didn't already have
            // grad_output's own shape got there by broadcasting, so its
            // gradient is grad_output summed back down over every axis
            // that broadcast -- reduce_to_shape is a no-op (single full
            // pass, no actual reduction) when a shape already matches,
            // but skip it outright in that case anyway rather than pay
            // for a copy neither operand needs.
            Tensor grad_a = (a_shape == grad_output.shape())
                                 ? grad_output
                                 : Tensor::zeros(a_shape, false);
            if (a_shape != grad_output.shape())
                cpu::reduce_to_shape(grad_output.data_ptr(), grad_output.shape().data(), grad_output.ndim(),
                                      a_shape.data(), static_cast<int64_t>(a_shape.size()), grad_a.data_ptr());
            Tensor grad_b = (b_shape == grad_output.shape())
                                 ? grad_output
                                 : Tensor::zeros(b_shape, false);
            if (b_shape != grad_output.shape())
                cpu::reduce_to_shape(grad_output.data_ptr(), grad_output.shape().data(), grad_output.ndim(),
                                      b_shape.data(), static_cast<int64_t>(b_shape.size()), grad_b.data_ptr());
            return {grad_a, grad_b};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

Tensor Tensor::sub(const Tensor& other) const {
    const Tensor& a = *this;
    const Tensor& b = other;

    bool exact = (a.shape() == b.shape());
    std::vector<int64_t> out_shape = exact ? a.shape() : broadcast_shapes(a.shape(), b.shape(), "sub");

    Tensor out = Tensor::zeros(out_shape, false);
    if (exact)
        cpu::sub(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel());
    else
        cpu::sub_broadcast(a.data_ptr(), a.shape().data(), a.ndim(), b.data_ptr(), b.shape().data(), b.ndim(),
                            out_shape.data(), static_cast<int64_t>(out_shape.size()), out.data_ptr());

    if (a.requires_grad() || b.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "sub";
        node->inputs = {a, b};
        auto a_shape = a.shape();
        auto b_shape = b.shape();
        node->backward_fn = [exact, a_shape, b_shape](const Tensor& grad_output) -> std::vector<Tensor> {
            Tensor neg = Tensor::zeros(grad_output.shape(), false);
            cpu::axpy_(neg.data_ptr(), grad_output.data_ptr(), -1.0f, grad_output.numel());
            if (exact) return {grad_output, neg};

            Tensor grad_a = (a_shape == grad_output.shape()) ? grad_output : Tensor::zeros(a_shape, false);
            if (a_shape != grad_output.shape())
                cpu::reduce_to_shape(grad_output.data_ptr(), grad_output.shape().data(), grad_output.ndim(),
                                      a_shape.data(), static_cast<int64_t>(a_shape.size()), grad_a.data_ptr());
            Tensor grad_b = (b_shape == neg.shape()) ? neg : Tensor::zeros(b_shape, false);
            if (b_shape != neg.shape())
                cpu::reduce_to_shape(neg.data_ptr(), neg.shape().data(), neg.ndim(),
                                      b_shape.data(), static_cast<int64_t>(b_shape.size()), grad_b.data_ptr());
            return {grad_a, grad_b};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

Tensor Tensor::mul(const Tensor& other) const {
    const Tensor& a = *this;
    const Tensor& b = other;

    bool exact = (a.shape() == b.shape());
    std::vector<int64_t> out_shape = exact ? a.shape() : broadcast_shapes(a.shape(), b.shape(), "mul");

    Tensor out = Tensor::zeros(out_shape, false);
    if (exact)
        cpu::mul(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel());
    else
        cpu::mul_broadcast(a.data_ptr(), a.shape().data(), a.ndim(), b.data_ptr(), b.shape().data(), b.ndim(),
                            out_shape.data(), static_cast<int64_t>(out_shape.size()), out.data_ptr());

    if (a.requires_grad() || b.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "mul";
        node->inputs = {a, b};
        node->backward_fn = [a, b, exact, out_shape](const Tensor& grad_output) -> std::vector<Tensor> {
            if (exact) {
                Tensor grad_a = Tensor::zeros(a.shape(), false);
                Tensor grad_b = Tensor::zeros(b.shape(), false);
                cpu::mul(grad_output.data_ptr(), b.data_ptr(), grad_a.data_ptr(), a.numel());
                cpu::mul(grad_output.data_ptr(), a.data_ptr(), grad_b.data_ptr(), a.numel());
                return {grad_a, grad_b};
            }

            // General case: d/da(a*b) = grad_output * b, d/db(a*b) =
            // grad_output * a -- computed at the full broadcast shape
            // first (mul_broadcast handles b or a broadcasting up to
            // grad_output's own shape, exactly the forward op's own
            // logic run again), then reduced down to each operand's
            // real shape.
            Tensor grad_a_full = Tensor::zeros(out_shape, false);
            cpu::mul_broadcast(grad_output.data_ptr(), grad_output.shape().data(), grad_output.ndim(),
                                b.data_ptr(), b.shape().data(), b.ndim(),
                                out_shape.data(), static_cast<int64_t>(out_shape.size()), grad_a_full.data_ptr());
            Tensor grad_b_full = Tensor::zeros(out_shape, false);
            cpu::mul_broadcast(grad_output.data_ptr(), grad_output.shape().data(), grad_output.ndim(),
                                a.data_ptr(), a.shape().data(), a.ndim(),
                                out_shape.data(), static_cast<int64_t>(out_shape.size()), grad_b_full.data_ptr());

            Tensor grad_a = (a.shape() == out_shape) ? grad_a_full : Tensor::zeros(a.shape(), false);
            if (a.shape() != out_shape)
                cpu::reduce_to_shape(grad_a_full.data_ptr(), out_shape.data(), static_cast<int64_t>(out_shape.size()),
                                      a.shape().data(), a.ndim(), grad_a.data_ptr());
            Tensor grad_b = (b.shape() == out_shape) ? grad_b_full : Tensor::zeros(b.shape(), false);
            if (b.shape() != out_shape)
                cpu::reduce_to_shape(grad_b_full.data_ptr(), out_shape.data(), static_cast<int64_t>(out_shape.size()),
                                      b.shape().data(), b.ndim(), grad_b.data_ptr());
            return {grad_a, grad_b};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

namespace {
Tensor compare(const Tensor& a, const Tensor& b, const char* op_name,
               void (*broadcast_kernel)(const float*, const int64_t*, int64_t, const float*, const int64_t*,
                                         int64_t, const int64_t*, int64_t, float*)) {
    bool exact = (a.shape() == b.shape());
    std::vector<int64_t> out_shape = exact ? a.shape() : broadcast_shapes(a.shape(), b.shape(), op_name);
    Tensor out = Tensor::zeros(out_shape, false);
    broadcast_kernel(a.data_ptr(), a.shape().data(), a.ndim(), b.data_ptr(), b.shape().data(), b.ndim(),
                      out_shape.data(), static_cast<int64_t>(out_shape.size()), out.data_ptr());
    return out;
}
} // namespace

Tensor Tensor::gt(const Tensor& other) const { return compare(*this, other, "gt", cpu::greater_broadcast); }
Tensor Tensor::lt(const Tensor& other) const { return compare(*this, other, "lt", cpu::less_broadcast); }
Tensor Tensor::eq(const Tensor& other) const { return compare(*this, other, "eq", cpu::equal_broadcast); }

Tensor Tensor::matmul(const Tensor& other) const {
    const Tensor& a = *this;
    const Tensor& b = other;

    if (a.ndim() == 2 && b.ndim() == 2) {
        // The common case keeps its own exact, unchanged fast path --
        // no batch-broadcast bookkeeping, no behavior change from
        // before batched matmul existed.
        if (a.shape()[1] != b.shape()[0])
            throw std::runtime_error("matmul: incompatible shapes");

        int64_t M = a.shape()[0], K = a.shape()[1], N = b.shape()[1];
        Tensor out = Tensor::zeros({M, N}, false);
        cpu::matmul(a.data_ptr(), b.data_ptr(), out.data_ptr(), M, K, N);

        if (a.requires_grad() || b.requires_grad()) {
            auto node = std::make_shared<GradNode>();
            node->name = "matmul";
            node->inputs = {a, b};
            node->backward_fn = [a, b, M, K, N](const Tensor& grad_output) -> std::vector<Tensor> {
                Tensor grad_a = Tensor::zeros({M, K}, false);
                cpu::matmul_nt(grad_output.data_ptr(), b.data_ptr(), grad_a.data_ptr(), M, N, K);
                Tensor grad_b = Tensor::zeros({K, N}, false);
                cpu::matmul_tn(a.data_ptr(), grad_output.data_ptr(), grad_b.data_ptr(), M, K, N);
                return {grad_a, grad_b};
            };
            out.set_grad_node(node);
            out.set_requires_grad(true);
        }
        return out;
    }

    // Batched path: at least one operand has rank > 2. The trailing two
    // dims of each are the real matrix dims (M,K) and (K,N); everything
    // before that broadcasts via the same NumPy-style rule add/sub/mul's
    // own broadcast_shapes already implements (a rank-2 operand simply
    // has an empty/rank-0 batch shape, which broadcasts against any
    // batch shape by reading its one matrix repeatedly -- the "a shared
    // weight applied across a batch" case).
    if (a.ndim() < 2 || b.ndim() < 2)
        throw std::runtime_error("matmul: both operands must be at least 2D");
    int64_t M = a.shape()[a.ndim() - 2], Ka = a.shape()[a.ndim() - 1];
    int64_t Kb = b.shape()[b.ndim() - 2], N = b.shape()[b.ndim() - 1];
    if (Ka != Kb)
        throw std::runtime_error("matmul: inner dimensions don't match");
    int64_t K = Ka;

    std::vector<int64_t> a_batch(a.shape().begin(), a.shape().end() - 2);
    std::vector<int64_t> b_batch(b.shape().begin(), b.shape().end() - 2);
    std::vector<int64_t> out_batch = broadcast_shapes(a_batch, b_batch, "matmul");

    std::vector<int64_t> out_shape = out_batch;
    out_shape.push_back(M);
    out_shape.push_back(N);

    Tensor out = Tensor::zeros(out_shape, false);
    cpu::batched_matmul(a.data_ptr(), a_batch.data(), static_cast<int64_t>(a_batch.size()), b.data_ptr(),
                         b_batch.data(), static_cast<int64_t>(b_batch.size()), out_batch.data(),
                         static_cast<int64_t>(out_batch.size()), M, K, N, out.data_ptr());

    if (a.requires_grad() || b.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "matmul";
        node->inputs = {a, b};
        auto a_shape = a.shape();
        auto b_shape = b.shape();
        node->backward_fn = [a, b, a_shape, b_shape, a_batch, b_batch, out_batch, M, K, N](
                                 const Tensor& grad_output) -> std::vector<Tensor> {
            // grad_a_full (out_batch,M,K) = grad_output (out_batch,M,N)
            // @ b(out_batch,K,N)^T, at the FULL broadcast batch shape
            // first (mirroring the 2D case's matmul_nt call, batched);
            // reduced down to a's own (possibly smaller) batch shape
            // afterward, exactly the same "compute at the broadcast
            // shape, then reduce_to_shape" pattern general add/sub/mul
            // broadcasting already established.
            std::vector<int64_t> full_a_shape = out_batch;
            full_a_shape.push_back(M);
            full_a_shape.push_back(K);
            Tensor grad_a_full = Tensor::zeros(full_a_shape, false);
            cpu::batched_matmul_nt(grad_output.data_ptr(), out_batch.data(), static_cast<int64_t>(out_batch.size()),
                                    b.data_ptr(), b_batch.data(), static_cast<int64_t>(b_batch.size()),
                                    out_batch.data(), static_cast<int64_t>(out_batch.size()), M, N, K,
                                    grad_a_full.data_ptr());

            std::vector<int64_t> full_b_shape = out_batch;
            full_b_shape.push_back(K);
            full_b_shape.push_back(N);
            Tensor grad_b_full = Tensor::zeros(full_b_shape, false);
            cpu::batched_matmul_tn(a.data_ptr(), a_batch.data(), static_cast<int64_t>(a_batch.size()),
                                    grad_output.data_ptr(), out_batch.data(), static_cast<int64_t>(out_batch.size()),
                                    out_batch.data(), static_cast<int64_t>(out_batch.size()), M, K, N,
                                    grad_b_full.data_ptr());

            Tensor grad_a = (a_shape == full_a_shape) ? grad_a_full : Tensor::zeros(a_shape, false);
            if (a_shape != full_a_shape)
                cpu::reduce_to_shape(grad_a_full.data_ptr(), full_a_shape.data(),
                                      static_cast<int64_t>(full_a_shape.size()), a_shape.data(), a.ndim(),
                                      grad_a.data_ptr());
            Tensor grad_b = (b_shape == full_b_shape) ? grad_b_full : Tensor::zeros(b_shape, false);
            if (b_shape != full_b_shape)
                cpu::reduce_to_shape(grad_b_full.data_ptr(), full_b_shape.data(),
                                      static_cast<int64_t>(full_b_shape.size()), b_shape.data(), b.ndim(),
                                      grad_b.data_ptr());
            return {grad_a, grad_b};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

Tensor Tensor::relu() const {
    const Tensor& x = *this;
    Tensor out = Tensor::zeros(x.shape(), false);
    cpu::relu_fwd(x.data_ptr(), out.data_ptr(), x.numel());

    if (x.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "relu";
        node->inputs = {x};
        node->backward_fn = [x](const Tensor& grad_output) -> std::vector<Tensor> {
            Tensor grad_x = Tensor::zeros(x.shape(), false);
            cpu::relu_bwd(x.data_ptr(), grad_output.data_ptr(), grad_x.data_ptr(), x.numel());
            return {grad_x};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

Tensor Tensor::sum() const {
    const Tensor& x = *this;
    Tensor out = Tensor::zeros({1}, false);
    out.data_ptr()[0] = cpu::reduce_sum(x.data_ptr(), x.numel());

    if (x.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "sum";
        node->inputs = {x};
        node->backward_fn = [x](const Tensor& grad_output) -> std::vector<Tensor> {
            Tensor grad_x = Tensor::zeros(x.shape(), false);
            cpu::fill(grad_x.data_ptr(), grad_output.data_ptr()[0], grad_x.numel());
            return {grad_x};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

Tensor Tensor::conv2d(const Tensor& weight, const Tensor& bias, int64_t stride, int64_t padding) const {
    const Tensor& x = *this;
    if (x.ndim() != 4)
        throw std::runtime_error("conv2d: input must be 4D (N, Cin, H, W)");
    if (weight.ndim() != 4)
        throw std::runtime_error("conv2d: weight must be 4D (Cout, Cin, kH, kW)");

    int64_t N = x.shape()[0], Cin = x.shape()[1], H = x.shape()[2], W = x.shape()[3];
    int64_t Cout = weight.shape()[0], Cin_w = weight.shape()[1], kH = weight.shape()[2], kW = weight.shape()[3];
    if (Cin != Cin_w)
        throw std::runtime_error("conv2d: input and weight channel counts don't match");
    if (bias.numel() != Cout)
        throw std::runtime_error("conv2d: bias must have Cout elements");

    int64_t Hout = (H + 2 * padding - kH) / stride + 1;
    int64_t Wout = (W + 2 * padding - kW) / stride + 1;
    if (Hout <= 0 || Wout <= 0)
        throw std::runtime_error("conv2d: kernel/stride/padding produce a non-positive output size");
    int64_t HWout = Hout * Wout;
    int64_t colRows = Cin * kH * kW;

    Tensor out = Tensor::zeros({N, Cout, Hout, Wout}, false);

    // One im2col + one matmul per batch item -- reusing the exact matmul
    // kernel (Accelerate-backed on this platform) every other op in this
    // codebase already goes through, rather than a hand-written
    // convolution inner loop.
    std::vector<float> col(static_cast<size_t>(colRows * HWout));
    for (int64_t n = 0; n < N; ++n) {
        const float* xn = x.data_ptr() + n * Cin * H * W;
        float* yn = out.data_ptr() + n * Cout * HWout;
        cpu::im2col(xn, col.data(), Cin, H, W, kH, kW, stride, padding, Hout, Wout);
        cpu::matmul(weight.data_ptr(), col.data(), yn, Cout, colRows, HWout);
    }
    cpu::add_bias_nchw(out.data_ptr(), bias.data_ptr(), out.data_ptr(), N, Cout, HWout);

    if (x.requires_grad() || weight.requires_grad() || bias.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "conv2d";
        node->inputs = {x, weight, bias};
        node->backward_fn = [x, weight, N, Cin, H, W, Cout, kH, kW, stride, padding, Hout, Wout, HWout,
                              colRows](const Tensor& grad_output) -> std::vector<Tensor> {
            Tensor grad_x = Tensor::zeros({N, Cin, H, W}, false);
            Tensor grad_w = Tensor::zeros({Cout, Cin, kH, kW}, false);
            Tensor grad_b = Tensor::zeros({Cout}, false);

            cpu::sum_over_batch_and_spatial(grad_output.data_ptr(), grad_b.data_ptr(), N, Cout, HWout);

            std::vector<float> col(static_cast<size_t>(colRows * HWout));
            std::vector<float> dcol(static_cast<size_t>(colRows * HWout));
            std::vector<float> grad_w_step(static_cast<size_t>(Cout * colRows));

            for (int64_t n = 0; n < N; ++n) {
                const float* xn = x.data_ptr() + n * Cin * H * W;
                const float* dyn = grad_output.data_ptr() + n * Cout * HWout;
                float* dxn = grad_x.data_ptr() + n * Cin * H * W;

                // grad_w's im2col needs the SAME col matrix the forward
                // pass built for this batch item -- recomputed here
                // (not cached from forward) to keep this closure's only
                // captured state the inputs themselves, matching every
                // other op's backward_fn in this file.
                cpu::im2col(xn, col.data(), Cin, H, W, kH, kW, stride, padding, Hout, Wout);

                // grad_w += dy_flat(Cout,HWout) @ col(colRows,HWout)^T -> (Cout,colRows)
                cpu::matmul_nt(dyn, col.data(), grad_w_step.data(), Cout, HWout, colRows);
                cpu::axpy_(grad_w.data_ptr(), grad_w_step.data(), 1.0f, Cout * colRows);

                // dcol = weight_flat(Cout,colRows)^T @ dy_flat(Cout,HWout) -> (colRows,HWout)
                cpu::matmul_tn(weight.data_ptr(), dyn, dcol.data(), Cout, colRows, HWout);
                cpu::col2im(dcol.data(), dxn, Cin, H, W, kH, kW, stride, padding, Hout, Wout);
            }

            return {grad_x, grad_w, grad_b};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

Tensor Tensor::max_pool2d(int64_t kernel_size, int64_t stride) const {
    const Tensor& x = *this;
    if (x.ndim() != 4)
        throw std::runtime_error("max_pool2d: input must be 4D (N, C, H, W)");

    int64_t N = x.shape()[0], C = x.shape()[1], H = x.shape()[2], W = x.shape()[3];
    int64_t Hout = (H - kernel_size) / stride + 1;
    int64_t Wout = (W - kernel_size) / stride + 1;
    if (Hout <= 0 || Wout <= 0)
        throw std::runtime_error("max_pool2d: kernel_size/stride produce a non-positive output size");

    Tensor out = Tensor::zeros({N, C, Hout, Wout}, false);
    // Shared (not copied) into the backward_fn closure below -- cheap
    // (a refcount bump, not an N*C*Hout*Wout-sized copy) and correct,
    // since nothing else ever mutates this buffer after forward fills it.
    auto argmax = std::make_shared<std::vector<int64_t>>(static_cast<size_t>(N * C * Hout * Wout));
    cpu::maxpool2d_fwd(x.data_ptr(), out.data_ptr(), argmax->data(), N, C, H, W, kernel_size, stride, Hout, Wout);

    if (x.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "max_pool2d";
        node->inputs = {x};
        auto x_shape = x.shape();
        node->backward_fn = [x_shape, argmax, N, C, H, W, Hout, Wout](
                                 const Tensor& grad_output) -> std::vector<Tensor> {
            Tensor grad_x = Tensor::zeros(x_shape, false);
            cpu::maxpool2d_bwd(grad_output.data_ptr(), argmax->data(), grad_x.data_ptr(), N, C, H, W, Hout, Wout);
            return {grad_x};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

Tensor Tensor::mean() const {
    const Tensor& x = *this;
    Tensor out = Tensor::zeros({1}, false);
    int64_t n = x.numel();
    out.data_ptr()[0] = cpu::reduce_sum(x.data_ptr(), n) / static_cast<float>(n);

    if (x.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "mean";
        node->inputs = {x};
        node->backward_fn = [x, n](const Tensor& grad_output) -> std::vector<Tensor> {
            Tensor grad_x = Tensor::zeros(x.shape(), false);
            cpu::fill(grad_x.data_ptr(), grad_output.data_ptr()[0] / static_cast<float>(n), grad_x.numel());
            return {grad_x};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

Tensor Tensor::sqrt() const {
    const Tensor& x = *this;
    Tensor out = Tensor::zeros(x.shape(), false);
    cpu::sqrt_fwd(x.data_ptr(), out.data_ptr(), x.numel());

    if (x.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "sqrt";
        node->inputs = {x};
        // Captures a plain std::vector snapshot of out's VALUES, not
        // `out` itself -- see this file's own note by Tensor::exp's
        // fix for why: capturing `out` here would make this closure
        // (owned by out's own GradNode, which out.impl_ owns) hold a
        // reference back to out.impl_, a genuine shared_ptr cycle
        // nothing ever breaks. Semantically identical (out's values
        // never change after construction) at the cost of one extra
        // copy, the same trade sqrt_bwd's siblings below all make now.
        auto out_shape = out.shape();
        auto out_vals = out.to_vector();
        node->backward_fn = [out_shape, out_vals](const Tensor& grad_output) -> std::vector<Tensor> {
            Tensor grad_x = Tensor::zeros(out_shape, false);
            cpu::sqrt_bwd(out_vals.data(), grad_output.data_ptr(), grad_x.data_ptr(), grad_x.numel());
            return {grad_x};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

Tensor Tensor::reciprocal() const {
    const Tensor& x = *this;
    Tensor out = Tensor::zeros(x.shape(), false);
    cpu::reciprocal_fwd(x.data_ptr(), out.data_ptr(), x.numel());

    if (x.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "reciprocal";
        node->inputs = {x};
        // See sqrt()'s own fix above for why this is out.to_vector(),
        // not out itself: capturing `out` (which out's OWN GradNode
        // ends up owned by) creates a shared_ptr cycle refcounting
        // never breaks -- a real, permanent per-call leak this
        // project's own memory-scaling investigation caught.
        auto out_shape = out.shape();
        auto out_vals = out.to_vector();
        node->backward_fn = [out_shape, out_vals](const Tensor& grad_output) -> std::vector<Tensor> {
            Tensor grad_x = Tensor::zeros(out_shape, false);
            cpu::reciprocal_bwd(out_vals.data(), grad_output.data_ptr(), grad_x.data_ptr(), grad_x.numel());
            return {grad_x};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

Tensor Tensor::div(const Tensor& other) const {
    return this->mul(other.reciprocal());
}

Tensor Tensor::exp() const {
    const Tensor& x = *this;
    Tensor out = Tensor::zeros(x.shape(), false);
    cpu::exp_fwd(x.data_ptr(), out.data_ptr(), x.numel());

    if (x.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "exp";
        node->inputs = {x};
        // d/dx exp(x) = exp(x) = out -- exp's own vjp is exactly a
        // multiply by its own forward output, no dedicated bwd kernel
        // needed. Captures out.to_vector() (a plain data copy), NOT
        // `out` itself: `out`'s own GradNode -- this very closure --
        // is owned by out.impl_, so capturing `out` here would put a
        // second reference to out.impl_ INSIDE the closure that
        // out.impl_ itself owns: a genuine shared_ptr cycle (TensorData
        // -> GradNode -> captured Tensor -> the same TensorData)
        // nothing in this project ever breaks -- every differentiable
        // exp() call leaked its entire output permanently until this
        // fix (found via this project's own memory-scaling
        // investigation: softmax, used in every attention layer,
        // composes exp() internally, and Adam/AdamW's sqrt() call --
        // same bug, fixed alongside this one -- meant every training
        // step using either optimizer had been leaking this whole
        // project).
        auto out_shape = out.shape();
        auto out_vals = out.to_vector();
        node->backward_fn = [out_shape, out_vals](const Tensor& grad_output) -> std::vector<Tensor> {
            Tensor grad_x = Tensor::zeros(out_shape, false);
            cpu::mul(grad_output.data_ptr(), out_vals.data(), grad_x.data_ptr(), grad_x.numel());
            return {grad_x};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

Tensor Tensor::log() const {
    const Tensor& x = *this;
    Tensor out = Tensor::zeros(x.shape(), false);
    cpu::log_fwd(x.data_ptr(), out.data_ptr(), x.numel());

    if (x.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "log";
        node->inputs = {x};
        node->backward_fn = [x](const Tensor& grad_output) -> std::vector<Tensor> {
            // d/dx log(x) = 1/x -- composed from the existing
            // reciprocal kernel + mul rather than its own bwd kernel.
            Tensor recip_x = Tensor::zeros(x.shape(), false);
            cpu::reciprocal_fwd(x.data_ptr(), recip_x.data_ptr(), x.numel());
            Tensor grad_x = Tensor::zeros(x.shape(), false);
            cpu::mul(grad_output.data_ptr(), recip_x.data_ptr(), grad_x.data_ptr(), x.numel());
            return {grad_x};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

Tensor Tensor::tanh() const {
    const Tensor& x = *this;
    Tensor out = Tensor::zeros(x.shape(), false);
    cpu::tanh_fwd(x.data_ptr(), out.data_ptr(), x.numel());

    if (x.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "tanh";
        node->inputs = {x};
        // See exp()'s own fix above for the shared_ptr cycle this
        // avoids by capturing out.to_vector() rather than out itself.
        auto out_shape = out.shape();
        auto out_vals = out.to_vector();
        node->backward_fn = [out_shape, out_vals](const Tensor& grad_output) -> std::vector<Tensor> {
            Tensor grad_x = Tensor::zeros(out_shape, false);
            cpu::tanh_bwd(out_vals.data(), grad_output.data_ptr(), grad_x.data_ptr(), grad_x.numel());
            return {grad_x};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

Tensor Tensor::sigmoid() const {
    const Tensor& x = *this;
    Tensor out = Tensor::zeros(x.shape(), false);
    cpu::sigmoid_fwd(x.data_ptr(), out.data_ptr(), x.numel());

    if (x.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "sigmoid";
        node->inputs = {x};
        // See exp()'s own fix above for the shared_ptr cycle this
        // avoids by capturing out.to_vector() rather than out itself.
        auto out_shape = out.shape();
        auto out_vals = out.to_vector();
        node->backward_fn = [out_shape, out_vals](const Tensor& grad_output) -> std::vector<Tensor> {
            Tensor grad_x = Tensor::zeros(out_shape, false);
            cpu::sigmoid_bwd(out_vals.data(), grad_output.data_ptr(), grad_x.data_ptr(), grad_x.numel());
            return {grad_x};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

Tensor Tensor::gelu() const {
    const Tensor& x = *this;
    Tensor out = Tensor::zeros(x.shape(), false);
    cpu::gelu_fwd(x.data_ptr(), out.data_ptr(), x.numel());

    if (x.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "gelu";
        node->inputs = {x};
        node->backward_fn = [x](const Tensor& grad_output) -> std::vector<Tensor> {
            Tensor grad_x = Tensor::zeros(x.shape(), false);
            cpu::gelu_bwd(x.data_ptr(), grad_output.data_ptr(), grad_x.data_ptr(), x.numel());
            return {grad_x};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

Tensor Tensor::leaky_relu(float negative_slope) const {
    const Tensor& x = *this;
    Tensor out = Tensor::zeros(x.shape(), false);
    cpu::leaky_relu_fwd(x.data_ptr(), out.data_ptr(), x.numel(), negative_slope);

    if (x.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "leaky_relu";
        node->inputs = {x};
        node->backward_fn = [x, negative_slope](const Tensor& grad_output) -> std::vector<Tensor> {
            Tensor grad_x = Tensor::zeros(x.shape(), false);
            cpu::leaky_relu_bwd(x.data_ptr(), grad_output.data_ptr(), grad_x.data_ptr(), x.numel(), negative_slope);
            return {grad_x};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

Tensor Tensor::sum(int64_t dim, bool keepdim) const {
    const Tensor& x = *this;
    int64_t nd = x.ndim();
    if (dim < 0 || dim >= nd)
        throw std::runtime_error("sum(dim): dim out of range");

    auto reduced_shape = x.shape();
    reduced_shape[dim] = 1;

    Tensor out = Tensor::zeros(reduced_shape, false);
    cpu::reduce_to_shape(x.data_ptr(), x.shape().data(), nd, reduced_shape.data(), nd, out.data_ptr());

    if (x.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "sum_dim";
        node->inputs = {x};
        auto x_shape = x.shape();
        node->backward_fn = [x_shape, reduced_shape, nd](const Tensor& grad_output) -> std::vector<Tensor> {
            Tensor grad_x = Tensor::zeros(x_shape, false);
            cpu::broadcast_to_shape(grad_output.data_ptr(), reduced_shape.data(), nd,
                                     x_shape.data(), nd, grad_x.data_ptr());
            return {grad_x};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }

    if (keepdim) return out;
    auto squeezed = x.shape();
    squeezed.erase(squeezed.begin() + dim);
    return out.reshape(squeezed);
}

Tensor Tensor::mean(int64_t dim, bool keepdim) const {
    const Tensor& x = *this;
    int64_t nd = x.ndim();
    if (dim < 0 || dim >= nd)
        throw std::runtime_error("mean(dim): dim out of range");
    int64_t dim_size = x.shape()[dim];

    // Composed from sum(dim) [above] + an existing broadcast-mul by a
    // shape-[1] scalar -- no dedicated kernel or backward of its own;
    // both already-autograd-aware ops carry the gradient correctly.
    Tensor summed = x.sum(dim, true);
    Tensor scale = Tensor::from_flat(std::vector<float>{1.0f / static_cast<float>(dim_size)},
                                      std::vector<int64_t>{1}, false);
    Tensor result = summed.mul(scale);

    if (keepdim) return result;
    auto squeezed = x.shape();
    squeezed.erase(squeezed.begin() + dim);
    return result.reshape(squeezed);
}

Tensor Tensor::max(int64_t dim, bool keepdim) const {
    const Tensor& x = *this;
    int64_t nd = x.ndim();
    if (dim < 0 || dim >= nd)
        throw std::runtime_error("max(dim): dim out of range");

    auto reduced_shape = x.shape();
    reduced_shape[dim] = 1;
    Tensor out = Tensor::zeros(reduced_shape, false);
    cpu::max_along_dim(x.data_ptr(), x.shape().data(), nd, dim, out.data_ptr());
    // Deliberately no GradNode attached here, regardless of
    // x.requires_grad() -- see this method's own declaration in
    // Tensor.hpp for why a gradient through max(dim) isn't needed (or
    // provided).

    if (keepdim) return out;
    auto squeezed = x.shape();
    squeezed.erase(squeezed.begin() + dim);
    return out.reshape(squeezed);
}

Tensor Tensor::softmax(int64_t dim) const {
    const Tensor& x = *this;
    Tensor m = x.max(dim, true);
    Tensor shifted = x.sub(m);
    Tensor exp_shifted = shifted.exp();
    Tensor denom = exp_shifted.sum(dim, true);
    return exp_shifted.div(denom);
}

Tensor Tensor::cross_entropy(const Tensor& targets) const {
    const Tensor& logits = *this;
    if (logits.ndim() != 2 || logits.shape() != targets.shape())
        throw std::runtime_error("cross_entropy: logits and targets must both be (batch, classes) and the same shape");

    // logsumexp(logits, dim=1) - sum(logits * targets, dim=1), never
    // softmax(logits).log() -- see this method's own declaration in
    // Tensor.hpp for why the log-sum-exp form is the numerically stable
    // one and the naive composition isn't.
    Tensor m = logits.max(1, true);
    Tensor shifted = logits.sub(m);
    Tensor lse = shifted.exp().sum(1, true).log().add(m);
    Tensor picked = logits.mul(targets).sum(1, true);
    Tensor per_example = lse.sub(picked);
    return per_example.mean();
}

Tensor Tensor::index_select(int64_t dim, const std::vector<int64_t>& indices) const {
    const Tensor& x = *this;
    int64_t nd = x.ndim();
    if (dim < 0 || dim >= nd)
        throw std::runtime_error("index_select: dim out of range");
    int64_t dim_size = x.shape()[dim];
    for (int64_t idx : indices)
        if (idx < 0 || idx >= dim_size)
            throw std::runtime_error("index_select: index out of range for dim of size " +
                                      std::to_string(dim_size));

    auto out_shape = x.shape();
    out_shape[dim] = static_cast<int64_t>(indices.size());
    Tensor out = Tensor::zeros(out_shape, false);
    cpu::index_select(x.data_ptr(), x.shape().data(), nd, dim, indices.data(),
                       static_cast<int64_t>(indices.size()), out.data_ptr());

    if (x.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "index_select";
        node->inputs = {x};
        auto x_shape = x.shape();
        node->backward_fn = [x_shape, dim, indices, nd](const Tensor& grad_output) -> std::vector<Tensor> {
            Tensor grad_x = Tensor::zeros(x_shape, false);
            cpu::index_select_bwd(grad_output.data_ptr(), x_shape.data(), nd, dim, indices.data(),
                                   static_cast<int64_t>(indices.size()), grad_x.data_ptr());
            return {grad_x};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

Tensor Tensor::reshape(std::vector<int64_t> new_shape) const {
    const Tensor& x = *this;
    int64_t n = numel_of(new_shape);
    if (n != x.numel())
        throw std::runtime_error("reshape: number of elements must match");

    Tensor out = Tensor::zeros(std::move(new_shape), false);
    cpu::copy(x.data_ptr(), out.data_ptr(), n);

    if (x.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "reshape";
        node->inputs = {x};
        auto orig_shape = x.shape();
        node->backward_fn = [orig_shape](const Tensor& grad_output) -> std::vector<Tensor> {
            Tensor grad_x = Tensor::zeros(orig_shape, false);
            cpu::copy(grad_output.data_ptr(), grad_x.data_ptr(), grad_x.numel());
            return {grad_x};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    const Tensor& x = *this;
    int64_t nd = x.ndim();
    if (dim0 < 0 || dim0 >= nd || dim1 < 0 || dim1 >= nd)
        throw std::runtime_error("transpose: dim out of range");

    auto out_shape = x.shape();
    std::swap(out_shape[dim0], out_shape[dim1]);
    Tensor out = Tensor::zeros(out_shape, false);
    cpu::transpose(x.data_ptr(), x.shape().data(), nd, dim0, dim1, out.data_ptr());

    if (x.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "transpose";
        node->inputs = {x};
        auto x_shape = x.shape();
        node->backward_fn = [x_shape, out_shape, dim0, dim1, nd](const Tensor& grad_output) -> std::vector<Tensor> {
            // transpose swapping the same two axes is its own inverse:
            // applying it again to grad_output (shaped like `out`, i.e.
            // x_shape with dim0/dim1 already swapped) restores x's
            // original shape and element order.
            Tensor grad_x = Tensor::zeros(x_shape, false);
            cpu::transpose(grad_output.data_ptr(), out_shape.data(), nd, dim0, dim1, grad_x.data_ptr());
            return {grad_x};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t stop) const {
    const Tensor& x = *this;
    int64_t nd = x.ndim();
    if (dim < 0 || dim >= nd)
        throw std::runtime_error("slice: dim out of range");
    if (start < 0 || stop > x.shape()[dim] || start >= stop)
        throw std::runtime_error("slice: invalid [start, stop) range");

    auto out_shape = x.shape();
    out_shape[dim] = stop - start;
    Tensor out = Tensor::zeros(out_shape, false);
    cpu::slice(x.data_ptr(), x.shape().data(), nd, dim, start, stop, out.data_ptr());

    if (x.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "slice";
        node->inputs = {x};
        auto x_shape = x.shape();
        node->backward_fn = [x_shape, dim, start, stop, nd](const Tensor& grad_output) -> std::vector<Tensor> {
            // Zero everywhere except the range that was actually read --
            // that range never received a contribution from anywhere
            // else, and grad_x is fresh (Tensor::zeros), so this is a
            // plain scatter, not an accumulate.
            Tensor grad_x = Tensor::zeros(x_shape, false);
            cpu::scatter_range(grad_output.data_ptr(), x_shape.data(), nd, dim, start, stop, grad_x.data_ptr());
            return {grad_x};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

Tensor Tensor::cat(const std::vector<Tensor>& tensors, int64_t dim) {
    if (tensors.empty())
        throw std::runtime_error("cat: need at least one tensor");
    int64_t nd = tensors[0].ndim();
    if (dim < 0 || dim >= nd)
        throw std::runtime_error("cat: dim out of range");

    auto out_shape = tensors[0].shape();
    int64_t total = 0;
    for (const auto& t : tensors) {
        if (t.ndim() != nd)
            throw std::runtime_error("cat: all tensors must have the same rank");
        for (int64_t d = 0; d < nd; ++d) {
            if (d != dim && t.shape()[d] != out_shape[d])
                throw std::runtime_error("cat: shapes must match on every dim except `dim`");
        }
        total += t.shape()[dim];
    }
    out_shape[dim] = total;

    Tensor out = Tensor::zeros(out_shape, false);
    int64_t offset = 0;
    bool any_grad = false;
    for (const auto& t : tensors) {
        cpu::scatter_range(t.data_ptr(), out_shape.data(), nd, dim, offset, offset + t.shape()[dim],
                            out.data_ptr());
        offset += t.shape()[dim];
        any_grad = any_grad || t.requires_grad();
    }

    if (any_grad) {
        auto node = std::make_shared<GradNode>();
        node->name = "cat";
        node->inputs = tensors;
        std::vector<int64_t> sizes;
        sizes.reserve(tensors.size());
        for (const auto& t : tensors) sizes.push_back(t.shape()[dim]);
        node->backward_fn = [sizes, dim](const Tensor& grad_output) -> std::vector<Tensor> {
            // The inverse of cat's own forward: each input's own
            // gradient is exactly the slice of grad_output at the same
            // offset that input was written to -- reusing Tensor::slice
            // directly rather than a separate backward-only kernel.
            std::vector<Tensor> grads;
            grads.reserve(sizes.size());
            int64_t off = 0;
            for (auto sz : sizes) {
                grads.push_back(grad_output.slice(dim, off, off + sz));
                off += sz;
            }
            return grads;
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

// ---------------------------------------------------------------------
// backward() — standard reverse-mode: topo-sort the graph reachable from
// this tensor, seed its own gradient with ones, then walk in reverse
// order accumulating gradients into every parent (grad_map handles
// tensors reused as inputs to more than one op).
// ---------------------------------------------------------------------

void Tensor::backward() {
    if (numel() != 1)
        throw std::runtime_error("backward() only supported on scalar (numel==1) tensors");

    std::vector<Tensor> topo;
    std::unordered_set<TensorData*> visited;

    std::function<void(const Tensor&)> visit;
    visit = [&](const Tensor& t) {
        TensorData* key = t.impl_ptr();
        if (visited.count(key)) return;
        visited.insert(key);
        if (auto node = t.grad_node()) {
            for (const Tensor& inp : node->inputs) visit(inp);
        }
        topo.push_back(t);
    };
    visit(*this);

    std::unordered_map<TensorData*, Tensor> grad_map;
    grad_map[impl_ptr()] = Tensor::ones_like(*this);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Tensor t = *it;
        auto found = grad_map.find(t.impl_ptr());
        if (found == grad_map.end()) continue;
        Tensor grad_output = found->second;

        auto node = t.grad_node();
        if (!node) {
            if (t.requires_grad()) {
                if (auto existing = t.grad())
                    t.impl_->grad = existing->add(grad_output).impl_;
                else
                    t.impl_->grad = grad_output.impl_;
            }
            continue;
        }

        auto grads_in = node->backward_fn(grad_output);
        for (size_t i = 0; i < node->inputs.size(); ++i) {
            TensorData* pkey = node->inputs[i].impl_ptr();
            auto pfound = grad_map.find(pkey);
            if (pfound == grad_map.end())
                grad_map[pkey] = grads_in[i];
            else
                grad_map[pkey] = pfound->second.add(grads_in[i]);
        }
    }
}

} // namespace kan
