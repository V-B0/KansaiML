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

    bool bias_broadcast = (b.ndim() == 1 && a.ndim() == 2 && a.shape()[1] == b.shape()[0]);
    if (!bias_broadcast && a.shape() != b.shape())
        throw std::runtime_error("add: shape mismatch");

    Tensor out = Tensor::zeros(a.shape(), false);
    if (bias_broadcast)
        cpu::add_bias_broadcast(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.shape()[0], a.shape()[1]);
    else
        cpu::add(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel());

    if (a.requires_grad() || b.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "add";
        node->inputs = {a, b};
        node->backward_fn = [bias_broadcast](const Tensor& grad_output) -> std::vector<Tensor> {
            Tensor grad_a = grad_output;
            Tensor grad_b;
            if (bias_broadcast) {
                grad_b = Tensor::zeros({grad_output.shape()[1]}, false);
                cpu::sum_over_batch(grad_output.data_ptr(), grad_b.data_ptr(),
                                     grad_output.shape()[0], grad_output.shape()[1]);
            } else {
                grad_b = grad_output;
            }
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
    if (a.shape() != b.shape())
        throw std::runtime_error("sub: shape mismatch");

    Tensor out = Tensor::zeros(a.shape(), false);
    cpu::sub(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel());

    if (a.requires_grad() || b.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "sub";
        node->inputs = {a, b};
        node->backward_fn = [](const Tensor& grad_output) -> std::vector<Tensor> {
            Tensor neg = Tensor::zeros(grad_output.shape(), false);
            cpu::axpy_(neg.data_ptr(), grad_output.data_ptr(), -1.0f, grad_output.numel());
            return {grad_output, neg};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

Tensor Tensor::mul(const Tensor& other) const {
    const Tensor& a = *this;
    const Tensor& b = other;
    if (a.shape() != b.shape())
        throw std::runtime_error("mul: shape mismatch");

    Tensor out = Tensor::zeros(a.shape(), false);
    cpu::mul(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel());

    if (a.requires_grad() || b.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "mul";
        node->inputs = {a, b};
        node->backward_fn = [a, b](const Tensor& grad_output) -> std::vector<Tensor> {
            Tensor grad_a = Tensor::zeros(a.shape(), false);
            Tensor grad_b = Tensor::zeros(b.shape(), false);
            cpu::mul(grad_output.data_ptr(), b.data_ptr(), grad_a.data_ptr(), a.numel());
            cpu::mul(grad_output.data_ptr(), a.data_ptr(), grad_b.data_ptr(), a.numel());
            return {grad_a, grad_b};
        };
        out.set_grad_node(node);
        out.set_requires_grad(true);
    }
    return out;
}

Tensor Tensor::matmul(const Tensor& other) const {
    const Tensor& a = *this;
    const Tensor& b = other;
    if (a.ndim() != 2 || b.ndim() != 2 || a.shape()[1] != b.shape()[0])
        throw std::runtime_error("matmul: incompatible shapes");

    int64_t M = a.shape()[0], K = a.shape()[1], N = b.shape()[1];
    Tensor out = Tensor::zeros({M, N}, false);
    cpu::matmul(a.data_ptr(), b.data_ptr(), out.data_ptr(), M, K, N);

    if (a.requires_grad() || b.requires_grad()) {
        auto node = std::make_shared<GradNode>();
        node->name = "matmul";
        node->inputs = {a, b};
        node->backward_fn = [a, b, M, K, N](const Tensor& grad_output) -> std::vector<Tensor> {
            // grad_a = grad_output (M,N) @ b(K,N)^T -> (M,K). b's transpose
            // is never materialized -- matmul_nt reads b's own (K,N)
            // layout directly (BLAS's CblasTrans, or the fallback's
            // swapped indexing).
            Tensor grad_a = Tensor::zeros({M, K}, false);
            cpu::matmul_nt(grad_output.data_ptr(), b.data_ptr(), grad_a.data_ptr(), M, N, K);

            // grad_b = a(M,K)^T @ grad_output(M,N) -> (K,N). Same idea,
            // transposing a instead of b.
            Tensor grad_b = Tensor::zeros({K, N}, false);
            cpu::matmul_tn(a.data_ptr(), grad_output.data_ptr(), grad_b.data_ptr(), M, K, N);

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
