#pragma once
#include "kansai/DType.hpp"
#include "kansai/Storage.hpp"
#include "kansai/StoragePool.hpp"
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace kan {

class Tensor;
struct GradNode;

// The reference-counted payload behind every Tensor handle. Tensor itself
// is a thin shared_ptr wrapper (same shape as PyTorch's Tensor/TensorImpl
// split) so copying a Tensor is cheap and graph nodes can hold parents by
// value without duplicating storage.
struct TensorData {
    StoragePtr storage;
    std::vector<int64_t> shape;
    DType dtype = DType::Float32;
    bool requires_grad = false;
    std::shared_ptr<TensorData> grad;      // accumulated gradient, if any
    std::shared_ptr<GradNode> grad_node;   // how this tensor was produced, if not a leaf
};

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::shared_ptr<TensorData> impl) : impl_(std::move(impl)) {}

    static Tensor zeros(std::vector<int64_t> shape, bool requires_grad = false);
    static Tensor ones(std::vector<int64_t> shape, bool requires_grad = false);
    static Tensor randn(std::vector<int64_t> shape, float std = 1.0f,
                         bool requires_grad = false, uint64_t seed = 0);
    static Tensor from_flat(std::vector<float> data, std::vector<int64_t> shape,
                             bool requires_grad = false);
    static Tensor zeros_like(const Tensor& other);
    static Tensor ones_like(const Tensor& other);

    bool defined() const { return static_cast<bool>(impl_); }
    const std::vector<int64_t>& shape() const { return impl_->shape; }
    int64_t numel() const;
    int64_t ndim() const { return static_cast<int64_t>(impl_->shape.size()); }

    bool requires_grad() const { return impl_->requires_grad; }
    void set_requires_grad(bool rg) { impl_->requires_grad = rg; }

    float* data_ptr() { return static_cast<float*>(impl_->storage->data()); }
    const float* data_ptr() const { return static_cast<const float*>(impl_->storage->data()); }

    std::optional<Tensor> grad() const;
    void zero_grad();

    std::shared_ptr<GradNode> grad_node() const;
    void set_grad_node(std::shared_ptr<GradNode> node);

    // Reverse-mode autodiff entry point. Only valid on a scalar (numel==1)
    // tensor — walks the graph this tensor's grad_node chain reaches and
    // accumulates gradients into every leaf tensor that requires_grad.
    void backward();

    // In-place update, bypassing autograd entirely. Used by optimizers.
    void add_(const Tensor& other, float alpha = 1.0f);

    Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;
    Tensor relu() const;
    Tensor sum() const;
    Tensor mean() const;

    // Elementwise sqrt/reciprocal, standard IEEE-754 semantics -- see
    // backend/cpu's own doc comment for the edge cases (sqrt of a
    // negative input, reciprocal of zero) this doesn't specially guard.
    Tensor sqrt() const;
    Tensor reciprocal() const;

    // a.div(b) == a.mul(b.reciprocal()) -- composed from two already-
    // autograd-aware ops rather than its own kernel or backward_fn, so
    // its gradient falls out of mul's and reciprocal's own chain rules
    // automatically. A real, un-fused cost (two passes -- reciprocal
    // then multiply -- instead of one division pass) in exchange for
    // adding a whole new op category for free; a dedicated fused divide
    // kernel is a real, unattempted future optimization if this ever
    // shows up as a bottleneck.
    Tensor div(const Tensor& other) const;

    // self: (N, Cin, H, W), weight: (Cout, Cin, kH, kW), bias: (Cout,).
    // Forward is im2col + the same matmul kernel every other op already
    // uses (one call per batch item); backward reuses matmul_nt/matmul_tn
    // the same way Tensor::matmul's own backward does, plus col2im for
    // the scatter-add back into the input's shape. NCHW only -- see the
    // project README for why (no layout optimizer exists yet to pick
    // between layouts, so there was nothing to gain from supporting both
    // on day one).
    Tensor conv2d(const Tensor& weight, const Tensor& bias, int64_t stride, int64_t padding) const;

    // General tensor manipulation -- unlike every op above, these don't
    // change values, only which position each element sits at. All
    // three copy (this codebase has no non-owning "view" that shares
    // another Tensor's Storage): reshape's copy is a straight memcpy
    // (row-major reshape never reorders bytes, just reinterprets the
    // same flat sequence under new shape boundaries), transpose's and
    // slice's genuinely move data. A real zero-copy view would need
    // StoragePool's pooling to become aware of aliased buffers (two
    // Tensors sharing one Storage, with different lifetimes) --
    // unattempted, real future work, not silently assumed safe here.
    Tensor reshape(std::vector<int64_t> new_shape) const;
    Tensor transpose(int64_t dim0, int64_t dim1) const;
    Tensor slice(int64_t dim, int64_t start, int64_t stop) const;

    // Concatenates `tensors` along `dim` -- every other dimension must
    // already match across all of them. Static (not a method) since
    // there's no single natural "self" among an arbitrary-length list
    // of tensors being joined, the same reason Tensor::zeros/ones/randn
    // are static rather than instance methods.
    static Tensor cat(const std::vector<Tensor>& tensors, int64_t dim);

    std::vector<float> to_vector() const;

    TensorData* impl_ptr() const { return impl_.get(); }
    StoragePtr storage_ptr() const { return impl_->storage; }

private:
    std::shared_ptr<TensorData> impl_;
};

// Redirects every subsequent Tensor::zeros() allocation -- and hence
// every op's output, since they all construct theirs via zeros() -- to
// draw from `pool` instead of malloc'ing fresh, until cleared (pass
// nullptr). See kir.run_planned()'s docstring for the safety protocol
// this requires from the caller.
void set_active_pool(StoragePool* pool);

// A single recorded op: which parent tensors produced this one, and the
// closure that turns an upstream gradient into gradients for each parent.
// This — plus Tensor::backward()'s graph walk — *is* Kansai's autograd
// engine for Milestone 0 (a tape-based one; Phase 2 replaces it with a
// source transformation over KIR, per the project plan).
struct GradNode {
    std::vector<Tensor> inputs;
    std::function<std::vector<Tensor>(const Tensor&)> backward_fn;
    std::string name;
};

} // namespace kan
