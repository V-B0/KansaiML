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

    // A new Tensor handle sharing this one's Storage (a real view, O(1),
    // no data copy -- Storage is already refcounted via shared_ptr for
    // exactly this kind of sharing) but with requires_grad=false and no
    // grad_node: cut from the autograd graph. Used wherever a value
    // needs to keep being READ or fed into further computation without
    // that computation growing back into the graph the value came from
    // -- BatchNorm1d's running-stats update is the motivating case (see
    // its own docstring in python/kansai/nn.py): folding a
    // still-attached batch mean/variance into a persistent buffer would
    // grow the graph across every training step, not just use its
    // current value. Eager-only: no KIR op or TraceValue method exists
    // for this, since the actual problem it solves for BatchNorm1d
    // (mutating self.running_mean/var as a Python-level side effect)
    // isn't something a traced graph can express regardless of whether
    // the value feeding it is detached -- adding "detach" tracing
    // support wouldn't make that path traceable, so it isn't attempted.
    Tensor detach() const;

    Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;

    // Elementwise comparisons (general-broadcasting, same rule add/sub/
    // mul's own broadcasting follows): 1.0 where the comparison holds,
    // 0.0 where it doesn't. Deliberately NEVER attach a GradNode or set
    // requires_grad, even if an input does -- a comparison's result is
    // piecewise-constant in its inputs (a step function), so its true
    // gradient is zero (or undefined right at the boundary) everywhere,
    // not "whatever add/mul would propagate through it." Combined with
    // `where` (cond.mul(a).add(ones_like(cond).sub(cond).mul(b)), pure
    // composition -- no new op needed), this is what makes real
    // boolean-style masking possible: MultiHeadAttention's own mask
    // arg had to be additive-only before this existed (see its doc
    // comment), specifically because Kansai had no comparison/where
    // primitive yet.
    Tensor gt(const Tensor& other) const;
    Tensor lt(const Tensor& other) const;
    Tensor eq(const Tensor& other) const;

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

    // exp/log, standard elementwise, natural base.
    Tensor exp() const;
    Tensor log() const;

    // tanh/sigmoid/gelu/leaky_relu: the rest of this project's
    // activation vocabulary beyond relu. gelu is the EXACT formulation
    // (via std::erf), not the tanh-based approximation some frameworks
    // default to -- see backend/cpu's own comment for why there was
    // nothing to gain from approximating when the exact form is a
    // direct standard-library call.
    Tensor tanh() const;
    Tensor sigmoid() const;
    Tensor gelu() const;
    Tensor leaky_relu(float negative_slope = 0.01f) const;

    // Reduces along one axis instead of every axis the way sum()/
    // mean() above do -- keepdim=false (the default) squeezes that
    // axis away afterward via reshape() rather than leaving it as a
    // literal size-1 dimension. mean(dim) is composed entirely from
    // sum(dim) + an existing broadcast-mul, no dedicated kernel or
    // backward of its own; max(dim) is forward-only, DELIBERATELY --
    // see backend/cpu's own comment on max_along_dim for exactly why a
    // gradient through it isn't needed (or provided).
    Tensor sum(int64_t dim, bool keepdim = false) const;
    Tensor mean(int64_t dim, bool keepdim = false) const;
    Tensor max(int64_t dim, bool keepdim = false) const;

    // Numerically stable: subtracts max(dim) before exponentiating
    // (mathematically a no-op -- softmax(x) == softmax(x - c) for any
    // constant c -- but the only thing standing between this and
    // overflowing exp() on realistic logit magnitudes). Composed
    // entirely from existing ops (max(dim) [forward-only, see above],
    // sub, exp, sum(dim), div), so it needs no dedicated kernel or vjp
    // rule of its own -- every op it's built from already has one.
    Tensor softmax(int64_t dim) const;

    // self: (batch, classes) raw logits (NOT pre-softmaxed). targets:
    // (batch, classes) ONE-HOT, not a class-index vector -- Kansai has
    // no integer gather/indexing op yet, so a one-hot target is what
    // makes this expressible from existing ops at all; converting a
    // class-index label vector to one-hot is the caller's job (a plain
    // Python loop, not a kernel this needs). Computed via the numerically
    // stable log-sum-exp identity (logsumexp(logits) - sum(logits *
    // targets, dim=1)), never softmax(logits).log() -- log(softmax(x))
    // is the textbook numerically UNSTABLE way to compute this (softmax
    // can legitimately underflow to exactly 0.0 before log ever sees
    // it, producing -inf and then NaN once multiplied by 0 for a
    // masked-out class), which is exactly why every real framework's
    // cross-entropy uses this log-sum-exp form instead of composing
    // softmax and log separately.
    Tensor cross_entropy(const Tensor& targets) const;

    // Selects, along `dim`, the slices at each position in `indices`
    // (repeats allowed, order preserved) -- out's shape is self's shape
    // with dim's extent replaced by indices.size(). `indices` are plain
    // integers, not a Tensor -- Kansai has no integer dtype, and an
    // index into a lookup table isn't a differentiable quantity anyway,
    // the same reason slice()'s own start/stop are plain ints. This is
    // what Embedding is built from: weight.index_select(0, token_ids)
    // looks up each token's own row of the embedding table, with a real
    // gradient that correctly accumulates when the same row is looked
    // up more than once in a batch.
    Tensor index_select(int64_t dim, const std::vector<int64_t>& indices) const;

    // self: (N, Cin, H, W), weight: (Cout, Cin, kH, kW), bias: (Cout,).
    // Forward is im2col + the same matmul kernel every other op already
    // uses (one call per batch item); backward reuses matmul_nt/matmul_tn
    // the same way Tensor::matmul's own backward does, plus col2im for
    // the scatter-add back into the input's shape. NCHW only -- see the
    // project README for why (no layout optimizer exists yet to pick
    // between layouts, so there was nothing to gain from supporting both
    // on day one).
    Tensor conv2d(const Tensor& weight, const Tensor& bias, int64_t stride, int64_t padding) const;

    // self: (N, C, H, W). 2D max pooling with a real, argmax-routed
    // gradient -- deliberately its OWN implementation, not built on
    // max(dim) above: max(dim) is intentionally non-differentiable (see
    // its own declaration for why that's correct for softmax's
    // max-subtraction specifically), and reusing it here would silently
    // give this a zero gradient everywhere, a real correctness trap for
    // a layer meant to backprop through. No padding parameter (unlike
    // conv2d) -- `(H - kernel_size) / stride + 1` must already be a
    // positive integer.
    Tensor max_pool2d(int64_t kernel_size, int64_t stride) const;

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

// Per-thread grad-tracking switch backing kansai.no_grad(): while
// disabled, every op's own `if (x.requires_grad())` check (Tensor.cpp)
// also requires this, so no op attaches a GradNode or turns on its
// output's requires_grad -- exactly what an inference/validation pass
// wants (estimate_val_loss in examples/tinyshakespeare/
// train_shakespeare.py built a full, immediately-discarded backward
// graph every call before this existed, a real, previously-documented
// inefficiency). Defaults to enabled; thread_local (see its own
// definition in Tensor.cpp) since DeviceMesh dispatches real,
// concurrently-overlapping threads that must not share this state.
bool grad_enabled();
void set_grad_enabled(bool enabled);

// NumPy-style right-aligned broadcasting: pads the shorter shape with
// implicit leading 1s, then each aligned pair of dims must either match
// or one of them must be 1 -- the standard rule add/sub/mul/matmul's
// own batch-dim broadcasting in Tensor.cpp all use. A free function
// (not Tensor-scoped `static`, which is where this originally lived)
// specifically so GradOps.cpp can share the identical implementation
// for its own batched matmul_nt/matmul_tn vjp ops rather than
// re-deriving the same broadcasting rule a second time. Throws on an
// incompatible pair rather than silently picking one side.
std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& a, const std::vector<int64_t>& b,
                                       const char* op_name);

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
