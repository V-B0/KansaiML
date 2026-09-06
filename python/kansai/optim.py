class SGD:
    def __init__(self, params, lr: float = 0.1):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            g = p.grad
            if g is not None:
                p.add_(g, -self.lr)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


from . import _core as core


class Adam:
    """The standard Adam optimizer (Kingma & Ba, 2014): per-parameter
    first- and second-moment running averages of the gradient, bias-
    corrected, dividing the step by the second moment's square root --
    the update every widely-used training recipe reaches for by
    default, where plain SGD needs a hand-tuned learning-rate schedule
    (and often momentum on top) to converge at a comparable rate.

    No weight decay here -- this is the original paper's algorithm, not
    AdamW's decoupled-weight-decay variant (see the AdamW class below),
    kept as two distinct classes for exactly the reason PyTorch keeps
    them distinct too: silently changing what "Adam" does by adding an
    undocumented decay term would be a correctness surprise, not a
    convenience.

    Implemented entirely from existing Tensor ops (mul, add, sub, sqrt,
    div, and general broadcasting for the scalar hyperparameters) rather
    than a dedicated C++ optimizer kernel -- the same "prototype in
    Python first" tradeoff distributed.py's split/concat and
    quantize.py's (de)quantization already made, and only newly possible
    at all now that sqrt/div exist (Adam is literally the reason they
    were added: nothing before this needed elementwise sqrt or division,
    and SGD's own single `add_` call never did either). A real,
    un-fused cost (several separate elementwise passes per parameter per
    step, each with its own allocation) versus a single fused CUDA-style
    kernel -- correct and clear before fast, the same call this
    project's whole history has made every time those traded off
    against each other.
    """

    def __init__(self, params, lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [core.zeros(list(p.shape)) for p in self.params]
        self.v = [core.zeros(list(p.shape)) for p in self.params]

    def step(self):
        self.t += 1
        bias_correction1 = 1.0 - self.beta1 ** self.t
        bias_correction2 = 1.0 - self.beta2 ** self.t

        # Scalar hyperparameters as shape-[1] tensors, broadcasting
        # against any parameter's own shape via general broadcasting
        # (add/sub/mul now support this for any shape, not just the old
        # bias-broadcast case) -- built once per step() call, reused
        # across every parameter this step, not reallocated per
        # parameter.
        beta1_t = core.from_flat([self.beta1], [1])
        one_minus_beta1_t = core.from_flat([1.0 - self.beta1], [1])
        beta2_t = core.from_flat([self.beta2], [1])
        one_minus_beta2_t = core.from_flat([1.0 - self.beta2], [1])
        eps_t = core.from_flat([self.eps], [1])
        lr_over_bc1_t = core.from_flat([self.lr / bias_correction1], [1])
        inv_bc2_t = core.from_flat([1.0 / bias_correction2], [1])

        for i, p in enumerate(self.params):
            g = p.grad
            if g is None:
                continue

            # m = beta1*m + (1-beta1)*g ; v = beta2*v + (1-beta2)*g^2
            self.m[i] = self.m[i].mul(beta1_t).add(g.mul(one_minus_beta1_t))
            self.v[i] = self.v[i].mul(beta2_t).add(g.mul(g).mul(one_minus_beta2_t))

            # update = lr * m_hat / (sqrt(v_hat) + eps), factored as
            # lr * m / (bias_correction1 * (sqrt(v/bias_correction2) + eps))
            # so the bias correction folds into lr_over_bc1_t /
            # inv_bc2_t instead of computing m_hat/v_hat as their own
            # separate tensors.
            v_hat = self.v[i].mul(inv_bc2_t)
            denom = v_hat.sqrt().add(eps_t)
            update = self.m[i].div(denom).mul(lr_over_bc1_t)
            p.add_(update, alpha=-1.0)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


class AdamW(Adam):
    """Adam with DECOUPLED weight decay (Loshchilov & Hutter, 2019) --
    the real, separate addition Adam's own docstring above says it
    deliberately doesn't make. The difference from plain L2
    regularization (which Adam doesn't implement either, but which is
    the OTHER common way people bolt "weight decay" onto Adam) matters:
    L2 would add `weight_decay * p` into the gradient itself, so it
    gets divided by the second-moment estimate along with everything
    else -- parameters with large gradients end up decayed less, which
    is backwards from the intent. Decoupled decay instead shrinks the
    parameter directly, `p *= (1 - lr * weight_decay)`, entirely
    outside the moment estimates -- the same order of operations
    PyTorch's own AdamW uses (decay first, against the pre-step
    parameter, then the ordinary Adam update on the now-decayed
    value), not a coincidence: it's what makes the decay rate behave
    the way `weight_decay` is documented to.

    Subclasses Adam rather than duplicating its moment bookkeeping --
    the only difference is one extra in-place step before the
    inherited Adam update runs. `p.add_(p, alpha=-lr*weight_decay)`
    computes exactly `p *= (1 - lr*weight_decay)` via the existing
    axpy_-backed add_ (safe to alias `p` against itself: axpy_ is a
    plain elementwise loop, `p[i] += alpha*p[i]`, no cross-index
    dependency), so this needs no new kernel either -- the same "reuse
    what already exists" approach every optimizer/composed-layer
    addition in this project has taken.
    """

    def __init__(self, params, lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        self.weight_decay = weight_decay

    def step(self):
        if self.weight_decay != 0.0:
            decay_alpha = -self.lr * self.weight_decay
            for p in self.params:
                if p.grad is not None:
                    p.add_(p, alpha=decay_alpha)
        super().step()


def clip_grad_norm_(params, max_norm: float) -> float:
    """The standard global-L2-norm gradient clipper: computes ONE norm
    across every parameter's gradient combined (not per-parameter --
    clipping each parameter's grad to its own norm independently would
    change the relative scale between parameters, distorting the
    update direction, not just its magnitude), and if that norm exceeds
    `max_norm`, scales every gradient down by the same factor so the
    combined norm becomes exactly `max_norm`. Left unchanged if the
    norm is already within bounds -- this only ever shrinks, never
    grows, a gradient.

    Exists for the same reason every serious training loop reaches for
    it: without it, a single unlucky batch producing a huge gradient
    (a real, common failure mode training a transformer -- the
    intended next use for this) can blow up Adam's moment estimates
    for the rest of training, not just that one step.

    Mutates every parameter's `.grad` IN PLACE via the existing
    axpy_-backed `add_` (`g.add_(g, alpha=clip_coef - 1.0)` computes
    exactly `g *= clip_coef`, the same self-aliasing trick AdamW's own
    decoupled decay already uses) rather than replacing `p.grad` with
    a new Tensor -- there's no setter for `.grad` exposed to Python at
    all (it's read-only, populated only by `backward()`), so in-place
    mutation of the existing storage is the only way this CAN work, not
    just the way it happens to be written. No new kernel: the norm
    itself is computed via `g.mul(g).sum()`, ops that already exist.

    Returns the pre-clipping total norm (matching PyTorch's own
    `clip_grad_norm_` return value), useful for logging even when no
    clipping actually happened this step.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0

    total_sq = 0.0
    for g in grads:
        total_sq += g.mul(g).sum().tolist()[0]
    total_norm = total_sq ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for g in grads:
            g.add_(g, alpha=clip_coef - 1.0)
    return total_norm


class StepLR:
    """Decays `optimizer.lr` by a factor of `gamma` every `step_size`
    calls to `step()` -- the simplest possible learning-rate schedule,
    a flat rate with periodic drops. Reads/writes `optimizer.lr`
    directly rather than needing any change to the optimizer classes
    themselves: every optimizer above (SGD/Adam/AdamW) already reads
    `self.lr` fresh inside its own `step()` rather than caching it once
    at construction, so mutating it externally between optimizer steps
    is already exactly how a schedule is meant to take effect -- this
    class is pure bookkeeping around that existing seam, not a new
    integration point.
    """

    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma


class CosineAnnealingLR:
    """Cosine-anneals `optimizer.lr` from its value AT CONSTRUCTION TIME
    down to `eta_min` over `T_max` calls to `step()`, following the
    standard half-cosine schedule (Loshchilov & Hutter's SGDR paper --
    the same authors as AdamW above): smooth, monotonic decay that
    starts and ends flat (zero slope at both `last_epoch=0` and
    `last_epoch=T_max`) rather than `StepLR`'s abrupt drops -- the
    schedule most commonly paired with transformer training in
    practice, the intended next use for this.

    `base_lr` is captured once from `optimizer.lr` at construction,
    not re-read every `step()` -- the schedule is always relative to
    where training started, so it stays well-defined even though
    `step()` itself mutates `optimizer.lr` on every call.
    """

    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.last_epoch = 0

    def step(self):
        import math
        self.last_epoch += 1
        progress = min(self.last_epoch, self.T_max) / self.T_max
        self.optimizer.lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
