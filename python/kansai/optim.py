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
    AdamW's decoupled-weight-decay variant, which is a real, separate,
    unattempted addition (PyTorch ships them as two distinct classes for
    exactly this reason: silently changing what "Adam" does by adding
    an undocumented decay term would be a correctness surprise, not a
    convenience).

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
