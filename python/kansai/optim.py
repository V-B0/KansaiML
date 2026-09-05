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
