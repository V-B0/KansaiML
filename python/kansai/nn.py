import math

from . import _core as core
from ._core import randn, zeros


class Module:
    def parameters(self):
        return [t for _, t in self.named_parameters()]

    def named_parameters(self, prefix: str = ""):
        """Yields (name, tensor) for every trainable parameter,
        name-qualified by attribute path -- "layers.0.weight" for the
        first sub-layer of a Sequential, say. Same recursion
        parameters() already did, just keeping the path that gets there
        instead of discarding it. That path is exactly what
        kansai.serialize needs to match a saved tensor back to the
        right attribute on load -- a flat, order-dependent list (what
        parameters() alone gives you) has no name to check a checkpoint
        against, only a position, which breaks the moment two model
        definitions differ in ways that don't change parameter count
        (a reordered layer, an extra non-trainable bumper)."""
        for key, value in vars(self).items():
            if isinstance(value, core.Tensor) and value.requires_grad:
                yield (f"{prefix}{key}", value)
            elif isinstance(value, Module):
                yield from value.named_parameters(f"{prefix}{key}.")
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        yield from item.named_parameters(f"{prefix}{key}.{i}.")

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, seed: int = 0):
        std = 1.0 / math.sqrt(in_features)
        self.weight = randn([in_features, out_features], std=std, requires_grad=True, seed=seed)
        self.bias = zeros([out_features], requires_grad=True)

    def forward(self, x):
        return x.matmul(self.weight).add(self.bias)


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, seed: int = 0):
        self.stride = stride
        self.padding = padding
        fan_in = in_channels * kernel_size * kernel_size
        std = 1.0 / math.sqrt(fan_in)
        self.weight = randn([out_channels, in_channels, kernel_size, kernel_size],
                             std=std, requires_grad=True, seed=seed)
        self.bias = zeros([out_channels], requires_grad=True)

    def forward(self, x):
        return x.conv2d(self.weight, self.bias, self.stride, self.padding)


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class GELU(Module):
    """The exact formulation (via erf), not the tanh-based approximation
    some frameworks default to -- see Tensor::gelu's own declaration in
    core/include/kansai/Tensor.hpp for why."""

    def forward(self, x):
        return x.gelu()


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01):
        self.negative_slope = negative_slope

    def forward(self, x):
        return x.leaky_relu(self.negative_slope)


class Softmax(Module):
    def __init__(self, dim: int):
        self.dim = dim

    def forward(self, x):
        return x.softmax(self.dim)


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
