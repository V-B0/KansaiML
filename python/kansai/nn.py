import math

from . import _core as core
from ._core import randn, zeros


class Module:
    def parameters(self):
        params = []
        for value in vars(self).values():
            if isinstance(value, core.Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

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


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
