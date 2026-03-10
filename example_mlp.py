#!/usr/bin/env python
"""
Minimal MLP example with custom SiLU and torch.compile
"""
import torch
import torch.nn as nn
from extension_cpp import ops


class Layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return ops.mysilu(torch.mm(x, self.weight.t()) + self.bias)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = Layer(4, 8)
        self.l2 = Layer(8, 8)
        self.l3 = nn.Linear(8, 2)  # output layer, no activation

    def forward(self, x):
        return self.l3(self.l2(self.l1(x)))


if __name__ == "__main__":
    model = MLP()
    model_compiled = torch.compile(model)

    x = torch.randn(5, 4)
    print("Input:", x)

    out = model_compiled(x)
    print("Output:", out)

    # Training step
    loss = out.sum()
    loss.backward()
    print("Loss:", loss.item())
