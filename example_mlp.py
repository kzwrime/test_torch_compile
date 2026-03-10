#!/usr/bin/env python
"""
Minimal MLP example with custom mysilu_out and torch.compile
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
        out = torch.mm(x, self.weight.t()) + self.bias
        # Use mysilu_out for activation
        activated = torch.empty_like(out)
        ops.mysilu_out(out, activated)
        return activated


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

    x = torch.randn(5, 4)
    print("Input shape:", x.shape)

    # Run inference
    out = model(x)
    print("Output:", out)
    print("Output shape:", out.shape)

    # Test with torch.compile
    print("\nTesting torch.compile...")
    model_compiled = torch.compile(model, mode="reduce-overhead")

    # Warmup
    for _ in range(3):
        _ = model_compiled(x)

    out_compiled = model_compiled(x)
    print("Compiled output:", out_compiled)

    # Verify outputs match
    if torch.allclose(out, out_compiled, rtol=1e-4):
        print("Outputs match: OK")
    else:
        print("Outputs differ!")
