#!/usr/bin/env python
"""
Test Case 2: Build a simple Layer using torch.mm and custom mysilu,
then build multi-layer network and test with torch.compile
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from extension_cpp import ops
import time


class MyLinearLayer(nn.Module):
    """A simple linear layer with SiLU activation using custom ops"""

    def __init__(self, in_features, out_features, activation=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.activation = activation

    def forward(self, x):
        # x: (N, in_features)
        # weight: (out_features, in_features)
        # Use torch.mm for matrix multiplication
        out = torch.mm(x, self.weight.t()) + self.bias
        # Use custom SiLU activation if enabled
        if self.activation:
            out = ops.mysilu(out)
        return out

    def forward_torch(self, x):
        """Reference implementation using torch ops"""
        out = torch.mm(x, self.weight.t()) + self.bias
        if self.activation:
            out = F.silu(out)
        return out


class MyMLP(nn.Module):
    """Multi-layer Perceptron using custom SiLU"""

    def __init__(self, in_features, hidden_features, out_features, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()

        # Input layer (with activation)
        self.layers.append(MyLinearLayer(in_features, hidden_features, activation=True))

        # Hidden layers (with activation)
        for _ in range(num_layers - 2):
            self.layers.append(MyLinearLayer(hidden_features, hidden_features, activation=True))

        # Output layer (no activation)
        self.layers.append(MyLinearLayer(hidden_features, out_features, activation=False))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class TorchMLP(nn.Module):
    """Reference MLP using torch ops"""

    def __init__(self, in_features, hidden_features, out_features, num_layers=3):
        super().__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.SiLU())

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.SiLU())

        # Output layer
        layers.append(nn.Linear(hidden_features, out_features))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def test_single_layer():
    """Test single layer forward pass"""
    print("=" * 50)
    print("Test 1: Single Layer Forward Pass")
    print("=" * 50)

    torch.manual_seed(42)

    layer = MyLinearLayer(in_features=4, out_features=3)
    x = torch.randn(2, 4)

    # Custom forward
    out_custom = layer.forward_torch(x)
    # SiLU forward
    out_silu = layer(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape (custom): {out_custom.shape}")
    print(f"Output shape (silu):  {out_silu.shape}")
    print(f"Outputs match: {torch.allclose(out_custom, out_silu, rtol=1e-4)}")

    assert torch.allclose(out_custom, out_silu, rtol=1e-4, atol=1e-5)
    print("PASSED\n")


def test_single_layer_backward():
    """Test single layer backward pass"""
    print("=" * 50)
    print("Test 2: Single Layer Backward Pass")
    print("=" * 50)

    torch.manual_seed(42)

    layer = MyLinearLayer(in_features=4, out_features=3)
    x = torch.randn(2, 4, requires_grad=True)

    # Custom backward
    x_custom = x.clone().detach().requires_grad_(True)
    layer_custom = MyLinearLayer(in_features=4, out_features=3)
    layer_custom.load_state_dict(layer.state_dict())

    out_custom = layer_custom.forward_torch(x_custom)
    out_custom.sum().backward()
    grad_custom = x_custom.grad.clone()

    # SiLU backward
    x_silu = x.clone().detach().requires_grad_(True)
    layer_silu = MyLinearLayer(in_features=4, out_features=3)
    layer_silu.load_state_dict(layer.state_dict())

    out_silu = layer_silu(x_silu)
    out_silu.sum().backward()
    grad_silu = x_silu.grad.clone()

    print(f"Input grad (custom): {grad_custom}")
    print(f"Input grad (silu):  {grad_silu}")
    print(f"Gradients match: {torch.allclose(grad_custom, grad_silu, rtol=1e-4)}")

    assert torch.allclose(grad_custom, grad_silu, rtol=1e-4, atol=1e-5)
    print("PASSED\n")


def test_mlp_forward():
    """Test MLP forward pass"""
    print("=" * 50)
    print("Test 3: MLP Forward Pass")
    print("=" * 50)

    torch.manual_seed(42)

    my_mlp = MyMLP(in_features=4, hidden_features=8, out_features=2, num_layers=3)
    torch_mlp = TorchMLP(in_features=4, hidden_features=8, out_features=2, num_layers=3)

    # Copy weights
    with torch.no_grad():
        for i, layer in enumerate(my_mlp.layers):
            torch_mlp.net[2 * i].weight.copy_(layer.weight)
            torch_mlp.net[2 * i].bias.copy_(layer.bias)

    x = torch.randn(3, 4)

    out_my = my_mlp(x)
    out_torch = torch_mlp(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape (my):   {out_my.shape}")
    print(f"Output shape (torch): {out_torch.shape}")
    print(f"Max difference: {(out_my - out_torch).abs().max().item():.10e}")

    assert torch.allclose(out_my, out_torch, rtol=1e-4, atol=1e-5)
    print("PASSED\n")


def test_torch_compile():
    """Test with torch.compile"""
    print("=" * 50)
    print("Test 4: torch.compile")
    print("=" * 50)

    torch.manual_seed(42)

    my_mlp = MyMLP(in_features=4, hidden_features=16, out_features=2, num_layers=3)
    my_mlp.eval()

    # Compile the model
    print("Compiling model...")
    my_mlp_compiled = torch.compile(my_mlp, mode="reduce-overhead")
    print("Compilation complete")

    x = torch.randn(5, 4)

    # Warmup
    for _ in range(3):
        _ = my_mlp_compiled(x)

    # Benchmark uncompiled
    torch.manual_seed(42)
    my_mlp_uncompiled = MyMLP(in_features=4, hidden_features=16, out_features=2, num_layers=3)
    my_mlp_uncompiled.eval()
    my_mlp_uncompiled.load_state_dict(my_mlp.state_dict())

    start = time.perf_counter()
    for _ in range(100):
        _ = my_mlp_uncompiled(x)
    time_uncompiled = time.perf_counter() - start

    # Benchmark compiled
    start = time.perf_counter()
    for _ in range(100):
        _ = my_mlp_compiled(x)
    time_compiled = time.perf_counter() - start

    print(f"\nTiming (100 iterations):")
    print(f"  Uncompiled: {time_uncompiled*1000:.2f} ms")
    print(f"  Compiled:   {time_compiled*1000:.2f} ms")
    print(f"  Speedup:    {time_uncompiled/time_compiled:.2f}x")

    # Verify outputs match
    out_uncompiled = my_mlp_uncompiled(x)
    out_compiled = my_mlp_compiled(x)

    print(f"\nOutputs match: {torch.allclose(out_uncompiled, out_compiled, rtol=1e-4, atol=1e-5)}")

    assert torch.allclose(out_uncompiled, out_compiled, rtol=1e-4, atol=1e-5)
    print("PASSED\n")


def test_training_step():
    """Test a simple training step"""
    print("=" * 50)
    print("Test 5: Training Step with Gradient")
    print("=" * 50)

    torch.manual_seed(42)

    model = MyMLP(in_features=4, hidden_features=8, out_features=2, num_layers=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(5, 4)
    target = torch.randn(5, 2)

    # Forward
    output = model(x)
    loss = nn.MSELoss()(output, target)

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Check gradients exist
    grad_exists = all(p.grad is not None for p in model.parameters() if p.requires_grad)

    print(f"Loss: {loss.item():.6f}")
    print(f"All gradients exist: {grad_exists}")

    # Check gradients are finite
    grad_finite = all(p.grad.isfinite().all() for p in model.parameters() if p.requires_grad and p.grad is not None)

    print(f"All gradients finite: {grad_finite}")

    # Optimizer step
    optimizer.step()

    print("Optimizer step completed")

    assert grad_exists and grad_finite
    print("PASSED\n")


if __name__ == "__main__":
    test_single_layer()
    test_single_layer_backward()
    test_mlp_forward()
    test_torch_compile()
    test_training_step()

    print("=" * 50)
    print("All Multi-Layer tests PASSED!")
    print("=" * 50)
