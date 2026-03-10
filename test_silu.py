#!/usr/bin/env python
"""
Test Case 1: Compare mysilu with torch.nn.functional.silu
"""
import torch
import torch.nn.functional as F
from extension_cpp import ops


def test_silu_basic():
    """Test basic SiLU functionality"""
    print("=" * 50)
    print("Test 1: Basic SiLU functionality")
    print("=" * 50)

    # Test 1D tensor
    x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    expected = F.silu(x)
    result = ops.mysilu(x)

    print(f"Input: {x}")
    print(f"PyTorch silu: {expected}")
    print(f"Our mysilu:   {result}")
    print(f"Difference:    {(expected - result).abs().max().item():.10e}")

    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6), "SiLU basic test failed!"
    print("PASSED\n")


def test_silu_2d():
    """Test SiLU on 2D tensor"""
    print("=" * 50)
    print("Test 2: SiLU on 2D tensor")
    print("=" * 50)

    x = torch.randn(4, 5)
    expected = F.silu(x)
    result = ops.mysilu(x)

    print(f"Input shape: {x.shape}")
    print(f"Max difference: {(expected - result).abs().max().item():.10e}")

    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6), "SiLU 2D test failed!"
    print("PASSED\n")


def test_silu_gradient():
    """Test SiLU gradient computation"""
    print("=" * 50)
    print("Test 3: SiLU gradient")
    print("=" * 50)

    x = torch.randn(3, 4, requires_grad=True)

    # PyTorch gradient
    x_torch = x.clone().detach().requires_grad_(True)
    y_torch = F.silu(x_torch)
    y_torch.sum().backward()
    grad_torch = x_torch.grad.clone()

    # Our gradient
    x_ours = x.clone().detach().requires_grad_(True)
    y_ours = ops.mysilu(x_ours)
    y_ours.sum().backward()
    grad_ours = x_ours.grad.clone()

    print(f"Input: {x}")
    print(f"PyTorch gradient: {grad_torch}")
    print(f"Our gradient:     {grad_ours}")
    print(f"Max difference:   {(grad_torch - grad_ours).abs().max().item():.10e}")

    assert torch.allclose(grad_ours, grad_torch, rtol=1e-5, atol=1e-6), "SiLU gradient test failed!"
    print("PASSED\n")


def test_silu_edge_cases():
    """Test SiLU edge cases"""
    print("=" * 50)
    print("Test 4: SiLU edge cases")
    print("=" * 50)

    # Test with zeros
    x = torch.zeros(10)
    result = ops.mysilu(x)
    expected = F.silu(x)
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)
    print("Zeros: PASSED")

    # Test with ones
    x = torch.ones(10)
    result = ops.mysilu(x)
    expected = F.silu(x)
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)
    print("Ones: PASSED")

    # Test with large positive values
    x = torch.tensor([100.0, 50.0, 10.0])
    result = ops.mysilu(x)
    expected = F.silu(x)
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)
    print("Large positive: PASSED")

    # Test with large negative values
    x = torch.tensor([-100.0, -50.0, -10.0])
    result = ops.mysilu(x)
    expected = F.silu(x)
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)
    print("Large negative: PASSED\n")


if __name__ == "__main__":
    test_silu_basic()
    test_silu_2d()
    test_silu_gradient()
    test_silu_edge_cases()

    print("=" * 50)
    print("All SiLU tests PASSED!")
    print("=" * 50)
