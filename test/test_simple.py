#!/usr/bin/env python
"""Simple correctness test for mysilu_out"""
import torch
import torch.nn.functional as F
from extension_cpp import ops


def test_basic():
    """Test basic functionality"""
    print("Test 1: Basic")
    a = torch.tensor([0.0, 1.0, 2.0, -1.0, -2.0])
    expected = F.silu(a)

    out = torch.empty_like(a)
    ops.mysilu_out(a, out)

    print(f"  input:    {a}")
    print(f"  expected: {expected}")
    print(f"  output:   {out}")
    print(f"  match:    {torch.allclose(out, expected, rtol=1e-5)}")
    assert torch.allclose(out, expected, rtol=1e-5)
    print("  PASSED\n")


def test_2d():
    """Test 2D tensor"""
    print("Test 2: 2D tensor")
    a = torch.randn(3, 4)
    expected = F.silu(a)

    out = torch.empty_like(a)
    ops.mysilu_out(a, out)

    print(f"  shape:    {a.shape}")
    print(f"  max_diff: {(out - expected).abs().max().item():.10e}")
    assert torch.allclose(out, expected, rtol=1e-5)
    print("  PASSED\n")


if __name__ == "__main__":
    test_basic()
    test_2d()
    print("All tests passed!")
