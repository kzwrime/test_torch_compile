#!/bin/bash
# Test script for extension_cpp

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Set LD_LIBRARY_PATH for PyTorch libraries
export LD_LIBRARY_PATH="$SCRIPT_DIR/.venv/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH"

echo "======================================"
echo "Testing extension_cpp"
echo "======================================"
echo ""

# Run Python test
"$SCRIPT_DIR/.venv/bin/python" << 'EOF'
import torch
from extension_cpp import ops

GREEN = "\033[0;32m"
NC = "\033[0m"

print("Testing mymuladd...")
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
c = 10.0
result = ops.mymuladd(a, b, c)
expected = a * b + c
print(f"  a = {a}")
print(f"  b = {b}")
print(f"  c = {c}")
print(f"  result = {result}")
print(f"  expected = {expected}")
assert torch.allclose(result, expected), "mymuladd test failed!"
print(f"  {GREEN}PASSED{NC}")

print()
print("Testing myadd_out...")
out = torch.empty_like(a)
ops.myadd_out(a, b, out)
expected_add = a + b
print(f"  a = {a}")
print(f"  b = {b}")
print(f"  out = {out}")
print(f"  expected = {expected_add}")
assert torch.allclose(out, expected_add), "myadd_out test failed!"
print(f"  {GREEN}PASSED{NC}")

print()
print("Testing autograd...")
a = torch.tensor([1.0, 2.0], requires_grad=True)
b = torch.tensor([3.0, 4.0], requires_grad=True)
c = 5.0
result = ops.mymuladd(a, b, c)
result.sum().backward()
print(f"  grad_a = {a.grad}")
print(f"  grad_b = {b.grad}")
assert torch.allclose(a.grad, b), "grad_a should equal b"
assert torch.allclose(b.grad, a), "grad_b should equal a"
print(f"  {GREEN}PASSED{NC}")

print()
print(f"{GREEN}======================================")
print("All tests passed!")
print("======================================{NC}")
EOF

echo ""
