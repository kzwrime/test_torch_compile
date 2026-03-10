import torch
from torch import Tensor

__all__ = ["mymuladd", "myadd_out", "mysilu"]


def mymuladd(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    return torch.ops.extension_cpp.mymuladd.default(a, b, c)


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("extension_cpp::mymuladd")
def _(a, b, c):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)


def _backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_a, grad_b = None, None
    if ctx.needs_input_grad[0]:
        grad_a = torch.ops.extension_cpp.mymul.default(grad, b)
    if ctx.needs_input_grad[1]:
        grad_b = torch.ops.extension_cpp.mymul.default(grad, a)
    return grad_a, grad_b, None


def _setup_context(ctx, inputs, output):
    a, b, c = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[1]:
        saved_a = a
    ctx.save_for_backward(saved_a, saved_b)


# This adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd(
    "extension_cpp::mymuladd", _backward, setup_context=_setup_context)


@torch.library.register_fake("extension_cpp::mymul")
def _(a, b):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)


def mysilu(a: Tensor) -> Tensor:
    """SiLU activation function: x * sigmoid(x)"""
    return torch.ops.extension_cpp.mysilu.default(a)


@torch.library.register_fake("extension_cpp::mysilu")
def _(a):
    torch._check(a.dtype == torch.float)
    return torch.empty_like(a)


def _silu_backward(ctx, grad):
    a, = ctx.saved_tensors
    # SiLU backward: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    #               = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    sigmoid_a = torch.sigmoid(a)
    grad_input = grad * sigmoid_a * (1 + a * (1 - sigmoid_a))
    return grad_input


def _silu_setup_context(ctx, inputs, output):
    a, = inputs
    ctx.save_for_backward(a)


torch.library.register_autograd(
    "extension_cpp::mysilu", _silu_backward, setup_context=_silu_setup_context)


def myadd_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    """Writes a + b into out"""
    torch.ops.extension_cpp.myadd_out.default(a, b, out)
