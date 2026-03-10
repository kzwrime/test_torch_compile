import torch
from torch import Tensor

__all__ = ["mysilu_out"]


def mysilu_out(a: Tensor, out: Tensor) -> None:
    """Writes SiLU(a) into out: out = a * sigmoid(a)"""
    torch.ops.extension_cpp.mysilu_out.default(a, out)


@torch.library.register_fake("extension_cpp::mysilu_out")
def _(a, out):
    torch._check(a.dtype == torch.float)
    torch._check(a.shape == out.shape)
