#!/usr/bin/env python
"""
Minimal MLP example with custom mysilu_out and torch.compile
Demonstrating Dynamic Shapes for varying token counts (vLLM style)
"""
import torch
import torch.nn as nn
import os

from extension_cpp import ops


class MLP(nn.Module):
    def __init__(self, in_out_dim=4, hidden_dim=8):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(hidden_dim, in_out_dim) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.W2 = nn.Parameter(torch.randn(in_out_dim, hidden_dim) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(in_out_dim))

    def forward(self, x):
        # First mm
        hidden = torch.mm(x, self.W1.t()) + self.b1
        # SiLU activation
        activated = torch.empty_like(hidden)
        ops.mysilu_out(hidden, activated)
        # Second mm
        out = torch.mm(activated, self.W2.t()) + self.b2
        return out


class MultiMLP(nn.Module):
    def __init__(self, num_mlps=3, in_out_dim=4, hidden_dim=8):
        super().__init__()
        self.mlps = nn.ModuleList([
            MLP(in_out_dim, hidden_dim) for _ in range(num_mlps)
        ])

    def forward(self, x):
        outputs = []
        for mlp in self.mlps:
            outputs.append(mlp(x))
        return outputs


if __name__ == "__main__":
    # Base MLP
    mlp = MLP(in_out_dim=4, hidden_dim=8)

    # 初始输入形状 (例如 5 个 token)
    x_init = torch.randn(5, 4)
    
    # MultiMLP
    multi_mlp = MultiMLP(num_mlps=20, in_out_dim=4, hidden_dim=8)

    print("--- 开启 torch.compile 并支持动态形状 ---")
    # 仅使用显式标记的方式：不传入 dynamic=True，而是依赖 mark_dynamic 精确控制
    multi_mlp_compiled = torch.compile(multi_mlp)

    # 关键修改: 显式标记维度 0 为动态维度
    # 明确告诉 Dynamo 张量的第 0 维是动态变化的，避免因为形状改变而陷入无休止的重新编译
    torch._dynamo.mark_dynamic(x_init, 0)

    # Warmup (使用初始形状)
    print(f"Warmup with shape: {x_init.shape}")
    for _ in range(3):
        _ = multi_mlp_compiled(x_init)

    # 测试动态形状: token 数发生变化 (例如从 5 变到 12，再变到 1)
    # 如果不开启 dynamic=True，每一次形状改变都会导致漫长的重新编译 (Recompilation)
    shapes_to_test = [(12, 4), (1, 4), (1024, 4), (33, 4)]
    
    for shape in shapes_to_test:
        x_new = torch.randn(*shape)
        # 必须显式标记新张量的动态维度，因为我们没有开启全局 dynamic=True
        torch._dynamo.mark_dynamic(x_new, 0)
        
        print(f"\nTesting dynamic shape: {x_new.shape}")
        
        # 验证输出
        out_eager = multi_mlp(x_new)
        out_compiled = multi_mlp_compiled(x_new)
        
        all_match = True
        for i, (orig, comp) in enumerate(zip(out_eager, out_compiled)):
            if not torch.allclose(orig, comp, rtol=1e-4):
                all_match = False
                print(f"  MLP {i+1} outputs differ!")
                
        if all_match:
            print("  Outputs match successfully without crashing!")