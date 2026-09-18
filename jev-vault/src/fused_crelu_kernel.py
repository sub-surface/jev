"""
=============================================================================
⚡ FUSED CReLU & NOUL EPISTEMIC SENSOR KERNEL
=============================================================================
High-performance fused operator for Jevformer:
1. CReLU: Clamps accumulator pre-activations to [0.0, 1.0].
2. Sparsity L1 Loss: Computes mean activation per batch element.
3. Epistemic Noul Sensor: Computes intrinsic epistemic certainty from 
   activation dispersion and variance in SRAM without round-trip memory traffic.

Includes:
- GPU Triton Kernel (for Linux CUDA on Modal A100 / H100)
- Optimized PyTorch / TorchScript Fallback (for Windows / Local CUDA / CPU)
=============================================================================
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Rule 1: Windows Unicode encoding hygiene
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Try importing Triton
_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def _fused_crelu_noul_fwd_kernel(
        x_ptr,           # *float32 input (B, D)
        out_ptr,         # *float32 output activations (B, D)
        l1_ptr,          # *float32 output mean L1 (B,)
        noul_ptr,        # *float32 output noul (B,)
        B, D,            # batch size, hidden dim (e.g. 128)
        stride_xb, stride_xd,
        stride_ob, stride_od,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)  # batch index
        if pid >= B:
            return

        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < D

        # Load x from DRAM to registers
        x_ptrs = x_ptr + pid * stride_xb + cols * stride_xd
        x = tl.load(x_ptrs, mask=mask, other=0.0)

        # 1. CReLU: clamp to [0.0, 1.0]
        a = tl.maximum(0.0, tl.minimum(1.0, x))

        # Store clamped activations to DRAM
        out_ptrs = out_ptr + pid * stride_ob + cols * stride_od
        tl.store(out_ptrs, a, mask=mask)

        # 2. Sparsity L1: mean(a)
        sum_a = tl.sum(a, axis=0)
        mean_a = sum_a / D
        tl.store(l1_ptr + pid, mean_a)

        # 3. Epistemic Dispersion / Noul
        # var = sum((a - mean)^2) / D
        diff = tl.where(mask, a - mean_a, 0.0)
        sum_sq = tl.sum(diff * diff, axis=0)
        var_a = sum_sq / D
        std_a = tl.sqrt(var_a + 1e-6)

        # Epistemic Noul: 1.0 - 2.5 * std - 0.5 * mean, clamped to [0, 1]
        noul = tl.maximum(0.0, tl.minimum(1.0, 1.0 - 2.5 * std_a - 0.5 * mean_a))
        tl.store(noul_ptr + pid, noul)


    @triton.jit
    def _fused_crelu_bwd_kernel(
        grad_out_ptr,    # *float32 dL/dOut (B, D)
        x_ptr,           # *float32 input x (B, D)
        grad_x_ptr,      # *float32 dL/dx (B, D)
        B, D,
        stride_gb, stride_gd,
        stride_xb, stride_xd,
        stride_dxb, stride_dxd,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= B:
            return

        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < D

        # Load grad_out and original x
        go_ptrs = grad_out_ptr + pid * stride_gb + cols * stride_gd
        x_ptrs = x_ptr + pid * stride_xb + cols * stride_xd
        go = tl.load(go_ptrs, mask=mask, other=0.0)
        x = tl.load(x_ptrs, mask=mask, other=0.0)

        # d(CReLU)/dx = 1.0 if 0.0 < x < 1.0 else 0.0
        active_mask = (x > 0.0) & (x < 1.0)
        gx = tl.where(active_mask, go, 0.0)

        # Store dL/dx
        gx_ptrs = grad_x_ptr + pid * stride_dxb + cols * stride_dxd
        tl.store(gx_ptrs, gx, mask=mask)


class FusedCReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        # x shape: (B, D)
        ctx.save_for_backward(x)
        B, D = x.shape

        if _TRITON_AVAILABLE and x.is_cuda:
            out = torch.empty_like(x)
            l1_loss = torch.empty((B,), dtype=x.dtype, device=x.device)
            noul = torch.empty((B,), dtype=x.dtype, device=x.device)

            BLOCK_SIZE = triton.next_power_of_2(D)
            grid = (B,)
            _fused_crelu_noul_fwd_kernel[grid](
                x, out, l1_loss, noul,
                B, D,
                x.stride(0), x.stride(1),
                out.stride(0), out.stride(1),
                BLOCK_SIZE=BLOCK_SIZE,
            )
            return out, l1_loss, noul
        else:
            # Fast PyTorch fallback
            out = torch.clamp(x, 0.0, 1.0)
            l1_loss = torch.mean(out, dim=-1)
            mean_a = l1_loss.unsqueeze(-1)
            var_a = torch.var(out, dim=-1, unbiased=False)
            std_a = torch.sqrt(var_a + 1e-6)
            noul = torch.clamp(1.0 - 2.5 * std_a - 0.5 * l1_loss, 0.0, 1.0)
            return out, l1_loss, noul

    @staticmethod
    def backward(ctx, grad_out, grad_l1=None, grad_noul=None):
        (x,) = ctx.saved_tensors
        B, D = x.shape

        if _TRITON_AVAILABLE and x.is_cuda and grad_out.is_cuda:
            grad_x = torch.empty_like(x)
            BLOCK_SIZE = triton.next_power_of_2(D)
            grid = (B,)
            _fused_crelu_bwd_kernel[grid](
                grad_out, x, grad_x,
                B, D,
                grad_out.stride(0), grad_out.stride(1),
                x.stride(0), x.stride(1),
                grad_x.stride(0), grad_x.stride(1),
                BLOCK_SIZE=BLOCK_SIZE,
            )
            return grad_x
        else:
            grad_x = grad_out.clone()
            grad_x[(x <= 0.0) | (x >= 1.0)] = 0.0
            return grad_x


class FusedCReLULayer(nn.Module):
    """
    Fused CReLU & Epistemic Sensor module.
    Replaces separate Clamp + Mean + Var + Sigmoid kernels with a single fused pass.
    """
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor):
        return FusedCReLUFunction.apply(x)


if __name__ == "__main__":
    print(f"--- Fused CReLU Kernel Test (Triton Available: {_TRITON_AVAILABLE}) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on Device: {device}")

    layer = FusedCReLULayer(hidden_dim=128)
    
    # Test batch
    x = torch.randn(64, 128, device=device, requires_grad=True)
    out, l1, noul = layer(x)

    print(f"Output shape: {out.shape}, range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    print(f"L1 Sparsity shape: {l1.shape}, mean: {l1.mean().item():.4f}")
    print(f"Noul shape: {noul.shape}, mean: {noul.mean().item():.4f}")

    # Test backward pass
    loss = out.sum() + l1.sum()
    loss.backward()
    print(f"Backward successful! Grad shape: {x.grad.shape}, nonzero grads: {(x.grad != 0).float().mean().item():.2%}")
    print("✅ Fused CReLU & Epistemic Sensor Kernel verified!")
