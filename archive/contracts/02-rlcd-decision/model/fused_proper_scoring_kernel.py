"""
=============================================================================
Fused Proper Scoring Rules Kernel (Brier, Spherical, Ranked Probability Score)
=============================================================================
High-performance proper scoring loss functions with exact closed-form gradients:
  1. Brier Score (Categorical decisions & MIT RLCR formulation)
  2. Spherical Scoring Rule: S_sph(p, y) = p_y / ||p||_2
  3. Ranked Probability Score (RPS): Strictly proper score for ordinal/score primitives
       RPS_b = -(1 / (K_b - 1)) * sum_{m=0}^{K_b-2} (P_{b,m} - Y_{b,m})^2
     with exact closed-form analytic gradient:
       d(RPS_b)/d(z_{b,i}) = -(2 / (tau * (K_b - 1))) * p_{b,i} * [ Gamma_{b,i} - Omega_b ]
     where:
       Gamma_{b,i} = sum_{m=i}^{K_b-2} (P_{b,m} - Y_{b,m})
       Omega_b     = sum_{m=0}^{K_b-2} P_{b,m} * (P_{b,m} - Y_{b,m})

Eliminates Python batch loops, torch.cumsum graph construction, and intermediate tensor allocations.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


# =============================================================================
# 1. Fused Ranked Probability Score (RPS) Autograd Function
# =============================================================================

class FusedRPSFunction(torch.autograd.Function):
    """
    Vectorized Ranked Probability Score with closed-form analytic gradient.
    Eliminates Python loops and PyTorch cumsum autograd graph tape.
    """
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        labels: torch.Tensor,
        num_options: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            logits: (B, max_k) unnormalized logits
            labels: (B,) ordinal target bin index (0 <= y < K_b)
            num_options: (B,) valid option count K_b
            temperature: scalar temperature
        Returns:
            mean_loss: scalar tensor (-RPS mean)
            probs: (B, max_k) normalized probabilities
        """
        B, max_k = logits.shape
        device = logits.device

        # Mask invalid options
        idx = torch.arange(max_k, device=device).unsqueeze(0).expand(B, max_k)
        valid_mask = idx < num_options.unsqueeze(1)

        z_masked = torch.where(valid_mask, logits / temperature, torch.tensor(-1e9, device=device, dtype=logits.dtype))
        probs = F.softmax(z_masked, dim=-1)
        probs = torch.where(valid_mask, probs, torch.zeros_like(probs))

        # Predicted CDF: P_{b, m} = sum_{j=0}^m p_{b, j}
        cdf_pred = probs.cumsum(dim=-1)

        # Target CDF: Y_{b, m} = 1 if m >= y_b else 0
        y_exp = labels.unsqueeze(1).expand(B, max_k)
        cdf_true = (idx >= y_exp).to(probs.dtype)

        # Difference: delta_{b, m} = P_{b, m} - Y_{b, m}
        diff = cdf_pred - cdf_true

        # Only evaluate thresholds 0 to K_b - 2
        # (threshold K_b - 1 always has P = Y = 1, so diff = 0)
        thresh_mask = idx < (num_options - 1).clamp(min=1).unsqueeze(1)
        diff_masked = torch.where(thresh_mask, diff, torch.zeros_like(diff))

        # Squared difference sum
        sq_diff = (diff_masked ** 2).sum(dim=-1)
        k_minus_1 = (num_options - 1).clamp(min=1).to(probs.dtype)
        # RPS reward in [ -1, 0 ]
        row_rps_reward = -sq_diff / k_minus_1
        # In multi-class with K=1, RPS is 0
        row_rps_reward = torch.where(num_options > 1, row_rps_reward, torch.zeros_like(row_rps_reward))

        # Policy loss: minimize negative reward (-mean RPS)
        loss = -row_rps_reward.mean()

        ctx.save_for_backward(probs, cdf_pred, diff_masked, num_options)
        ctx.temperature = temperature

        return loss, probs

    @staticmethod
    def backward(ctx, grad_loss, grad_probs=None):
        probs, cdf_pred, diff_masked, num_options = ctx.saved_tensors
        temperature = ctx.temperature
        B, max_k = probs.shape
        device = probs.device

        # Incoming loss gradient scaled by 1/B
        grad_scaled = grad_loss / B

        # Analytic closed-form gradient for RPS loss:
        # Loss = sum_{b} sq_diff_b / (B * (K_b - 1))
        # Gamma_{b, i} = sum_{m=i}^{K_b-2} diff_{b, m} (reverse cumsum along options)
        gamma = torch.flip(torch.cumsum(torch.flip(diff_masked, dims=[-1]), dim=-1), dims=[-1])

        # Omega_b = sum_{m=0}^{K_b-2} P_{b, m} * diff_{b, m}
        omega = (cdf_pred * diff_masked).sum(dim=-1, keepdim=True)

        k_minus_1 = (num_options - 1).clamp(min=1).to(probs.dtype).unsqueeze(1)

        # Gradient w.r.t logits z_{b, i}:
        # d(Loss)/d(z_{b, i}) = (2 / (tau * (K_b - 1))) * p_{b, i} * [ Gamma_{b, i} - Omega_b ] * grad_scaled
        grad_z = (2.0 / (temperature * k_minus_1)) * probs * (gamma - omega) * grad_scaled

        # Zero out invalid options and rows with K <= 1
        idx = torch.arange(max_k, device=device).unsqueeze(0).expand(B, max_k)
        valid_mask = idx < num_options.unsqueeze(1)
        grad_z = torch.where(valid_mask & (num_options.unsqueeze(1) > 1), grad_z, torch.zeros_like(grad_z))

        return grad_z, None, None, None


# =============================================================================
# 2. Fused Spherical Scoring Autograd Function
# =============================================================================

class FusedSphericalFunction(torch.autograd.Function):
    """
    Spherical Scoring Rule: S_sph = p_y / ||p||_2
    Policy loss: L = -S_sph.mean()
    Closed-form gradient:
      dL/dz_i = -(1 / (tau * B)) * p_i * [ (delta_{iy} / ||p||) - (p_y * p_i / ||p||^3) ]
    """
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        labels: torch.Tensor,
        num_options: torch.Tensor,
        temperature: float = 1.0,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, max_k = logits.shape
        device = logits.device

        idx = torch.arange(max_k, device=device).unsqueeze(0).expand(B, max_k)
        valid_mask = idx < num_options.unsqueeze(1)

        z_masked = torch.where(valid_mask, logits / temperature, torch.tensor(-1e9, device=device, dtype=logits.dtype))
        probs = F.softmax(z_masked, dim=-1)
        probs = torch.where(valid_mask, probs, torch.zeros_like(probs))

        # p_y
        p_y = probs.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)
        # ||p||_2
        l2_norm = probs.norm(p=2, dim=1).clamp(min=eps)

        score = p_y / l2_norm
        loss = -score.mean()

        ctx.save_for_backward(probs, labels, num_options, l2_norm, p_y)
        ctx.temperature = temperature

        return loss, probs

    @staticmethod
    def backward(ctx, grad_loss, grad_probs=None):
        probs, labels, num_options, l2_norm, p_y = ctx.saved_tensors
        temperature = ctx.temperature
        B, max_k = probs.shape
        device = probs.device

        grad_scaled = grad_loss / B

        y_one_hot = F.one_hot(labels.clamp(min=0), num_classes=max_k).to(probs.dtype)
        inv_norm = 1.0 / l2_norm.unsqueeze(1)
        norm_cube = (l2_norm ** 3).unsqueeze(1)
        p_y_exp = p_y.unsqueeze(1)

        # Vector term: (delta_{iy} / ||p||) - (p_y * p_i / ||p||^3)
        term = (y_one_hot * inv_norm) - (p_y_exp * probs / norm_cube)

        # Gradient w.r.t logits (negative sign because Loss = -Score):
        grad_z = -(1.0 / temperature) * probs * term * grad_scaled

        idx = torch.arange(max_k, device=device).unsqueeze(0).expand(B, max_k)
        valid_mask = idx < num_options.unsqueeze(1)
        grad_z = torch.where(valid_mask, grad_z, torch.zeros_like(grad_z))

        return grad_z, None, None, None, None


# =============================================================================
# Clean Public APIs
# =============================================================================

def fused_rps_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_options: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Execute fused Ranked Probability Score loss with exact analytic gradients."""
    return FusedRPSFunction.apply(logits, labels, num_options, temperature)


def fused_spherical_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_options: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Execute fused Spherical Score loss with exact analytic gradients."""
    return FusedSphericalFunction.apply(logits, labels, num_options, temperature)
