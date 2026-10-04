"""
Unit tests verifying mathematical equivalence of Fused RPS & Spherical Kernel vs. PyTorch Autograd.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
cur = Path(__file__).resolve().parent
while cur != cur.parent:
    if (cur / "contracts").exists() or (cur / ".git").exists():
        break
    cur = cur.parent
if str(cur) not in sys.path:
    sys.path.insert(0, str(cur))

import torch
import torch.nn.functional as F
from decision_model.model.fused_proper_scoring_kernel import fused_rps_loss, fused_spherical_loss


def test_rps_equivalence():
    print("Testing Ranked Probability Score (RPS) Equivalence...", flush=True)
    torch.manual_seed(1337)
    B = 6
    max_k = 5
    num_options = torch.tensor([3, 4, 5, 2, 4, 3], dtype=torch.long)
    labels = torch.tensor([1, 2, 0, 1, 3, 2], dtype=torch.long)
    temperature = 1.15

    logits_raw = torch.randn(B, max_k, dtype=torch.float64, requires_grad=True)

    # 1. Reference PyTorch autograd
    logits_ref = logits_raw.clone().detach().requires_grad_(True)
    idx = torch.arange(max_k).unsqueeze(0).expand(B, max_k)
    mask = idx < num_options.unsqueeze(1)
    z_masked = torch.where(mask, logits_ref / temperature, torch.tensor(-1e9, dtype=torch.float64))
    probs_ref = F.softmax(z_masked, dim=-1)
    probs_ref_clean = torch.where(mask, probs_ref, torch.zeros_like(probs_ref))

    cdf_pred = probs_ref_clean.cumsum(dim=-1)
    y_exp = labels.unsqueeze(1).expand(B, max_k)
    cdf_true = (idx >= y_exp).to(torch.float64)
    diff = cdf_pred - cdf_true
    thresh_mask = idx < (num_options - 1).clamp(min=1).unsqueeze(1)
    sq_diff = torch.where(thresh_mask, diff**2, torch.zeros_like(diff)).sum(dim=-1)
    k_minus_1 = (num_options - 1).clamp(min=1).to(torch.float64)
    row_rps = -sq_diff / k_minus_1
    loss_ref = -row_rps.mean()
    loss_ref.backward()
    grad_ref = logits_ref.grad.clone()

    # 2. Fused Kernel
    logits_fused = logits_raw.clone().detach().requires_grad_(True)
    loss_fused, probs_fused = fused_rps_loss(
        logits=logits_fused,
        labels=labels,
        num_options=num_options,
        temperature=temperature,
    )
    loss_fused.backward()
    grad_fused = logits_fused.grad.clone()

    loss_diff = torch.abs(loss_ref - loss_fused).item()
    grad_diff = torch.max(torch.abs(grad_ref - grad_fused)).item()

    print(f"  RPS Loss Difference:     {loss_diff:.10e}")
    print(f"  RPS Max Grad Difference: {grad_diff:.10e}")

    assert loss_diff < 1e-6, f"RPS loss mismatch: {loss_diff}"
    assert grad_diff < 1e-6, f"RPS grad mismatch: {grad_diff}"
    print("  ✓ RPS mathematical equivalence verified!", flush=True)


def test_spherical_equivalence():
    print("Testing Spherical Scoring Rule Equivalence...", flush=True)
    torch.manual_seed(2026)
    B = 6
    max_k = 5
    num_options = torch.tensor([3, 4, 5, 2, 4, 3], dtype=torch.long)
    labels = torch.tensor([1, 2, 0, 1, 3, 2], dtype=torch.long)
    temperature = 1.05

    logits_raw = torch.randn(B, max_k, dtype=torch.float64, requires_grad=True)

    # 1. Reference PyTorch autograd
    logits_ref = logits_raw.clone().detach().requires_grad_(True)
    idx = torch.arange(max_k).unsqueeze(0).expand(B, max_k)
    mask = idx < num_options.unsqueeze(1)
    z_masked = torch.where(mask, logits_ref / temperature, torch.tensor(-1e9, dtype=torch.float64))
    probs_ref = F.softmax(z_masked, dim=-1)
    probs_ref_clean = torch.where(mask, probs_ref, torch.zeros_like(probs_ref))

    p_y = probs_ref_clean.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)
    l2_norm = probs_ref_clean.norm(p=2, dim=1).clamp(min=1e-8)
    score = p_y / l2_norm
    loss_ref = -score.mean()
    loss_ref.backward()
    grad_ref = logits_ref.grad.clone()

    # 2. Fused Kernel
    logits_fused = logits_raw.clone().detach().requires_grad_(True)
    loss_fused, probs_fused = fused_spherical_loss(
        logits=logits_fused,
        labels=labels,
        num_options=num_options,
        temperature=temperature,
    )
    loss_fused.backward()
    grad_fused = logits_fused.grad.clone()

    loss_diff = torch.abs(loss_ref - loss_fused).item()
    grad_diff = torch.max(torch.abs(grad_ref - grad_fused)).item()

    print(f"  Spherical Loss Difference:     {loss_diff:.10e}")
    print(f"  Spherical Max Grad Difference: {grad_diff:.10e}")

    assert loss_diff < 1e-6, f"Spherical loss mismatch: {loss_diff}"
    assert grad_diff < 1e-6, f"Spherical grad mismatch: {grad_diff}"
    print("  ✓ Spherical mathematical equivalence verified!", flush=True)


if __name__ == "__main__":
    test_rps_equivalence()
    test_spherical_equivalence()
    print("\nALL PROPER SCORING KERNEL EQUIVALENCE TESTS PASSED!", flush=True)
