"""
Unit tests verifying mathematical equivalence of Fused Brier Kernel vs. PyTorch Autograd.
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
from decision_model.model.fused_brier_kernel import fused_brier_loss


def test_forward_and_backward_equivalence():
    print("Running Fused Brier Kernel vs PyTorch Autograd equivalence test...", flush=True)

    torch.manual_seed(42)
    B = 8
    max_k = 6
    num_options = torch.tensor([2, 3, 4, 5, 2, 6, 3, 4], dtype=torch.long)
    labels = torch.tensor([1, 0, 3, 2, 0, 5, 1, 2], dtype=torch.long)
    temperature = 1.25

    # Random logits
    logits_raw = torch.randn(B, max_k, dtype=torch.float64, requires_grad=True)

    # 1. Standard PyTorch Implementation
    logits_ref = logits_raw.clone().detach().requires_grad_(True)
    idx = torch.arange(max_k).unsqueeze(0).expand(B, max_k)
    mask = idx < num_options.unsqueeze(1)
    z_masked = torch.where(mask, logits_ref / temperature, torch.tensor(-1e9, dtype=torch.float64))
    probs_ref = F.softmax(z_masked, dim=-1)
    probs_ref_clean = torch.where(mask, probs_ref, torch.zeros_like(probs_ref))

    y_one_hot = F.one_hot(labels, num_classes=max_k).to(torch.float64)
    diff = probs_ref_clean - y_one_hot
    brier_sq = torch.where(mask, diff**2, torch.zeros_like(diff))
    row_loss = brier_sq.sum(dim=-1) / num_options.to(torch.float64)
    loss_ref = row_loss.mean()

    loss_ref.backward()
    grad_ref = logits_ref.grad.clone()

    # 2. Fused Kernel Implementation
    logits_fused = logits_raw.clone().detach().requires_grad_(True)
    loss_fused, probs_fused, noul_fused = fused_brier_loss(
        logits=logits_fused,
        labels=labels,
        num_options=num_options,
        temperature=temperature,
        normalize_cardinality=True,
    )

    loss_fused.backward()
    grad_fused = logits_fused.grad.clone()

    # Numerical Comparisons
    loss_diff = torch.abs(loss_ref - loss_fused).item()
    probs_diff = torch.max(torch.abs(probs_ref_clean - probs_fused)).item()
    grad_diff = torch.max(torch.abs(grad_ref - grad_fused)).item()

    print(f"  Loss Difference:      {loss_diff:.10e}")
    print(f"  Max Probs Difference: {probs_diff:.10e}")
    print(f"  Max Grad Difference:  {grad_diff:.10e}")

    assert loss_diff < 1e-6, f"Loss mismatch: {loss_diff}"
    assert probs_diff < 1e-6, f"Probs mismatch: {probs_diff}"
    assert grad_diff < 1e-6, f"Gradient mismatch: {grad_diff}"

    # Verify padding positions have exactly 0 gradient
    for b in range(B):
        k = num_options[b].item()
        if k < max_k:
            assert torch.all(grad_fused[b, k:] == 0.0), f"Non-zero gradient in padded position for row {b}"

    # Verify Noul calculation
    print(f"  Sample Noul values:   {[round(n.item(), 4) for n in noul_fused[:4]]}")
    assert torch.all((noul_fused >= 0.0) & (noul_fused <= 1.0)), "Noul out of bounds [0, 1]"

    print("ALL TESTS PASSED: Fused Brier kernel is mathematically identical to PyTorch autograd!", flush=True)


if __name__ == "__main__":
    test_forward_and_backward_equivalence()
