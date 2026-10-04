"""
Benchmark comparing Sequential Marker Looping vs Batched Tensor Gathering.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
from pathlib import Path
cur = Path(__file__).resolve().parent
while cur != cur.parent:
    if (cur / "contracts").exists() or (cur / ".git").exists():
        break
    cur = cur.parent
if str(cur) not in sys.path:
    sys.path.insert(0, str(cur))

import torch
import torch.nn as nn
from decision_model.model.decision_heads import OptionScorer


def test_batched_gather_benchmark():
    print("Testing Batched Tensor Gathering vs Sequential Looping...", flush=True)

    B = 32
    seq_len = 512
    hidden_size = 2560
    max_k = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scorer = OptionScorer(hidden_size=hidden_size, head_hidden=256).to(device)
    scorer.eval()
    last_hidden = torch.randn(B, seq_len, hidden_size, device=device)

    # Random marker positions per example
    import random
    marker_positions = [
        sorted(random.sample(range(50, 450), random.randint(2, max_k)))
        for _ in range(B)
    ]

    # --- 1. Sequential Looping (Current Method) ---
    start_seq = time.perf_counter()
    batch_logits_seq = []
    for b_idx, positions in enumerate(marker_positions):
        pos_tensor = torch.tensor(positions, device=device, dtype=torch.long)
        h_opts = last_hidden[b_idx, pos_tensor, :]
        logits = scorer(h_opts)
        batch_logits_seq.append(logits)
    seq_time = (time.perf_counter() - start_seq) * 1000

    # --- 2. Batched Tensor Gather (New Method) ---
    start_batch = time.perf_counter()
    padded_pos = torch.zeros((B, max_k), device=device, dtype=torch.long)
    valid_mask = torch.zeros((B, max_k), device=device, dtype=torch.bool)
    for b, pos in enumerate(marker_positions):
        k = len(pos)
        padded_pos[b, :k] = torch.tensor(pos, device=device, dtype=torch.long)
        valid_mask[b, :k] = True

    pos_expanded = padded_pos.unsqueeze(-1).expand(-1, -1, hidden_size)
    h_opts_batched = torch.gather(last_hidden, dim=1, index=pos_expanded)  # (B, max_k, D)
    logits_batched = scorer(h_opts_batched)  # (B, max_k)
    logits_batched = torch.where(valid_mask, logits_batched, torch.tensor(float("-inf"), device=device))
    batch_time = (time.perf_counter() - start_batch) * 1000

    # Verify numerical equivalence for all valid options
    max_diff = 0.0
    for b in range(B):
        k = len(marker_positions[b])
        diff = torch.max(torch.abs(batch_logits_seq[b] - logits_batched[b, :k])).item()
        if diff > max_diff:
            max_diff = diff

    print(f"  Sequential Looping Time: {seq_time:.2f} ms")
    print(f"  Batched Gather Time:     {batch_time:.2f} ms")
    print(f"  Speedup:                 {seq_time / max(batch_time, 1e-4):.2f}x")
    print(f"  Max Output Difference:   {max_diff:.10e}")

    assert max_diff < 1e-5, f"Numerical mismatch: {max_diff}"
    print("  ✓ Batched Tensor Gather verified and numerically identical!", flush=True)


if __name__ == "__main__":
    test_batched_gather_benchmark()
