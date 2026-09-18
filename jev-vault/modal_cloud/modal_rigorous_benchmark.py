"""
Modal Cloud: Rigorous Scientific Benchmark (Jevformer vs. CALM vs. Uniform)
===========================================================================

Removes all surface markers/shortcuts. Tests on Permutation Path Tracing (PCPR):
  - Directed graph of permutations represented as shuffled edge pairs.
  - Query: trace path of length k in [0..max_hops] starting at node S.
  - Zero prefix hints. Tokens are purely numerical identifiers.
  - Theoretical constraint: A circuit of depth L cannot resolve k > L hops without recurrence.

Compares:
  1. Jevformer (Trained with RLCD Proper Scoring Brier Loss)
  2. CALM Baseline (Schuster et al., 2022: Early exit gated by Softmax Entropy)
  3. Uniform Compute Baseline (Always runs max deliberation steps)
  4. Uniform Shallow Baseline (Always exits at System 1)

Evaluates:
  - Accuracy vs. FLOPs Pareto frontier
  - Expected Calibration Error (ECE) of Noul vs. Softmax Entropy
  - Compute Monotonicity against true latent path length k*
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
import modal

APP_NAME = "jevformer-rigorous-benchmark"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("jevformer-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4.0",
        "numpy>=1.26.0",
    )
)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Permutation Path Tracing Dataset (PCPR - No Surface Shortcuts)
# ----------------------------------------------------------------------

def generate_pcpr_batch(
    batch_size: int,
    num_nodes: int = 32,
    max_hops: int = 4,
    seq_len: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates permutation path tracing tasks:
      - Node space: 0 to num_nodes-1
      - Generates a random directed graph (permutation mapping)
      - Query: start at node S, trace k hops.
      - Edge pairs (u -> v) are placed in random order across sequence tokens.
      - Query is placed at the end: (S, ?).
      - NO PREFIX TOKENS. NO MARKERS. NO LEAKAGE.
    """
    tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)
    labels = torch.zeros(batch_size, dtype=torch.long)
    true_hops = torch.randint(0, max_hops + 1, (batch_size,))

    num_edges = (seq_len - 2) // 2

    for i in range(batch_size):
        # 1. Random permutation of nodes: mapping[u] = v
        perm = torch.randperm(num_nodes)
        
        # 2. Select a query start node S
        start_node = int(torch.randint(0, num_nodes, (1,)).item())
        k = int(true_hops[i].item())

        # Compute ground truth destination after k hops
        cur = start_node
        for _ in range(k):
            cur = int(perm[cur].item())
        labels[i] = cur % 8  # 8 output classes

        # 3. Form edge tokens: (u, perm[u])
        # Randomly select subset of edges to populate sequence
        all_u = torch.randperm(num_nodes)[:num_edges]
        edge_pairs = []
        for u in all_u:
            edge_pairs.append((int(u.item()), int(perm[u].item())))

        # Shuffle edges so position leaks zero order information
        perm_order = torch.randperm(len(edge_pairs))
        idx = 0
        for p_idx in perm_order:
            u, v = edge_pairs[p_idx]
            tokens[i, idx] = u + 10     # offset for node vocabulary
            tokens[i, idx + 1] = v + 10
            idx += 2
            if idx >= seq_len - 2:
                break

        # Append query at the end: [Start Node, Query Token (1)]
        tokens[i, -2] = start_node + 10
        tokens[i, -1] = 1  # query indicator

    return tokens, labels, true_hops


# ----------------------------------------------------------------------
# Model Architectures: Jevformer & CALM
# ----------------------------------------------------------------------

class RigorousJevHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int = 8, max_steps: int = 4):
        super().__init__()
        self.choice_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )
        self.noul_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.score_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, max_steps + 1),
        )

    def forward(self, h_pool: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        choice_logits = self.choice_proj(h_pool)
        noul_raw = self.noul_proj(h_pool) / torch.clamp(self.temperature, min=0.1, max=10.0)
        noul_prob = torch.sigmoid(noul_raw)
        score_logits = self.score_proj(h_pool)
        return choice_logits, noul_prob, score_logits


class RigorousDeliberator(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_classes: int = 8):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.05,
            batch_first=True,
            norm_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, num_classes)

    def forward(self, h_seq: torch.Tensor, steps: int) -> torch.Tensor:
        cur = h_seq
        for _ in range(steps):
            cur = self.layer(cur)
        h_pool = self.norm(cur[:, -1, :])  # query token position
        return self.out_proj(h_pool)


class RigorousBenchmarkTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 128,
        d_model: int = 256,
        num_heads: int = 8,
        s1_layers: int = 3,
        max_hops: int = 4,
        num_classes: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_hops = max_hops

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.05,
            batch_first=True,
            norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(encoder_layer, num_layers=s1_layers)
        self.jev_head = RigorousJevHead(d_model, num_classes, max_steps=max_hops)
        self.deliberator = RigorousDeliberator(d_model, num_heads, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "jev",       # "jev", "calm", "uniform_s1", "uniform_s2"
        threshold: float = 0.70, # Noul threshold (jev) or Entropy threshold (calm)
        max_delib_steps: int = 4,
    ) -> dict[str, Any]:
        b, seq_len = x.shape
        h = self.token_emb(x) + self.pos_emb[:, :seq_len, :]
        h_seq = self.trunk(h)
        h_query = h_seq[:, -1, :]

        s1_logits, noul_conf, score_logits = self.jev_head(h_query)
        noul_p = noul_conf.squeeze(-1)

        # CALM Baseline metric: Normalized Softmax Entropy in [0, 1]
        probs_s1 = F.softmax(s1_logits, dim=-1)
        entropy = -torch.sum(probs_s1 * torch.log(probs_s1 + 1e-9), dim=-1)
        norm_entropy = entropy / math.log(probs_s1.size(-1))
        calm_confidence = 1.0 - norm_entropy

        # Routing logic
        if mode == "jev":
            s1_exit_mask = noul_p >= threshold
            pred_steps = torch.argmax(score_logits, dim=-1).clamp(min=1, max=self.max_hops)
            steps_to_run = int(pred_steps.max().item())
        elif mode == "calm":
            s1_exit_mask = calm_confidence >= threshold
            steps_to_run = max_delib_steps
            pred_steps = torch.ones(b, dtype=torch.long, device=x.device) * max_delib_steps
        elif mode == "uniform_s1":
            s1_exit_mask = torch.ones(b, dtype=torch.bool, device=x.device)
            steps_to_run = 0
            pred_steps = torch.zeros(b, dtype=torch.long, device=x.device)
        elif mode == "uniform_s2":
            s1_exit_mask = torch.zeros(b, dtype=torch.bool, device=x.device)
            steps_to_run = max_delib_steps
            pred_steps = torch.ones(b, dtype=torch.long, device=x.device) * max_delib_steps
        else:
            raise ValueError(f"Unknown mode: {mode}")

        s2_logits = None
        if not s1_exit_mask.all() and steps_to_run > 0:
            s2_logits = self.deliberator(h_seq, steps=steps_to_run)
            final_logits = torch.where(s1_exit_mask.unsqueeze(-1), s1_logits, s2_logits)
        else:
            final_logits = s1_logits

        return {
            "final_logits": final_logits,
            "s1_logits": s1_logits,
            "s2_logits": s2_logits,
            "noul_p": noul_p,
            "calm_confidence": calm_confidence,
            "s1_exit_mask": s1_exit_mask,
            "score_logits": score_logits,
            "pred_steps": pred_steps,
        }


# ----------------------------------------------------------------------
# Modal Training & Benchmark Function
# ----------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={"/checkpoints": volume},
)
def run_rigorous_benchmark_cloud(
    steps: int = 5000,
    batch_size: int = 128,
    lr: float = 6e-4,
    d_model: int = 256,
    num_heads: int = 8,
    s1_layers: int = 3,
    max_hops: int = 4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Worker] Running Rigorous Benchmark on GPU: {torch.cuda.get_device_name(0)}")

    model = RigorousBenchmarkTransformer(
        vocab_size=128,
        d_model=d_model,
        num_heads=num_heads,
        s1_layers=s1_layers,
        max_hops=max_hops,
        num_classes=8,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model Architecture]: Rigorous Benchmark Model with {param_count:,} parameters.")
    print("Dataset: Permutation Path Tracing (PCPR) — ZERO surface cues or prefix hints.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)

    print("\n--- Training Jevformer with Joint RLCD Calibration ---")
    start_time = time.time()

    for step in range(1, steps + 1):
        model.train()
        x, y, true_hops = generate_pcpr_batch(batch_size, max_hops=max_hops)
        x, y, true_hops = x.to(device), y.to(device), true_hops.to(device)

        out = model(x, mode="jev", threshold=0.5, max_delib_steps=max_hops)
        s1_logits = out["s1_logits"]
        s2_logits = model.deliberator(model.trunk(model.token_emb(x) + model.pos_emb[:, :x.size(1), :]), steps=max_hops)
        noul_p = out["noul_p"]
        score_logits = out["score_logits"]

        # 1. Task Cross-Entropy
        loss_s1 = F.cross_entropy(s1_logits, y)
        loss_s2 = F.cross_entropy(s2_logits, y)

        # 2. Strict RLCD Calibration Target:
        # z = 1 if S1 is correct on this graph traversal, 0 otherwise
        s1_correct = (torch.argmax(s1_logits, dim=-1) == y).float().detach()
        loss_calib = F.mse_loss(noul_p, s1_correct)

        # 3. Depth Prediction Loss
        loss_depth = F.cross_entropy(score_logits, true_hops)

        # 4. Compute penalty
        loss_compute = torch.mean(1.0 - noul_p) * 0.10

        total_loss = loss_s1 + loss_s2 + (3.0 * loss_calib) + (0.5 * loss_depth) + loss_compute

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 1000 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                s1_acc = (torch.argmax(s1_logits, dim=-1) == y).float().mean().item()
                s2_acc = (torch.argmax(s2_logits, dim=-1) == y).float().mean().item()
                
                # Check confidence on true k=0 vs k>=3
                k0_mask = true_hops == 0
                k_deep_mask = true_hops >= 3
                conf_k0 = noul_p[k0_mask].mean().item() if k0_mask.any() else 0.0
                conf_deep = noul_p[k_deep_mask].mean().item() if k_deep_mask.any() else 0.0

                print(
                    f"Step {step:04d}/{steps} | Loss: {total_loss.item():.3f} | "
                    f"S1 Acc: {s1_acc*100:.1f}% | S2 Acc: {s2_acc*100:.1f}% | "
                    f"Conf (Hop=0): {conf_k0:.3f} | Conf (Hop>=3): {conf_deep:.3f}"
                )

    print(f"\n[Training Complete]: Elapsed {time.time() - start_time:.1f}s")

    # ------------------------------------------------------------------
    # Comprehensive Holdout Evaluation: Jevformer vs CALM vs Baselines
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" Rigorous Comparative Evaluation on 3,000 Unseen PCPR Tasks")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        x_eval, y_eval, true_hops_eval = generate_pcpr_batch(3000, max_hops=max_hops)
        x_eval, y_eval = x_eval.to(device), y_eval.to(device)

        # Baseline A: Pure Uniform S1 (Constant Depth)
        res_s1 = model(x_eval, mode="uniform_s1")
        acc_s1 = (torch.argmax(res_s1["final_logits"], dim=-1) == y_eval).float().mean().item() * 100.0

        # Baseline B: Pure Uniform S2 (Full Recurrence 100% FLOPs)
        res_s2 = model(x_eval, mode="uniform_s2", max_delib_steps=max_hops)
        acc_s2 = (torch.argmax(res_s2["final_logits"], dim=-1) == y_eval).float().mean().item() * 100.0

        print(f"Baseline A (Uniform S1 - 15% FLOPs) : Acc = {acc_s1:.2f}%")
        print(f"Baseline B (Uniform S2 - 100% FLOPs): Acc = {acc_s2:.2f}%")

        # Sweep Jevformer across thresholds
        jev_results = []
        for tau in [0.20, 0.40, 0.60, 0.80, 0.95]:
            out_j = model(x_eval, mode="jev", threshold=tau)
            acc_j = (torch.argmax(out_j["final_logits"], dim=-1) == y_eval).float().mean().item() * 100.0
            exit_pct = out_j["s1_exit_mask"].float().mean().item() * 100.0
            rel_flops = exit_pct * 0.15 + (100.0 - exit_pct) * 1.0
            jev_results.append({"tau": tau, "acc": acc_j, "exit_rate": exit_pct, "flops": rel_flops})
            print(f"Jevformer (τ={tau:.2f}) -> Acc: {acc_j:.2f}% | S1 Exit: {exit_pct:.1f}% | Relative FLOPs: {rel_flops:.1f}%")

        # Sweep CALM Baseline (Softmax Entropy Thresholding - Schuster et al. 2022)
        calm_results = []
        for tau_c in [0.20, 0.40, 0.60, 0.80, 0.95]:
            out_c = model(x_eval, mode="calm", threshold=tau_c, max_delib_steps=max_hops)
            acc_c = (torch.argmax(out_c["final_logits"], dim=-1) == y_eval).float().mean().item() * 100.0
            exit_pct_c = out_c["s1_exit_mask"].float().mean().item() * 100.0
            rel_flops_c = exit_pct_c * 0.15 + (100.0 - exit_pct_c) * 1.0
            calm_results.append({"tau": tau_c, "acc": acc_c, "exit_rate": exit_pct_c, "flops": rel_flops_c})
            print(f"CALM (Entropy τ={tau_c:.2f}) -> Acc: {acc_c:.2f}% | S1 Exit: {exit_pct_c:.1f}% | Relative FLOPs: {rel_flops_c:.1f}%")

        # Expected Calibration Error (ECE) Comparison
        # Compare Jev Noul calibration vs CALM Softmax Entropy calibration
        noul_all = model(x_eval, mode="jev")["noul_p"].cpu().numpy()
        calm_conf_all = model(x_eval, mode="calm")["calm_confidence"].cpu().numpy()
        s1_pred_all = (torch.argmax(model(x_eval, mode="uniform_s1")["final_logits"], dim=-1) == y_eval).float().cpu().numpy()

        def compute_ece(confs, corrects, num_bins=10):
            bins = np.linspace(0.0, 1.0, num_bins + 1)
            ece = 0.0
            n = len(confs)
            for i in range(num_bins):
                m = (confs >= bins[i]) & (confs < bins[i + 1])
                if np.any(m):
                    bin_acc = np.mean(corrects[m])
                    bin_conf = np.mean(confs[m])
                    ece += (np.sum(m) / n) * abs(bin_acc - bin_conf)
            return float(ece)

        import numpy as np
        ece_jev = compute_ece(noul_all, s1_pred_all)
        ece_calm = compute_ece(calm_conf_all, s1_pred_all)

        print(f"\n[Calibration Comparison]:")
        print(f"  * Jevformer ECE (RLCD Proper Scoring): {ece_jev * 100:.2f}%")
        print(f"  * CALM ECE (Raw Softmax Entropy)    : {ece_calm * 100:.2f}%")

    # Save Checkpoint
    checkpoint_path = "/checkpoints/jevformer_rigorous_benchmark.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "jev_results": jev_results,
            "calm_results": calm_results,
            "acc_s1": acc_s1,
            "acc_s2": acc_s2,
            "ece_jev": ece_jev,
            "ece_calm": ece_calm,
        },
        checkpoint_path,
    )
    volume.commit()
    print(f"\n[Saved]: Checkpoint and benchmark results committed to {checkpoint_path}")

    return {
        "acc_s1": acc_s1,
        "acc_s2": acc_s2,
        "jev_results": jev_results,
        "calm_results": calm_results,
        "ece_jev": ece_jev,
        "ece_calm": ece_calm,
    }


@app.local_entrypoint()
def main():
    print("=" * 70)
    print(" Launching Rigorous Scientific Benchmark: Jevformer vs CALM on Modal")
    print("=" * 70)
    res = run_rigorous_benchmark_cloud.remote(
        steps=4000,
        batch_size=128,
        d_model=256,
        num_heads=8,
        s1_layers=3,
        max_hops=4,
    )
    print("\nBenchmark Complete! Summary of Findings:")
    print(f"  Jev ECE: {res['ece_jev']*100:.2f}% vs CALM ECE: {res['ece_calm']*100:.2f}%")
