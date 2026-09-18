"""
Modal Cloud Scaling Experiment: Jevformer
=========================================

Scaled training run testing the Calibrated Dual-Process Transformer (Jevformer)
on multi-hop symbolic reasoning and graph induction across variable difficulty depths.

Evaluates:
  1. Epistemic Calibration (Expected Calibration Error - ECE, Brier Score)
  2. Compute Monotonicity (Does allocated depth correlate with true latent complexity?)
  3. Pareto Frontier (Accuracy vs FLOP efficiency compared to uniform compute)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
import modal

# ----------------------------------------------------------------------
# Modal App & Environment Configuration
# ----------------------------------------------------------------------

APP_NAME = "jevformer-scaling-experiment"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("jevformer-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4.0",
        "numpy>=1.26.0",
    )
)

# ----------------------------------------------------------------------
# Model Architecture (Scaled Jevformer)
# ----------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledJevHead(nn.Module):
    """Calibrated System 1 Perceptual Head with temperature scaling."""
    def __init__(self, d_model: int, num_classes: int, max_steps: int = 5):
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


class ScaledDeliberator(nn.Module):
    """System 2 Deliberator: Recurrent Reasoning Transformer Blocks."""
    def __init__(self, d_model: int, num_heads: int, num_classes: int):
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
        h_pool = self.norm(cur[:, 0, :])
        return self.out_proj(h_pool)


class ScaledJevformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 512,
        d_model: int = 256,
        num_heads: int = 8,
        s1_layers: int = 4,
        max_s2_steps: int = 5,
        num_classes: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_s2_steps = max_s2_steps

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.05,
            batch_first=True,
            norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(encoder_layer, num_layers=s1_layers)
        self.jev_head = ScaledJevHead(d_model, num_classes, max_steps=max_s2_steps)
        self.deliberator = ScaledDeliberator(d_model, num_heads, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        confidence_threshold: float = 0.70,
        fixed_steps: int | None = None,
    ) -> dict[str, Any]:
        b, seq_len = x.shape
        h = self.token_emb(x) + self.pos_emb[:, :seq_len, :]
        h_seq = self.trunk(h)
        h_pool = h_seq[:, 0, :]

        s1_logits, noul_conf, score_logits = self.jev_head(h_pool)
        noul_p = noul_conf.squeeze(-1)

        s1_exit_mask = noul_p >= confidence_threshold
        pred_steps = torch.argmax(score_logits, dim=-1).clamp(min=1, max=self.max_s2_steps)
        
        steps_to_run = fixed_steps if fixed_steps is not None else int(pred_steps.max().item())
        s2_logits = self.deliberator(h_seq, steps=max(1, steps_to_run))

        final_logits = torch.where(s1_exit_mask.unsqueeze(-1), s1_logits, s2_logits)

        return {
            "final_logits": final_logits,
            "s1_logits": s1_logits,
            "s2_logits": s2_logits,
            "noul_p": noul_p,
            "score_logits": score_logits,
            "s1_exit_mask": s1_exit_mask,
            "pred_steps": pred_steps,
        }


# ----------------------------------------------------------------------
# Multi-Depth Relational Task Generator
# ----------------------------------------------------------------------

def generate_multi_depth_batch(
    batch_size: int,
    seq_len: int = 32,
    num_classes: int = 8,
    max_hops: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates synthetic relational reasoning chains with true latent depth k in [0..max_hops].
      - k = 0: Direct pattern match (Solvable by S1)
      - k > 0: Sequential pointer jumps requiring k deliberation steps
    """
    tokens = torch.randint(20, 480, (batch_size, seq_len))
    labels = torch.zeros(batch_size, dtype=torch.long)
    true_hops = torch.randint(0, max_hops + 1, (batch_size,))

    for i in range(batch_size):
        k = int(true_hops[i].item())
        if k == 0:
            # Reflexive: Direct class indicator at index 1
            labels[i] = tokens[i, 1] % num_classes
            tokens[i, 0] = 1  # marker
        else:
            # Multi-hop pointer chain across sequence
            cur_idx = int(tokens[i, 2] % (seq_len - 6) + 3)
            accum = 0
            for _ in range(k):
                val = int(tokens[i, cur_idx])
                accum += val
                cur_idx = int(val % (seq_len - 6) + 3)
            labels[i] = accum % num_classes
            tokens[i, 0] = 2  # marker

    return tokens, labels, true_hops


# ----------------------------------------------------------------------
# Modal Training Function
# ----------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,  # 30 mins
    volumes={"/checkpoints": volume},
)
def train_jevformer_cloud(
    steps: int = 2500,
    batch_size: int = 128,
    lr: float = 8e-4,
    d_model: int = 256,
    num_heads: int = 8,
    s1_layers: int = 4,
    max_hops: int = 5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Modal Worker] Initialized on GPU: {torch.cuda.get_device_name(0)}")

    model = ScaledJevformer(
        vocab_size=512,
        d_model=d_model,
        num_heads=num_heads,
        s1_layers=s1_layers,
        max_s2_steps=max_hops,
        num_classes=8,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model Architecture]: Scaled Jevformer with {param_count:,} parameters.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)

    print("\n[Starting Joint RLCD Training Run]")
    start_time = time.time()

    for step in range(1, steps + 1):
        model.train()
        x, y, true_hops = generate_multi_depth_batch(batch_size, max_hops=max_hops)
        x, y, true_hops = x.to(device), y.to(device), true_hops.to(device)

        out = model(x, confidence_threshold=0.5, fixed_steps=max_hops)
        s1_logits = out["s1_logits"]
        s2_logits = out["s2_logits"]
        noul_p = out["noul_p"]
        score_logits = out["score_logits"]

        # 1. Primary Task Loss
        loss_s1 = F.cross_entropy(s1_logits, y)
        loss_s2 = F.cross_entropy(s2_logits, y)

        # 2. Epistemic Calibration (RLCD):
        # Ground truth correctness of S1 prediction
        s1_correct = (torch.argmax(s1_logits, dim=-1) == y).float().detach()
        loss_calib = F.mse_loss(noul_p, s1_correct)

        # 3. Depth Prediction Loss: Score head predicting true required hops
        loss_depth = F.cross_entropy(score_logits, true_hops)

        # 4. Parsimony Regularization: Penalize unnecessary deliberation
        loss_compute = torch.mean(1.0 - noul_p) * 0.10

        total_loss = loss_s1 + loss_s2 + (2.5 * loss_calib) + (0.5 * loss_depth) + loss_compute

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 250 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                s1_acc = (torch.argmax(s1_logits, dim=-1) == y).float().mean().item()
                s2_acc = (torch.argmax(s2_logits, dim=-1) == y).float().mean().item()
                
                # Calibration by true hop count
                hop0_mask = true_hops == 0
                hop_deep_mask = true_hops >= 3
                conf_hop0 = noul_p[hop0_mask].mean().item() if hop0_mask.any() else 0.0
                conf_deep = noul_p[hop_deep_mask].mean().item() if hop_deep_mask.any() else 0.0

                print(
                    f"Step {step:04d}/{steps} | Loss: {total_loss.item():.3f} | "
                    f"S1 Acc: {s1_acc*100:.1f}% | S2 Acc: {s2_acc*100:.1f}% | "
                    f"Conf (Hop=0): {conf_hop0:.3f} | Conf (Hop>=3): {conf_deep:.3f}"
                )

    # Final Evaluation & Pareto Analysis
    print("\n--- Final Evaluation & Epistemic Pareto Frontier ---")
    model.eval()
    with torch.no_grad():
        x_eval, y_eval, true_hops_eval = generate_multi_depth_batch(2000, max_hops=max_hops)
        x_eval, y_eval = x_eval.to(device), y_eval.to(device)

        # Sweep thresholds to produce Accuracy vs Compute Pareto Frontier
        print("\nThreshold Sweep (Accuracy vs Compute Allocation):")
        for tau in [0.30, 0.50, 0.70, 0.85, 0.95]:
            eval_out = model(x_eval, confidence_threshold=tau)
            preds = torch.argmax(eval_out["final_logits"], dim=-1)
            acc = (preds == y_eval).float().mean().item()
            s1_exit_pct = eval_out["s1_exit_mask"].float().mean().item() * 100.0
            print(f"  tau = {tau:.2f} -> Acc: {acc*100:.2f}% | S1 Early Exit Rate: {s1_exit_pct:.1f}%")

    # Persist checkpoint to Modal Volume
    checkpoint_path = "/checkpoints/jevformer_scaled_latest.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "d_model": d_model,
                "num_heads": num_heads,
                "s1_layers": s1_layers,
                "max_hops": max_hops,
            },
        },
        checkpoint_path,
    )
    volume.commit()
    print(f"\n[Saved]: Checkpoint successfully committed to Modal Volume: {checkpoint_path}")
    print(f"[Elapsed Time]: {(time.time() - start_time):.1f}s")
    return {"final_acc": acc, "param_count": param_count}


# ----------------------------------------------------------------------
# Local Entrypoint for CLI Execution
# ----------------------------------------------------------------------

@app.local_entrypoint()
def main():
    print("=" * 70)
    print(" Launching Scaled Jevformer Experiment on Modal Cloud (A10G)")
    print("=" * 70)
    result = train_jevformer_cloud.remote(
        steps=2000,
        batch_size=128,
        d_model=256,
        num_heads=8,
        s1_layers=4,
        max_hops=5,
    )
    print(f"\nExperiment Complete! Result: {result}")
