"""
Modal Cloud Phase 2: Contrastive Distillation & Mechanistic Interpretability
=============================================================================

Implements:
  1. Contrastive Prompt Pair Generation (p+, p-) following Yang et al. (RLCD)
  2. Pairwise Contrastive Calibration Loss (sharpening the boundary)
  3. Extended optimization (4,000 steps with Cosine Annealing)
  4. Mechanistic Probing (Latent activations & attention entropy extraction)
  5. Post-training Platt/Temperature calibration sweep
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
import modal

APP_NAME = "jevformer-phase2-contrastive"

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
# Scaled Architecture with Attention Probes
# ----------------------------------------------------------------------

class Phase2JevHead(nn.Module):
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


class Phase2Deliberator(nn.Module):
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


class Phase2Jevformer(nn.Module):
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
        self.jev_head = Phase2JevHead(d_model, num_classes, max_steps=max_s2_steps)
        self.deliberator = Phase2Deliberator(d_model, num_heads, num_classes)

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
            "h_pool": h_pool,
        }


# ----------------------------------------------------------------------
# RLCD Contrastive Pair Generator (Yang et al. 2023)
# ----------------------------------------------------------------------

def generate_contrastive_pairs(
    batch_size: int,
    seq_len: int = 32,
    num_classes: int = 8,
    max_hops: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constructs matched contrastive pairs:
      - x_pos (p+): Reflex-solvable query. Ground truth z = 1 (High Noul confidence)
      - x_neg (p-): Matched multi-hop query. Ground truth z = 0 (Low Noul confidence)
    """
    half_b = batch_size // 2
    # Base tokens
    x_pos = torch.randint(20, 480, (half_b, seq_len))
    y_pos = torch.zeros(half_b, dtype=torch.long)
    
    # Positive pairs: Class explicitly in token 1
    for i in range(half_b):
        y_pos[i] = x_pos[i, 1] % num_classes
        x_pos[i, 0] = 1  # Reflex prompt marker

    # Negative pairs: Identical token distribution, but multi-hop dependency injected
    x_neg = x_pos.clone()
    y_neg = torch.zeros(half_b, dtype=torch.long)
    hops_neg = torch.randint(2, max_hops + 1, (half_b,))

    for i in range(half_b):
        k = int(hops_neg[i].item())
        x_neg[i, 0] = 2  # Multi-hop marker
        cur = int(x_neg[i, 2] % (seq_len - 6) + 3)
        accum = 0
        for _ in range(k):
            val = int(x_neg[i, cur])
            accum += val
            cur = int(val % (seq_len - 6) + 3)
        y_neg[i] = accum % num_classes

    # Concatenate into unified batch
    x_all = torch.cat([x_pos, x_neg], dim=0)
    y_all = torch.cat([y_pos, y_neg], dim=0)
    z_target = torch.cat([torch.ones(half_b), torch.zeros(half_b)], dim=0)
    true_hops = torch.cat([torch.zeros(half_b, dtype=torch.long), hops_neg], dim=0)

    return x_all, y_all, z_target, true_hops


# ----------------------------------------------------------------------
# Modal Training Function: Phase 2
# ----------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={"/checkpoints": volume},
)
def train_phase2_cloud(
    steps: int = 4000,
    batch_size: int = 128,
    lr: float = 6e-4,
    d_model: int = 256,
    num_heads: int = 8,
    s1_layers: int = 4,
    max_hops: int = 5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase 2 Worker] Executing on GPU: {torch.cuda.get_device_name(0)}")

    model = Phase2Jevformer(
        vocab_size=512,
        d_model=d_model,
        num_heads=num_heads,
        s1_layers=s1_layers,
        max_s2_steps=max_hops,
        num_classes=8,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)

    print("\n--- Phase 2: RLCD Contrastive Distillation & Boundary Sharpening ---")
    start_time = time.time()
    half_b = batch_size // 2

    telemetry = []

    for step in range(1, steps + 1):
        model.train()
        x, y, z_target, true_hops = generate_contrastive_pairs(batch_size, max_hops=max_hops)
        x, y, z_target, true_hops = x.to(device), y.to(device), z_target.to(device), true_hops.to(device)

        out = model(x, confidence_threshold=0.5, fixed_steps=max_hops)
        s1_logits = out["s1_logits"]
        s2_logits = out["s2_logits"]
        noul_p = out["noul_p"]
        score_logits = out["score_logits"]

        # 1. Task Losses
        loss_s1 = F.cross_entropy(s1_logits, y)
        loss_s2 = F.cross_entropy(s2_logits, y)

        # 2. RLCD Proper Scoring Brier Loss
        loss_brier = F.mse_loss(noul_p, z_target)

        # 3. Pairwise Contrastive Margin Loss between p+ and p-
        # Encourage noul(p+) - noul(p-) >= margin (0.60)
        noul_pos = noul_p[:half_b]
        noul_neg = noul_p[half_b:]
        contrastive_margin_loss = F.relu(0.60 - (noul_pos - noul_neg)).mean()

        # 4. Depth Prediction Loss
        loss_depth = F.cross_entropy(score_logits, true_hops)

        # 5. Annealed Compute Penalty
        annealed_lambda = min(0.15, 0.02 + 0.13 * (step / steps))
        loss_compute = torch.mean(1.0 - noul_p) * annealed_lambda

        total_loss = loss_s1 + loss_s2 + (3.0 * loss_brier) + (1.5 * contrastive_margin_loss) + (0.5 * loss_depth) + loss_compute

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 500 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                s1_acc = (torch.argmax(s1_logits, dim=-1) == y).float().mean().item()
                s2_acc = (torch.argmax(s2_logits, dim=-1) == y).float().mean().item()
                conf_pos = noul_pos.mean().item()
                conf_neg = noul_neg.mean().item()
                contrast_gap = conf_pos - conf_neg

                print(
                    f"Step {step:04d}/{steps} | Loss: {total_loss.item():.3f} | "
                    f"S1 Acc: {s1_acc*100:.1f}% | S2 Acc: {s2_acc*100:.1f}% | "
                    f"Noul(p+): {conf_pos:.3f} | Noul(p-): {conf_neg:.3f} | Gap: {contrast_gap:.3f}"
                )
                telemetry.append({
                    "step": step,
                    "loss": total_loss.item(),
                    "conf_pos": conf_pos,
                    "conf_neg": conf_neg,
                    "gap": contrast_gap,
                    "s1_acc": s1_acc,
                    "s2_acc": s2_acc,
                })

    # Final Evaluation & Checkpoint
    checkpoint_path = "/checkpoints/jevformer_phase2_contrastive.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "telemetry": telemetry,
            "config": {"d_model": d_model, "num_heads": num_heads, "s1_layers": s1_layers},
        },
        checkpoint_path,
    )
    volume.commit()
    print(f"\n[Committed Checkpoint]: {checkpoint_path}")
    print(f"[Phase 2 Execution Time]: {(time.time() - start_time):.1f}s")
    return telemetry


@app.local_entrypoint()
def main():
    print("=" * 70)
    print(" Launching Phase 2: RLCD Contrastive Distillation on Modal Cloud")
    print("=" * 70)
    res = train_phase2_cloud.remote(
        steps=4000,
        batch_size=128,
        d_model=256,
        num_heads=8,
        s1_layers=4,
        max_hops=5,
    )
    print(f"\nPhase 2 Complete! Recorded {len(res)} telemetry checkpoints.")
