"""
Jevformer: The Calibrated Dual-Process Transformer
===================================================

A prototype exploring the fusion of:
1. System 1 (Jev): Constant-depth forward pass with calibrated Noul/Choice/Score heads.
2. System 2 (LLM Deliberator): Recurrent reasoning depth / test-time compute.

Trained with RLCD (Reinforcement Learning for Calibrated Decisions) + Compute Penalties,
the model learns *epistemic self-awareness*: it exits early on intuitive tasks and
allocates deep recurrent compute only when its confidence is genuinely low.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 1. Architecture: The Jevformer Module
# ----------------------------------------------------------------------

class JevHead(nn.Module):
    """
    System 1 Perceptual Head.
    Directly evaluates intermediate representations to output:
      - Choice: Task class logits
      - Noul: Calibrated probability P(System 1 is sufficient)
      - Score: Predicted reasoning depth needed
    """
    def __init__(self, d_model: int, num_classes: int, max_reasoning_steps: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        # Choice head (Task logits)
        self.choice_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes)
        )

        # Noul head (Calibrated probability of S1 sufficiency)
        # Using a dedicated projection with temperature scaling
        self.noul_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        self.temp = nn.Parameter(torch.ones(1) * 1.5)

        # Score head (Predicted reasoning steps needed: 0 to K)
        self.score_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, max_reasoning_steps + 1)
        )

    def forward(self, h_pool: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          choice_logits: [batch, num_classes]
          noul_prob:     [batch, 1]  (calibrated probability in [0, 1])
          score_logits:  [batch, max_steps + 1]
        """
        choice_logits = self.choice_proj(h_pool)
        
        # Sigmoid with learned temperature for calibrated epistemic confidence
        noul_raw = self.noul_proj(h_pool) / torch.clamp(self.temp, min=0.1, max=10.0)
        noul_prob = torch.sigmoid(noul_raw)
        
        score_logits = self.score_proj(h_pool)
        return choice_logits, noul_prob, score_logits


class System2Deliberator(nn.Module):
    """
    System 2 Deliberator (Recurrent Reasoning Blocks).
    Performs iterative internal computation (inference-time search/refinement).
    """
    def __init__(self, d_model: int, num_classes: int, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.refined_proj = nn.Linear(d_model, num_classes)

    def step(self, h_seq: torch.Tensor) -> torch.Tensor:
        """Executes a single step of recurrent deliberation."""
        return self.layer(h_seq)

    def forward(self, h_seq: torch.Tensor, steps: int) -> torch.Tensor:
        """Executes K steps of sequential reasoning."""
        cur = h_seq
        for _ in range(steps):
            cur = self.step(cur)
        h_out = self.final_norm(cur[:, 0, :])
        return self.refined_proj(h_out)


class Jevformer(nn.Module):
    """
    The Calibrated Dual-Process Transformer.
    Integrates System 1 (Jev) and System 2 (Deliberation) within one substrate.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_classes: int = 4,
        s1_layers: int = 2,
        max_s2_steps: int = 3,
        num_heads: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_s2_steps = max_s2_steps

        # Shared perceptual trunk (early layers)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 128, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True
        )
        self.trunk = nn.TransformerEncoder(encoder_layer, num_layers=s1_layers)

        # System 1: Calibrated Jev Head
        self.jev_head = JevHead(d_model, num_classes, max_reasoning_steps=max_s2_steps)

        # System 2: Deliberative Reasoner
        self.deliberator = System2Deliberator(d_model, num_classes, num_heads=num_heads)

    def forward(
        self,
        x: torch.Tensor,
        confidence_threshold: float = 0.70,
        force_s2: bool = False,
    ) -> dict[str, Any]:
        """
        Forward pass with adaptive compute routing.
        """
        b, seq_len = x.shape
        h = self.token_emb(x) + self.pos_emb[:, :seq_len, :]
        h_seq = self.trunk(h)
        h_pool = h_seq[:, 0, :]  # CLS-style pooled token

        # 1. System 1 Forward Pass (Jev Evaluation)
        s1_choice_logits, noul_conf, score_logits = self.jev_head(h_pool)

        # 2. Dynamic Routing Decision
        # If confidence >= threshold, early exit. Otherwise, enter System 2 deliberation.
        s1_exit_mask = (noul_conf.squeeze(-1) >= confidence_threshold) & (~torch.tensor(force_s2))

        # 3. System 2 Deliberation for samples below threshold
        s2_logits = None
        s2_steps_allocated = torch.zeros(b, dtype=torch.long, device=x.device)

        if not s1_exit_mask.all():
            # Estimate needed steps from Score head
            pred_steps = torch.argmax(score_logits, dim=-1).clamp(min=1, max=self.max_s2_steps)
            s2_steps_allocated = torch.where(s1_exit_mask, 0, pred_steps)
            
            # Execute deliberation for the batch
            # For simplicity in vectorized code, we run the max needed steps
            max_steps = int(pred_steps.max().item())
            s2_logits = self.deliberator(h_seq, steps=max_steps)

        # Final composite output
        if s2_logits is not None:
            mask_expanded = s1_exit_mask.unsqueeze(-1)
            final_logits = torch.where(mask_expanded, s1_choice_logits, s2_logits)
        else:
            final_logits = s1_choice_logits

        return {
            "final_logits": final_logits,
            "s1_choice_logits": s1_choice_logits,
            "s2_logits": s2_logits,
            "noul_confidence": noul_conf.squeeze(-1),
            "score_logits": score_logits,
            "s1_exit_mask": s1_exit_mask,
            "s2_steps": s2_steps_allocated,
        }


# ----------------------------------------------------------------------
# 2. Synthetic Benchmark: Reflex vs. Multi-hop Logic
# ----------------------------------------------------------------------

def generate_synthetic_data(batch_size: int = 128, seq_len: int = 16, num_classes: int = 4):
    """
    Creates a mixture of:
    - 50% Reflex Tasks (Solvable by System 1 shallow feature matching)
    - 50% Multi-hop Tasks (Requires composition / deep sequential graph walk)
    """
    tokens = torch.randint(10, 100, (batch_size, seq_len))
    labels = torch.zeros(batch_size, dtype=torch.long)
    is_hard_task = torch.rand(batch_size) > 0.5

    for i in range(batch_size):
        if not is_hard_task[i]:
            # Reflex task: Class is determined directly by first token modulo num_classes
            labels[i] = tokens[i, 1] % num_classes
            tokens[i, 0] = 1  # prefix indicating easy
        else:
            # Hard multi-hop task: pointer arithmetic across 3 hops
            hop1 = int(tokens[i, 2] % (seq_len - 4) + 3)
            hop2 = int(tokens[i, hop1] % (seq_len - 4) + 3)
            hop3 = int(tokens[i, hop2] % (seq_len - 4) + 3)
            labels[i] = (tokens[i, hop1] + tokens[i, hop2] + tokens[i, hop3]) % num_classes
            tokens[i, 0] = 2  # prefix indicating multi-hop

    return tokens, labels, is_hard_task


# ----------------------------------------------------------------------
# 3. Training Loop: The RLCD + Compute Penalty Loss
# ----------------------------------------------------------------------

def train_jevformer_experiment(epochs: int = 350, batch_size: int = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device]: Utilizing {device} for Jevformer execution.")

    model = Jevformer(vocab_size=120, d_model=64, num_classes=4, s1_layers=2, max_s2_steps=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)

    print("\n--- Phase 1: Joint Pre-training & RLCD Calibration ---")
    print("Training Jevformer where:")
    print("  1. System 1 learns fast predictions + calibrated Noul confidence.")
    print("  2. System 2 learns iterative multi-hop deliberation.")
    print("  3. RLCD penalizes miscalibration (Brier score) and compute overhead.\n")

    for epoch in range(1, epochs + 1):
        x, y, is_hard = generate_synthetic_data(batch_size=batch_size)
        x, y, is_hard = x.to(device), y.to(device), is_hard.to(device)

        # Force both branches to compute during training so both receive gradients
        out = model(x, confidence_threshold=0.5, force_s2=False)

        s1_logits = out["s1_choice_logits"]
        s2_logits = model.deliberator(model.trunk(model.token_emb(x) + model.pos_emb[:, :x.size(1), :]), steps=3)
        noul_p = out["noul_confidence"]

        # 1. Task Cross-Entropy Loss
        loss_s1 = F.cross_entropy(s1_logits, y)
        loss_s2 = F.cross_entropy(s2_logits, y)

        # 2. Epistemic Calibration Target:
        # z = 1 if S1 got the answer right, 0 otherwise (Strict Ground Truth Calibration)
        s1_pred = torch.argmax(s1_logits, dim=-1)
        z = (s1_pred == y).float().detach()

        # RLCD Loss: Brier Score on Noul confidence
        loss_calib = F.mse_loss(noul_p, z)

        # 3. Compute Penalty: Incentivize S1 when confident, penalize unnecessary S2
        loss_compute = torch.mean(1.0 - noul_p) * 0.15

        total_loss = loss_s1 + loss_s2 + (2.0 * loss_calib) + loss_compute

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 70 == 0 or epoch == epochs:
            with torch.no_grad():
                s1_acc = (s1_pred == y).float().mean().item()
                s2_acc = (torch.argmax(s2_logits, dim=-1) == y).float().mean().item()
                avg_conf_easy = noul_p[~is_hard].mean().item() if (~is_hard).any() else 0.0
                avg_conf_hard = noul_p[is_hard].mean().item() if is_hard.any() else 0.0
                print(
                    f"Epoch {epoch:03d} | Total Loss: {total_loss.item():.3f} | "
                    f"S1 Acc: {s1_acc*100:.1f}% | S2 Acc: {s2_acc*100:.1f}% | "
                    f"Noul Conf [Easy]: {avg_conf_easy:.3f} | [Hard]: {avg_conf_hard:.3f}"
                )

    print("\n--- Phase 2: Inference & Epistemic Gating Verification ---")
    model.eval()
    with torch.no_grad():
        x_test, y_test, is_hard_test = generate_synthetic_data(batch_size=500)
        x_test, y_test = x_test.to(device), y_test.to(device)

        # Run adaptive routing with confidence threshold tau = 0.65
        t0 = time.perf_counter()
        eval_res = model(x_test, confidence_threshold=0.65)
        eval_time_ms = (time.perf_counter() - t0) * 1000.0

        final_pred = torch.argmax(eval_res["final_logits"], dim=-1)
        final_acc = (final_pred == y_test).float().mean().item()

        s1_exits = eval_res["s1_exit_mask"]
        s1_exit_pct = s1_exits.float().mean().item() * 100.0

        # Calibration on Easy vs Hard
        easy_exit_pct = s1_exits[~is_hard_test].float().mean().item() * 100.0
        hard_exit_pct = s1_exits[is_hard_test].float().mean().item() * 100.0

        print(f"Test Accuracy across mixture: {final_acc * 100:.2f}%")
        print(f"Overall Early-Exit Rate (System 1): {s1_exit_pct:.1f}% of queries handled in constant time")
        print(f"  * Easy / Reflex tasks routed to S1 (Early Exit): {easy_exit_pct:.1f}%")
        print(f"  * Hard / Multi-hop tasks routed to S2 (Deliberation): {100.0 - hard_exit_pct:.1f}%")
        print(f"Total batch evaluation latency: {eval_time_ms:.2f} ms")


if __name__ == "__main__":
    train_jevformer_experiment()
