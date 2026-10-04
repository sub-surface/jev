"""
=============================================================================
⚡ ERET: EPISTEMIC RECURRENT EQUILIBRIUM TRANSFORMER (JEVFORMER 2.0)
=============================================================================
Contemporary 2026 Bounded-Compute Neural Chess Engine:
1. Outer Arrow of Time (plies): Emits actions and searches candidate branches.
2. Inner Arrow of Time (micro-time k): Weight-tied Krasnoselskii-Mann equilibrium
   relaxation in latent space:
      Noul_k = σ(W_noul h_k) ∈ [0, 1]
      γ_k = (1 - Noul_k)
      h_{k+1} = (1 - γ_k) * h_k + γ_k * Block(h_k)
   Halts dynamically when Noul_k >= τ (Adaptive Test-Time Deliberation).
3. Pure Bitter-Lesson: Zero PeSTO, zero piece-square tables, zero spite check bonuses.
=============================================================================
"""

import os
import sys
import math
import time
from typing import Tuple, Dict, Any, Optional, List
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Rule 1: Windows Unicode encoding hygiene
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class ResidualBlock(nn.Module):
    """Squeeze-and-Excitation / Residual Block for 2D Chess Feature Maps."""
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        # SE Attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # SE channel gating
        b, c, _, _ = out.shape
        w = self.se(out).view(b, c, 1, 1)
        out = out * w
        out = F.relu(out + res)
        return out


class ERETChessEngine(nn.Module):
    """
    Epistemic Recurrent Equilibrium Transformer for Chess.
    Features:
    - Input stem: 13 planes -> 64 channels
    - Looped Krasnoselskii-Mann Equilibrium Block (weight-tied)
    - Dynamic Noul Halting (1 step for quiet plies, up to 4 for crises)
    - 128-neuron CReLU Accumulator
    - Dual Heads: Value V(s) in [-1, +1], Policy Prior P(s) in R^4096, Noul in [0, 1]
    """
    def __init__(self, max_unrolls: int = 4, tau_halt: float = 0.70):
        super().__init__()
        self.max_unrolls = max_unrolls
        self.tau_halt = tau_halt

        # Input Stem
        self.stem = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Weight-Tied Looped Equilibrium Block
        self.eq_block = ResidualBlock(channels=64)

        # Latent Epistemic Halting Sensor (predicts convergence & certainty)
        self.noul_sensor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 128-neuron CReLU Accumulator
        self.fc_accum = nn.Linear(64 * 8 * 8, 128)
        self.ln_accum = nn.LayerNorm(128)

        # Value Head
        self.val_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        # Policy Head (4096 logits)
        self.policy_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 4096)
        )

    def forward(
        self,
        x: torch.Tensor,
        force_unrolls: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns:
          val: (B,) outcome prediction
          pol_logits: (B, 4096) move priors
          noul: (B,) epistemic confidence
          crelu: (B, 128) clamped activations
          unrolls_taken: int
        """
        B = x.shape[0]
        h = self.stem(x)

        # Inner Krasnoselskii-Mann Iteration
        # h_{k+1} = (1 - γ_k) h_k + γ_k * Block(h_k)
        # where γ_k = 1 - Noul_k
        unrolls_taken = 0
        final_noul = torch.zeros((B,), device=x.device)

        for k in range(self.max_unrolls):
            unrolls_taken += 1
            noul_k = self.noul_sensor(h).squeeze(-1)  # shape (B,)
            final_noul = noul_k

            if force_unrolls is not None:
                if k >= force_unrolls:
                    break
            elif torch.all(noul_k >= self.tau_halt) and k >= 1:
                # Early exit: fixed point attractor reached
                break

            gamma_k = (1.0 - noul_k).view(B, 1, 1, 1)
            # Damped equilibrium transition
            h_next = (1.0 - gamma_k) * h + gamma_k * self.eq_block(h)
            h = h_next

        # Accumulator & Readout
        flat_h = h.view(B, -1)
        accum = self.ln_accum(self.fc_accum(flat_h))
        crelu = torch.clamp(accum, 0.0, 1.0)

        val = self.val_head(crelu).squeeze(-1)
        pol_logits = self.policy_head(crelu)

        return val, pol_logits, final_noul, crelu, unrolls_taken


def encode_board_13(board: chess.Board) -> np.ndarray:
    t = np.zeros((13, 8, 8), dtype=np.float32)
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            r = 7 - chess.square_rank(sq)
            c = chess.square_file(sq)
            pt = p.piece_type - 1
            plane = pt if p.color == chess.WHITE else pt + 6
            t[plane, r, c] = 1.0
    if board.turn == chess.WHITE:
        t[12, :, :] = 1.0
    return t


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing ERET Engine on {device}...")
    model = ERETChessEngine().to(device)
    model.eval()

    dummy_x = torch.randn(4, 13, 8, 8, device=device)
    with torch.no_grad():
        v, p, noul, crelu, unrolls = model(dummy_x)

    print(f"Val shape: {v.shape}, range: [{v.min().item():.2f}, {v.max().item():.2f}]")
    print(f"Policy shape: {p.shape}")
    print(f"Noul shape: {noul.shape}, mean: {noul.mean().item():.3f}")
    print(f"CReLU shape: {crelu.shape}, sparsity: {(crelu == 0.0).float().mean().item():.1%}")
    print(f"Equilibrium Unrolls: {unrolls}")
    print("✅ ERET Architecture verified!")
