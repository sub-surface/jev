"""
=============================================================================
Per-Cardinality Temperature Scaling (Guo et al. 2017 Extension)
=============================================================================
Fits temperature scaling parameters per option-cardinality bucket:
  - Bucket 2: [2 options] (binary / noul)
  - Bucket 3_5: [3 to 5 options]
  - Bucket 6_10: [6 to 10 options]
  - Bucket 11_plus: [11+ options]

As noted on Laya/TypeSafe model cards, modern neural networks exhibit different
calibration distortion profiles depending on the cardinality of the candidate
set. Fitting temperature separately per cardinality bucket resolves overconfidence
at both binary and multi-choice regimes.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CardinalityTemperatureScaler(nn.Module):
    """
    Learns and applies temperature scaling parameters per option cardinality bucket.
    """
    def __init__(self):
        super().__init__()
        # Initialized to T = 1.0 (no scaling)
        # Stored in log-space for unconstrained positive temperature: T = exp(log_T)
        self.log_T_2 = nn.Parameter(torch.zeros(1))       # 2 options
        self.log_T_3_5 = nn.Parameter(torch.zeros(1))     # 3-5 options
        self.log_T_6_10 = nn.Parameter(torch.zeros(1))    # 6-10 options
        self.log_T_11_plus = nn.Parameter(torch.zeros(1)) # 11+ options

    def get_bucket_key(self, num_options: int) -> str:
        if num_options <= 2:
            return "2"
        elif num_options <= 5:
            return "3_5"
        elif num_options <= 10:
            return "6_10"
        else:
            return "11_plus"

    def get_temperature(self, num_options: int) -> torch.Tensor:
        key = self.get_bucket_key(num_options)
        if key == "2":
            return torch.exp(self.log_T_2)
        elif key == "3_5":
            return torch.exp(self.log_T_3_5)
        elif key == "6_10":
            return torch.exp(self.log_T_6_10)
        else:
            return torch.exp(self.log_T_11_plus)

    def scale_logits(self, logits: torch.Tensor, num_options: int) -> torch.Tensor:
        """Scale 1D or 2D logits tensor by the bucket temperature."""
        T = self.get_temperature(num_options)
        return logits / T

    def scale_batch(self, batch_logits: list[torch.Tensor]) -> list[torch.Tensor]:
        """Scale a list of variable-length logit tensors."""
        scaled = []
        for l in batch_logits:
            num_opts = len(l)
            scaled.append(self.scale_logits(l, num_opts))
        return scaled

    def fit(
        self,
        val_logits: list[torch.Tensor],
        val_labels: list[int],
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> dict[str, float]:
        """
        Fit temperature parameters per bucket on validation logits and labels
        by minimizing negative log-likelihood (cross-entropy) via L-BFGS.
        """
        # Group logits and labels by bucket
        buckets: dict[str, tuple[list[torch.Tensor], list[int]]] = {
            "2": ([], []),
            "3_5": ([], []),
            "6_10": ([], []),
            "11_plus": ([], []),
        }

        for logits, label in zip(val_logits, val_labels):
            k = len(logits)
            b_key = self.get_bucket_key(k)
            buckets[b_key][0].append(logits.detach().cpu().float())
            buckets[b_key][1].append(label)

        fitted_temps = {}

        for b_key, (b_logits, b_labels) in buckets.items():
            if not b_logits:
                fitted_temps[b_key] = 1.0
                continue

            # Select target parameter
            if b_key == "2":
                param = self.log_T_2
            elif b_key == "3_5":
                param = self.log_T_3_5
            elif b_key == "6_10":
                param = self.log_T_6_10
            else:
                param = self.log_T_11_plus

            optimizer = optim.LBFGS([param], lr=lr, max_iter=max_iter)
            l2_reg = 0.1  # Prior shrinkage towards T = 1.0 (log_T = 0.0)

            def eval_closure():
                optimizer.zero_grad()
                loss = 0.0
                # Clamp log_param to prevent catastrophic temperature collapse (T in [0.5, 3.0])
                T = torch.exp(param.clamp(min=-0.693, max=1.098))
                for l_item, y_item in zip(b_logits, b_labels):
                    scaled_l = l_item / T
                    target = torch.tensor([y_item], dtype=torch.long)
                    loss = loss + F.cross_entropy(scaled_l.unsqueeze(0), target)
                loss = (loss / len(b_logits)) + l2_reg * (param ** 2)
                loss.backward()
                return loss

            optimizer.step(eval_closure)
            # Final clamped temperature value
            with torch.no_grad():
                param.clamp_(min=-0.693, max=1.098)
            fitted_temps[b_key] = torch.exp(param).item()
            print(
                f"  Bucket [{b_key:>7s}]: {len(b_logits)} examples -> "
                f"fitted temperature T = {fitted_temps[b_key]:.4f}",
                flush=True,
            )

        return fitted_temps

    def save(self, path: Path):
        """Save fitted temperatures to JSON."""
        temps = {
            "T_2": torch.exp(self.log_T_2).item(),
            "T_3_5": torch.exp(self.log_T_3_5).item(),
            "T_6_10": torch.exp(self.log_T_6_10).item(),
            "T_11_plus": torch.exp(self.log_T_11_plus).item(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(temps, f, indent=2)

    def load(self, path: Path):
        """Load fitted temperatures from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            temps = json.load(f)
        with torch.no_grad():
            self.log_T_2.copy_(torch.tensor([temps.get("T_2", 1.0)]).log())
            self.log_T_3_5.copy_(torch.tensor([temps.get("T_3_5", 1.0)]).log())
            self.log_T_6_10.copy_(torch.tensor([temps.get("T_6_10", 1.0)]).log())
            self.log_T_11_plus.copy_(torch.tensor([temps.get("T_11_plus", 1.0)]).log())
