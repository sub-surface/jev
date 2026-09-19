"""
=============================================================================
RLCD Trainer — Reinforcement Learning for Calibrated Decisions
=============================================================================
Implementation of TypeSafe's RLCD training methodology:
  1. Forward pass extracts candidate option logits z_i at marker tokens.
  2. Gaussian exploration noise is injected into the logits: z_noisy = z + sigma * eps.
  3. The policy reports the probability distribution p_noisy = Softmax(z_noisy).
  4. Reward is computed via strictly proper scoring rules:
       R = w_log * S_log(p, y) + w_sph * S_sph(p, y) + w_rps * RPS(p, y).
  5. Gradients flow through the reparameterized action distribution, directly
     optimizing the expected proper scoring rule reward.
  6. Exploration noise sigma is annealed across epochs.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import math
from typing import Optional, Callable
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from decision_model.config import RLCDConfig
from decision_model.model.decision_heads import JevDecisionModel
from decision_model.model.option_marker import TokenizedBatch, encode_examples
from decision_model.training.scoring_rules import (
    combined_reward,
    add_exploration_noise,
)
from decision_model.data.tasksource_loader import JevExample


class RLCDTrainer:
    """
    Trains JevDecisionModel via RLCD using strictly proper scoring rule rewards.
    """
    def __init__(
        self,
        model: JevDecisionModel,
        config: RLCDConfig,
        lr: float = 5e-5,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        self.max_grad_norm = max_grad_norm
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configure trainable parameters (LoRA + OptionScorer)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )

    def train_epoch(
        self,
        train_examples: list[JevExample],
        tokenizer,
        marker_token_ids: list[int],
        epoch_idx: int,
        total_epochs: int,
        batch_size: int = 4,
        max_length: int = 512,
        log_interval: int = 25,
    ) -> float:
        """Run one training epoch of RLCD."""
        self.model.train()

        # Compute current exploration noise sigma (linear annealing)
        progress = epoch_idx / max(1, total_epochs - 1)
        sigma = self.config.exploration_sigma_start + progress * (
            self.config.exploration_sigma_end - self.config.exploration_sigma_start
        )

        num_batches = math.ceil(len(train_examples) / batch_size)
        total_loss = 0.0
        total_reward = 0.0

        # Shuffle examples
        import random
        indices = list(range(len(train_examples)))
        random.shuffle(indices)

        for step in range(num_batches):
            batch_idxs = indices[step * batch_size : (step + 1) * batch_size]
            batch_examples = [train_examples[i] for i in batch_idxs]

            tokenized = encode_examples(
                batch_examples,
                tokenizer,
                marker_token_ids,
                max_length=max_length,
                device=self.device,
                shuffle_options=True,
            )

            self.optimizer.zero_grad()

            # Padded forward pass
            padded_logits, _, valid_mask = self.model.forward_padded(tokenized)
            batch_sz, max_k = padded_logits.shape

            # Add Gaussian exploration noise to valid option logits
            noise = torch.randn_like(padded_logits) * sigma
            noisy_logits = torch.where(valid_mask, padded_logits + noise, padded_logits)

            # Re-normalize over valid options
            noisy_probs = torch.softmax(noisy_logits, dim=-1)

            num_opts_t = torch.tensor(tokenized.num_options, device=self.device, dtype=torch.long)

            # Compute strictly proper scoring rewards with cardinality normalization
            rewards = combined_reward(
                probs=noisy_probs,
                labels=tokenized.labels,
                is_ordinal=tokenized.is_ordinal,
                num_options=num_opts_t,
                log_weight=self.config.log_score_weight,
                spherical_weight=self.config.spherical_score_weight,
                rps_weight=self.config.rps_weight,
                normalize_by_cardinality=True,
            )

            # RLCD objective: maximize expected reward -> minimize negative reward
            loss = -rewards.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.max_grad_norm,
            )
            self.optimizer.step()

            total_loss += loss.item()
            total_reward += rewards.mean().item()

            if (step + 1) % log_interval == 0 or (step + 1) == num_batches:
                avg_r = total_reward / (step + 1)
                avg_l = total_loss / (step + 1)
                print(
                    f"Epoch {epoch_idx+1}/{total_epochs} | "
                    f"Batch {step+1}/{num_batches} | "
                    f"Sigma: {sigma:.4f} | "
                    f"Mean Reward: {avg_r:.4f} | "
                    f"Loss: {avg_l:.4f}",
                    flush=True,
                )

        return total_loss / num_batches
