"""
=============================================================================
Cross-Entropy Baseline Trainer
=============================================================================
Standard cross-entropy fine-tuning baseline (matching open reproductions like
Kev, openjev-lm, and jevlike). Used as the direct control baseline to isolate
the calibration benefits of RLCD and proper scoring rules.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import math
from typing import Optional
import torch
import torch.nn.functional as F

from decision_model.model.decision_heads import JevDecisionModel
from decision_model.model.option_marker import encode_examples
from decision_model.data.tasksource_loader import JevExample


class CrossEntropyTrainer:
    """
    Standard cross-entropy trainer for JevDecisionModel.
    """
    def __init__(
        self,
        model: JevDecisionModel,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.max_grad_norm = max_grad_norm
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        """Run one training epoch of standard Cross-Entropy fine-tuning."""
        self.model.train()

        num_batches = math.ceil(len(train_examples) / batch_size)
        total_loss = 0.0

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
            )

            self.optimizer.zero_grad()

            padded_logits, _, _ = self.model.forward_padded(tokenized)

            # Standard cross-entropy loss against ground truth target
            loss = F.cross_entropy(padded_logits, tokenized.labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.max_grad_norm,
            )
            self.optimizer.step()

            total_loss += loss.item()

            if (step + 1) % log_interval == 0 or (step + 1) == num_batches:
                avg_l = total_loss / (step + 1)
                print(
                    f"[CE Baseline] Epoch {epoch_idx+1}/{total_epochs} | "
                    f"Batch {step+1}/{num_batches} | "
                    f"CE Loss: {avg_l:.4f}",
                    flush=True,
                )

        return total_loss / num_batches
