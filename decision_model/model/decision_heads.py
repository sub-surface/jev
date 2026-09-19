"""
=============================================================================
Decision Heads & Jev Decision Model Architecture
=============================================================================
Unified architecture embedding calibrated decision heads directly on a transformer
backbone (e.g. Qwen2.5-0.5B) with optional LoRA parameter-efficient fine-tuning.

Extracts representations at option-marker tokens [OPT_A]..[OPT_Z] and maps them
to calibrated probabilities over Choice, Noul, and Score primitives.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase
from peft import LoraConfig, get_peft_model

from decision_model.config import ModelConfig
from decision_model.model.option_marker import setup_tokenizer, TokenizedBatch


class OptionScorer(nn.Module):
    """
    Scoring head that maps a hidden state h_i at an option marker position
    to a scalar logit s_i.
    """
    def __init__(self, hidden_size: int, head_hidden: int = 256, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.GELU(),
            nn.LayerNorm(head_hidden),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (num_options, hidden_size) or (batch, num_options, hidden_size)
        Returns:
            logits: (num_options,) or (batch, num_options)
        """
        return self.net(h).squeeze(-1)


class JevDecisionModel(nn.Module):
    """
    The Jev Calibrated Decision Model.
    Wraps a transformer backbone with option-marker representation pooling
    and calibrated decision heads.
    """
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-0.5B",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[list[str]] = None,
        head_hidden: int = 256,
        torch_dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    ):
        super().__init__()
        self.base_model_name = base_model_name
        self.torch_dtype = torch_dtype

        # Load backbone
        print(f"Loading base model {base_model_name}...", flush=True)
        self.backbone = AutoModel.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        hidden_size = self.backbone.config.hidden_size

        # Apply LoRA if requested
        if use_lora:
            targets = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=targets,
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, peft_config)
            print("LoRA adapter attached. Trainable parameters:", flush=True)
            self.backbone.print_trainable_parameters()

        # Shared option marker scorer
        self.scorer = OptionScorer(
            hidden_size=hidden_size,
            head_hidden=head_hidden,
        ).to(dtype=torch_dtype)

    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize backbone embeddings after adding option marker tokens."""
        self.backbone.resize_token_embeddings(new_num_tokens)

    def forward_batch(
        self,
        batch: TokenizedBatch,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Forward pass on a TokenizedBatch.

        Returns:
            batch_logits: List of 1D tensors, where each tensor contains the logits
                          for that example's options. Length equals batch size.
            batch_probs: List of 1D probability distributions (softmax of logits).
        """
        outputs = self.backbone(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        )
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        batch_logits = []
        batch_probs = []

        for b_idx, positions in enumerate(batch.marker_positions):
            # Gather hidden states at each option marker position
            # positions: list of ints for example b_idx
            pos_tensor = torch.tensor(positions, device=last_hidden.device, dtype=torch.long)
            # h_opts: (k, hidden_size)
            h_opts = last_hidden[b_idx, pos_tensor, :]

            # Score each option marker
            logits = self.scorer(h_opts)  # (k,)
            probs = F.softmax(logits, dim=-1)

            batch_logits.append(logits)
            batch_probs.append(probs)

        return batch_logits, batch_probs

    def forward_padded(
        self,
        batch: TokenizedBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Padded forward pass returning dense tensors for vectorized loss computation.

        Returns:
            padded_logits: (batch, max_k) with -inf mask on invalid options
            padded_probs: (batch, max_k) with 0.0 mask on invalid options
            valid_mask: (batch, max_k) boolean mask
        """
        batch_logits, batch_probs = self.forward_batch(batch)
        batch_size = len(batch_logits)
        max_k = max(len(l) for l in batch_logits)
        device = batch.input_ids.device

        padded_logits = torch.full((batch_size, max_k), float("-inf"), device=device, dtype=torch.float32)
        padded_probs = torch.zeros((batch_size, max_k), device=device, dtype=torch.float32)
        valid_mask = torch.zeros((batch_size, max_k), device=device, dtype=torch.bool)

        for b in range(batch_size):
            k = len(batch_logits[b])
            padded_logits[b, :k] = batch_logits[b].to(torch.float32)
            padded_probs[b, :k] = batch_probs[b].to(torch.float32)
            valid_mask[b, :k] = True

        return padded_logits, padded_probs, valid_mask


def build_jev_decision_model(
    config: ModelConfig,
    device: Optional[torch.device] = None,
) -> tuple[JevDecisionModel, PreTrainedTokenizerBase, list[int]]:
    """
    Factory function to instantiate tokenizer with option markers,
    resize embeddings, and create JevDecisionModel.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Initializing tokenizer from {config.base_model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    tokenizer, marker_token_ids = setup_tokenizer(tokenizer)

    model = JevDecisionModel(
        base_model_name=config.base_model,
        use_lora=True,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        head_hidden=config.decision_head_hidden,
    )

    # Resize token embeddings to include [OPT_A]..[OPT_Z]
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    return model, tokenizer, marker_token_ids
