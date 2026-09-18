"""
Tri-Process Neural Architecture: NNUE (System 0) + Jev (System 1) + In-Context Transformer (System 2)
=====================================================================================================
Authors: Leon & Ilya Sutskever persona

The Fundamental Principle:
  Intelligence across complex, combinatorial domains requires a multi-scale hierarchy:
    - System 0 (NNUE): Sparse, incrementally updatable feature accumulator with Clipped ReLU.
      Executes discrete counterfactual search at microsecond latency with zero representation drift.
    - System 1 (TypeSafe Jev): Calibrated epistemic sensor emitting (Value, Noul).
      Acts as the dynamic governor: allowing reflexive play when Noul >= tau, and triggering
      deeper search or escalation when Noul < tau.
    - System 2 (In-Context Transformer): Causal multi-head self-attention module (nanoGPT style).
      Ingests sequence traces of failed branches, past games, or demonstration pairs in-context,
      synthesizing continuous modulation vectors and symbolic sub-goals to steer System 0 and 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# 1. System 0: NNUE Sparse Accumulator Engine
# ----------------------------------------------------------------------

class NNUESparseAccumulator(nn.Module):
    """
    Stockfish-style Efficiently Updatable Neural Network (NNUE).
    First layer is a linear feature accumulator: a = W * x + b.
    Activation: Clipped ReLU (CReLU) in [0.0, 1.0].
    Incremental update:
      a_new = a_old - sum(W[:, removed]) + sum(W[:, added])
    This eliminates matrix-vector multiplication during search, enabling
    O(k * D) branch evaluations where k is the number of changed features.
    """
    def __init__(self, num_features: int, accumulator_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.num_features = num_features
        self.accumulator_dim = accumulator_dim

        # Accumulator weight and bias
        self.w_accum = nn.Parameter(torch.randn(num_features, accumulator_dim) * (1.0 / math.sqrt(num_features)))
        self.b_accum = nn.Parameter(torch.zeros(accumulator_dim))

        # Hidden layer and output
        self.fc_hidden = nn.Linear(accumulator_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward_from_features(self, active_feature_indices: list[int], modulation: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass from scratch given list of active feature indices."""
        if not active_feature_indices:
            accum = self.b_accum.clone()
        else:
            accum = self.w_accum[active_feature_indices].sum(dim=0) + self.b_accum

        if modulation is not None:
            accum = accum + modulation

        crelu = torch.clamp(accum, min=0.0, max=1.0)
        h = torch.clamp(self.fc_hidden(crelu), min=0.0, max=1.0)
        out = self.fc_out(h).squeeze(-1)
        return out, accum

    def forward_batch_accum(self, accum_batch: torch.Tensor, modulation: torch.Tensor | None = None) -> torch.Tensor:
        """Batch forward evaluation directly from accumulated state vectors."""
        if modulation is not None:
            accum_batch = accum_batch + modulation
        crelu = torch.clamp(accum_batch, min=0.0, max=1.0)
        h = torch.clamp(self.fc_hidden(crelu), min=0.0, max=1.0)
        return self.fc_out(h).squeeze(-1)

    def incremental_update(
        self,
        old_accum: torch.Tensor,
        removed_indices: list[int],
        added_indices: list[int],
    ) -> torch.Tensor:
        """Incremental accumulator update in O(k * D) time."""
        delta = torch.zeros_like(old_accum)
        if removed_indices:
            delta -= self.w_accum[removed_indices].sum(dim=0)
        if added_indices:
            delta += self.w_accum[added_indices].sum(dim=0)
        return old_accum + delta


# ----------------------------------------------------------------------
# 2. System 1: TypeSafe Jev Calibrated Epistemic Head
# ----------------------------------------------------------------------

class JevEpistemicHead(nn.Module):
    """
    TypeSafe Jev Calibrated Head:
      Input: Feature representation (from NNUE accumulator or state embedding).
      Outputs:
        - Value V(s) in [-1, 1] or [0, 1]: estimated objective / win advantage.
        - Noul(s) in [0, 1]: calibrated epistemic certainty (probability of safety/soundness).
    """
    def __init__(self, in_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.val_head = nn.Linear(hidden_dim, 1)
        self.noul_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.net(h)
        val = torch.tanh(self.val_head(feat)).squeeze(-1)
        noul = self.noul_head(feat).squeeze(-1)
        return val, noul


# ----------------------------------------------------------------------
# 3. System 2: In-Context Causal Transformer (nanoGPT-style Deliberator)
# ----------------------------------------------------------------------

@dataclass
class TransformerConfig:
    vocab_size: int = 256
    seq_len: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.seq_len, cfg.seq_len)).view(1, 1, cfg.seq_len, cfg.seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class InContextTransformer(nn.Module):
    """
    System 2 In-Context Meta-Reasoner:
      Ingests tokens representing:
        - Demonstration examples (ARC grids or math decompositions)
        - Historical games or failed search branches
      Outputs:
        1. Contextual Modulation Vector m in R^{accumulator_dim} to bias System 0 NNUE.
        2. Sub-goal / invariant tokens for heuristic pruning.
    """
    def __init__(self, cfg: TransformerConfig, modulation_dim: int = 128):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Modulation head (emits vector to steer NNUE accumulator)
        self.mod_head = nn.Sequential(
            nn.Linear(cfg.d_model, modulation_dim),
            nn.Tanh(),
        )
        # Next-token / Subgoal prediction head
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = idx.size()
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # Pooled representation from the final context token
        last_tok = x[:, -1, :]
        modulation = self.mod_head(last_tok)
        logits = self.lm_head(x)
        return modulation, logits


# ----------------------------------------------------------------------
# 4. Integrated Tri-Process Agent
# ----------------------------------------------------------------------

class TriProcessAgent(nn.Module):
    """
    Unified Tri-Process Agent:
      - System 0: NNUESparseAccumulator (discrete high-speed search)
      - System 1: JevEpistemicHead (calibrated epistemic gate)
      - System 2: InContextTransformer (in-context deliberation & modulation)
    """
    def __init__(
        self,
        num_features: int,
        accumulator_dim: int = 128,
        transformer_cfg: TransformerConfig | None = None,
    ):
        super().__init__()
        self.nnue = NNUESparseAccumulator(num_features=num_features, accumulator_dim=accumulator_dim)
        self.jev = JevEpistemicHead(in_dim=accumulator_dim)
        t_cfg = transformer_cfg or TransformerConfig(vocab_size=256, seq_len=128, d_model=128)
        self.transformer = InContextTransformer(t_cfg, modulation_dim=accumulator_dim)

    def deliberate_context(self, context_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """System 2 in-context forward pass: returns (modulation_vector, next_token_logits)."""
        return self.transformer(context_tokens)

    def evaluate_position(
        self,
        active_features: list[int],
        modulation: torch.Tensor | None = None,
    ) -> tuple[float, float, torch.Tensor]:
        """
        System 0 + System 1 joint evaluation:
          Returns:
            - val: scalar position score
            - noul: epistemic certainty in [0, 1]
            - accum: raw accumulator tensor for incremental operations
        """
        nnue_score, accum = self.nnue.forward_from_features(active_features, modulation=modulation)
        val, noul = self.jev(accum.unsqueeze(0))
        return val.item(), noul.item(), accum
