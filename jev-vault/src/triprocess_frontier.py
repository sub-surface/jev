"""
Frontier Tri-Process Architecture: NNUE (System 0) + Jev (System 1) + In-Context Transformer (System 2)
========================================================================================================
Authors: Leon & The Research Collective
Architectural Principles:
  1. System 0 (NNUE Sparse Accumulator): Microsecond discrete counterfactual evaluations.
     Supports 4 coupling modes:
       - 'dense_additive': a_tilde = a + m  (causes CReLU leakage)
       - 'multiplicative': a_tilde = a * sigmoid(m) (preserves zeros, 0% leakage)
       - 'discrete_mask': a_tilde = a * M_discrete (subspace invariant)
       - 'vq_bottleneck': a_tilde conditioned on vector-quantized codebook tokens
  2. System 1 (TypeSafe Jev): Calibrated epistemic sensor (Value, Noul).
     Trained with Brier proper scoring to govern search depth.
  3. System 2 (In-Context Causal Transformer): Deliberator with discrete codebook bottleneck.
     Emits discrete invariants (factor tokens, rule tokens, or VQ indices) to steer System 0.
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import math
from enum import Enum
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CouplingMode(str, Enum):
    DENSE_ADDITIVE = "dense_additive"
    MULTIPLICATIVE = "multiplicative"
    DISCRETE_MASK = "discrete_mask"
    VQ_BOTTLENECK = "vq_bottleneck"


# ----------------------------------------------------------------------
# 1. Vector Quantization Bottleneck (Shannon Channel Discretization)
# ----------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """
    Vector Quantization Bottleneck (Shannon Channel Discretization).
    Forces continuous deliberation vectors into K discrete codebook vectors.
    Channel capacity = log2(K) bits.
    Trained with Straight-Through Estimator (STE).
    """
    def __init__(self, num_embeddings: int = 16, embedding_dim: int = 128, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / math.sqrt(num_embeddings), 1.0 / math.sqrt(num_embeddings))

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input: z_e [B, D]
        Returns:
          z_q: Quantized latent [B, D] (with STE gradient)
          loss: VQ codebook loss + commitment loss
          indices: [B] discrete codebook indices
        """
        # Distance ||z_e - e_k||^2 = ||z_e||^2 + ||e_k||^2 - 2 z_e e_k
        d = (
            torch.sum(z_e ** 2, dim=-1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=-1)
            - 2 * torch.matmul(z_e, self.embedding.weight.t())
        )
        encoding_indices = torch.argmin(d, dim=-1)
        z_q = self.embedding(encoding_indices)

        # Loss terms: codebook loss + commitment loss
        loss_codebook = F.mse_loss(z_q, z_e.detach())
        loss_commitment = F.mse_loss(z_e, z_q.detach())
        loss = loss_codebook + self.commitment_cost * loss_commitment

        # Straight-Through Estimator: gradients flow from z_q back to z_e
        z_q = z_e + (z_q - z_e).detach()

        return z_q, loss, encoding_indices


# ----------------------------------------------------------------------
# 2. System 0: NNUE Sparse Accumulator with Pluggable Coupling
# ----------------------------------------------------------------------

class NNUEFrontierAccumulator(nn.Module):
    """
    Stockfish-style Efficiently Updatable Neural Network (NNUE).
    First layer: a = W * x + b
    Activation: Clipped ReLU (CReLU) in [0.0, 1.0].
    """
    def __init__(
        self,
        num_features: int,
        accumulator_dim: int = 128,
        hidden_dim: int = 64,
        coupling_mode: CouplingMode = CouplingMode.DISCRETE_MASK,
    ):
        super().__init__()
        self.num_features = num_features
        self.accumulator_dim = accumulator_dim
        self.coupling_mode = coupling_mode

        self.w_accum = nn.Parameter(torch.randn(num_features, accumulator_dim) * (1.0 / math.sqrt(num_features)))
        self.b_accum = nn.Parameter(torch.zeros(accumulator_dim))

        self.fc_hidden = nn.Linear(accumulator_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward_from_features(
        self,
        active_feature_indices: list[int],
        modulation: torch.Tensor | None = None,
        discrete_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates position from sparse active features.
        Applies modulation according to coupling_mode.
        """
        if not active_feature_indices:
            accum = self.b_accum.clone()
        else:
            valid_feats = [f for f in active_feature_indices if f < self.num_features]
            if valid_feats:
                accum = self.w_accum[valid_feats].sum(dim=0) + self.b_accum
            else:
                accum = self.b_accum.clone()

        # Apply Coupling Physics
        if modulation is not None:
            if self.coupling_mode == CouplingMode.DENSE_ADDITIVE:
                accum = accum + modulation
            elif self.coupling_mode == CouplingMode.MULTIPLICATIVE:
                accum = accum * torch.sigmoid(modulation)
            elif self.coupling_mode == CouplingMode.VQ_BOTTLENECK:
                # Modulate multiplicatively using VQ codebook vector (preserves 0-thresholds)
                accum = accum * torch.sigmoid(modulation)

        if discrete_mask is not None:
            accum = accum * discrete_mask

        crelu = torch.clamp(accum, min=0.0, max=1.0)
        h = torch.clamp(self.fc_hidden(crelu), min=0.0, max=1.0)
        out = self.fc_out(h).squeeze(-1)
        return out, accum

    def forward_batch_accum(
        self,
        accum_batch: torch.Tensor,
        modulation: torch.Tensor | None = None,
        discrete_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Batch evaluation directly from accumulated state vectors."""
        accum = accum_batch
        if modulation is not None:
            if self.coupling_mode == CouplingMode.DENSE_ADDITIVE:
                accum = accum + modulation
            elif self.coupling_mode in (CouplingMode.MULTIPLICATIVE, CouplingMode.VQ_BOTTLENECK):
                accum = accum * torch.sigmoid(modulation)
        if discrete_mask is not None:
            accum = accum * discrete_mask

        crelu = torch.clamp(accum, min=0.0, max=1.0)
        h = torch.clamp(self.fc_hidden(crelu), min=0.0, max=1.0)
        return self.fc_out(h).squeeze(-1)


# ----------------------------------------------------------------------
# 3. System 1: TypeSafe Jev Calibrated Epistemic Head
# ----------------------------------------------------------------------

class JevEpistemicHead(nn.Module):
    """
    TypeSafe Jev Calibrated Epistemic Sensor:
      Maps feature state h to:
        - Value V(s) in [-1, 1] (win/solution probability)
        - Noul(s) in [0, 1] (calibrated epistemic confidence)
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
# 4. System 2: In-Context Causal Transformer with VQ Channel
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att_weights = F.softmax(att, dim=-1)

        y = att_weights @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y), att_weights


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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, att_weights = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, att_weights


class InContextDeliberator(nn.Module):
    """
    System 2 In-Context Deliberator:
      Ingests sequence tokens (demonstration pairs or match traces).
      Emits:
        1. Discrete symbolic next-token / rule prediction logits.
        2. Vector-Quantized (Shannon) discrete latent vector z_q.
        3. Raw unquantized latent vector z_e.
        4. Attention weights from all layers for MechInterp.
    """
    def __init__(
        self,
        cfg: TransformerConfig,
        modulation_dim: int = 128,
        vq_num_codes: int = 16,
    ):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Raw continuous modulation projection
        self.mod_proj = nn.Linear(cfg.d_model, modulation_dim)

        # Vector Quantizer discrete bottleneck
        self.vq = VectorQuantizer(num_embeddings=vq_num_codes, embedding_dim=modulation_dim)

        # Symbolic next-token head
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        B, T = idx.size()
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]

        layer_attns = []
        for block in self.blocks:
            x, att_w = block(x)
            layer_attns.append(att_w)

        x = self.ln_f(x)
        last_tok = x[:, -1, :]

        # Continuous latent
        z_e = torch.tanh(self.mod_proj(last_tok))

        # Vector Quantized discrete latent
        z_q, vq_loss, vq_indices = self.vq(z_e)

        # Token prediction logits
        logits = self.lm_head(x)

        return {
            "logits": logits,
            "z_e": z_e,
            "z_q": z_q,
            "vq_loss": vq_loss,
            "vq_indices": vq_indices,
            "attentions": torch.stack(layer_attns, dim=1),  # [B, L, H, T, T]
        }


# ----------------------------------------------------------------------
# 5. Unified Frontier Tri-Process Agent
# ----------------------------------------------------------------------

class FrontierTriProcessAgent(nn.Module):
    """
    Unified Frontier Tri-Process Agent:
      System 0: NNUEFrontierAccumulator
      System 1: JevEpistemicHead
      System 2: InContextDeliberator
    """
    def __init__(
        self,
        num_features: int,
        accumulator_dim: int = 128,
        transformer_cfg: TransformerConfig | None = None,
        coupling_mode: CouplingMode = CouplingMode.VQ_BOTTLENECK,
        vq_num_codes: int = 16,
    ):
        super().__init__()
        self.coupling_mode = coupling_mode
        self.nnue = NNUEFrontierAccumulator(
            num_features=num_features,
            accumulator_dim=accumulator_dim,
            coupling_mode=coupling_mode,
        )
        self.jev = JevEpistemicHead(in_dim=accumulator_dim)
        t_cfg = transformer_cfg or TransformerConfig(vocab_size=256, seq_len=128, d_model=128)
        self.transformer = InContextDeliberator(
            t_cfg,
            modulation_dim=accumulator_dim,
            vq_num_codes=vq_num_codes,
        )

    def deliberate_context(self, context_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.transformer(context_tokens)

    def evaluate_position(
        self,
        active_features: list[int],
        modulation: torch.Tensor | None = None,
        discrete_mask: torch.Tensor | None = None,
    ) -> tuple[float, float, torch.Tensor]:
        score, accum = self.nnue.forward_from_features(
            active_features,
            modulation=modulation,
            discrete_mask=discrete_mask,
        )
        val, noul = self.jev(accum.unsqueeze(0))
        return val.item(), noul.item(), accum
