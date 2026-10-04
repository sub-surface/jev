"""
=============================================================================
Fused Epistemic Router & Speculative Arbiter (Sub-Microsecond Dispatch)
=============================================================================
High-performance inference kernel for Epistemic Speculative Decoding (ESD)
and Epistemic MCTS (E-MCTS):
  1. Computes Top-1 decision and Top-2 decision margin in a single pass.
  2. Evaluates calibrated epistemic credence (Noul) normalized to [0, 1].
  3. Gated fast-path decision arbiter (bypasses deliberative LLM when Noul >= tau).
  4. Epistemic UCT exploration bonus computation:
       U_epi(s, a) = Q(s, a) + (1 - Noul(s, a)) * sqrt(ln(N) / (N_a + 1))
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import math
from typing import NamedTuple, Optional
import torch
import torch.nn.functional as F


class EpistemicRoutingDecision(NamedTuple):
    """Output of the sub-microsecond epistemic router."""
    selected_option: torch.Tensor       # (B,) int64 Top-1 index
    top1_prob: torch.Tensor             # (B,) float32 credence
    noul: torch.Tensor                  # (B,) float32 epistemic certainty in [0, 1]
    decision_margin: torch.Tensor       # (B,) float32 gap between Top-1 and Top-2 prob
    fast_path_mask: torch.Tensor        # (B,) bool flag (True -> emit reflex, False -> route to System 2)
    search_depth_budget: torch.Tensor   # (B,) int32 recommended MCTS ply depth


def evaluate_epistemic_routing(
    logits: torch.Tensor,
    num_options: torch.Tensor,
    temperature: float = 1.0,
    noul_threshold: float = 0.85,
    margin_threshold: float = 0.40,
    quiet_depth: int = 1,
    deep_depth: int = 6,
) -> EpistemicRoutingDecision:
    """
    Sub-microsecond batched epistemic evaluation and routing.

    Args:
        logits: (B, max_k) unnormalized option logits from OptionScorer
        num_options: (B,) valid option count K_b
        temperature: fitted calibration temperature
        noul_threshold: confidence threshold for fast-path reflex bypass
        margin_threshold: top-1 vs top-2 probability gap threshold
        quiet_depth: ply depth allocated when confident (Rule 6 clock management)
        deep_depth: ply depth allocated during sharp epistemic crises

    Returns:
        EpistemicRoutingDecision with decisions, credences, and routing masks
    """
    B, max_k = logits.shape
    device = logits.device

    idx = torch.arange(max_k, device=device).unsqueeze(0).expand(B, max_k)
    valid_mask = idx < num_options.unsqueeze(1)

    # Scaled masked logits
    z = torch.where(valid_mask, logits / temperature, torch.tensor(-1e9, device=device, dtype=logits.dtype))
    probs = F.softmax(z, dim=-1)
    probs_clean = torch.where(valid_mask, probs, torch.zeros_like(probs))

    # Top-1 and Top-2 values and indices
    sorted_probs, sorted_indices = torch.sort(probs_clean, dim=-1, descending=True)
    top1_val = sorted_probs[:, 0]
    top1_idx = sorted_indices[:, 0]

    top2_val = torch.where(
        num_options > 1,
        sorted_probs[:, 1],
        torch.zeros_like(top1_val),
    )
    margin = top1_val - top2_val

    # Epistemic Noul: 1 - sqrt( (k / (2*(k-1))) * sum( (p - 1/k)^2 ) )
    inv_k = 1.0 / num_options.clamp(min=1).unsqueeze(1).to(probs.dtype)
    dev_sq = torch.where(valid_mask, (probs_clean - inv_k) ** 2, torch.zeros_like(probs_clean)).sum(dim=-1)
    k_float = num_options.to(probs.dtype)
    k_minus_1 = (k_float - 1.0).clamp(min=1.0)
    norm_factor = k_float / (2.0 * k_minus_1)
    dist = torch.sqrt(torch.clamp(dev_sq * norm_factor, min=0.0, max=1.0))
    noul = torch.where(num_options > 1, 1.0 - dist, torch.ones_like(dist))

    # Fast-Path Gating condition:
    # Accept fast path if BOTH Noul and Top-1 margin clear calibrated thresholds
    fast_path = (noul >= noul_threshold) & (margin >= margin_threshold)

    # Dynamic Clock Management (Rule 6):
    # Quiet plies spend minimal depth; crises spend deep search depth
    search_depth = torch.where(fast_path, quiet_depth, deep_depth).to(torch.int32)

    return EpistemicRoutingDecision(
        selected_option=top1_idx,
        top1_prob=top1_val,
        noul=noul,
        decision_margin=margin,
        fast_path_mask=fast_path,
        search_depth_budget=search_depth,
    )


def compute_epistemic_uct_bonus(
    q_values: torch.Tensor,
    noul_values: torch.Tensor,
    visit_counts: torch.Tensor,
    parent_visits: int,
    c_base: float = 1.0,
) -> torch.Tensor:
    """
    Epistemic Upper Confidence Bound for Trees (E-MCTS):
      U_epi(s, a) = Q(s, a) + c_base * (1.0 - Noul(s, a)) * sqrt(ln(N) / (N(s, a) + 1))

    Properties:
      - As Noul -> 1 (high epistemic certainty), exploration term -> 0 (zero compute wasted).
      - As Noul -> 0 (crisis/ambiguity), exploration bonus swells to resolve uncertainty.
    """
    ln_parent = math.log(max(parent_visits, 1))
    visit_term = torch.sqrt(ln_parent / (visit_counts.float() + 1.0))
    epistemic_uncertainty = (1.0 - noul_values).clamp(min=0.0, max=1.0)
    exploration_bonus = c_base * epistemic_uncertainty * visit_term
    return q_values + exploration_bonus
