"""Strictly proper scoring rules as training losses (lower is better).

Every function takes padded logits/probs of shape (B, K) plus an optional
boolean ``mask`` (B, K) marking which options exist, so batches can mix
cardinalities. Labels are int64 indices of shape (B,).

Note on terminology: minimising the expectation of any of these under the
data distribution *is* proper-scoring-rule training. Doing it by gradient
descent through a softmax is supervised learning, not RL — cross-entropy is
already the log score. RL only enters when the thing being scored is a
*sampled* action whose correctness is not differentiable (e.g. an LLM's
generated answer plus a verbalised confidence, as in RLCR).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    return F.log_softmax(logits.float(), dim=-1)


def log_loss(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Negative log score (= cross-entropy). Unbounded; punishes confident errors hardest."""
    return -masked_log_softmax(logits, mask).gather(1, y[:, None]).squeeze(1)


def brier_loss(probs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Multiclass Brier score sum_k (p_k - 1[k=y])^2. Bounded in [0, 2].

    Padded (non-existent) options must carry probability 0, which is what a
    masked softmax produces, so they contribute nothing.
    """
    onehot = F.one_hot(y, probs.shape[1]).to(probs.dtype)
    return ((probs - onehot) ** 2).sum(1)


def spherical_loss(probs: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Negative spherical score -p_y / ||p||_2. Bounded in [-1, 0)."""
    return -probs.gather(1, y[:, None]).squeeze(1) / probs.norm(dim=1).clamp_min(eps)


def rps_loss(probs: torch.Tensor, y: torch.Tensor, num_options: torch.Tensor | None = None) -> torch.Tensor:
    """Ranked probability score for ordinal targets (strictly proper; distance-aware).

    RPS = 1/(K-1) * sum_{k<K-1} (CDF_p(k) - 1[k >= y])^2, normalised per example
    by its own K so binary and 10-bin questions are on the same scale.
    """
    B, K = probs.shape
    idx = torch.arange(K, device=probs.device)[None, :]
    cdf_gap = (probs.cumsum(1) - (idx >= y[:, None]).to(probs.dtype)) ** 2
    if num_options is None:
        num_options = torch.full((B,), K, device=probs.device)
    valid = idx < (num_options[:, None] - 1)  # thresholds 0..K_i-2
    return (cdf_gap * valid).sum(1) / (num_options - 1).clamp_min(1).to(probs.dtype)


def proper_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor | None = None,
    rule: str = "log",
    ordinal: torch.Tensor | None = None,
    rps_weight: float = 1.0,
) -> torch.Tensor:
    """Convenience: per-example loss under one named rule, plus RPS on ordinal rows.

    A positive combination of strictly proper rules is strictly proper, so
    mixing is legitimate — but there is no free lunch: all of them share the
    same population optimum (the true conditional distribution). Differences
    between rules show up only through finite-sample / optimisation effects
    (gradient magnitude on confident errors, robustness to label noise).
    """
    probs = masked_log_softmax(logits, mask).exp()
    if rule == "log":
        loss = log_loss(logits, y, mask)
    elif rule == "brier":
        loss = brier_loss(probs, y)
    elif rule == "spherical":
        loss = spherical_loss(probs, y)
    else:
        raise ValueError(f"unknown rule {rule!r}")
    if ordinal is not None and ordinal.any():
        k = mask.sum(1) if mask is not None else None
        loss = loss + rps_weight * rps_loss(probs, y, k) * ordinal.to(loss.dtype)
    return loss
