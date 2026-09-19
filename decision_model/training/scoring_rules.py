"""
=============================================================================
Strictly Proper Scoring Rules for RLCD
=============================================================================
Mathematical implementations of the three scoring rules used in TypeSafe's
Reinforcement Learning for Calibrated Decisions (RLCD):

1. Log Score (Logarithmic Scoring Rule)
2. Spherical Score
3. Ranked Probability Score (RPS) — for ordinal/score primitives

All three are STRICTLY PROPER: the expected score is maximized if and only
if the reported probability distribution equals the true data-generating
distribution. This is the mathematical guarantee that optimizing these
rewards produces calibrated outputs.

References:
  - Gneiting & Raftery (2007), "Strictly Proper Scoring Rules, Prediction,
    and Estimation"
  - TypeSafe AI / Laya model card: RLCD method description
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from typing import Optional


def log_score(
    probs: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Logarithmic scoring rule (strictly proper).

    S_log(p, y) = log(p_y)

    Higher is better. Range: (-inf, 0].

    Args:
        probs: (batch, num_options) predicted probabilities, must sum to 1
        labels: (batch,) integer labels indexing the true option

    Returns:
        (batch,) log scores
    """
    # Gather probability assigned to the true label
    p_y = probs.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)
    return torch.log(p_y.clamp(min=eps))


def spherical_score(
    probs: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Spherical scoring rule (strictly proper).

    S_sph(p, y) = p_y / ||p||_2

    Higher is better. Range: (0, 1].

    Args:
        probs: (batch, num_options) predicted probabilities
        labels: (batch,) integer labels

    Returns:
        (batch,) spherical scores
    """
    p_y = probs.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)
    l2_norm = probs.norm(p=2, dim=1).clamp(min=eps)
    return p_y / l2_norm


def ranked_probability_score(
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Ranked Probability Score — negated for use as reward (strictly proper).

    RPS(p, y) = -(1/(K-1)) * sum_{k=1}^{K-1} (CDF_pred(k) - CDF_true(k))^2

    This accounts for ordinal distance: predicting a nearby bin to the true
    answer is penalized less than predicting a distant bin. Essential for
    the `score` primitive where labels have meaningful order.

    Higher (less negative) is better. Range: [-(1), 0].

    Args:
        probs: (batch, K) predicted probabilities over K ordered bins
        labels: (batch,) integer labels (0 to K-1)

    Returns:
        (batch,) negated RPS (higher = better calibrated)
    """
    K = probs.shape[1]
    if K <= 1:
        return torch.zeros(probs.shape[0], device=probs.device)

    # Predicted CDF: cumulative sum of probabilities
    cdf_pred = probs.cumsum(dim=1)  # (batch, K)

    # True CDF: step function at the label
    # cdf_true[i, k] = 1 if k >= label[i], else 0
    batch_indices = torch.arange(K, device=probs.device).unsqueeze(0)  # (1, K)
    label_expanded = labels.unsqueeze(1)  # (batch, 1)
    cdf_true = (batch_indices >= label_expanded).float()  # (batch, K)

    # RPS = mean squared difference of CDFs (excluding last position which is always 1)
    squared_diff = (cdf_pred[:, :-1] - cdf_true[:, :-1]) ** 2
    rps = squared_diff.mean(dim=1)

    # Return negated so that higher = better (reward convention)
    return -rps


def combined_reward(
    probs: torch.Tensor,
    labels: torch.Tensor,
    is_ordinal: torch.Tensor,
    log_weight: float = 1.0,
    spherical_weight: float = 0.5,
    rps_weight: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Combined strictly proper scoring rule reward for RLCD.

    For categorical (choice/noul) tasks:
        R = w_log * log_score + w_sph * spherical_score

    For ordinal (score) tasks:
        R = w_log * log_score + w_sph * spherical_score + w_rps * RPS

    Args:
        probs: (batch, max_options) predicted probabilities (padded)
        labels: (batch,) integer labels
        is_ordinal: (batch,) boolean mask — True for score tasks
        log_weight, spherical_weight, rps_weight: scoring rule weights

    Returns:
        (batch,) combined reward scores
    """
    r_log = log_score(probs, labels, eps=eps)
    r_sph = spherical_score(probs, labels, eps=eps)

    reward = log_weight * r_log + spherical_weight * r_sph

    # Add RPS for ordinal tasks
    if is_ordinal.any():
        r_rps = ranked_probability_score(probs, labels)
        reward = reward + rps_weight * r_rps * is_ordinal.float()

    return reward


def add_exploration_noise(
    logits: torch.Tensor,
    sigma: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Add Gaussian exploration noise to logits before softmax.

    This is the exploration mechanism for RLCD: the model reports a noisy
    probability distribution, and the reward signal teaches it to report
    calibrated probabilities despite the noise. The noise anneals from
    sigma_start to sigma_end during training.

    Args:
        logits: (batch, num_options) raw logits before softmax
        sigma: standard deviation of Gaussian noise
        generator: optional torch Generator for reproducibility

    Returns:
        (batch, num_options) noisy logits
    """
    if sigma <= 0:
        return logits
    noise = torch.randn_like(logits, generator=generator) * sigma
    return logits + noise


# ─── Verification utilities ───────────────────────────────────────────────────

def verify_proper_scoring(n_options: int = 5, n_samples: int = 10000):
    """
    Verify that all scoring rules are strictly proper by showing that
    the expected reward is maximized when the reported distribution
    equals the true distribution.

    This is a sanity check, not a formal proof — but it should be run
    once to confirm the implementation is correct.
    """
    print(f"Verifying strict properness with {n_options} options...", flush=True)
    torch.manual_seed(42)

    # True distribution (randomly generated)
    true_probs = F.softmax(torch.randn(n_options), dim=0)
    print(f"  True distribution: {true_probs.tolist()}", flush=True)

    # Generate samples from the true distribution
    labels = torch.multinomial(true_probs, n_samples, replacement=True)

    # Test 1: Report the true distribution → should get highest expected reward
    true_batch = true_probs.unsqueeze(0).expand(n_samples, -1)
    labels_batch = labels

    true_log = log_score(true_batch, labels_batch).mean().item()
    true_sph = spherical_score(true_batch, labels_batch).mean().item()

    # Test 2: Report a uniform distribution
    uniform = torch.ones(n_options) / n_options
    uniform_batch = uniform.unsqueeze(0).expand(n_samples, -1)

    uniform_log = log_score(uniform_batch, labels_batch).mean().item()
    uniform_sph = spherical_score(uniform_batch, labels_batch).mean().item()

    # Test 3: Report a random wrong distribution
    wrong_probs = F.softmax(torch.randn(n_options), dim=0)
    wrong_batch = wrong_probs.unsqueeze(0).expand(n_samples, -1)

    wrong_log = log_score(wrong_batch, labels_batch).mean().item()
    wrong_sph = spherical_score(wrong_batch, labels_batch).mean().item()

    print(f"  Expected log  score: true={true_log:.4f}, uniform={uniform_log:.4f}, wrong={wrong_log:.4f}", flush=True)
    print(f"  Expected sph  score: true={true_sph:.4f}, uniform={uniform_sph:.4f}, wrong={wrong_sph:.4f}", flush=True)

    # Verify strict properness: true should be highest
    assert true_log >= uniform_log, f"Log score not proper: true={true_log} < uniform={uniform_log}"
    assert true_log >= wrong_log, f"Log score not proper: true={true_log} < wrong={wrong_log}"
    assert true_sph >= uniform_sph, f"Spherical score not proper: true={true_sph} < uniform={uniform_sph}"
    assert true_sph >= wrong_sph, f"Spherical score not proper: true={true_sph} < wrong={wrong_sph}"

    # Test RPS properness with ordinal labels
    labels_ord = labels  # Reuse; treat as ordinal
    true_rps = ranked_probability_score(true_batch, labels_ord).mean().item()
    uniform_rps = ranked_probability_score(uniform_batch, labels_ord).mean().item()
    wrong_rps = ranked_probability_score(wrong_batch, labels_ord).mean().item()
    print(f"  Expected RPS score: true={true_rps:.4f}, uniform={uniform_rps:.4f}, wrong={wrong_rps:.4f}", flush=True)
    assert true_rps >= uniform_rps, f"RPS not proper: true={true_rps} < uniform={uniform_rps}"

    print("  All properness checks passed!", flush=True)
    return True


if __name__ == "__main__":
    verify_proper_scoring(n_options=3)
    verify_proper_scoring(n_options=5)
    verify_proper_scoring(n_options=10)
    print("\nAll scoring rule verification passed.", flush=True)
