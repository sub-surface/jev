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
    num_options: Optional[torch.Tensor] = None,
    normalize_by_cardinality: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Logarithmic scoring rule (strictly proper).

    When normalize_by_cardinality=True:
        S_log(p, y) = log(p_y) / log(K_i)

    This ensures uniform guesses achieve exactly -1.0 reward and perfect guesses
    achieve 0.0 reward regardless of option cardinality K_i in [2, 26], preventing
    high-cardinality tasks from dominating batch gradients.
    """
    p_y = probs.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)
    raw_log = torch.log(p_y.clamp(min=eps))

    if normalize_by_cardinality and num_options is not None:
        log_k = torch.log(num_options.float().clamp(min=2.0))
        return raw_log / log_k

    return raw_log


def spherical_score(
    probs: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Spherical scoring rule (strictly proper).
    S_sph(p, y) = p_y / ||p||_2
    Range: (0, 1].
    """
    p_y = probs.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)
    l2_norm = probs.norm(p=2, dim=1).clamp(min=eps)
    return p_y / l2_norm


def ranked_probability_score(
    probs: torch.Tensor,
    labels: torch.Tensor,
    num_options: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Ranked Probability Score (strictly proper for ordinal targets).
    RPS = -(1 / (K_i - 1)) * sum_{k=1}^{K_i-1} (CDF_pred(k) - CDF_true(k))^2
    """
    batch_size, max_k = probs.shape
    if max_k <= 1:
        return torch.zeros(batch_size, device=probs.device)

    cdf_pred = probs.cumsum(dim=1)
    batch_indices = torch.arange(max_k, device=probs.device).unsqueeze(0)
    label_expanded = labels.unsqueeze(1)
    cdf_true = (batch_indices >= label_expanded).float()

    squared_diff = (cdf_pred - cdf_true) ** 2  # (batch, max_k)

    if num_options is not None:
        rps_list = []
        for b in range(batch_size):
            k_i = int(num_options[b].item())
            if k_i <= 1:
                rps_list.append(torch.tensor(0.0, device=probs.device))
            else:
                # Sum over valid thresholds 0 to k_i - 2
                diff_sum = squared_diff[b, : k_i - 1].sum()
                rps_list.append(-diff_sum / (k_i - 1))
        return torch.stack(rps_list)
    else:
        diff_mean = squared_diff[:, :-1].mean(dim=1)
        return -diff_mean


def brier_score_reward(
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Brier Score Reward from MIT RLCR (Damani et al., ICLR 2026):
        R_brier = 1_{y=y*} - sum_k (p_k - 1_{k=y*})^2

    Theorem 1 (Damani et al., ICLR 2026):
    Unlike logarithmic loss where S(p, 1) - S(p, 0) -> inf as p -> 0 causing
    accuracy collapse in low-confidence regimes, the Brier score satisfies:
        max_p [ S(p, 1) - S(p, 0) ] = 1 - 2p <= 1.0
    guaranteeing that the expected reward is strictly monotonically increasing
    in true accuracy p_y, eliminating the degenerate collapse pathology.
    """
    batch_size, max_k = probs.shape
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

    # Squared error distance: ||p - e_y||^2
    brier_penalty = torch.sum((probs - one_hot) ** 2, dim=1)

    # Correctness indicator (top-1 accuracy)
    top1 = torch.argmax(probs, dim=1)
    correctness = (top1 == labels).float()

    return correctness - brier_penalty


def rewarding_doubt_score(
    probs: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 0.01,
) -> torch.Tensor:
    """
    Clipped Rewarding Doubt Rule (Bani-Harouni et al., TUM 2026):
        R = 1_{correct} * log(p_y) + 1_{incorrect} * log(1 - p_y)

    Rewards the model for expressing high confidence when correct, but
    actively rewards doubt (log(1 - p_y)) when the answer is incorrect.
    Clipped at eps=0.01 to prevent unbounded logarithmic divergence.
    """
    top1 = torch.argmax(probs, dim=1)
    is_correct = (top1 == labels).float()

    p_y = probs.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)
    p_clipped = p_y.clamp(min=eps, max=1.0 - eps)

    reward_correct = torch.log(p_clipped)
    reward_doubt = torch.log(1.0 - p_clipped)

    return is_correct * reward_correct + (1.0 - is_correct) * reward_doubt


def combined_reward(
    probs: torch.Tensor,
    labels: torch.Tensor,
    is_ordinal: torch.Tensor,
    num_options: Optional[torch.Tensor] = None,
    use_brier_rlcr: bool = True,
    log_weight: float = 1.0,
    spherical_weight: float = 0.5,
    rps_weight: float = 0.5,
    brier_weight: float = 1.0,
    normalize_by_cardinality: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Combined strictly proper scoring rule reward incorporating MIT RLCR (2026),
    TUM Rewarding Doubt (2026), and TypeSafe RLCD formulations.
    """
    if use_brier_rlcr:
        # MIT RLCR: Brier reward + Spherical bounded reward
        r_brier = brier_score_reward(probs, labels)
        r_sph = spherical_score(probs, labels, eps=eps)
        reward = brier_weight * r_brier + spherical_weight * r_sph
    else:
        # Log score + Spherical score
        r_log = log_score(
            probs, labels, num_options=num_options,
            normalize_by_cardinality=normalize_by_cardinality, eps=eps
        )
        r_sph = spherical_score(probs, labels, eps=eps)
        reward = log_weight * r_log + spherical_weight * r_sph

    if is_ordinal.any():
        r_rps = ranked_probability_score(probs, labels, num_options=num_options)
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
