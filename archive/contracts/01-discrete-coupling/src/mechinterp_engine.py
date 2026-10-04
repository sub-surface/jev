"""
Mechanistic Interpretability Engine for Tri-Process & Dual-Scale Architectures
=============================================================================
Authors: Leon & The Research Collective
Grounding: Andrej Karpathy, Geoff Hinton, Claude Shannon, Linus Torvalds

Provides rigorous, high-speed diagnostics for extracting the activation physics,
spectral geometry, and epistemic calibration of:
  - System 0: NNUE Sparse Accumulators (CReLU activation sparsity, dead neuron leakage)
  - System 1: TypeSafe Jev Epistemic Heads (Brier proper scoring, ECE, reliability diagrams)
  - System 2: In-Context Causal Transformers (attention entropy, demonstration attribution)
  - Coupling Channels: Shannon rate-distortion, mutual information, codebook usage
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import math
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# 1. Activation Biology & CReLU Threshold Diagnostics (Geoff Hinton)
# ----------------------------------------------------------------------

@dataclass
class CReLUPhysiologyReport:
    baseline_sparsity: float       # Fraction of units = 0 in unmodulated state
    modulated_sparsity: float      # Fraction of units = 0 in modulated state
    leakage_rate: float            # Inactive units flipped to >0 by modulation
    extinction_rate: float         # Active units suppressed to 0 by modulation
    saturation_rate: float         # Units pushed to max clamp (1.0)
    mean_shift: float              # Average delta in activation
    max_shift: float               # Peak perturbation magnitude


def analyze_crelu_physics(
    base_accum: torch.Tensor,
    modulated_accum: torch.Tensor,
) -> CReLUPhysiologyReport:
    """
    Computes activation physics across Clipped ReLU layers:
      CReLU(x) = clamp(x, 0, 1)
    Quantifies the exact degree of representation interference caused by modulation.
    """
    with torch.no_grad():
        c_base = torch.clamp(base_accum, min=0.0, max=1.0)
        c_mod = torch.clamp(modulated_accum, min=0.0, max=1.0)

        total_units = float(c_base.numel())
        base_inactive = (c_base == 0.0)
        base_active = (c_base > 0.0)

        mod_inactive = (c_mod == 0.0)
        mod_saturated = (c_mod >= 1.0)

        # Spurious leakage: previously inactive units now active
        leaked = (base_inactive & (c_mod > 0.0)).sum().item()
        # Extinction: previously active units now dead
        extinguished = (base_active & mod_inactive).sum().item()
        # Saturated
        saturated = mod_saturated.sum().item()

        delta = (c_mod - c_base).abs()

        return CReLUPhysiologyReport(
            baseline_sparsity=float(base_inactive.sum().item() / total_units),
            modulated_sparsity=float(mod_inactive.sum().item() / total_units),
            leakage_rate=float(leaked / max(1.0, base_inactive.sum().item())),
            extinction_rate=float(extinguished / max(1.0, base_active.sum().item())),
            saturation_rate=float(saturated / total_units),
            mean_shift=float(delta.mean().item()),
            max_shift=float(delta.max().item()),
        )


# ----------------------------------------------------------------------
# 2. Spectral Geometry & Effective Dimensionality (Claude Shannon & Karpathy)
# ----------------------------------------------------------------------

@dataclass
class SpectralGeometryReport:
    singular_values: np.ndarray
    effective_rank: float          # Exponential of spectral entropy
    participation_ratio: float     # (sum s_i^2)^2 / sum s_i^4
    condition_number: float        # s_max / s_min
    top1_variance_ratio: float     # s_1^2 / sum s_i^2
    top5_variance_ratio: float     # sum_{i=1}^5 s_i^2 / sum s_i^2


def compute_spectral_geometry(activation_matrix: torch.Tensor, eps: float = 1e-12) -> SpectralGeometryReport:
    """
    Performs Singular Value Decomposition on batch activations [B, D]
    to measure the geometric dimensionality and information compression.
    """
    with torch.no_grad():
        # Center activations
        X = activation_matrix.float()
        if X.dim() > 2:
            X = X.view(-1, X.size(-1))
        X_centered = X - X.mean(dim=0, keepdim=True)

        B, D = X_centered.size()
        min_dim = min(B, D)
        if min_dim < 2:
            return SpectralGeometryReport(
                singular_values=np.array([1.0]),
                effective_rank=1.0,
                participation_ratio=1.0,
                condition_number=1.0,
                top1_variance_ratio=1.0,
                top5_variance_ratio=1.0,
            )

        # SVD on centered matrix
        _, S, _ = torch.linalg.svd(X_centered, full_matrices=False)
        s_np = S.cpu().numpy()

        # Variance explained
        var = s_np ** 2
        sum_var = np.sum(var) + eps
        p_var = var / sum_var

        # Shannon spectral entropy & Effective Rank
        p_norm = s_np / (np.sum(s_np) + eps)
        spectral_entropy = -np.sum(p_norm * np.log(p_norm + eps))
        effective_rank = float(np.exp(spectral_entropy))

        # Participation Ratio
        pr = float((np.sum(var) ** 2) / (np.sum(var ** 2) + eps))

        # Condition Number
        cond = float(s_np[0] / (s_np[-1] + eps))

        top1 = float(var[0] / sum_var)
        top5 = float(np.sum(var[:min(5, len(var))]) / sum_var)

        return SpectralGeometryReport(
            singular_values=s_np,
            effective_rank=effective_rank,
            participation_ratio=pr,
            condition_number=cond,
            top1_variance_ratio=top1,
            top5_variance_ratio=top5,
        )


# ----------------------------------------------------------------------
# 3. Epistemic Proper Scoring & Calibration Analyzer (TypeSafe Jev)
# ----------------------------------------------------------------------

@dataclass
class EpistemicCalibrationReport:
    brier_score: float
    expected_calibration_error: float  # ECE in %
    maximum_calibration_error: float   # MCE in %
    mean_confidence: float
    empirical_accuracy: float
    bin_confidences: list[float]
    bin_accuracies: list[float]
    bin_counts: list[int]


def compute_epistemic_calibration(
    confidences: np.ndarray | list[float],
    ground_truth: np.ndarray | list[int | bool],
    num_bins: int = 10,
) -> EpistemicCalibrationReport:
    """
    Evaluates proper scoring and Expected Calibration Error (ECE) for Jev's Noul head.
    ECE = sum_b (n_b / N) * |acc(b) - conf(b)|
    """
    confs = np.array(confidences, dtype=np.float64)
    labels = np.array(ground_truth, dtype=np.float64)
    assert len(confs) == len(labels), "Confidences and ground truth must match in length"

    total = len(confs)
    if total == 0:
        return EpistemicCalibrationReport(0.0, 0.0, 0.0, 0.0, 0.0, [], [], [])

    # Brier Score = (1/N) * sum (p_i - y_i)^2
    brier = float(np.mean((confs - labels) ** 2))

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_confs = []
    bin_accs = []
    bin_cnts = []

    for i in range(num_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        if i == num_bins - 1:
            in_bin = (confs >= bin_lower) & (confs <= bin_upper)
        else:
            in_bin = (confs >= bin_lower) & (confs < bin_upper)

        cnt = int(np.sum(in_bin))
        bin_cnts.append(cnt)

        if cnt > 0:
            mean_conf = float(np.mean(confs[in_bin]))
            mean_acc = float(np.mean(labels[in_bin]))
            diff = abs(mean_acc - mean_conf)
            ece += (cnt / float(total)) * diff
            mce = max(mce, diff)
            bin_confs.append(mean_conf)
            bin_accs.append(mean_acc)
        else:
            bin_confs.append((bin_lower + bin_upper) / 2.0)
            bin_accs.append(0.0)

    return EpistemicCalibrationReport(
        brier_score=brier,
        expected_calibration_error=float(ece * 100.0),
        maximum_calibration_error=float(mce * 100.0),
        mean_confidence=float(np.mean(confs)),
        empirical_accuracy=float(np.mean(labels) * 100.0),
        bin_confidences=bin_confs,
        bin_accuracies=bin_accs,
        bin_counts=bin_cnts,
    )


# ----------------------------------------------------------------------
# 4. System 2 Attention Entropy & Attribution Tracker
# ----------------------------------------------------------------------

@dataclass
class AttentionDiagnosticsReport:
    mean_attention_entropy: float      # Lower entropy = sharper pointer induction
    demonstration_attribution: float   # Fraction of attention from test query to demos
    head_entropies: list[float]        # Per-head sharpness


def analyze_transformer_attention(
    attn_weights: torch.Tensor,
    demo_token_range: tuple[int, int],
    test_token_range: tuple[int, int],
) -> AttentionDiagnosticsReport:
    """
    Inspects attention weights [B, H, T, T] to quantify:
      - How sharply heads attend (attention entropy)
      - Demonstration attribution (how much test queries attend to demo tokens)
    """
    with torch.no_grad():
        # Average over batch: [H, T, T]
        if attn_weights.dim() == 4:
            attn = attn_weights.mean(dim=0)
        else:
            attn = attn_weights

        H, T, _ = attn.size()
        eps = 1e-12

        # Entropy per row: -sum p log p
        entropies = -(attn * torch.log(attn + eps)).sum(dim=-1)  # [H, T]
        head_ent = entropies.mean(dim=-1).cpu().tolist()
        overall_ent = float(entropies.mean().item())

        # Cross-attribution from test range to demo range
        d_start, d_end = demo_token_range
        t_start, t_end = test_token_range

        if t_end > t_start and d_end > d_start:
            sub_attn = attn[:, t_start:t_end, d_start:d_end]
            demo_attr = float(sub_attn.sum(dim=-1).mean().item())
        else:
            demo_attr = 0.0

        return AttentionDiagnosticsReport(
            mean_attention_entropy=overall_ent,
            demonstration_attribution=demo_attr,
            head_entropies=head_ent,
        )
