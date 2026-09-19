"""
=============================================================================
Calibration Metrics & Reliability Evaluation
=============================================================================
Implements rigorous epistemic calibration metrics:
  - Expected Calibration Error (ECE, 15-bin equal-width)
  - Maximum Calibration Error (MCE)
  - Brier Score (multi-class & binary)
  - Negative Log-Likelihood (NLL)
  - Reliability Diagrams data generator
  - Risk-Coverage Curves & Selective Classification Accuracy
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import math
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
import torch


@dataclass
class ReliabilityBin:
    bin_idx: int
    lower: float
    upper: float
    accuracy: float
    confidence: float
    count: int


@dataclass
class CalibrationReport:
    accuracy: float
    ece: float
    mce: float
    brier_score: float
    nll: float
    num_samples: int
    reliability_bins: list[ReliabilityBin]
    risk_coverage: dict[str, float]  # e.g. {"cov_1.0": 0.82, "cov_0.8": 0.91, ...}

    def summary(self, label: str = "Evaluation") -> str:
        lines = [
            f"─── {label} Calibration Report ───",
            f"Samples:      {self.num_samples:,}",
            f"Top-1 Acc:    {self.accuracy * 100:.2f}%",
            f"ECE (15-bin): {self.ece * 100:.2f}%",
            f"MCE:          {self.mce * 100:.2f}%",
            f"Brier Score:  {self.brier_score:.4f}",
            f"Mean NLL:     {self.nll:.4f}",
            f"Coverage Acc: 100% cov -> {self.risk_coverage.get('cov_1.0', 0)*100:.1f}%, "
            f"80% cov -> {self.risk_coverage.get('cov_0.8', 0)*100:.1f}%, "
            f"50% cov -> {self.risk_coverage.get('cov_0.5', 0)*100:.1f}%",
        ]
        return "\n".join(lines)


def compute_calibration_metrics(
    probs_list: list[np.ndarray],
    labels_list: list[int],
    num_bins: int = 15,
) -> CalibrationReport:
    """
    Compute calibration metrics across a list of probability distributions
    and ground-truth label indices. Supports variable option counts per example.

    Args:
        probs_list: List of 1D numpy arrays summing to 1.0
        labels_list: List of ground-truth integer label indices
        num_bins: Number of bins for ECE (default: 15)

    Returns:
        CalibrationReport with full metrics
    """
    assert len(probs_list) == len(labels_list), "Mismatch between probs and labels length"
    N = len(probs_list)
    if N == 0:
        raise ValueError("Empty predictions list passed to compute_calibration_metrics")

    confidences = np.zeros(N, dtype=np.float64)
    accuracies = np.zeros(N, dtype=np.float64)
    brier_sum = 0.0
    nll_sum = 0.0

    for i, (p, y) in enumerate(zip(probs_list, labels_list)):
        p = np.asarray(p, dtype=np.float64)
        k = len(p)
        pred_label = int(np.argmax(p))
        conf = float(p[pred_label])
        is_correct = 1.0 if pred_label == y else 0.0

        confidences[i] = conf
        accuracies[i] = is_correct

        # Brier score: sum_{c} (p_c - y_c)^2
        one_hot = np.zeros(k, dtype=np.float64)
        if 0 <= y < k:
            one_hot[y] = 1.0
        brier_sum += float(np.sum((p - one_hot) ** 2))

        # NLL: -log(p_y)
        p_y = max(1e-12, float(p[y]) if 0 <= y < k else 1e-12)
        nll_sum += -math.log(p_y)

    top1_accuracy = float(np.mean(accuracies))
    mean_brier = brier_sum / N
    mean_nll = nll_sum / N

    # ─── Binned ECE & MCE ─────────────────────────────────────────────────────────
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    reliability_bins = []
    ece = 0.0
    mce = 0.0

    for b in range(num_bins):
        lower = bin_boundaries[b]
        upper = bin_boundaries[b + 1]

        # Samples falling into this confidence interval
        if b == num_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)

        count = int(np.sum(in_bin))
        if count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            gap = abs(bin_acc - bin_conf)
            ece += (count / N) * gap
            mce = max(mce, gap)
        else:
            bin_acc = 0.0
            bin_conf = (lower + upper) / 2.0

        reliability_bins.append(ReliabilityBin(
            bin_idx=b,
            lower=float(lower),
            upper=float(upper),
            accuracy=bin_acc,
            confidence=bin_conf,
            count=count,
        ))

    # ─── Risk-Coverage Curve ──────────────────────────────────────────────────────
    # Sort samples by confidence descending
    sorted_order = np.argsort(-confidences)
    sorted_accs = accuracies[sorted_order]

    coverage_levels = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    risk_coverage = {}
    for cov in coverage_levels:
        cutoff = max(1, int(N * cov))
        cov_acc = float(np.mean(sorted_accs[:cutoff]))
        risk_coverage[f"cov_{cov}"] = cov_acc

    return CalibrationReport(
        accuracy=top1_accuracy,
        ece=float(ece),
        mce=float(mce),
        brier_score=float(mean_brier),
        nll=float(mean_nll),
        num_samples=N,
        reliability_bins=reliability_bins,
        risk_coverage=risk_coverage,
    )
