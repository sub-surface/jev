"""
=============================================================================
Evaluation Harness — In-Task vs Zero-Shot Family Generalization
=============================================================================
Evaluates JevDecisionModel on both:
  1. In-Task Validation Suite (seen task families, held-out examples)
  2. Zero-Shot Task Family Suite (entire task families never seen during training)

Computes calibration metrics (ECE, Brier, accuracy, NLL, risk-coverage)
both with and without per-cardinality temperature scaling, and measures
the generalization gap honestly.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
from pathlib import Path
from typing import Optional
import numpy as np
import torch

from decision_model.model.decision_heads import JevDecisionModel
from decision_model.model.option_marker import encode_examples
from decision_model.model.temperature_scaling import CardinalityTemperatureScaler
from decision_model.data.tasksource_loader import JevExample
from decision_model.data.splits import DataSplit
from decision_model.evaluation.calibration_metrics import (
    compute_calibration_metrics,
    CalibrationReport,
)


def evaluate_dataset(
    model: JevDecisionModel,
    examples: list[JevExample],
    tokenizer,
    marker_token_ids: list[int],
    batch_size: int = 4,
    max_length: int = 512,
    temp_scaler: Optional[CardinalityTemperatureScaler] = None,
    device: Optional[torch.device] = None,
) -> tuple[CalibrationReport, list[torch.Tensor], list[int]]:
    """
    Run full evaluation on an example list.

    Returns:
        report: CalibrationReport with accuracy, ECE, Brier, NLL, etc.
        all_logits: Raw unscaled or scaled logits per example
        all_labels: Ground-truth label indices
    """
    model.eval()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_probs: list[np.ndarray] = []
    all_logits: list[torch.Tensor] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch_examples = examples[i : i + batch_size]
            tokenized = encode_examples(
                batch_examples,
                tokenizer,
                marker_token_ids,
                max_length=max_length,
                device=device,
            )

            batch_logits, _ = model.forward_batch(tokenized)

            for b_idx, l in enumerate(batch_logits):
                raw_l = l.detach().cpu().float()
                k = len(raw_l)

                if temp_scaler is not None:
                    scaled_l = temp_scaler.scale_logits(raw_l, k)
                    probs = torch.softmax(scaled_l, dim=-1).numpy()
                    all_logits.append(scaled_l)
                else:
                    probs = torch.softmax(raw_l, dim=-1).numpy()
                    all_logits.append(raw_l)

                all_probs.append(probs)
                all_labels.append(int(tokenized.labels[b_idx].item()))

    report = compute_calibration_metrics(all_probs, all_labels)
    return report, all_logits, all_labels


def run_full_evaluation_suite(
    model: JevDecisionModel,
    split: DataSplit,
    tokenizer,
    marker_token_ids: list[int],
    temp_scaler: Optional[CardinalityTemperatureScaler] = None,
    batch_size: int = 4,
    device: Optional[torch.device] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Run comprehensive evaluation on both In-Task and Zero-Shot splits,
    computing the generalization gap.
    """
    print("\n" + "=" * 60, flush=True)
    print("RUNNING IN-TASK EVALUATION SUITE", flush=True)
    print("=" * 60, flush=True)
    in_task_report, _, _ = evaluate_dataset(
        model=model,
        examples=split.in_task_val_examples,
        tokenizer=tokenizer,
        marker_token_ids=marker_token_ids,
        batch_size=batch_size,
        temp_scaler=temp_scaler,
        device=device,
    )
    print(in_task_report.summary(label="In-Task Validation"), flush=True)

    print("\n" + "=" * 60, flush=True)
    print("RUNNING ZERO-SHOT FAMILY EVALUATION SUITE", flush=True)
    print("=" * 60, flush=True)
    zero_shot_report, _, _ = evaluate_dataset(
        model=model,
        examples=split.zero_shot_examples,
        tokenizer=tokenizer,
        marker_token_ids=marker_token_ids,
        batch_size=batch_size,
        temp_scaler=temp_scaler,
        device=device,
    )
    print(zero_shot_report.summary(label="Zero-Shot Task Families"), flush=True)

    # Generalization gap
    acc_gap = in_task_report.accuracy - zero_shot_report.accuracy
    ece_gap = zero_shot_report.ece - in_task_report.ece
    brier_gap = zero_shot_report.brier_score - in_task_report.brier_score

    print("\n" + "=" * 60, flush=True)
    print("EPISTEMIC GENERALIZATION GAP (In-Task vs Zero-Shot Family)", flush=True)
    print("=" * 60, flush=True)
    print(f"Accuracy Drop:   {acc_gap * 100:+.2f}% ({in_task_report.accuracy*100:.2f}% -> {zero_shot_report.accuracy*100:.2f}%)", flush=True)
    print(f"ECE Degradation: {ece_gap * 100:+.2f}% ({in_task_report.ece*100:.2f}% -> {zero_shot_report.ece*100:.2f}%)", flush=True)
    print(f"Brier Increase:  {brier_gap:+.4f} ({in_task_report.brier_score:.4f} -> {zero_shot_report.brier_score:.4f})", flush=True)

    results = {
        "in_task": {
            "accuracy": in_task_report.accuracy,
            "ece": in_task_report.ece,
            "mce": in_task_report.mce,
            "brier_score": in_task_report.brier_score,
            "nll": in_task_report.nll,
            "num_samples": in_task_report.num_samples,
            "risk_coverage": in_task_report.risk_coverage,
        },
        "zero_shot": {
            "accuracy": zero_shot_report.accuracy,
            "ece": zero_shot_report.ece,
            "mce": zero_shot_report.mce,
            "brier_score": zero_shot_report.brier_score,
            "nll": zero_shot_report.nll,
            "num_samples": zero_shot_report.num_samples,
            "risk_coverage": zero_shot_report.risk_coverage,
        },
        "generalization_gap": {
            "accuracy_drop": acc_gap,
            "ece_degradation": ece_gap,
            "brier_increase": brier_gap,
        },
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "eval_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_dir / 'eval_results.json'}", flush=True)

    return results
