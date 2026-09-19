"""
=============================================================================
TypeSafe Jev API Head-to-Head Comparison Harness
=============================================================================
Follows the kev-vs-jev comparative methodology:
  1. Batches frozen decision questions through the official TypeSafe Jev API
     using JEV_API_KEY from .env.
  2. Compares predictions between the local/cloud RLCD model and hosted Jev.
  3. Computes:
       - Top-1 accuracy for both models
       - McNemar's test and 95% Wilson score / bootstrap confidence interval
         on the accuracy gap: (Acc_RLCD - Acc_HostedJev) +- 1.96 * SE
       - Full error-case audit: logs EVERY single disagreement instance
  4. Strict budget protection: tracks API call count, caps maximum spend
     (staying well under the $5/month limit).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import json
import math
from pathlib import Path
from typing import Optional
import numpy as np
import requests
from dotenv import load_dotenv

from decision_model.data.tasksource_loader import JevExample
from decision_model.data.task_registry import JevPrimitive

# Load .env
load_dotenv()
JEV_API_KEY = os.getenv("JEV_API_KEY")
TYPESAFE_API_URL = os.getenv("TYPESAFE_API_URL", "https://api.typesafe.ai/v1/decision")


class TypeSafeJevClient:
    """Client for querying TypeSafe's hosted Jev API."""
    def __init__(self, api_key: Optional[str] = None, max_calls: int = 200):
        self.api_key = api_key or os.getenv("JEV_API_KEY")
        self.max_calls = max_calls
        self.calls_made = 0

        if not self.api_key:
            print("⚠️ Warning: JEV_API_KEY not found in environment or .env", flush=True)

    def query_decision(self, prompt: str, options: list[str], primitive: JevPrimitive) -> Optional[dict]:
        """
        Query TypeSafe Jev API for a single decision.
        """
        if not self.api_key:
            return None

        if self.calls_made >= self.max_calls:
            print(f"🛑 Budget safeguard triggered: reached max calls limit ({self.max_calls})", flush=True)
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "prompt": prompt,
            "options": options,
            "type": primitive.value,
        }

        try:
            resp = requests.post(TYPESAFE_API_URL, headers=headers, json=payload, timeout=10)
            self.calls_made += 1
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"Jev API returned status {resp.status_code}: {resp.text[:100]}", flush=True)
                return None
        except Exception as e:
            print(f"Jev API query error: {e}", flush=True)
            return None


def compute_gap_confidence_interval(
    model_correct: list[bool],
    jev_correct: list[bool],
    alpha: float = 0.05,
) -> dict:
    """
    Compute 95% confidence interval on the performance difference
    using paired bootstrap resampling.
    """
    n = len(model_correct)
    assert n == len(jev_correct)

    m_acc = np.mean(model_correct)
    j_acc = np.mean(jev_correct)
    diff = m_acc - j_acc

    # Paired bootstrap
    n_boot = 5000
    rng = np.random.RandomState(42)
    boot_diffs = []
    for _ in range(n_boot):
        indices = rng.choice(n, n, replace=True)
        boot_m = np.mean([model_correct[i] for i in indices])
        boot_j = np.mean([jev_correct[i] for i in indices])
        boot_diffs.append(boot_m - boot_j)

    ci_lower = np.percentile(boot_diffs, 100 * (alpha / 2))
    ci_upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))

    # McNemar's contingency table
    # n00: both wrong, n01: model wrong, jev right
    # n10: model right, jev wrong, n11: both right
    n01 = sum(1 for m, j in zip(model_correct, jev_correct) if not m and j)
    n10 = sum(1 for m, j in zip(model_correct, jev_correct) if m and not j)
    mcnemar_stat = ((abs(n10 - n01) - 1) ** 2) / max(1, (n10 + n01))

    return {
        "model_accuracy": float(m_acc),
        "hosted_jev_accuracy": float(j_acc),
        "accuracy_diff": float(diff),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "mcnemar_stat": float(mcnemar_stat),
        "both_correct": sum(1 for m, j in zip(model_correct, jev_correct) if m and j),
        "both_wrong": sum(1 for m, j in zip(model_correct, jev_correct) if not m and not j),
        "model_only_correct": n10,
        "jev_only_correct": n01,
        "total_evaluated": n,
    }


def run_jev_head_to_head(
    local_predictions: list[tuple[int, list[float]]],  # (pred_idx, probs)
    examples: list[JevExample],
    max_api_calls: int = 50,
    out_dir: Optional[Path] = None,
) -> Optional[dict]:
    """
    Run frozen-question comparison against hosted Jev API.
    Publishes summary and every disagreement case.
    """
    client = TypeSafeJevClient(max_calls=max_api_calls)
    if not client.api_key:
        print("Skipping hosted Jev comparison: no API key found.", flush=True)
        return None

    print(f"\nComparing up to {min(max_api_calls, len(examples))} frozen questions against hosted Jev...", flush=True)

    model_correct = []
    jev_correct = []
    disagreements = []

    for i in range(min(max_api_calls, len(examples))):
        ex = examples[i]
        local_pred, local_probs = local_predictions[i]

        jev_resp = client.query_decision(ex.instruction, ex.options, ex.primitive)
        if jev_resp is None:
            continue

        # Extract Jev's predicted option
        jev_pred = jev_resp.get("choice_index", jev_resp.get("prediction"))
        if jev_pred is None:
            continue

        m_ok = (local_pred == ex.label)
        j_ok = (jev_pred == ex.label)

        model_correct.append(m_ok)
        jev_correct.append(j_ok)

        if local_pred != jev_pred:
            disagreements.append({
                "example_idx": i,
                "task_id": ex.task_id,
                "instruction": ex.instruction[:200],
                "options": ex.options,
                "ground_truth": ex.label,
                "local_model_pred": local_pred,
                "local_model_probs": [round(float(p), 4) for p in local_probs],
                "hosted_jev_pred": jev_pred,
                "hosted_jev_resp": jev_resp,
            })

    if not model_correct:
        print("No valid responses received from Jev API.", flush=True)
        return None

    stats = compute_gap_confidence_interval(model_correct, jev_correct)
    print("\n" + "=" * 60, flush=True)
    print("HOSTED JEV VS RLCD MODEL COMPARATIVE REPORT", flush=True)
    print("=" * 60, flush=True)
    print(f"Evaluated Questions:     {stats['total_evaluated']}", flush=True)
    print(f"Local RLCD Accuracy:     {stats['model_accuracy']*100:.2f}%", flush=True)
    print(f"Hosted Jev Accuracy:     {stats['hosted_jev_accuracy']*100:.2f}%", flush=True)
    print(f"Accuracy Gap (RLCD-Jev): {stats['accuracy_diff']*100:+.2f}% [95% CI: {stats['ci_95_lower']*100:+.2f}%, {stats['ci_95_upper']*100:+.2f}%]", flush=True)
    print(f"Disagreements:           {len(disagreements)} / {stats['total_evaluated']}", flush=True)

    result = {
        "stats": stats,
        "disagreements": disagreements,
    }

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "jev_head_to_head.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Detailed audit saved to {out_dir / 'jev_head_to_head.json'}", flush=True)

    return result
