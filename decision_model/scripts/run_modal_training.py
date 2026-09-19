"""
=============================================================================
Modal Cloud Training Launcher & Budget Guard
=============================================================================
Launches calibrated decision model training on Modal cloud GPUs (A10G or L40S)
with automated pre-flight budget checks, cost accounting, and guardrails.

Usage:
  python decision_model/scripts/run_modal_training.py --tasks 20 --examples 500
  python decision_model/scripts/run_modal_training.py --ce-epochs 1 --rlcd-epochs 2
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import argparse
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv

# Insert project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_model.config import (
    BudgetConfig,
    get_cumulative_spend,
    log_budget_spend,
)

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Jev Decision Model Modal Cloud Training Launcher")
    parser.add_argument("--tasks", type=int, default=15, help="Number of tasksource tasks to ingest")
    parser.add_argument("--examples", type=int, default=250, help="Max examples per task")
    parser.add_argument("--ce-epochs", type=int, default=1, help="Cross-entropy warmup epochs")
    parser.add_argument("--rlcd-epochs", type=int, default=2, help="RLCD proper scoring epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr-ce", type=float, default=2e-4, help="Learning rate for CE warmup")
    parser.add_argument("--lr-rlcd", type=float, default=5e-5, help="Learning rate for RLCD")
    parser.add_argument("--gpu-type", type=str, default="A10G", choices=["A10G", "L40S"], help="Modal GPU type")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B", help="HuggingFace base model")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--dry-run", action="store_true", help="Print command and estimated cost without launching")
    args = parser.parse_args()

    budget_cfg = BudgetConfig()
    cum_spend = get_cumulative_spend()
    remaining = budget_cfg.total_budget - cum_spend

    # Hourly rate
    rate_hr = 1.10 if args.gpu_type == "A10G" else 1.95
    # Accurate throughput-based estimate
    est_examples = args.tasks * args.examples
    est_batches_per_ep = est_examples / max(1, args.batch_size)
    sec_per_batch = 1.0 if args.gpu_type == "L40S" else 1.8
    est_train_sec = est_batches_per_ep * sec_per_batch * (args.ce_epochs + args.rlcd_epochs)
    est_seconds = 180 + est_train_sec
    est_hours = est_seconds / 3600.0
    est_cost = est_hours * rate_hr

    print("=" * 70, flush=True)
    print("MODAL SCALED TRAINING PRE-FLIGHT CHECK", flush=True)
    print("=" * 70, flush=True)
    print(f"Base Model:          {args.base_model}", flush=True)
    print(f"LoRA Rank / Alpha:   r={args.lora_r}, alpha={args.lora_alpha}", flush=True)
    print(f"Total Budget:        ${budget_cfg.total_budget:.2f} USD", flush=True)
    print(f"Cumulative Spend:    ${cum_spend:.4f} USD", flush=True)
    print(f"Remaining Budget:    ${remaining:.4f} USD", flush=True)
    print(f"Eval Reserve (20%):  ${budget_cfg.eval_reserve:.2f} USD", flush=True)
    print(f"Available for Train: ${remaining - budget_cfg.eval_reserve:.2f} USD", flush=True)
    print(f"Selected Hardware:   NVIDIA {args.gpu_type} (${rate_hr:.2f}/hr)", flush=True)
    print(f"Estimated Cost:      ~${est_cost:.2f} USD ({est_hours*60:.1f} mins)", flush=True)

    # Budget Guardrail: max 1/3 of remaining budget on a single run
    max_allowed = remaining * budget_cfg.max_single_run_fraction
    if est_cost > max_allowed:
        print(f"\n🛑 SAFETY GUARD TRIGGERED: Estimated run cost (${est_cost:.2f}) exceeds 1/3 of remaining budget (${max_allowed:.2f})!", flush=True)
        print("Please reduce epochs or examples per task before dispatching.", flush=True)
        return

    if args.dry_run:
        print("\n[DRY RUN] Pre-flight validation passed. Exiting without launching.", flush=True)
        return

    # Dispatch to Modal
    print("\n🚀 Dispatching training job to Modal cloud...", flush=True)
    modal_script = PROJECT_ROOT / "decision_model" / "modal_runs" / "train_a10.py"

    cmd = [
        sys.executable,
        "-m",
        "modal",
        "run",
        str(modal_script),
        "--gpu-type", args.gpu_type,
        "--base-model", args.base_model,
        "--num-tasks", str(args.tasks),
        "--examples-per-task", str(args.examples),
        "--ce-epochs", str(args.ce_epochs),
        "--rlcd-epochs", str(args.rlcd_epochs),
        "--batch-size", str(args.batch_size),
        "--lr-ce", str(args.lr_ce),
        "--lr-rlcd", str(args.lr_rlcd),
        "--lora-r", str(args.lora_r),
        "--lora-alpha", str(args.lora_alpha),
    ]

    print(f"Executing: {' '.join(cmd)}", flush=True)
    start_t = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - start_t
    actual_cost = elapsed * (rate_hr / 3600.0)

    log_budget_spend(
        run_name=f"modal_train_{args.gpu_type}_{int(start_t)}",
        gpu_type=args.gpu_type,
        duration_seconds=elapsed,
        cost_usd=actual_cost,
        notes=f"tasks={args.tasks}, ex={args.examples}, ce_ep={args.ce_epochs}, rlcd_ep={args.rlcd_epochs}",
    )

    print("\n" + "=" * 70, flush=True)
    print(f"MODAL TRAINING JOB COMPLETED | Duration: {elapsed:.1f}s | Cost: ${actual_cost:.4f} USD", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
