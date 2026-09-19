"""
=============================================================================
Evaluation Suite Entrypoint
=============================================================================
Runs rigorous dual-axis evaluation (in-task vs zero-shot family),
fits per-cardinality temperature scaling, and optionally benchmarks against
the hosted TypeSafe Jev API and DiffusionGemma.

Usage:
  python decision_model/scripts/run_evaluation.py
  python decision_model/scripts/run_evaluation.py --compare-jev --max-api-calls 30
  python decision_model/scripts/run_evaluation.py --diffusion-gemma
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import json
import argparse
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_model.config import ModelConfig, CHECKPOINTS_DIR, LOGS_DIR
from decision_model.model.decision_heads import build_jev_decision_model
from decision_model.model.temperature_scaling import CardinalityTemperatureScaler
from decision_model.data.splits import create_task_family_split
from decision_model.evaluation.eval_harness import evaluate_dataset, run_full_evaluation_suite
from decision_model.evaluation.jev_api_comparison import run_jev_head_to_head
from decision_model.scripts.run_local_prototype import build_synthetic_tasksource_slice


def main():
    parser = argparse.ArgumentParser(description="Jev Decision Model Evaluation Suite")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint (.pt)")
    parser.add_argument("--compare-jev", action="store_true", help="Run head-to-head comparison vs hosted Jev API")
    parser.add_argument("--max-api-calls", type=int, default=30, help="Max API calls to Jev API")
    parser.add_argument("--diffusion-gemma", action="store_true", help="Launch DiffusionGemma zero-shot eval on Modal")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70, flush=True)
    print(f"JEV EVALUATION HARNESS | Device: {device}", flush=True)
    print("=" * 70, flush=True)

    # 1. Prepare benchmark evaluation data
    specs, task_examples = build_synthetic_tasksource_slice()
    split = create_task_family_split(specs, task_examples, train_family_fraction=0.80, in_task_val_fraction=0.15)

    # 2. Load model
    model_cfg = ModelConfig()
    model, tokenizer, marker_ids = build_jev_decision_model(model_cfg, device=device)

    temp_scaler = CardinalityTemperatureScaler()

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint weights from {args.checkpoint}...", flush=True)
        ckpt = torch.load(args.checkpoint, map_location=device)
        if "scorer_state_dict" in ckpt:
            model.scorer.load_state_dict(ckpt["scorer_state_dict"])
        if "fitted_temperatures" in ckpt:
            temp_scaler.fit_from_dict(ckpt["fitted_temperatures"])
        print("Checkpoint weights loaded successfully.", flush=True)

    # 3. Fit or update per-cardinality temperature scaling on validation split
    print("\nCalibrating per-cardinality temperature scaling on held-out validation...", flush=True)
    _, val_logits, val_labels = evaluate_dataset(
        model=model,
        examples=split.in_task_val_examples,
        tokenizer=tokenizer,
        marker_token_ids=marker_ids,
        batch_size=args.batch_size,
        device=device,
    )
    temp_scaler.fit(val_logits, val_labels)

    # 4. Run Dual-Axis Evaluation
    out_dir = LOGS_DIR / "evaluation_run"
    results = run_full_evaluation_suite(
        model=model,
        split=split,
        tokenizer=tokenizer,
        marker_token_ids=marker_ids,
        temp_scaler=temp_scaler,
        batch_size=args.batch_size,
        device=device,
        output_dir=out_dir,
    )

    # 5. Optional: Hosted Jev API Comparison
    if args.compare_jev:
        print("\n" + "=" * 60, flush=True)
        print("LAUNCHING HOSTED JEV API COMPARISON", flush=True)
        print("=" * 60, flush=True)

        # Collect local predictions on zero-shot examples
        local_preds = []
        for l in val_logits[:args.max_api_calls]:
            scaled = temp_scaler.scale_logits(l, len(l))
            probs = torch.softmax(scaled, dim=-1).tolist()
            pred = int(torch.argmax(scaled).item())
            local_preds.append((pred, probs))

        run_jev_head_to_head(
            local_predictions=local_preds,
            examples=split.in_task_val_examples[:args.max_api_calls],
            max_api_calls=args.max_api_calls,
            out_dir=out_dir,
        )

    # 6. Optional: Launch DiffusionGemma on Modal
    if args.diffusion_gemma:
        print("\n" + "=" * 60, flush=True)
        print("LAUNCHING DIFFUSIONGEMMA ZERO-SHOT EVAL ON MODAL CLOUD", flush=True)
        print("=" * 60, flush=True)
        import subprocess
        gemma_script = PROJECT_ROOT / "decision_model" / "modal_runs" / "eval_diffusion_gemma.py"
        subprocess.run([sys.executable, "-m", "modal", "run", str(gemma_script)], check=False)


if __name__ == "__main__":
    main()
