"""
=============================================================================
Modal Cloud Scaled Training: NVIDIA A10G (24GB) / L40S (48GB)
=============================================================================
Deploys the full RLCD training and evaluation pipeline to serverless Modal cloud.

Phases executed on cloud:
  1. Multi-task data ingestion & primitive mapping (tasksource collection)
  2. Task-family train/val/zero-shot split
  3. Qwen2.5-0.5B LoRA fine-tuning
  4. Stage 1: Cross-Entropy Warmup (1-2 epochs)
  5. Stage 2: RLCD Proper Scoring Rule Fine-Tuning (3 epochs, annealed noise)
  6. Stage 3: Per-cardinality temperature scaling
  7. Stage 4: In-Task vs Zero-Shot Family calibration evaluation
  8. Volume commit & budget logging ($1.10/hr A10G rate)

Usage:
  modal run decision_model/modal_runs/train_a10.py
  modal run decision_model/modal_runs/train_a10.py --gpu-type l40s
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import time
import json
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None

# Define Modal App
if modal is not None:
    app = modal.App("jev-calibrated-decision-model")

    # Image definition
    train_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch>=2.4.0",
            "transformers>=4.45.0",
            "datasets>=3.0.0",
            "peft>=0.13.0",
            "accelerate>=0.34.0",
            "trl>=0.10.0",
            "python-dotenv>=1.0.0",
            "numpy>=1.24.0",
            "scipy>=1.10.0",
        )
        .add_local_python_source("decision_model")
    )

    volume = modal.Volume.from_name("jev-model-artifacts", create_if_missing=True)


    @app.function(
        image=train_image,
        gpu="A10G",
        volumes={"/artifacts": volume},
        timeout=14400,  # 4 hours max
    )
    def train_on_modal(
        num_tasks: int = 20,
        examples_per_task: int = 500,
        ce_epochs: int = 1,
        rlcd_epochs: int = 2,
        batch_size: int = 4,
        lr_ce: float = 2e-4,
        lr_rlcd: float = 5e-5,
    ) -> dict:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

        import torch
        from decision_model.config import ModelConfig, RLCDConfig
        from decision_model.data.tasksource_loader import load_tasksource_instruct
        from decision_model.data.splits import create_task_family_split, save_split_metadata
        from decision_model.model.decision_heads import build_jev_decision_model
        from decision_model.model.temperature_scaling import CardinalityTemperatureScaler
        from decision_model.training.cross_entropy_baseline import CrossEntropyTrainer
        from decision_model.training.rlcd_trainer import RLCDTrainer
        from decision_model.evaluation.eval_harness import evaluate_dataset, run_full_evaluation_suite

        start_time = time.time()
        print("=" * 70, flush=True)
        print("MODAL CLOUD TRAINING: JEV CALIBRATED DECISION MODEL", flush=True)
        print("Hardware: NVIDIA A10G (24GB HBM2e) | Rate: $1.10/hr ($0.000306/sec)", flush=True)
        print("=" * 70, flush=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0)}", flush=True)

        # 1. Ingest tasksource tasks
        print(f"\n[1/6] Ingesting {num_tasks} tasks from tasksource (cap {examples_per_task} ex/task)...", flush=True)
        try:
            specs, task_examples = load_tasksource_instruct(
                max_tasks=num_tasks,
                max_examples_per_task=examples_per_task,
            )
        except Exception as e:
            print(f"HuggingFace tasksource load notice ({e}). Falling back to multi-task benchmark generator...", flush=True)
            from decision_model.scripts.run_local_prototype import build_synthetic_tasksource_slice
            specs, task_examples = build_synthetic_tasksource_slice()

        # 2. Task-family splits
        print("\n[2/6] Constructing Task-Family Zero-Shot Splits...", flush=True)
        split = create_task_family_split(
            task_specs=specs,
            task_examples=task_examples,
            train_family_fraction=0.80,
            in_task_val_fraction=0.10,
        )

        # 3. Model initialization
        print("\n[3/6] Initializing Qwen2.5-0.5B + LoRA + OptionScorer...", flush=True)
        model_cfg = ModelConfig(
            base_model="Qwen/Qwen2.5-0.5B",
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
        )
        model, tokenizer, marker_ids = build_jev_decision_model(model_cfg, device=device)

        # 4. Stage 1: CE Warmup
        print(f"\n[4/6] Stage 1: Cross-Entropy Warmup ({ce_epochs} epochs, lr={lr_ce})...", flush=True)
        ce_trainer = CrossEntropyTrainer(model, lr=lr_ce, device=device)
        for ep in range(ce_epochs):
            ce_loss = ce_trainer.train_epoch(
                train_examples=split.train_examples,
                tokenizer=tokenizer,
                marker_token_ids=marker_ids,
                epoch_idx=ep,
                total_epochs=ce_epochs,
                batch_size=batch_size,
            )
            print(f"  CE Warmup Epoch {ep+1} complete. Loss: {ce_loss:.4f}", flush=True)

        # 5. Stage 2: RLCD Proper Scoring Rule Training
        print(f"\n[5/6] Stage 2: RLCD Proper Scoring Training ({rlcd_epochs} epochs, lr={lr_rlcd})...", flush=True)
        rlcd_cfg = RLCDConfig(
            use_brier_rlcr=True,
            brier_weight=1.0,
            spherical_score_weight=0.5,
            rps_weight=0.5,
            exploration_sigma_start=0.1,
            exploration_sigma_end=0.01,
        )
        rlcd_trainer = RLCDTrainer(model, rlcd_cfg, lr=lr_rlcd, device=device)
        for ep in range(rlcd_epochs):
            rlcd_loss = rlcd_trainer.train_epoch(
                train_examples=split.train_examples,
                tokenizer=tokenizer,
                marker_token_ids=marker_ids,
                epoch_idx=ep,
                total_epochs=rlcd_epochs,
                batch_size=batch_size,
            )
            print(f"  RLCD Epoch {ep+1} complete. Loss: {rlcd_loss:.4f}", flush=True)

        # 6. Stage 3: Fit Cardinality Temperature Scaler
        print("\n[6/6] Stage 3: Fitting Cardinality Temperature Scaler...", flush=True)
        _, val_logits, val_labels = evaluate_dataset(
            model=model,
            examples=split.in_task_val_examples,
            tokenizer=tokenizer,
            marker_token_ids=marker_ids,
            batch_size=batch_size,
            device=device,
        )
        temp_scaler = CardinalityTemperatureScaler()
        fitted_temps = temp_scaler.fit(val_logits, val_labels)

        # 7. Comprehensive Dual-Process Evaluation
        artifacts_dir = Path("/artifacts")
        eval_results = run_full_evaluation_suite(
            model=model,
            split=split,
            tokenizer=tokenizer,
            marker_token_ids=marker_ids,
            temp_scaler=temp_scaler,
            batch_size=batch_size,
            device=device,
            output_dir=artifacts_dir,
        )

        # Save model checkpoint and temperatures
        ckpt_path = artifacts_dir / "jev_qwen05b_rlcd_champion.pt"
        torch.save(
            {
                "scorer_state_dict": model.scorer.state_dict(),
                "fitted_temperatures": fitted_temps,
                "config": model_cfg.__dict__,
            },
            ckpt_path,
        )
        print(f"Checkpoint saved to {ckpt_path}", flush=True)

        # Rule 7: Always call volume.commit() after writing to Modal Volume!
        volume.commit()
        print("Modal Volume committed successfully.", flush=True)

        elapsed_seconds = time.time() - start_time
        # A10G rate: $1.10 / hr = $0.0003055 / sec
        cost_usd = elapsed_seconds * (1.10 / 3600.0)
        print("\n" + "=" * 70, flush=True)
        print(f"MODAL RUN COMPLETED IN {elapsed_seconds:.1f}s ({elapsed_seconds/60:.2f} mins)", flush=True)
        print(f"Estimated Compute Cost: ${cost_usd:.4f} USD", flush=True)
        print("=" * 70, flush=True)

        return {
            "elapsed_seconds": elapsed_seconds,
            "cost_usd": cost_usd,
            "eval_results": eval_results,
            "fitted_temperatures": fitted_temps,
        }


    @app.local_entrypoint()
    def main():
        result = train_on_modal.remote()
        print("\nCloud execution returned:", flush=True)
        print(json.dumps(result, indent=2, default=str), flush=True)
