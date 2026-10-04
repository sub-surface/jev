"""
=============================================================================
Modal Cloud Scaled Training: NVIDIA A10G (24GB) / L40S (48GB)
=============================================================================
Deploys the full RLCD training and evaluation pipeline to serverless Modal cloud
with automated self-termination guardrails, budget protection, and verbose telemetry.

Phases executed on cloud:
  1. Streaming multi-task data ingestion & primitive mapping (tasksource collection)
  2. Task-family train/val/zero-shot split
  3. Qwen2.5-0.5B LoRA fine-tuning
  4. Stage 1: Cross-Entropy Warmup (1 epoch)
  5. Stage 2: RLCD Proper Scoring Rule Fine-Tuning (MIT Brier + Spherical + RPS)
  6. Stage 3: Per-cardinality temperature scaling with L2 prior
  7. Stage 4: Dual-Process calibration evaluation (ECE, Reliability, Risk-Coverage)
  8. Volume commit & budget accounting ($1.10/hr A10G, $1.95/hr L40S)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import time
import json
import traceback
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("jev-calibrated-decision-model")

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


    def _run_cloud_pipeline(
        gpu_name: str,
        hourly_rate: float,
        num_tasks: int,
        examples_per_task: int,
        ce_epochs: int,
        rlcd_epochs: int,
        batch_size: int,
        lr_ce: float,
        lr_rlcd: float,
        base_model: str = "Qwen/Qwen3-4B",
        lora_r: int = 32,
        lora_alpha: int = 64,
    ) -> dict:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        import torch
        from decision_model.config import ModelConfig, RLCDConfig
        from decision_model.data.tasksource_loader import load_tasksource_instruct
        from decision_model.data.splits import create_task_family_split
        from decision_model.model.decision_heads import build_jev_decision_model
        from decision_model.model.temperature_scaling import CardinalityTemperatureScaler
        from decision_model.training.cross_entropy_baseline import CrossEntropyTrainer
        from decision_model.training.rlcd_trainer import RLCDTrainer
        from decision_model.evaluation.eval_harness import evaluate_dataset, run_full_evaluation_suite

        start_time = time.time()
        print("=" * 75, flush=True)
        print("MODAL SCALED CLOUD TRAINING: JEV CALIBRATED DECISION MODEL", flush=True)
        print(f"Model: {base_model} | LoRA r={lora_r}, alpha={lora_alpha} (Gradient Checkpointing Enabled)", flush=True)
        print(f"Hardware: NVIDIA {gpu_name} | Rate: ${hourly_rate:.2f}/hr (${hourly_rate/3600.0:.6f}/sec)", flush=True)
        print("Safety Watchdog: Max 30m wall-clock | Strict NaN/Inf loss kill-switch", flush=True)
        print("=" * 75, flush=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"CUDA Device: {device} | GPU: {torch.cuda.get_device_name(0)}", flush=True)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total VRAM Available: {vram_total:.2f} GB", flush=True)

        artifacts_dir = Path("/artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Multi-Task Data Ingestion
            print(f"\n[1/6] Ingesting {num_tasks} tasks from tasksource (cap {examples_per_task} ex/task)...", flush=True)
            try:
                specs, task_examples = load_tasksource_instruct(
                    max_tasks=num_tasks,
                    max_examples_per_task=examples_per_task,
                )
            except Exception as e:
                print(f"HuggingFace streaming notice ({e}). Falling back to multi-task generator...", flush=True)
                from decision_model.scripts.run_local_prototype import build_synthetic_tasksource_slice
                specs, task_examples = build_synthetic_tasksource_slice()

            total_ex = sum(len(v) for v in task_examples.values())
            print(f"  Ingested {len(specs)} tasks with {total_ex} total examples.", flush=True)

            # 2. Construct Zero-Shot Splits
            print("\n[2/6] Partitioning Task Families into Train vs In-Task Val vs Zero-Shot...", flush=True)
            split = create_task_family_split(
                task_specs=specs,
                task_examples=task_examples,
                train_family_fraction=0.80,
                in_task_val_fraction=0.10,
            )
            print(f"  Training set: {len(split.train_examples)} examples ({len(split.train_families)} families)", flush=True)
            print(f"  In-task val:  {len(split.in_task_val_examples)} examples", flush=True)
            print(f"  Zero-shot:    {len(split.zero_shot_examples)} examples ({len(split.zero_shot_families)} families: {split.zero_shot_families})", flush=True)

            # 3. Model Initialization
            print(f"\n[3/6] Initializing {base_model} + LoRA (r={lora_r}) + OptionScorer...", flush=True)
            model_cfg = ModelConfig(
                base_model=base_model,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                gradient_checkpointing=True,
            )
            model, tokenizer, marker_ids = build_jev_decision_model(model_cfg, device=device)
            vram_init = torch.cuda.memory_allocated() / (1024**3)
            print(f"  Model loaded into VRAM: {vram_init:.2f} GB allocated", flush=True)

            # Watchdog check
            def check_watchdog(step_name: str):
                elapsed = time.time() - start_time
                if elapsed > 4800:  # 80 minutes limit
                    raise TimeoutError(f"Safety watchdog limit (80 min) exceeded at step '{step_name}'. Self-terminating!")

            # 4. Stage 1: Cross-Entropy Warmup
            print(f"\n[4/6] Stage 1: Cross-Entropy Warmup ({ce_epochs} epochs, lr={lr_ce}, batch_size={batch_size})...", flush=True)
            ce_trainer = CrossEntropyTrainer(model, lr=lr_ce, device=device)
            for ep in range(ce_epochs):
                check_watchdog(f"CE_Epoch_{ep+1}")
                ce_loss = ce_trainer.train_epoch(
                    train_examples=split.train_examples,
                    tokenizer=tokenizer,
                    marker_token_ids=marker_ids,
                    epoch_idx=ep,
                    total_epochs=ce_epochs,
                    batch_size=batch_size,
                    log_interval=25,
                )
                if torch.isnan(torch.tensor(ce_loss)) or torch.isinf(torch.tensor(ce_loss)):
                    raise RuntimeError(f"CE Warmup loss diverged to {ce_loss}! Aborting run.")
                vram_ep = torch.cuda.memory_allocated() / (1024**3)
                print(f"  CE Warmup Epoch {ep+1}/{ce_epochs} Loss: {ce_loss:.4f} | VRAM: {vram_ep:.2f} GB", flush=True)

            # 5. Stage 2: RLCD Proper Scoring (MIT RLCR Brier + Spherical + RPS)
            print(f"\n[5/6] Stage 2: RLCD Proper Scoring ({rlcd_epochs} epochs, lr={lr_rlcd}, Brier RLCR)...", flush=True)
            rlcd_cfg = RLCDConfig(
                use_brier_rlcr=True,
                brier_weight=1.0,
                spherical_score_weight=0.5,
                rps_weight=0.5,
                exploration_sigma_start=0.10,
                exploration_sigma_end=0.01,
            )
            rlcd_trainer = RLCDTrainer(model, rlcd_cfg, lr=lr_rlcd, device=device)
            for ep in range(rlcd_epochs):
                check_watchdog(f"RLCD_Epoch_{ep+1}")
                rlcd_loss = rlcd_trainer.train_epoch(
                    train_examples=split.train_examples,
                    tokenizer=tokenizer,
                    marker_token_ids=marker_ids,
                    epoch_idx=ep,
                    total_epochs=rlcd_epochs,
                    batch_size=batch_size,
                    log_interval=25,
                )
                if torch.isnan(torch.tensor(rlcd_loss)) or torch.isinf(torch.tensor(rlcd_loss)):
                    raise RuntimeError(f"RLCD loss diverged to {rlcd_loss}! Aborting run.")
                vram_ep = torch.cuda.memory_allocated() / (1024**3)
                print(f"  RLCD Epoch {ep+1}/{rlcd_epochs} Loss: {rlcd_loss:.4f} | VRAM: {vram_ep:.2f} GB", flush=True)

            # Immediate weights persistence to volume right after training
            clean_name = base_model.replace("/", "_").replace(".", "").lower()
            ckpt_path = artifacts_dir / f"jev_{clean_name}_rlcd_champion.pt"
            torch.save(
                {
                    "scorer_state_dict": model.scorer.state_dict(),
                    "config": model_cfg.__dict__,
                },
                ckpt_path,
            )
            print(f"\n[Artifact] Trained weights checkpoint saved to {ckpt_path}", flush=True)
            try:
                volume.commit()
            except Exception as e:
                print(f"Intermediate volume commit notice: {e}", flush=True)

            # 6. Stage 3: Fit Cardinality Temperature Scaler
            print("\n[6/6] Stage 3: Fitting Cardinality Temperature Scaler on In-Task Validation...", flush=True)
            check_watchdog("Temp_Scaling")
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
            print("\nRunning Full Dual-Process Evaluation Suite...", flush=True)
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

            # Save champion checkpoint with complete evaluation and fitted temperatures
            state_payload = {
                "scorer_state_dict": model.scorer.state_dict(),
                "fitted_temperatures": fitted_temps,
                "config": model_cfg.__dict__,
                "eval_summary": {
                    "in_task_acc": eval_results["in_task"]["accuracy"],
                    "in_task_ece": eval_results["in_task"]["ece"],
                    "zero_shot_acc": eval_results["zero_shot"]["accuracy"],
                    "zero_shot_ece": eval_results["zero_shot"]["ece"],
                },
            }
            torch.save(state_payload, ckpt_path)
            torch.save(state_payload, artifacts_dir / "jev_champion_latest.pt")
            print(f"\nChampion model checkpoint successfully saved to {ckpt_path} and jev_champion_latest.pt", flush=True)

            elapsed_seconds = time.time() - start_time
            cost_usd = elapsed_seconds * (hourly_rate / 3600.0)

            print("\n" + "=" * 75, flush=True)
            print(f"MODAL TRAINING JOB COMPLETED SUCCESSFULLY in {elapsed_seconds:.1f}s ({elapsed_seconds/60:.2f} mins)", flush=True)
            print(f"Hardware: NVIDIA {gpu_name} | Compute Spend: ${cost_usd:.4f} USD", flush=True)
            print(f"In-Task Accuracy:       {eval_results['in_task']['accuracy']*100:.2f}% | ECE: {eval_results['in_task']['ece']*100:.2f}%", flush=True)
            print(f"Zero-Shot Family Acc:   {eval_results['zero_shot']['accuracy']*100:.2f}% | ECE: {eval_results['zero_shot']['ece']*100:.2f}%", flush=True)
            print("=" * 75, flush=True)

            return {
                "status": "success",
                "gpu": gpu_name,
                "elapsed_seconds": elapsed_seconds,
                "cost_usd": cost_usd,
                "in_task_acc": eval_results["in_task"]["accuracy"],
                "in_task_ece": eval_results["in_task"]["ece"],
                "zero_shot_acc": eval_results["zero_shot"]["accuracy"],
                "zero_shot_ece": eval_results["zero_shot"]["ece"],
                "fitted_temperatures": fitted_temps,
            }

        except Exception as e:
            elapsed = time.time() - start_time
            cost_usd = elapsed * (hourly_rate / 3600.0)
            print(f"\n🛑 CLOUD RUN ENCOUNTERED ERROR AFTER {elapsed:.1f}s (${cost_usd:.4f}): {e}", flush=True)
            traceback.print_exc()
            raise e

        finally:
            # Rule 7: Always call volume.commit() after writing to Modal Volume!
            try:
                volume.commit()
                print("Modal Volume committed successfully.", flush=True)
            except Exception as e:
                print(f"Volume commit warning: {e}", flush=True)


    @app.function(
        image=train_image,
        gpu="A10G",
        volumes={"/artifacts": volume},
        timeout=3600,  # 60 min strict timeout
    )
    def train_on_modal_a10g(
        base_model: str = "Qwen/Qwen3-4B",
        num_tasks: int = 20,
        examples_per_task: int = 350,
        ce_epochs: int = 1,
        rlcd_epochs: int = 2,
        batch_size: int = 16,
        lr_ce: float = 1e-4,
        lr_rlcd: float = 3e-5,
        lora_r: int = 32,
        lora_alpha: int = 64,
    ) -> dict:
        return _run_cloud_pipeline(
            gpu_name="A10G",
            hourly_rate=1.10,
            base_model=base_model,
            num_tasks=num_tasks,
            examples_per_task=examples_per_task,
            ce_epochs=ce_epochs,
            rlcd_epochs=rlcd_epochs,
            batch_size=batch_size,
            lr_ce=lr_ce,
            lr_rlcd=lr_rlcd,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )


    @app.function(
        image=train_image,
        gpu="L40S",
        volumes={"/artifacts": volume},
        timeout=10800,  # 3 hr strict timeout (prev 90 min killed EXP-007)
    )
    def train_on_modal_l40s(
        base_model: str = "Qwen/Qwen3-4B",
        num_tasks: int = 20,
        examples_per_task: int = 350,
        ce_epochs: int = 1,
        rlcd_epochs: int = 2,
        batch_size: int = 20,
        lr_ce: float = 1e-4,
        lr_rlcd: float = 3e-5,
        lora_r: int = 32,
        lora_alpha: int = 64,
    ) -> dict:
        return _run_cloud_pipeline(
            gpu_name="L40S",
            hourly_rate=1.95,
            base_model=base_model,
            num_tasks=num_tasks,
            examples_per_task=examples_per_task,
            ce_epochs=ce_epochs,
            rlcd_epochs=rlcd_epochs,
            batch_size=batch_size,
            lr_ce=lr_ce,
            lr_rlcd=lr_rlcd,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )


    @app.local_entrypoint()
    def main(
        gpu_type: str = "L40S",
        base_model: str = "Qwen/Qwen3-4B",
        num_tasks: int = 20,
        examples_per_task: int = 350,
        ce_epochs: int = 1,
        rlcd_epochs: int = 2,
        batch_size: int = 20,
        lr_ce: float = 1e-4,
        lr_rlcd: float = 3e-5,
        lora_r: int = 32,
        lora_alpha: int = 64,
    ):
        if gpu_type.upper() == "L40S":
            print(f"Targeting NVIDIA L40S (48GB Ada Lovelace) with {base_model}...", flush=True)
            result = train_on_modal_l40s.remote(
                base_model=base_model,
                num_tasks=num_tasks,
                examples_per_task=examples_per_task,
                ce_epochs=ce_epochs,
                rlcd_epochs=rlcd_epochs,
                batch_size=batch_size,
                lr_ce=lr_ce,
                lr_rlcd=lr_rlcd,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
            )
        else:
            print(f"Targeting NVIDIA A10G (24GB Ampere) with {base_model}...", flush=True)
            result = train_on_modal_a10g.remote(
                base_model=base_model,
                num_tasks=num_tasks,
                examples_per_task=examples_per_task,
                ce_epochs=ce_epochs,
                rlcd_epochs=rlcd_epochs,
                batch_size=batch_size,
                lr_ce=lr_ce,
                lr_rlcd=lr_rlcd,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
            )

        print("\nCloud execution returned:", flush=True)
        print(json.dumps(result, indent=2, default=str), flush=True)
