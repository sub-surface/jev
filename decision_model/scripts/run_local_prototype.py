"""
=============================================================================
Phase 2: Local Prototyping Script (RTX 2060, $0 Cloud Spend)
=============================================================================
Validates the entire calibrated decision pipeline locally:
  1. Data ingestion & primitive mapping (choice, noul, score)
  2. Option-marker tokenization ([OPT_A]..[OPT_Z])
  3. Qwen2.5-0.5B backbone + LoRA + OptionScorer initialization on local GPU
  4. Cross-entropy loss pass
  5. RLCD loss pass with strictly proper scoring rules & exploration noise
  6. Per-cardinality temperature scaling fitting
  7. Dual evaluation harness (in-task vs zero-shot family)

Usage:
  python decision_model/scripts/run_local_prototype.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from decision_model.config import (
    ExperimentConfig,
    ModelConfig,
    RLCDConfig,
    DataConfig,
    PROJECT_ROOT,
)
from decision_model.data.task_registry import JevPrimitive, TaskFamily, TaskSpec
from decision_model.data.tasksource_loader import JevExample
from decision_model.data.splits import create_task_family_split
from decision_model.model.decision_heads import build_jev_decision_model
from decision_model.model.temperature_scaling import CardinalityTemperatureScaler
from decision_model.training.rlcd_trainer import RLCDTrainer
from decision_model.training.cross_entropy_baseline import CrossEntropyTrainer
from decision_model.evaluation.eval_harness import evaluate_dataset, run_full_evaluation_suite


def build_synthetic_tasksource_slice() -> tuple[list[TaskSpec], dict[str, list[JevExample]]]:
    """
    Build a realistic, multi-family, multi-primitive prototype slice
    mirroring the tasksource harmonization structure.
    Enables instant, zero-network, local test verification.
    """
    specs = [
        # Family 1: NLI (Binary / Noul)
        TaskSpec("snli_binary", JevPrimitive.NOUL, TaskFamily.NLI, num_options=2),
        # Family 2: Sentiment (Choice, 3-class)
        TaskSpec("sst3_sentiment", JevPrimitive.CHOICE, TaskFamily.SENTIMENT, num_options=3),
        # Family 3: QA (Choice, 4-class)
        TaskSpec("arc_qa", JevPrimitive.CHOICE, TaskFamily.QA, num_options=4),
        # Family 4: Fact Verification (Held-out family for zero-shot eval)
        TaskSpec("fever_fact_verify", JevPrimitive.NOUL, TaskFamily.FACT_VERIFICATION, num_options=2),
        # Family 5: Ethics / Review (Score, 5 ordinal rating bins)
        TaskSpec("review_score", JevPrimitive.SCORE, TaskFamily.ETHICS, num_options=5, is_ordinal=True),
    ]

    task_examples: dict[str, list[JevExample]] = {}

    # 1. SNLI Binary (Noul)
    nli_data = [
        ("Premise: Two dogs run on grass.\nHypothesis: Animals are moving outside.", ["True", "False"], 0),
        ("Premise: A boy eats an apple.\nHypothesis: The boy sleeps in bed.", ["True", "False"], 1),
        ("Premise: A car stops at a red light.\nHypothesis: The vehicle is stationary.", ["True", "False"], 0),
        ("Premise: Water boils in a kettle.\nHypothesis: The water is freezing cold.", ["True", "False"], 1),
    ] * 6  # 24 samples
    task_examples["snli_binary"] = [
        JevExample("snli_binary", JevPrimitive.NOUL, TaskFamily.NLI, p, opts, y, 2)
        for p, opts, y in nli_data
    ]

    # 2. SST3 Sentiment (Choice, 3 options)
    sst_data = [
        ("Review: An extraordinary and uplifting cinematic triumph.", ["positive", "neutral", "negative"], 0),
        ("Review: The movie was neither particularly good nor notably bad.", ["positive", "neutral", "negative"], 1),
        ("Review: Painfully slow, incoherent, and utterly unwatchable.", ["positive", "neutral", "negative"], 2),
        ("Review: Visually stunning masterpiece with breathtaking depth.", ["positive", "neutral", "negative"], 0),
    ] * 6
    task_examples["sst3_sentiment"] = [
        JevExample("sst3_sentiment", JevPrimitive.CHOICE, TaskFamily.SENTIMENT, p, opts, y, 3)
        for p, opts, y in sst_data
    ]

    # 3. ARC QA (Choice, 4 options)
    qa_data = [
        ("Question: Which celestial body is at the center of the solar system?", ["Mars", "Sun", "Jupiter", "Moon"], 1),
        ("Question: What state of matter is water vapor?", ["Solid", "Liquid", "Gas", "Plasma"], 2),
        ("Question: What biological process converts light into sugars?", ["Respiration", "Photosynthesis", "Digestion", "Fermentation"], 1),
        ("Question: What force attracts objects toward Earth's center?", ["Magnetism", "Friction", "Gravity", "Tension"], 2),
    ] * 6
    task_examples["arc_qa"] = [
        JevExample("arc_qa", JevPrimitive.CHOICE, TaskFamily.QA, p, opts, y, 4)
        for p, opts, y in qa_data
    ]

    # 4. FEVER Fact Verification (Zero-Shot Family)
    fever_data = [
        ("Claim: The Eiffel Tower is located in Paris, France.", ["Supported", "Refuted"], 0),
        ("Claim: Mount Everest is located in South America.", ["Supported", "Refuted"], 1),
        ("Claim: Oxygen is essential for human cellular respiration.", ["Supported", "Refuted"], 0),
        ("Claim: The Pacific Ocean is smaller than the Mediterranean Sea.", ["Supported", "Refuted"], 1),
    ] * 6
    task_examples["fever_fact_verify"] = [
        JevExample("fever_fact_verify", JevPrimitive.NOUL, TaskFamily.FACT_VERIFICATION, p, opts, y, 2)
        for p, opts, y in fever_data
    ]

    # 5. Review Score (Score, 5 ordinal bins)
    score_data = [
        ("Rate the product: Absolute perfection, exceeded all expectations!", ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"], 4),
        ("Rate the product: Terrible quality, arrived broken and useless.", ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"], 0),
        ("Rate the product: Decent item, meets basic needs but has flaws.", ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"], 2),
        ("Rate the product: Very good value, minor cosmetic scratch.", ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"], 3),
    ] * 6
    task_examples["review_score"] = [
        JevExample("review_score", JevPrimitive.SCORE, TaskFamily.ETHICS, p, opts, y, 5, is_ordinal=True)
        for p, opts, y in score_data
    ]

    return specs, task_examples


def main():
    print("=" * 70, flush=True)
    print("PHASE 2: LOCAL END-TO-END PROTOTYPE VALIDATION", flush=True)
    print("Hardware: Local GPU (RTX 2060, $0 Modal Spend)", flush=True)
    print("=" * 70, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB", flush=True)

    # 1. Prepare data slice & task-family splits
    print("\n[1/6] Preparing Tasksource Multi-Family Dataset Slice...", flush=True)
    specs, task_examples = build_synthetic_tasksource_slice()
    split = create_task_family_split(specs, task_examples, train_family_fraction=0.75, in_task_val_fraction=0.20, seed=42)

    # 2. Build Jev Decision Model on Qwen2.5-0.5B
    print("\n[2/6] Initializing Qwen2.5-0.5B Decision Model + LoRA...", flush=True)
    model_cfg = ModelConfig(
        base_model="Qwen/Qwen2.5-0.5B",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )
    model, tokenizer, marker_ids = build_jev_decision_model(model_cfg, device=device)

    # 3. Test Step: Cross-Entropy Baseline Forward & Backward
    print("\n[3/6] Validating Cross-Entropy Baseline Step...", flush=True)
    ce_trainer = CrossEntropyTrainer(model, lr=2e-4, device=device)
    ce_loss = ce_trainer.train_epoch(
        train_examples=split.train_examples[:8],
        tokenizer=tokenizer,
        marker_token_ids=marker_ids,
        epoch_idx=0,
        total_epochs=1,
        batch_size=2,
        log_interval=2,
    )
    print(f"  CE Validation Passed. Epoch Loss: {ce_loss:.4f}", flush=True)

    # 4. Test Step: RLCD Proper Scoring Rules Forward & Backward
    print("\n[4/6] Validating RLCD Proper Scoring Training Loop...", flush=True)
    rlcd_cfg = RLCDConfig(
        exploration_sigma_start=0.1,
        exploration_sigma_end=0.05,
        log_score_weight=1.0,
        spherical_score_weight=0.5,
        rps_weight=0.5,
    )
    rlcd_trainer = RLCDTrainer(model, rlcd_cfg, lr=5e-5, device=device)
    rlcd_loss = rlcd_trainer.train_epoch(
        train_examples=split.train_examples[:8],
        tokenizer=tokenizer,
        marker_token_ids=marker_ids,
        epoch_idx=0,
        total_epochs=1,
        batch_size=2,
        log_interval=2,
    )
    print(f"  RLCD Validation Passed. Epoch Loss: {rlcd_loss:.4f}", flush=True)

    # 5. Fit Per-Cardinality Temperature Scaling
    print("\n[5/6] Fitting Per-Cardinality Temperature Scaler on In-Task Validation...", flush=True)
    _, val_logits, val_labels = evaluate_dataset(
        model=model,
        examples=split.in_task_val_examples,
        tokenizer=tokenizer,
        marker_token_ids=marker_ids,
        batch_size=2,
        device=device,
    )
    temp_scaler = CardinalityTemperatureScaler()
    fitted_temps = temp_scaler.fit(val_logits, val_labels, lr=0.05, max_iter=20)
    print(f"  Fitted Temperature Map: {fitted_temps}", flush=True)

    # 6. Run Complete In-Task vs Zero-Shot Family Evaluation Suite
    print("\n[6/6] Executing Comprehensive Evaluation Suite...", flush=True)
    eval_out = PROJECT_ROOT / "decision_model" / "logs" / "local_prototype"
    results = run_full_evaluation_suite(
        model=model,
        split=split,
        tokenizer=tokenizer,
        marker_token_ids=marker_ids,
        temp_scaler=temp_scaler,
        batch_size=2,
        device=device,
        output_dir=eval_out,
    )

    print("\n" + "=" * 70, flush=True)
    print("PHASE 2 SUCCESS: LOCAL PIPELINE VALIDATED END-TO-END AT ZERO COST!", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
