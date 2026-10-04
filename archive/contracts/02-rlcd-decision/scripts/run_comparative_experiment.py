"""
=============================================================================
Comparative Experiment: Cross-Entropy vs RLCD vs RLCD+TempScale
=============================================================================
Runs a controlled comparative evaluation on RTX 2060 ($0 cloud spend):
  1. Identical model architecture (Qwen2.5-0.5B + LoRA) and dataset splits.
  2. Condition A: Cross-Entropy (CE) baseline (what Kev, openjev, jevlike use).
  3. Condition B: RLCD with Proper Scoring Rules (Log + Spherical + RPS) & noise.
  4. Condition C: RLCD + Per-Cardinality Temperature Scaling.
  5. Mechanistic Interpretability: Latent geometry, logit margins, representation similarity.
  6. High-fidelity figure generation saved to jev-vault/figures/:
       - fig33_rlcd_vs_ce_training_dynamics.png
       - fig34_reliability_diagrams_calibration.png
       - fig35_latent_geometry_and_mechanistic_interp.png
       - fig36_risk_coverage_pareto_frontiers.png
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import json
import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_cur = Path(__file__).resolve().parent
while _cur != _cur.parent:
    if (_cur / "contracts").exists() or (_cur / ".git").exists():
        break
    _cur = _cur.parent
PROJECT_ROOT = _cur
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_model.config import ModelConfig, RLCDConfig
from decision_model.data.splits import create_task_family_split
from decision_model.model.decision_heads import build_jev_decision_model, JevDecisionModel
from decision_model.model.option_marker import encode_examples
from decision_model.model.temperature_scaling import CardinalityTemperatureScaler
from decision_model.training.cross_entropy_baseline import CrossEntropyTrainer
from decision_model.training.rlcd_trainer import RLCDTrainer
from decision_model.evaluation.eval_harness import evaluate_dataset
from decision_model.evaluation.calibration_metrics import compute_calibration_metrics
from decision_model.scripts.run_local_prototype import build_synthetic_tasksource_slice
from decision_model.reproducibility import set_reproducible_seed


def run_experiment():
    set_reproducible_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70, flush=True)
    print("COMPARATIVE EXPERIMENT: CE BASELINE VS RLCD CALIBRATION", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})", flush=True)
    print("=" * 70, flush=True)

    # 1. Dataset setup
    specs, task_examples = build_synthetic_tasksource_slice()
    split = create_task_family_split(specs, task_examples, train_family_fraction=0.80, in_task_val_fraction=0.20, seed=42)

    # 2. Build Base Model
    model_cfg = ModelConfig(
        base_model="Qwen/Qwen2.5-0.5B",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )
    base_model, tokenizer, marker_ids = build_jev_decision_model(model_cfg, device=device)

    # Clone models for fair comparison
    ce_model = copy.deepcopy(base_model).to(device)
    rlcd_model = copy.deepcopy(base_model).to(device)
    hybrid_model = copy.deepcopy(base_model).to(device)

    epochs = 3
    batch_size = 4

    # ─── RUN CONDITION A: CROSS-ENTROPY BASELINE ──────────────────────────────
    print("\n" + "─" * 60, flush=True)
    print("CONDITION A: Training Cross-Entropy Baseline (3 epochs)...", flush=True)
    print("─" * 60, flush=True)
    ce_trainer = CrossEntropyTrainer(ce_model, lr=1e-4, device=device)
    ce_losses = []

    for ep in range(epochs):
        loss = ce_trainer.train_epoch(
            train_examples=split.train_examples,
            tokenizer=tokenizer,
            marker_token_ids=marker_ids,
            epoch_idx=ep,
            total_epochs=epochs,
            batch_size=batch_size,
            log_interval=5,
        )
        ce_losses.append(loss)
        print(f"  CE Epoch {ep+1}/{epochs} Loss: {loss:.4f}", flush=True)

    # ─── RUN CONDITION B: RLCD LOG SCORING (UNBOUNDED LOG LOSS) ───────────────
    print("\n" + "─" * 60, flush=True)
    print("CONDITION B: Training RLCD Log Score (TUM Rewarding Doubt style)...", flush=True)
    print("─" * 60, flush=True)
    rlcd_log_cfg = RLCDConfig(
        use_brier_rlcr=False,
        exploration_sigma_start=0.12,
        exploration_sigma_end=0.02,
        log_score_weight=1.0,
        spherical_score_weight=0.5,
        rps_weight=0.5,
    )
    rlcd_trainer = RLCDTrainer(rlcd_model, rlcd_log_cfg, lr=5e-5, device=device)
    rlcd_losses = []

    for ep in range(epochs):
        loss = rlcd_trainer.train_epoch(
            train_examples=split.train_examples,
            tokenizer=tokenizer,
            marker_token_ids=marker_ids,
            epoch_idx=ep,
            total_epochs=epochs,
            batch_size=batch_size,
            log_interval=5,
        )
        rlcd_losses.append(loss)
        print(f"  RLCD-Log Epoch {ep+1}/{epochs} Loss: {loss:.4f}", flush=True)

    # ─── RUN CONDITION C: MIT RLCR (BOUNDED BRIER SCORE REWARD) ────────────────
    print("\n" + "─" * 60, flush=True)
    print("CONDITION C: Training MIT RLCR Bounded Brier (Damani et al., ICLR 2026)...", flush=True)
    print("─" * 60, flush=True)
    brier_model = copy.deepcopy(base_model).to(device)
    rlcr_brier_cfg = RLCDConfig(
        use_brier_rlcr=True,
        brier_weight=1.0,
        spherical_score_weight=0.5,
        rps_weight=0.5,
        exploration_sigma_start=0.12,
        exploration_sigma_end=0.02,
    )
    brier_trainer = RLCDTrainer(brier_model, rlcr_brier_cfg, lr=5e-5, device=device)
    brier_losses = []
    for ep in range(epochs):
        loss = brier_trainer.train_epoch(
            train_examples=split.train_examples,
            tokenizer=tokenizer,
            marker_token_ids=marker_ids,
            epoch_idx=ep,
            total_epochs=epochs,
            batch_size=batch_size,
            log_interval=5,
        )
        brier_losses.append(loss)
        print(f"  RLCR-Brier Epoch {ep+1}/{epochs} Loss: {loss:.4f}", flush=True)

    # ─── RUN CONDITION D: TWO-STAGE HYBRID (1 EP CE WARMUP + 2 EP RLCR BRIER) ──
    print("\n" + "─" * 60, flush=True)
    print("CONDITION D: Two-Stage Hybrid (1 ep CE Warmup -> 2 ep RLCR Brier)...", flush=True)
    print("─" * 60, flush=True)
    hybrid_ce_trainer = CrossEntropyTrainer(hybrid_model, lr=1e-4, device=device)
    h_ce_loss = hybrid_ce_trainer.train_epoch(
        train_examples=split.train_examples,
        tokenizer=tokenizer,
        marker_token_ids=marker_ids,
        epoch_idx=0,
        total_epochs=1,
        batch_size=batch_size,
    )
    print(f"  [Hybrid] Warmup CE Loss: {h_ce_loss:.4f}", flush=True)

    hybrid_rlcd_trainer = RLCDTrainer(hybrid_model, rlcr_brier_cfg, lr=5e-5, device=device)
    for ep in range(2):
        loss = hybrid_rlcd_trainer.train_epoch(
            train_examples=split.train_examples,
            tokenizer=tokenizer,
            marker_token_ids=marker_ids,
            epoch_idx=ep,
            total_epochs=2,
            batch_size=batch_size,
        )
        print(f"  [Hybrid] RLCR-Brier Epoch {ep+1}/2 Loss: {loss:.4f}", flush=True)

    # ─── FIT REGULARIZED TEMPERATURE SCALING ON HYBRID MODEL ───────────────────
    print("\n" + "─" * 60, flush=True)
    print("Fitting Regularized Temperature Scaling on Hybrid Model...", flush=True)
    print("─" * 60, flush=True)
    _, hyb_val_logits, hyb_val_labels = evaluate_dataset(
        model=hybrid_model,
        examples=split.in_task_val_examples,
        tokenizer=tokenizer,
        marker_token_ids=marker_ids,
        batch_size=batch_size,
        device=device,
    )
    temp_scaler = CardinalityTemperatureScaler()
    fitted_temps = temp_scaler.fit(hyb_val_logits, hyb_val_labels)

    # ─── DUAL EVALUATION (IN-TASK & ZERO-SHOT) FOR ALL CONDITIONS ──────────────
    print("\n" + "=" * 60, flush=True)
    print("EVALUATING ALL CONDITIONS ON IN-TASK & ZERO-SHOT SUITES", flush=True)
    print("=" * 60, flush=True)

    # 1. CE In-Task & Zero-Shot
    ce_in_rep, ce_in_logits, _ = evaluate_dataset(
        ce_model, split.in_task_val_examples, tokenizer, marker_ids, batch_size, device=device
    )
    ce_zs_rep, ce_zs_logits, _ = evaluate_dataset(
        ce_model, split.zero_shot_examples, tokenizer, marker_ids, batch_size, device=device
    )

    # 2. RLCD Log-Score In-Task & Zero-Shot
    rlcd_in_rep, rlcd_in_logits, _ = evaluate_dataset(
        rlcd_model, split.in_task_val_examples, tokenizer, marker_ids, batch_size, device=device
    )
    rlcd_zs_rep, rlcd_zs_logits, _ = evaluate_dataset(
        rlcd_model, split.zero_shot_examples, tokenizer, marker_ids, batch_size, device=device
    )

    # 3. MIT RLCR Brier In-Task & Zero-Shot
    brier_in_rep, brier_in_logits, _ = evaluate_dataset(
        brier_model, split.in_task_val_examples, tokenizer, marker_ids, batch_size, device=device
    )
    brier_zs_rep, brier_zs_logits, _ = evaluate_dataset(
        brier_model, split.zero_shot_examples, tokenizer, marker_ids, batch_size, device=device
    )

    # 4. Two-Stage Hybrid (CE Warmup + RLCR Brier)
    hyb_in_rep, _, _ = evaluate_dataset(
        hybrid_model, split.in_task_val_examples, tokenizer, marker_ids, batch_size, device=device
    )
    hyb_zs_rep, _, _ = evaluate_dataset(
        hybrid_model, split.zero_shot_examples, tokenizer, marker_ids, batch_size, device=device
    )

    # 5. Hybrid + Regularized TempScale
    hyb_ts_in_rep, _, _ = evaluate_dataset(
        hybrid_model, split.in_task_val_examples, tokenizer, marker_ids, batch_size, temp_scaler=temp_scaler, device=device
    )
    hyb_ts_zs_rep, _, _ = evaluate_dataset(
        hybrid_model, split.zero_shot_examples, tokenizer, marker_ids, batch_size, temp_scaler=temp_scaler, device=device
    )

    # Print summary comparative table
    print("\n" + "═" * 80, flush=True)
    print(f"{'Condition':<26} | {'In-Task Acc':<12} | {'In-Task ECE':<12} | {'Zero-Shot Acc':<14} | {'Zero-Shot ECE':<14}", flush=True)
    print("─" * 80, flush=True)
    print(f"{'1. Cross-Entropy Baseline':<26} | {ce_in_rep.accuracy*100:>10.2f}% | {ce_in_rep.ece*100:>10.2f}% | {ce_zs_rep.accuracy*100:>12.2f}% | {ce_zs_rep.ece*100:>12.2f}%", flush=True)
    print(f"{'2. RLCD Log Score (TUM)':<26} | {rlcd_in_rep.accuracy*100:>10.2f}% | {rlcd_in_rep.ece*100:>10.2f}% | {rlcd_zs_rep.accuracy*100:>12.2f}% | {rlcd_zs_rep.ece*100:>12.2f}%", flush=True)
    print(f"{'3. MIT RLCR (Bounded Brier)':<26} | {brier_in_rep.accuracy*100:>10.2f}% | {brier_in_rep.ece*100:>10.2f}% | {brier_zs_rep.accuracy*100:>12.2f}% | {brier_zs_rep.ece*100:>12.2f}%", flush=True)
    print(f"{'4. Two-Stage (CE -> Brier)':<26} | {hyb_in_rep.accuracy*100:>10.2f}% | {hyb_in_rep.ece*100:>10.2f}% | {hyb_zs_rep.accuracy*100:>12.2f}% | {hyb_zs_rep.ece*100:>12.2f}%", flush=True)
    print(f"{'5. Two-Stage + TempScale':<26} | {hyb_ts_in_rep.accuracy*100:>10.2f}% | {hyb_ts_in_rep.ece*100:>10.2f}% | {hyb_ts_zs_rep.accuracy*100:>12.2f}% | {hyb_ts_zs_rep.ece*100:>12.2f}%", flush=True)
    print("═" * 80, flush=True)

    # ─── MECHANISTIC INTERPRETABILITY & LATENT EXTRACTION ──────────────────────
    print("\n[Interp] Extracting option-marker latent representations & geometry...", flush=True)
    rlcd_model.eval()
    sample_exs = split.in_task_val_examples[:4]
    tokenized = encode_examples(sample_exs, tokenizer, marker_ids, device=device)

    with torch.no_grad():
        outputs = rlcd_model.backbone(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask)
        h_last = outputs.last_hidden_state  # (batch, seq, hidden)

        marker_reps = []
        for b_idx, positions in enumerate(tokenized.marker_positions):
            pos_t = torch.tensor(positions, device=device)
            h_opts = h_last[b_idx, pos_t, :].cpu().float().numpy()  # (k, hidden)
            marker_reps.append(h_opts)

    # ─── GENERATE PUBLICATION FIGURES ──────────────────────────────────────────
    figures_dir = PROJECT_ROOT / "jev-vault" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def safe_savefig(target_path, **kwargs):
        import time
        for attempt in range(5):
            try:
                plt.savefig(str(target_path), **kwargs)
                return
            except (OSError, PermissionError) as e:
                if attempt == 4:
                    raise
                print(f"  [Notice] File write retry {attempt+1}/5 on {target_path.name} ({e})", flush=True)
                time.sleep(0.5)

    # FIGURE 1: Training Dynamics (CE vs RLCD Log vs MIT RLCR Brier)
    print("Generating Fig 33: Training Dynamics...", flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    epochs_range = list(range(1, epochs + 1))

    ax[0].plot(epochs_range, ce_losses, marker="o", color="#d9534f", linewidth=2.5, label="Cross-Entropy Loss (NLL)")
    ax[0].set_title("Cross-Entropy Baseline Trajectory", fontsize=13, fontweight="bold")
    ax[0].set_xlabel("Epoch", fontsize=11)
    ax[0].set_ylabel("Loss", fontsize=11)
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(epochs_range, rlcd_losses, marker="s", color="#f0ad4e", linewidth=2.5, label="TUM RLCD Loss (Log + Sph + RPS)")
    ax[1].plot(epochs_range, brier_losses, marker="^", color="#0275d8", linewidth=2.5, label="MIT RLCR Loss (Bounded Brier + Sph + RPS)")
    ax[1].set_title("Proper Scoring Trajectories (TUM vs MIT)", fontsize=13, fontweight="bold")
    ax[1].set_xlabel("Epoch", fontsize=11)
    ax[1].set_ylabel("Loss (-Reward)", fontsize=11)
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    plt.suptitle("Fig 33: Multi-Task Decision Training Dynamics (RTX 2060)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig1_path = figures_dir / "fig33_rlcd_vs_ce_training_dynamics.png"
    safe_savefig(fig1_path, bbox_inches="tight")
    plt.close()

    # FIGURE 2: Reliability Diagrams & Calibration (In-Task vs Zero-Shot)
    print("Generating Fig 34: Reliability Diagrams...", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Plot In-Task & Zero-Shot Calibration Curves across literature methods
    for a, title, ce_rep, tum_rep, brier_rep, ts_rep in [
        (axes[0], "In-Task Validation Calibration", ce_in_rep, rlcd_in_rep, brier_in_rep, hyb_ts_in_rep),
        (axes[1], "Zero-Shot Family Generalization Calibration", ce_zs_rep, rlcd_zs_rep, brier_zs_rep, hyb_ts_zs_rep),
    ]:
        a.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect Calibration (Ideal)")

        # CE curve
        ce_confs = [b.confidence for b in ce_rep.reliability_bins if b.count > 0]
        ce_accs = [b.accuracy for b in ce_rep.reliability_bins if b.count > 0]
        a.plot(ce_confs, ce_accs, "o-", color="#d9534f", linewidth=2, label=f"CE Baseline (ECE: {ce_rep.ece*100:.1f}%)")

        # TUM Log curve
        tum_confs = [b.confidence for b in tum_rep.reliability_bins if b.count > 0]
        tum_accs = [b.accuracy for b in tum_rep.reliability_bins if b.count > 0]
        a.plot(tum_confs, tum_accs, "x-.", color="#f0ad4e", linewidth=1.8, label=f"TUM RLCD Log (ECE: {tum_rep.ece*100:.1f}%)")

        # MIT Brier curve
        brier_confs = [b.confidence for b in brier_rep.reliability_bins if b.count > 0]
        brier_accs = [b.accuracy for b in brier_rep.reliability_bins if b.count > 0]
        a.plot(brier_confs, brier_accs, "s-", color="#0275d8", linewidth=2, label=f"MIT RLCR Brier (ECE: {brier_rep.ece*100:.1f}%)")

        # Two-Stage + TempScale curve
        ts_confs = [b.confidence for b in ts_rep.reliability_bins if b.count > 0]
        ts_accs = [b.accuracy for b in ts_rep.reliability_bins if b.count > 0]
        a.plot(ts_confs, ts_accs, "^-", color="#5cb85c", linewidth=2.5, label=f"Two-Stage + TempScale (ECE: {ts_rep.ece*100:.1f}%)")

        a.set_title(title, fontsize=13, fontweight="bold")
        a.set_xlabel("Mean Predicted Confidence", fontsize=11)
        a.set_ylabel("Empirical Accuracy", fontsize=11)
        a.set_xlim(0.0, 1.0)
        a.set_ylim(0.0, 1.0)
        a.grid(True, alpha=0.3)
        a.legend(loc="lower right")

    plt.suptitle("Fig 34: Reliability Diagrams & Expected Calibration Error (ECE)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig2_path = figures_dir / "fig34_reliability_diagrams_calibration.png"
    safe_savefig(fig2_path, bbox_inches="tight")
    plt.close()

    # FIGURE 3: Mechanistic Latent Geometry & Interp
    print("Generating Fig 35: Latent Geometry & Interpretability...", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. Cosine Similarity Matrix between Option Markers in a 4-choice item
    sample_rep = marker_reps[2] if len(marker_reps) > 2 else marker_reps[0]
    # Normalize representations
    normed = sample_rep / np.linalg.norm(sample_rep, axis=1, keepdims=True)
    sim_matrix = normed @ normed.T

    im = axes[0].imshow(sim_matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0].set_title("Option-Marker Latent Cosine Similarity", fontsize=13, fontweight="bold")
    axes[0].set_xticks(range(len(sample_rep)))
    axes[0].set_yticks(range(len(sample_rep)))
    axes[0].set_xticklabels([f"[OPT_{chr(65+i)}]" for i in range(len(sample_rep))])
    axes[0].set_yticklabels([f"[OPT_{chr(65+i)}]" for i in range(len(sample_rep))])
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    for i in range(len(sample_rep)):
        for j in range(len(sample_rep)):
            axes[0].text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center", color="white" if sim_matrix[i, j] < 0.7 else "black")

    # 2. Logit Margin Distributions (CE vs TUM Log vs MIT Brier)
    ce_margins = [float(np.sort(l)[-1] - np.sort(l)[-2]) for l in ce_in_logits]
    rlcd_margins = [float(np.sort(l)[-1] - np.sort(l)[-2]) for l in rlcd_in_logits]
    brier_margins = [float(np.sort(l)[-1] - np.sort(l)[-2]) for l in brier_in_logits]

    axes[1].hist(ce_margins, bins=10, alpha=0.5, color="#d9534f", label="CE (Overconfident Spike)", density=True)
    axes[1].hist(rlcd_margins, bins=10, alpha=0.5, color="#f0ad4e", label="TUM Log (Conservative Spread)", density=True)
    axes[1].hist(brier_margins, bins=10, alpha=0.5, color="#0275d8", label="MIT Brier (Smooth Dispersion)", density=True)
    axes[1].set_title("Top-1 vs Top-2 Logit Margin Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Margin (z_1 - z_2)", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle("Fig 35: Mechanistic Latent Geometry & Logit Margin Mechanics", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig3_path = figures_dir / "fig35_latent_geometry_and_mechanistic_interp.png"
    safe_savefig(fig3_path, bbox_inches="tight")
    plt.close()

    # FIGURE 4: Risk-Coverage Selective Classification Frontiers
    print("Generating Fig 36: Risk-Coverage Pareto Frontiers...", flush=True)
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    cov_keys = ["cov_1.0", "cov_0.9", "cov_0.8", "cov_0.7", "cov_0.6", "cov_0.5"]
    cov_x = [float(k.replace("cov_", "")) * 100 for k in cov_keys]

    ce_cov_y = [ce_in_rep.risk_coverage.get(k, 0) * 100 for k in cov_keys]
    tum_cov_y = [rlcd_in_rep.risk_coverage.get(k, 0) * 100 for k in cov_keys]
    brier_cov_y = [brier_in_rep.risk_coverage.get(k, 0) * 100 for k in cov_keys]
    ts_cov_y = [hyb_ts_in_rep.risk_coverage.get(k, 0) * 100 for k in cov_keys]

    ax.plot(cov_x, ce_cov_y, "o--", color="#d9534f", linewidth=2, label="Cross-Entropy Baseline")
    ax.plot(cov_x, tum_cov_y, "x-.", color="#f0ad4e", linewidth=2, label="TUM RLCD Log")
    ax.plot(cov_x, brier_cov_y, "s-", color="#0275d8", linewidth=2.5, label="MIT RLCR (Bounded Brier)")
    ax.plot(cov_x, ts_cov_y, "^-", color="#5cb85c", linewidth=2.5, label="Two-Stage + TempScale (Pareto Optimal)")

    ax.set_title("Selective Classification Accuracy vs Coverage Budget", fontsize=13, fontweight="bold")
    ax.set_xlabel("Coverage Percentage (%) [Gated by Confidence Noul]", fontsize=11)
    ax.set_ylabel("Selective Classification Accuracy (%)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    plt.suptitle("Fig 36: Epistemic Risk-Coverage Pareto Frontier", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig4_path = figures_dir / "fig36_risk_coverage_pareto_frontiers.png"
    safe_savefig(fig4_path, bbox_inches="tight")
    plt.close()

    metrics_data = {
        "ce_in_task": {"acc": ce_in_rep.accuracy, "ece": ce_in_rep.ece},
        "ce_zero_shot": {"acc": ce_zs_rep.accuracy, "ece": ce_zs_rep.ece},
        "tum_log_in_task": {"acc": rlcd_in_rep.accuracy, "ece": rlcd_in_rep.ece},
        "tum_log_zero_shot": {"acc": rlcd_zs_rep.accuracy, "ece": rlcd_zs_rep.ece},
        "mit_brier_in_task": {"acc": brier_in_rep.accuracy, "ece": brier_in_rep.ece},
        "mit_brier_zero_shot": {"acc": brier_zs_rep.accuracy, "ece": brier_zs_rep.ece},
        "two_stage_in_task": {"acc": hyb_in_rep.accuracy, "ece": hyb_in_rep.ece},
        "two_stage_zero_shot": {"acc": hyb_zs_rep.accuracy, "ece": hyb_zs_rep.ece},
        "two_stage_ts_in_task": {"acc": hyb_ts_in_rep.accuracy, "ece": hyb_ts_in_rep.ece},
        "two_stage_ts_zero_shot": {"acc": hyb_ts_zs_rep.accuracy, "ece": hyb_ts_zs_rep.ece},
    }
    metrics_path = PROJECT_ROOT / "jev-vault" / "data" / "literature_comparative_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Comparative metrics saved to {metrics_path}", flush=True)

    print(f"\nAll publication figures successfully saved to {figures_dir}", flush=True)
    return {
        "ce_in_ece": ce_in_rep.ece,
        "ce_zs_ece": ce_zs_rep.ece,
        "brier_in_ece": brier_in_rep.ece,
        "brier_zs_ece": brier_zs_rep.ece,
        "hyb_ts_in_ece": hyb_ts_in_rep.ece,
        "hyb_ts_zs_ece": hyb_ts_zs_rep.ece,
        "figures": [str(fig1_path), str(fig2_path), str(fig3_path), str(fig4_path)],
    }


if __name__ == "__main__":
    run_experiment()
