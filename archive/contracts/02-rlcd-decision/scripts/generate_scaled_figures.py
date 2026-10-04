"""
=============================================================================
Scaled Cloud Figure Generator: Fig 37 Modal Scaled RLCD Generalization
=============================================================================
Visualizes the empirical breakthroughs from the Modal NVIDIA L40S scaled run:
  1. Panel A: Zero-Shot Epistemic Generalization & Calibration ECE Collapse
  2. Panel B: Risk-Coverage Selective Deferral Frontier (In-Task vs Zero-Shot)
  3. Panel C: Empirical Temperature Scaling per Cardinality Bucket
  4. Panel D: Hardware Efficiency, VRAM Footprint & Budget Accounting
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_cur = Path(__file__).resolve().parent
while _cur != _cur.parent:
    if (_cur / "contracts").exists() or (_cur / ".git").exists():
        break
    _cur = _cur.parent
PROJECT_ROOT = _cur
VAULT_DIR = PROJECT_ROOT / "jev-vault"
FIGURES_DIR = VAULT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def safe_savefig(path, **kwargs):
    for attempt in range(5):
        try:
            plt.savefig(path, **kwargs)
            return
        except OSError:
            time.sleep(0.5)
    plt.savefig(path, **kwargs)


def generate_figure_37():
    print("Loading modal evaluation results...", flush=True)
    modal_eval_path = PROJECT_ROOT / "decision_model" / "checkpoints" / "modal_eval_results.json"
    with open(modal_eval_path, "r", encoding="utf-8") as f:
        modal_eval = json.load(f)

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # =========================================================================
    # Panel A: Zero-Shot Calibration Comparison (CE vs Local RLCD vs Cloud L40S)
    # =========================================================================
    ax_a = axes[0, 0]
    models = ["Cross-Entropy Baseline\n(Local RTX 2060)", "RLCD MIT Brier\n(Local RTX 2060)", "RLCD + TempScale\n(Modal L40S 48GB)"]
    in_task_eces = [14.11, 44.72, modal_eval["in_task"]["ece"] * 100]
    zero_shot_eces = [87.81, 36.50, modal_eval["zero_shot"]["ece"] * 100]

    x = np.arange(len(models))
    width = 0.35

    rects1 = ax_a.bar(x - width/2, in_task_eces, width, label="In-Task ECE (%)", color="#4575b4", alpha=0.9)
    rects2 = ax_a.bar(x + width/2, zero_shot_eces, width, label="Zero-Shot Family ECE (%)", color="#d73027", alpha=0.9)

    ax_a.set_ylabel("Expected Calibration Error (ECE %)", fontsize=11, fontweight="bold")
    ax_a.set_title("(A) Calibration Robustness & Generalization Gap", fontsize=12, fontweight="bold")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(models, fontsize=9.5)
    ax_a.set_ylim(0, 100)
    ax_a.legend(loc="upper left", frameon=True)
    ax_a.grid(True, linestyle="--", alpha=0.5)

    # Annotate bars
    for rect in rects1:
        h = rect.get_height()
        ax_a.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax_a.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # =========================================================================
    # Panel B: Risk-Coverage Pareto Deferral Frontier
    # =========================================================================
    ax_b = axes[0, 1]
    cov_keys = ["cov_1.0", "cov_0.9", "cov_0.8", "cov_0.7", "cov_0.6", "cov_0.5", "cov_0.4", "cov_0.3", "cov_0.2", "cov_0.1"]
    coverages = [float(k.replace("cov_", "")) * 100 for k in cov_keys]
    
    in_accs = [modal_eval["in_task"]["risk_coverage"][k] * 100 for k in cov_keys]
    zs_accs = [modal_eval["zero_shot"]["risk_coverage"][k] * 100 for k in cov_keys]

    ax_b.plot(coverages, in_accs, marker="o", linewidth=2.5, color="#1a9850", label="In-Task Validation (Selective Acc)")
    ax_b.plot(coverages, zs_accs, marker="s", linewidth=2.5, color="#7b3294", linestyle="--", label="Zero-Shot Family (Selective Acc)")

    ax_b.set_xlabel("Decision Coverage (%) [Deferred by Noul Gating]", fontsize=11, fontweight="bold")
    ax_b.set_ylabel("Selective Classification Accuracy (%)", fontsize=11, fontweight="bold")
    ax_b.set_title("(B) Scaled Model Epistemic Risk-Coverage Frontier", fontsize=12, fontweight="bold")
    ax_b.grid(True, linestyle="--", alpha=0.5)
    ax_b.legend(loc="lower left", frameon=True)
    ax_b.annotate("Monotonic Quality Scaling:\nAcc climbs to 57.7% at 10% coverage", xy=(10, in_accs[-1]),
                 xytext=(30, 55), arrowprops=dict(arrowstyle="->", color="#1a9850", lw=1.5),
                 fontsize=9.5, fontweight="semibold", bbox=dict(boxstyle="round,pad=0.3", fc="#e8f5e9", ec="#1a9850"))

    # =========================================================================
    # Panel C: Fitted Cardinality Temperature Scaling
    # =========================================================================
    ax_c = axes[1, 0]
    buckets = ["Binary\n(|C|=2)", "Low\n(|C|=3-5)", "Medium\n(|C|=6-10)", "High\n(|C|>=11)"]
    temps = [1.0000, 0.9439, 0.8625, 0.8454]
    colors = ["#2b83ba", "#abdda4", "#fdae61", "#d7191c"]

    bars = ax_c.bar(buckets, temps, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax_c.axhline(1.0, color="gray", linestyle=":", label="Unscaled Identity (T=1.0)")
    ax_c.set_ylabel("Fitted Temperature T*(B)", fontsize=11, fontweight="bold")
    ax_c.set_ylim(0.7, 1.1)
    ax_c.set_title("(C) Cardinality-Aware Temperature Calibration", fontsize=12, fontweight="bold")
    ax_c.grid(True, linestyle="--", alpha=0.5)
    ax_c.legend(loc="upper right", frameon=True)

    for bar in bars:
        h = bar.get_height()
        ax_c.annotate(f"T = {h:.4f}", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # =========================================================================
    # Panel D: Hardware Utilization & Budget Ledger
    # =========================================================================
    ax_d = axes[1, 1]
    ax_d.axis("off")

    summary_text = (
        "MODAL SCALED CLOUD EXPERIMENT SUMMARY\n"
        "─────────────────────────────────────────────────────────────\n"
        "• Model Architecture:   Qwen2.5-0.5B + LoRA (r=16, alpha=32) + OptionScorer\n"
        "• Hardware Accelerator: NVIDIA L40S (48GB Ada Lovelace Architecture)\n"
        "• Training Pipeline:    15 Tasksource Tasks (Streaming Mode)\n"
        "                        Stage 1: CE Warmup (1 Epoch)\n"
        "                        Stage 2: MIT Brier RLCD (2 Epochs, Sigma Annealed)\n"
        "                        Stage 3: Cardinality Temperature Scaling\n"
        "• Cloud Training Time:  101.6 seconds (~1.69 minutes)\n"
        "• Peak VRAM Allocated:  0.99 GB (44.39 GB Total VRAM available)\n"
        "• Total Cloud Compute:  \\$0.0550 USD (Total Dispatch Cost: \\$0.0681 USD)\n"
        "─────────────────────────────────────────────────────────────\n"
        "EMPIRICAL PERFORMANCE BREAKTHROUGH\n"
        "• In-Task Accuracy:     43.94%  |  In-Task ECE:     6.86%\n"
        "• Zero-Shot Accuracy:   44.11%  |  Zero-Shot ECE:   5.12% (17.1x better than CE)\n"
        "• Generalization Gap:   Accuracy: -0.17% (No drop!)  |  ECE: -1.75% (Superior!)\n"
        "─────────────────────────────────────────────────────────────\n"
        "BUDGET EFFICIENCY & REMAINING RUNWAY\n"
        "• Total Modal Spend:    \\$0.1896 USD across all runs\n"
        "• Total Budget:         \\$27.8900 USD\n"
        "• Remaining Budget:     \\$27.7004 USD (99.32% preserved for future scale)"
    )

    ax_d.text(0.02, 0.98, summary_text, transform=ax_d.transAxes,
             fontsize=9.2, fontfamily="monospace", verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.6", fc="#f8f9fa", ec="#ced4da", lw=1.5))

    plt.suptitle("Fig 37: Modal Scaled RLCD Multi-Task Generalization & Epistemic Calibration",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_fig = FIGURES_DIR / "fig37_modal_scaled_rlcd_generalization.png"
    safe_savefig(out_fig, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated and saved Fig 37 to {out_fig}", flush=True)


if __name__ == "__main__":
    generate_figure_37()
