"""
Generate Figure 38: Qwen3-4B Scaled RLCD Multi-Task Empirical Calibration & Scaling Frontier.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Figure 38: Scaled RLCD on Qwen3-4B (NVIDIA L40S) — Empirical Calibration & Risk Frontier", fontsize=15, fontweight="bold")

# Palette
c_05b = "#4A90E2"
c_4b = "#50E3C2"
c_dark = "#1E2A38"
c_warn = "#FF6B6B"

# --- Panel 1: Model Capacity Scaling (0.5B vs 4B) ---
ax1 = axes[0, 0]
models = ["Qwen2.5-0.5B\n(LoRA r=16)", "Qwen3-4B\n(LoRA r=32)"]
x = np.arange(len(models))
width = 0.35

in_task_acc = [43.94, 86.99]
zero_shot_acc = [44.11, 70.18]

rects1 = ax1.bar(x - width/2, in_task_acc, width, label="In-Task Validation Acc", color="#3B82F6", alpha=0.9)
rects2 = ax1.bar(x + width/2, zero_shot_acc, width, label="Zero-Shot Family Acc", color="#10B981", alpha=0.9)

ax1.set_ylabel("Top-1 Accuracy (%)", fontsize=11, fontweight="bold")
ax1.set_title("(A) Accuracy Leap Across Scale (0.5B vs. 4B)", fontsize=12, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=10)
ax1.set_ylim(0, 105)
ax1.legend(loc="upper left")

for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha="center", va="bottom", fontweight="bold")
for rect in rects2:
    h = rect.get_height()
    ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha="center", va="bottom", fontweight="bold")

# --- Panel 2: Selective Risk-Coverage Monotonicity (Qwen3-4B) ---
ax2 = axes[0, 1]
coverages = [100, 80, 50]
in_task_cov_acc = [87.0, 92.9, 97.8]
zero_shot_cov_acc = [70.2, 76.2, 80.9]

ax2.plot(coverages, in_task_cov_acc, "o-", color="#3B82F6", linewidth=2.5, markersize=8, label="In-Task (Selective Deferral)")
ax2.plot(coverages, zero_shot_cov_acc, "s--", color="#10B981", linewidth=2.5, markersize=8, label="Zero-Shot Family (Paraphrase)")
ax2.axhline(97.8, color="#EF4444", linestyle=":", label="Peak Accuracy: 97.8% (50% cov)")

ax2.set_xlabel("Data Coverage (%) [Ranked by Epistemic Noul]", fontsize=11, fontweight="bold")
ax2.set_ylabel("Selective Accuracy (%)", fontsize=11, fontweight="bold")
ax2.set_title("(B) Risk-Coverage Frontier (Epistemic Selective Deferral)", fontsize=12, fontweight="bold")
ax2.set_ylim(65, 102)
ax2.invert_xaxis()
ax2.legend(loc="lower right")

for c, a in zip(coverages, in_task_cov_acc):
    ax2.annotate(f"{a:.1f}%", (c, a), textcoords="offset points", xytext=(0, 7), ha="center", fontweight="bold", color="#1E40AF")
for c, a in zip(coverages, zero_shot_cov_acc):
    ax2.annotate(f"{a:.1f}%", (c, a), textcoords="offset points", xytext=(0, -15), ha="center", fontweight="bold", color="#065F46")

# --- Panel 3: Proper Scoring Training Dynamics (Stage 1 vs Stage 2) ---
ax3 = axes[1, 0]
batches_ce = [25, 50, 75, 100, 125, 150, 168]
loss_ce = [0.9022, 0.7933, 0.7505, 0.7055, 0.6744, 0.6412, 0.6196]

batches_rlcd_ep1 = [168 + 25, 168 + 50, 168 + 75, 168 + 100, 168 + 125, 168 + 150, 168 + 168]
reward_rlcd_ep1 = [1.0673, 1.0696, 1.0216, 1.0244, 1.0246, 1.0372, 1.0402]

batches_rlcd_ep2 = [336 + 25, 336 + 50, 336 + 75, 336 + 100, 336 + 125, 336 + 150, 336 + 168]
reward_rlcd_ep2 = [1.1544, 1.1509, 1.1675, 1.1641, 1.1750, 1.1893, 1.1945]

ax3.plot(batches_ce, loss_ce, "o-", color="#F59E0B", linewidth=2, label="Stage 1: CE Loss (Warmup)")
ax3.plot(batches_rlcd_ep1, reward_rlcd_ep1, "s-", color="#6366F1", linewidth=2, label="Stage 2 Ep 1: Brier Reward (σ=0.10)")
ax3.plot(batches_rlcd_ep2, reward_rlcd_ep2, "^-", color="#10B981", linewidth=2.5, label="Stage 2 Ep 2: Brier Reward (σ=0.01)")

ax3.axvline(168, color="gray", linestyle="--", alpha=0.7)
ax3.axvline(336, color="gray", linestyle="--", alpha=0.7)
ax3.text(84, 0.45, "Stage 1\n(CE Warmup)", ha="center", fontsize=9, fontstyle="italic")
ax3.text(252, 0.45, "Stage 2 Ep 1\n(RLCD σ=0.10)", ha="center", fontsize=9, fontstyle="italic")
ax3.text(420, 0.45, "Stage 2 Ep 2\n(RLCD σ=0.01)", ha="center", fontsize=9, fontstyle="italic")

ax3.set_xlabel("Training Batch Index (Batch Size 20)", fontsize=11, fontweight="bold")
ax3.set_ylabel("Loss / Mean Reward Metric", fontsize=11, fontweight="bold")
ax3.set_title("(C) Multi-Stage Policy Convergence (NVIDIA L40S)", fontsize=12, fontweight="bold")
ax3.legend(loc="upper left")

# --- Panel 4: Per-Cardinality Fitted Calibration Temperatures ---
ax4 = axes[1, 1]
buckets = ["Binary (K=2)", "Multi (K=3-5)", "Wide (K=6-10)", "Deep (K>=11)"]
temps = [1.2766, 1.2129, 0.8676, 1.8153]
counts = [245, 100, 23, 1]

bars = ax4.bar(buckets, temps, color="#8B5CF6", alpha=0.85, width=0.5)
ax4.axhline(1.0, color="#EF4444", linestyle="--", label="Unscaled Temperature (T=1.0)")

ax4.set_ylabel("Fitted Temperature T*(|C|)", fontsize=11, fontweight="bold")
ax4.set_title("(D) Stage 3 Cardinality-Aware Temperature Scaling", fontsize=12, fontweight="bold")
ax4.set_ylim(0, 2.2)
ax4.legend(loc="upper right")

for bar, count, t in zip(bars, counts, temps):
    ax4.annotate(f"T={t:.2f}\n(n={count})", xy=(bar.get_x() + bar.get_width()/2, t), xytext=(0, 5),
                 textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
output_path = Path("C:/Users/Leon/Desktop/Psychograph/jev/jev-vault/figures/fig38_qwen3_4b_scaling_and_calibration.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300)
plt.close()
print(f"Figure 38 successfully saved to {output_path}", flush=True)
