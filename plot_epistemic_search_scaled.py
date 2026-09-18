"""
Plot Scaled Epistemic Search Benchmark Results
==============================================
Generates publication-quality figure:
  - Figure 15: The Bitter Lesson Revealed: Tree Search vs. Latent Equilibrium Collapse
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#222222"
plt.rcParams["axes.linewidth"] = 0.8

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)
fig.patch.set_facecolor("#FAFAFA")

# Data from Modal A10G Cloud Run
labels_12 = ["ERET (K=4)", "S1 Greedy", "Adaptive ETS (τ=0.85)", "Uniform Search (D=5)"]
succ_12 = [0.0, 76.0, 77.3, 81.3]
exp_12 = [0.0, 0.0, 13.7, 63.2]
col_12 = [92, 1, 1, 0]

labels_16 = ["ERET (K=4)", "S1 Greedy", "Adaptive ETS (τ=0.85)", "Uniform Search (D=5)"]
succ_16 = [1.0, 61.0, 73.0, 81.0]
exp_16 = [0.0, 0.0, 53.0, 84.0]
col_16 = [93, 0, 0, 0]

colors = ["#D32F2F", "#1976D2", "#7B1FA2", "#388E3C"]

# Subplot 1: Success Rate Comparison (12x12 vs 16x16)
x = np.arange(len(labels_12))
width = 0.35

ax1 = axes[0]
ax1.set_facecolor("#FFFFFF")
bars1 = ax1.bar(x - width/2, succ_12, width, label="12x12 (In-Distribution)", color="#26A69A", alpha=0.9, edgecolor="#004D40")
bars2 = ax1.bar(x + width/2, succ_16, width, label="16x16 (Out-of-Distribution Hard)", color="#FFA726", alpha=0.9, edgecolor="#E65100")

ax1.set_ylabel("Closed-Loop Success Rate (%)", fontsize=11, fontweight="bold")
ax1.set_title("A. Closed-Loop Autonomous Navigation", fontsize=12, fontweight="bold", pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(labels_12, rotation=15, ha="right", fontsize=9.5)
ax1.set_ylim(0, 100)
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.legend(frameon=True, facecolor="#FAFAFA", fontsize=9.5)

for b in bars1:
    h = b.get_height()
    ax1.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width()/2, h + 1.5), ha="center", va="bottom", fontsize=8.5, fontweight="bold")
for b in bars2:
    h = b.get_height()
    ax1.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width()/2, h + 1.5), ha="center", va="bottom", fontsize=8.5, fontweight="bold")

# Subplot 2: Compute Pareto Frontier on 16x16 Hard Labyrinths
ax2 = axes[1]
ax2.set_facecolor("#FFFFFF")

ax2.scatter([exp_16[0]], [succ_16[0]], color="#D32F2F", s=140, zorder=5, label="ERET Latent Loop (Collapse)", marker="X")
ax2.scatter([exp_16[1]], [succ_16[1]], color="#1976D2", s=140, zorder=5, label="S1 Greedy (0 Expansions)", marker="o")
ax2.scatter([exp_16[2]], [succ_16[2]], color="#7B1FA2", s=180, zorder=5, label="Adaptive ETS (Epistemic Gated)", marker="*")
ax2.scatter([exp_16[3]], [succ_16[3]], color="#388E3C", s=140, zorder=5, label="Uniform Search (Fixed Budget)", marker="s")

# Plot Pareto connection
ax2.plot([exp_16[1], exp_16[2], exp_16[3]], [succ_16[1], succ_16[2], succ_16[3]], linestyle="--", color="#7B1FA2", alpha=0.6, linewidth=1.5)

ax2.set_xlabel("Search Expansions per Maze (Inference Compute)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Success Rate on 16x16 (%)", fontsize=11, fontweight="bold")
ax2.set_title("B. Inference Compute Pareto Frontier (16x16)", fontsize=12, fontweight="bold", pad=12)
ax2.set_xlim(-5, 95)
ax2.set_ylim(0, 100)
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.legend(frameon=True, facecolor="#FAFAFA", fontsize=9)

ax2.annotate("Adaptive ETS:\n+12.0% Acc over S1\nwith 37% fewer FLOPs", 
             xy=(exp_16[2], succ_16[2]), xytext=(exp_16[2] - 30, succ_16[2] - 25),
             arrowprops=dict(arrowstyle="->", color="#7B1FA2", lw=1.2),
             fontsize=9, fontweight="bold", color="#7B1FA2",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3E5F5", edgecolor="#BA68C8"))

# Subplot 3: Collision Rates (The Reality of Latent Drift)
ax3 = axes[2]
ax3.set_facecolor("#FFFFFF")
y_pos = np.arange(len(labels_16))

col_rates_16 = [col_16[0]/100 * 100, col_16[1]/100 * 100, col_16[2]/100 * 100, col_16[3]/100 * 100]
bars3 = ax3.barh(y_pos, col_rates_16, color=["#D32F2F", "#1976D2", "#7B1FA2", "#388E3C"], height=0.55, edgecolor="#222222", alpha=0.85)

ax3.set_yticks(y_pos)
ax3.set_yticklabels(labels_16, fontsize=9.5)
ax3.set_xlabel("Wall Collision Rate (%)", fontsize=11, fontweight="bold")
ax3.set_title("C. Collision Reliability (16x16)", fontsize=12, fontweight="bold", pad=12)
ax3.set_xlim(0, 105)
ax3.grid(axis="x", linestyle="--", alpha=0.4)

for b in bars3:
    w = b.get_width()
    ax3.annotate(f"{w:.1f}%", xy=(w + 2, b.get_y() + b.get_height()/2), va="center", fontsize=9, fontweight="bold")

plt.tight_layout()
fig_path = "figures/fig15_epistemic_search_scaled_pareto.png"
plt.savefig(fig_path, dpi=300)
print(f"Saved publication figure to {fig_path}")
