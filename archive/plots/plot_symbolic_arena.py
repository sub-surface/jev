"""
Plot Symbolic Program Deduction Arena Results
=============================================
Generates publication-quality figure:
  - Figure 18: Symbolic Program Deduction: Depth Scaling and Epistemic Tree-of-Thought
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#222222"
plt.rcParams["axes.linewidth"] = 0.8

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)
fig.patch.set_facecolor("#FAFAFA")

hops = ["4-Hop (In-Dist)", "8-Hop (OOD Deep)", "12-Hop (Extreme)"]
s1_acc = [90.0, 80.0, 77.5]
tot_acc = [92.5, 95.0, 85.0]
ets_acc = [92.5, 95.0, 85.0]
eret_acc = [0.0, 0.0, 0.0]

exp_tot = [24.8, 48.6, 77.5]
exp_ets = [20.2, 42.5, 71.2]

# Subplot 1: Success Rate across Deductive Horizons
ax1 = axes[0]
ax1.set_facecolor("#FFFFFF")
x = np.arange(len(hops))
w = 0.22

bars1 = ax1.bar(x - 1.5*w, eret_acc, w, label="ERET Latent Loop", color="#D32F2F", alpha=0.9, edgecolor="#222222")
bars2 = ax1.bar(x - 0.5*w, s1_acc, w, label="S1 Greedy (Reflex)", color="#1976D2", alpha=0.9, edgecolor="#222222")
bars3 = ax1.bar(x + 0.5*w, tot_acc, w, label="Uniform Tree-of-Thought", color="#388E3C", alpha=0.9, edgecolor="#222222")
bars4 = ax1.bar(x + 1.5*w, ets_acc, w, label="Adaptive Epistemic ToT", color="#7B1FA2", alpha=0.9, edgecolor="#222222")

ax1.set_ylabel("Deductive Success Rate (%)", fontsize=11, fontweight="bold")
ax1.set_title("A. Program Deduction Scaling Across Depths", fontsize=12, fontweight="bold", pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(hops, fontsize=10, fontweight="bold")
ax1.set_ylim(0, 110)
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.legend(frameon=True, facecolor="#FAFAFA", fontsize=9, loc="lower left")

for b in bars2:
    h = b.get_height()
    ax1.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width()/2, h + 1.5), ha="center", va="bottom", fontsize=8, fontweight="bold")
for b in bars4:
    h = b.get_height()
    ax1.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width()/2, h + 1.5), ha="center", va="bottom", fontsize=8, fontweight="bold", color="#4A148C")

# Subplot 2: Inference Expansions (Compute Savings)
ax2 = axes[1]
ax2.set_facecolor("#FFFFFF")
w2 = 0.35

bars_tot = ax2.bar(x - w2/2, exp_tot, w2, label="Uniform ToT (Search)", color="#388E3C", alpha=0.85, edgecolor="#222222")
bars_ets = ax2.bar(x + w2/2, exp_ets, w2, label="Adaptive Epistemic ToT", color="#7B1FA2", alpha=0.85, edgecolor="#222222")

ax2.set_ylabel("Search Expansions per Program", fontsize=11, fontweight="bold")
ax2.set_title("B. Search Compute Scaling with Depth", fontsize=12, fontweight="bold", pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(hops, fontsize=10, fontweight="bold")
ax2.set_ylim(0, 95)
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.legend(frameon=True, facecolor="#FAFAFA", fontsize=9.5)

for b1, b2 in zip(bars_tot, bars_ets):
    h1 = b1.get_height()
    h2 = b2.get_height()
    saving = (h1 - h2) / h1 * 100
    ax2.annotate(f"-{saving:.0f}%", xy=(b2.get_x() + b2.get_width()/2, h2 + 2.0), ha="center", va="bottom", fontsize=9, fontweight="bold", color="#7B1FA2")

# Subplot 3: Cross-Horizon Pareto Summary
ax3 = axes[2]
ax3.set_facecolor("#FFFFFF")

# Plot 12-hop Pareto points
ax3.scatter([0.0], [0.0], color="#D32F2F", s=140, zorder=5, label="ERET (Total Failure)", marker="X")
ax3.scatter([0.0], [s1_acc[2]], color="#1976D2", s=140, zorder=5, label="S1 Greedy (0 Expansions)", marker="o")
ax3.scatter([exp_ets[2]], [ets_acc[2]], color="#7B1FA2", s=200, zorder=5, label="Adaptive Epistemic ToT", marker="*")
ax3.scatter([exp_tot[2]], [tot_acc[2]], color="#388E3C", s=140, zorder=5, label="Uniform ToT", marker="s")

ax3.plot([0.0, exp_ets[2]], [s1_acc[2], ets_acc[2]], linestyle="--", color="#7B1FA2", alpha=0.6, linewidth=1.5)

ax3.set_xlabel("Expansions on 12-Hop Extreme Horizon", fontsize=11, fontweight="bold")
ax3.set_ylabel("Accuracy (%) on 12-Hop", fontsize=11, fontweight="bold")
ax3.set_title("C. 12-Hop Extreme Horizon Pareto Frontier", fontsize=12, fontweight="bold", pad=12)
ax3.set_xlim(-5, 90)
ax3.set_ylim(-5, 100)
ax3.grid(True, linestyle="--", alpha=0.4)
ax3.legend(frameon=True, facecolor="#FAFAFA", fontsize=9.5)

ax3.annotate("Adaptive ToT:\nMatches 85% Accuracy\nwith 8% fewer expansions",
             xy=(exp_ets[2], ets_acc[2]), xytext=(exp_ets[2] - 38, ets_acc[2] - 22),
             arrowprops=dict(arrowstyle="->", color="#7B1FA2", lw=1.2),
             fontsize=9, fontweight="bold", color="#7B1FA2",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3E5F5", edgecolor="#BA68C8"))

plt.tight_layout()
out_fig = "figures/fig18_symbolic_deduction_scaling.png"
plt.savefig(out_fig, dpi=300)
print(f"Saved Figure 18 to {out_fig}")
