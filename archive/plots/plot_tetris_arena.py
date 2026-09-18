"""
Plot Tetris Epistemic Arena Results
===================================
Generates publication-quality figure:
  - Figure 16: The Real-Time Tetris Benchmark: Latent Collapse vs. Epistemic Search Mastery
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

modes = [
    "ERET_K4 (Latent Loop)",
    "S1_Greedy (Reflexive)",
    "Uniform_Search",
    "Adaptive_ETS (τ=0.80)",
]

pieces = [12.5, 60.8, 102.6, 111.8]
lines = [0.0, 12.6, 29.1, 33.2]
holes = [30.9, 25.7, 14.2, 4.2]
expansions = [0.00, 0.00, 4.00, 3.09]

colors = ["#D32F2F", "#1976D2", "#388E3C", "#7B1FA2"]

# Subplot 1: Lines Cleared & Pieces Survived
ax1 = axes[0]
ax1.set_facecolor("#FFFFFF")
x = np.arange(len(modes))
w = 0.35

bars1 = ax1.bar(x - w/2, lines, w, label="Lines Cleared", color="#7B1FA2", alpha=0.9, edgecolor="#4A148C")
bars2 = ax1.bar(x + w/2, pieces, w, label="Pieces Survived", color="#26A69A", alpha=0.9, edgecolor="#004D40")

ax1.set_ylabel("Count", fontsize=11, fontweight="bold")
ax1.set_title("A. Survival & Line Clearing in Real-Time Tetris", fontsize=12, fontweight="bold", pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(["ERET (K=4)", "S1 Reflex", "Uniform Lookahead", "Adaptive ETS (τ=0.8)"], rotation=15, ha="right", fontsize=9.5)
ax1.set_ylim(0, 135)
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.legend(frameon=True, facecolor="#FAFAFA", fontsize=9.5)

for b in bars1:
    h = b.get_height()
    ax1.annotate(f"{h:.1f}", xy=(b.get_x() + b.get_width()/2, h + 1.5), ha="center", va="bottom", fontsize=8.5, fontweight="bold")
for b in bars2:
    h = b.get_height()
    ax1.annotate(f"{h:.1f}", xy=(b.get_x() + b.get_width()/2, h + 1.5), ha="center", va="bottom", fontsize=8.5, fontweight="bold")

# Subplot 2: Hole Creation (Board Contamination)
ax2 = axes[1]
ax2.set_facecolor("#FFFFFF")
bars3 = ax2.bar(x, holes, color=colors, alpha=0.85, edgecolor="#222222", width=0.5)

ax2.set_ylabel("Final Hole Count (Lower is Cleaner)", fontsize=11, fontweight="bold")
ax2.set_title("B. Board Health & Hole Formation", fontsize=12, fontweight="bold", pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(["ERET (K=4)", "S1 Reflex", "Uniform Lookahead", "Adaptive ETS (τ=0.8)"], rotation=15, ha="right", fontsize=9.5)
ax2.set_ylim(0, 40)
ax2.grid(axis="y", linestyle="--", alpha=0.4)

for b in bars3:
    h = b.get_height()
    ax2.annotate(f"{h:.1f} holes", xy=(b.get_x() + b.get_width()/2, h + 1.0), ha="center", va="bottom", fontsize=9, fontweight="bold")

# Subplot 3: Compute Efficiency (Lines Cleared per Search Expansion)
ax3 = axes[2]
ax3.set_facecolor("#FFFFFF")

ax3.scatter([expansions[0]], [lines[0]], color="#D32F2F", s=140, zorder=5, label="ERET (Collapse)", marker="X")
ax3.scatter([expansions[1]], [lines[1]], color="#1976D2", s=140, zorder=5, label="S1 Reflex (0 Expansions)", marker="o")
ax3.scatter([expansions[3]], [lines[3]], color="#7B1FA2", s=200, zorder=5, label="Adaptive ETS (Pareto Superior)", marker="*")
ax3.scatter([expansions[2]], [lines[2]], color="#388E3C", s=140, zorder=5, label="Uniform Lookahead", marker="s")

ax3.plot([expansions[1], expansions[3]], [lines[1], lines[3]], linestyle="--", color="#7B1FA2", alpha=0.6, linewidth=1.5)

ax3.set_xlabel("Search Expansions per Piece", fontsize=11, fontweight="bold")
ax3.set_ylabel("Lines Cleared", fontsize=11, fontweight="bold")
ax3.set_title("C. Inference Compute Pareto Frontier (Tetris)", fontsize=12, fontweight="bold", pad=12)
ax3.set_xlim(-0.3, 4.5)
ax3.set_ylim(-2, 40)
ax3.grid(True, linestyle="--", alpha=0.4)
ax3.legend(frameon=True, facecolor="#FAFAFA", fontsize=9.5)

ax3.annotate("Adaptive ETS:\nHighest Lines (33.2)\nLowest Holes (4.2)\n23% Less Compute",
             xy=(expansions[3], lines[3]), xytext=(expansions[3] - 1.8, lines[3] - 12),
             arrowprops=dict(arrowstyle="->", color="#7B1FA2", lw=1.2),
             fontsize=9, fontweight="bold", color="#7B1FA2",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3E5F5", edgecolor="#BA68C8"))

plt.tight_layout()
out_fig = "figures/fig16_tetris_epistemic_arena.png"
plt.savefig(out_fig, dpi=300)
print(f"Saved Figure 16 to {out_fig}")
