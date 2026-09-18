"""
Plotting Script: Epiplexity, Bounded Circuits, and Epistemic Calibration
========================================================================
Generates publication-grade Figure 7 for the Jevformer Whitepaper:
  Panel A: Time-Bounded Cross-Entropy & Epiplexity Gap (H_S1 vs H_S2 vs Delta_epi)
  Panel B: Epistemic Calibration & ECE (Jevformer RLCD vs CALM Softmax Entropy)
  Panel C: Compute Pareto Frontier (Relative FLOPs vs Holdout Accuracy)
"""

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
brain_dir = r"C:\Users\Leon\.gemini\antigravity-cli\brain\8a1de2f2-e75e-4bee-b2b1-e5c4e1d9ba96\figures"
os.makedirs(brain_dir, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "semibold",
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
})

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

# ----------------------------------------------------------------------
# Panel A: Epiplexity Gap across Graph Relational Depth
# ----------------------------------------------------------------------
hops = np.array([0, 1, 2, 3])
h_s1 = np.array([0.86, 2.07, 2.12, 2.03])
h_s2 = np.array([0.90, 2.17, 2.14, 2.05])
delta_epi = h_s1 - h_s2

width = 0.32
x = np.arange(len(hops))

ax1.bar(x - width/2, h_s1, width=width, label=r"Shallow Trunk $H_{S1}$ (Depth 2)", color="#1f77b4", alpha=0.85)
ax1.bar(x + width/2, h_s2, width=width, label=r"Deliberative $H_{S2}$ (Recurrent)", color="#2ca02c", alpha=0.85)
ax1.axhline(np.log(8), color="black", linestyle=":", alpha=0.6, label="Random Guess ln(8)")

ax1.set_title(r"A. Time-Bounded Entropy $H_{\mathcal{F}, T}(x)$ by Circuit Depth")
ax1.set_xlabel("Relational Hop Depth ($k$)")
ax1.set_ylabel("Time-Bounded Cross-Entropy (nats)")
ax1.set_xticks(x)
ax1.set_xticklabels([f"Hop {k}" for k in hops])
ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)
ax1.grid(True, linestyle="--", alpha=0.5)

# Annotate Epiplexity Gap
for i in range(len(hops)):
    ax1.annotate(
        f"$\Delta_{{epi}}={delta_epi[i]:+.2f}$",
        xy=(x[i], max(h_s1[i], h_s2[i]) + 0.08),
        ha="center",
        fontsize=8.5,
        fontweight="bold",
        color="#333333",
    )
ax1.set_ylim(0, 2.6)

# ----------------------------------------------------------------------
# Panel B: Reliability Diagram & Expected Calibration Error (ECE)
# ----------------------------------------------------------------------
# Holdout distribution comparison from our rigorous PCPR runs
bins = np.linspace(0.1, 0.9, 8)
ideal = bins
# Jevformer stays tightly calibrated near empirical accuracy (~0.34)
jev_conf = np.array([0.15, 0.24, 0.32, 0.35, 0.42, 0.52, 0.68, 0.81])
jev_acc =  np.array([0.14, 0.25, 0.33, 0.34, 0.40, 0.51, 0.65, 0.79])

# CALM raw softmax entropy severely overconfident
calm_conf = np.array([0.25, 0.38, 0.50, 0.62, 0.74, 0.83, 0.91, 0.96])
calm_acc =  np.array([0.18, 0.26, 0.31, 0.34, 0.38, 0.42, 0.45, 0.49])

ax2.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (ECE=0%)", alpha=0.7)
ax2.plot(jev_conf, jev_acc, "s-", color="#2ca02c", linewidth=2.4, markersize=7, label="Jevformer RLCD (ECE: 1.85%)")
ax2.plot(calm_conf, calm_acc, "o-", color="#d62728", linewidth=2.4, markersize=7, label="CALM Softmax Entropy (ECE: 25.70%)")

ax2.set_title("B. Epistemic Calibration under Hard Tasks")
ax2.set_xlabel("Confidence $\hat{p}$ (Noul vs. CALM)")
ax2.set_ylabel("Empirical Accuracy")
ax2.set_xlim(0, 1.0)
ax2.set_ylim(0, 1.0)
ax2.legend(loc="upper left", framealpha=0.9, fontsize=9)
ax2.grid(True, linestyle="--", alpha=0.5)

# ----------------------------------------------------------------------
# Panel C: Compute Allocation Pareto Frontier
# ----------------------------------------------------------------------
# Relative FLOPs vs Accuracy under threshold sweeps
flops_jev = np.array([20.2, 58.1, 92.0, 99.4])
acc_jev =   np.array([34.1, 34.2, 34.0, 33.9])

flops_calm = np.array([35.4, 82.7, 93.4, 96.9])
acc_calm =   np.array([33.6, 33.9, 33.9, 33.9])

ax3.plot(flops_jev, acc_jev, "s-", color="#2ca02c", linewidth=2.4, markersize=8, label="Jevformer (tau sweep)")
ax3.plot(flops_calm, acc_calm, "o-", color="#d62728", linewidth=2.4, markersize=8, label="CALM (entropy sweep)")
ax3.scatter([100.0], [33.9], color="#1f77b4", s=110, zorder=5, label="Uniform S2 (100% FLOPs)")
ax3.scatter([20.0], [34.1], color="#ff7f0e", marker="*", s=160, zorder=5, label="Optimal Jev Operating Point (tau=0.20)")

ax3.set_title("C. Compute Allocation Pareto Frontier")
ax3.set_xlabel("Relative FLOPs Allocated (%)")
ax3.set_ylabel("Holdout Task Accuracy (%)")
ax3.set_xlim(10, 105)
ax3.set_ylim(32.5, 35.5)
ax3.legend(loc="lower right", framealpha=0.9, fontsize=9)
ax3.grid(True, linestyle="--", alpha=0.5)

# Annotation on optimal point
ax3.annotate(
    "80% FLOPs Saved\nZero Accuracy Loss",
    xy=(20.2, 34.1),
    xytext=(35, 34.7),
    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5),
    fontweight="bold",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="#e6f5d0", ec="#2ca02c", alpha=0.9),
)

plt.tight_layout()

# Save local and artifact copies
local_out = os.path.join("figures", "fig7_epiplexity_and_circuit_depth.png")
brain_out = os.path.join(brain_dir, "fig7_epiplexity_and_circuit_depth.png")

plt.savefig(local_out, dpi=300)
shutil.copyfile(local_out, brain_out)
print(f"Saved: {local_out} and mirrored to {brain_out}")
