"""
Plotting Script for Modal Scaled ERET (Jevformer 2.0) Results
=============================================================
Generates:
  - figures/fig13_eret_modal_scaling.png
  - figures/fig14_eret_computational_biology.png
Mirrors both to brain/<conversation-id>/figures/
"""

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

os.makedirs("figures", exist_ok=True)
brain_dir = r"C:\Users\Leon\.gemini\antigravity-cli\brain\8a1de2f2-e75e-4bee-b2b1-e5c4e1d9ba96\figures"
os.makedirs(brain_dir, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})

def plot_results(results_path: str = "eret_scaled_results.pt"):
    if not os.path.exists(results_path):
        print(f"File {results_path} not found yet.")
        return

    data = torch.load(results_path, map_location="cpu", weights_only=False)
    print("Loaded Modal results data successfully.")

    # ------------------------------------------------------------------
    # Figure 13: ERET Modal Scaling, Pareto Frontier & Calibration
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # Subplot 1: Optimization Dynamics
    ax1 = axes[0]
    steps = np.arange(1, len(data["step_losses"]) + 1) * 200
    ax1.plot(steps, data["step_losses"], color="#2563eb", lw=2.2, label="Total Loss (CE + 2.5 Brier)")
    ax1.set_xlabel("Training Steps (NVIDIA A10G)")
    ax1.set_ylabel("Loss Magnitude", color="#2563eb")
    ax1.tick_params(axis='y', labelcolor="#2563eb")

    ax1_twin = ax1.twinx()
    ax1_twin.plot(steps, data["step_accuracies"], color="#16a34a", lw=2.2, linestyle="--", label="Holdout Accuracy (%)")
    ax1_twin.set_ylabel("Next-Token Accuracy (%)", color="#16a34a")
    ax1_twin.tick_params(axis='y', labelcolor="#16a34a")
    ax1.set_title("A: Modal A10G Optimization Dynamics", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Pareto Frontier (Accuracy vs Relative Compute)
    ax2 = axes[1]
    eret_res = data["eret_results"]
    calm_res = data["calm_results"]
    fixed_res = data["fixed_results"]

    # ERET curve
    eret_flops = [v["relative_flops"] * 100 for v in eret_res.values()]
    eret_acc = [v["accuracy"] for v in eret_res.values()]
    ax2.plot(eret_flops, eret_acc, "o-", color="#2563eb", lw=2.5, markersize=7, label="ERET (E-ACT Noul Gate)")

    # CALM curve
    calm_flops = [v["relative_flops"] * 100 for v in calm_res.values()]
    calm_acc = [v["accuracy"] for v in calm_res.values()]
    ax2.plot(calm_flops, calm_acc, "s--", color="#dc2626", lw=2.0, markersize=6, label="CALM (Softmax Entropy Gate)")

    # Fixed compute points
    k_to_flops = {1: 1/6 * 100, 2: 2/6 * 100, 4: 4/6 * 100, 6: 100.0}
    for k, acc in fixed_res.items():
        fl = k_to_flops[k]
        ax2.scatter(fl, acc, marker="^", s=90, color="#7c3aed", zorder=5)
        ax2.annotate(f"Fixed K={k}", (fl, acc), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, fontweight="bold", color="#7c3aed")

    ax2.set_xlabel("Relative Inference Compute (% FLOPs of K=6)")
    ax2.set_ylabel("Next-Token Planning Accuracy (%)")
    ax2.set_title("B: Accuracy vs. Compute Pareto Frontier", fontweight="bold")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    # Subplot 3: ECE Calibration Comparison
    ax3 = axes[2]
    models = ["ERET (Calibrated Noul)", "CALM (Softmax Entropy)"]
    eces = [data["eret_ece"], data["calm_ece"]]
    colors = ["#16a34a", "#dc2626"]
    bars = ax3.bar(models, eces, color=colors, width=0.45, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, eces):
        y = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, y + 0.05, f"ECE = {val:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax3.set_ylabel("Expected Calibration Error (ECE %)")
    ax3.set_ylim(0, max(eces) * 1.3)
    ax3.set_title("C: Calibration Under Unsolvable Constraints", fontweight="bold")
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    p13 = os.path.join("figures", "fig13_eret_modal_scaling.png")
    plt.savefig(p13, dpi=300)
    shutil.copy(p13, os.path.join(brain_dir, "fig13_eret_modal_scaling.png"))
    plt.close()
    print(f"Saved {p13} and mirrored to brain.")

    # ------------------------------------------------------------------
    # Figure 14: Computational Biology: Topological Allocation & Contraction
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # Subplot 1: Topological Allocation (Corridors vs Junctions)
    ax1 = axes[0]
    topos = ["Corridor\n(Single Path Choice)", "Branch Junction\n(Multiple Alternative Paths)"]
    unrolls = [data["corridor_unrolls"], data["junction_unrolls"]]
    bar_colors = ["#3b82f6", "#f97316"]
    bars1 = ax1.bar(topos, unrolls, color=bar_colors, width=0.45, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars1, unrolls):
        y = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, y + 0.1, f"{val:.2f} Unrolls", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax1.axhline(6.0, color="gray", linestyle=":", label="Max Compute Bound (K=6)")
    ax1.set_ylabel("Average Allocated Inner Unrolls (k)")
    ax1.set_ylim(0, 7.0)
    ax1.set_title("A: Topological Allocation: Deliberation at Junctions", fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(axis='y', alpha=0.3)

    # Subplot 2: Banach Equilibrium Contraction Norm
    ax2 = axes[1]
    res = data["residual_decay"]
    k_steps = np.arange(1, len(res) + 1)
    ax2.plot(k_steps, res, "o-", color="#9333ea", lw=2.5, markersize=8)
    ax2.set_xlabel("Inner Krasnoselskii-Mann Unroll Step (k)")
    ax2.set_ylabel(r"Residual Step Norm $\|s_{k+1} - s_k\|_2$")
    ax2.set_title(r"B: Banach Contraction Dynamics to Latent Equilibrium", fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", alpha=0.3)

    # Subplot 3: Fixed Compute Depth Scaling
    ax3 = axes[2]
    k_vals = list(fixed_res.keys())
    k_accs = [fixed_res[k] for k in k_vals]
    ax3.plot(k_vals, k_accs, "s-", color="#059669", lw=2.5, markersize=8)
    for k, acc in zip(k_vals, k_accs):
        ax3.annotate(f"{acc:.1f}%", (k, acc), textcoords="offset points", xytext=(0, 10), ha='center', fontweight="bold", fontsize=10)
    ax3.set_xlabel("Inner Equilibrium Unroll Depth (K)")
    ax3.set_ylabel("Planning Accuracy (%)")
    ax3.set_title("C: Scaling of Accuracy with Equilibrium Depth", fontweight="bold")
    ax3.set_xticks(k_vals)
    ax3.set_ylim(min(k_accs) - 5, max(k_accs) + 8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    p14 = os.path.join("figures", "fig14_eret_computational_biology.png")
    plt.savefig(p14, dpi=300)
    shutil.copy(p14, os.path.join(brain_dir, "fig14_eret_computational_biology.png"))
    plt.close()
    print(f"Saved {p14} and mirrored to brain.")

if __name__ == "__main__":
    plot_results()
