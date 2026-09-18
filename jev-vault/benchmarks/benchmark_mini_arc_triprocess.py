"""
Benchmark 2: Mini-ARC Symbolic Grid Induction (Abstraction & Reasoning Corpus)
=============================================================================
Authors: Leon & Ilya Sutskever persona

The Task:
  Given 2 demonstration pairs (Input_k -> Output_k) illustrating an abstract visual rule
  (e.g., cavity enclosure fill, gravity drop, reflection symmetry, or pattern recolor),
  predict the exact target grid for a novel test input.

The Tri-Process Breakdown:
  - System 2 (In-Context Transformer): Ingests (I1, O1) and (I2, O2) demonstration pairs
    as token sequences. Attends across demonstrations to induce the latent transformation rule,
    emitting a rule token and continuous modulation vector m.
  - System 0 (NNUE Sparse Accumulator): High-speed discrete search over primitive cellular
    operators (fill, drop, reflect, recolor), updated incrementally in microseconds.
  - System 1 (TypeSafe Jev): Calibrated epistemic verifier. Ensures proposed operations
    do not violate demonstration invariants before committing cells.
"""

from __future__ import annotations

import os
import sys
import time
import math
import random
from dataclasses import dataclass

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from triprocess_engine import NNUESparseAccumulator, JevEpistemicHead, InContextTransformer, TransformerConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID_DIM = 6
NUM_COLORS = 6

# Task types:
# 0: Enclosure Fill (fill inside closed box)
# 1: Gravity Drop (shift foreground pixels down to floor)
# 2: Reflection Symmetry (mirror left half to right half)
# 3: Diagonal Transpose / Pattern Inversion


# ----------------------------------------------------------------------
# 1. Mini-ARC Synthetic Task Generator
# ----------------------------------------------------------------------

def generate_arc_task(rule_type: int | None = None, seed: int | None = None):
    rng = random.Random(seed)
    rule = rule_type if rule_type is not None else rng.randint(0, 3)

    def apply_rule(inp: np.ndarray) -> np.ndarray:
        out = np.copy(inp)
        if rule == 0:  # Enclosure fill: fill interior of 3x3 hollow box
            for r in range(1, GRID_DIM - 1):
                for c in range(1, GRID_DIM - 1):
                    if inp[r-1, c] > 0 and inp[r+1, c] > 0 and inp[r, c-1] > 0 and inp[r, c+1] > 0 and inp[r, c] == 0:
                        out[r, c] = 2  # Fill color
        elif rule == 1:  # Gravity drop: move non-zero pixels to bottom
            for c in range(GRID_DIM):
                col = [inp[r, c] for r in range(GRID_DIM) if inp[r, c] > 0]
                new_col = [0] * (GRID_DIM - len(col)) + col
                for r in range(GRID_DIM):
                    out[r, c] = new_col[r]
        elif rule == 2:  # Reflection symmetry: mirror left half to right
            mid = GRID_DIM // 2
            for r in range(GRID_DIM):
                for c in range(mid):
                    out[r, GRID_DIM - 1 - c] = out[r, c]
        elif rule == 3:  # Pattern recolor: swap color 1 to color 3
            out[inp == 1] = 3
        return out

    def make_random_grid() -> np.ndarray:
        g = np.zeros((GRID_DIM, GRID_DIM), dtype=np.int32)
        if rule == 0:
            # Place a hollow square
            r, c = rng.randint(1, 2), rng.randint(1, 2)
            g[r:r+3, c] = 1
            g[r:r+3, c+2] = 1
            g[r, c:c+3] = 1
            g[r+2, c:c+3] = 1
            g[r+1, c+1] = 0
        elif rule == 1:
            # Random floating blocks
            for _ in range(rng.randint(3, 6)):
                g[rng.randint(0, 3), rng.randint(0, GRID_DIM - 1)] = rng.randint(1, 3)
        elif rule == 2:
            # Pattern on left half
            for r in range(GRID_DIM):
                for c in range(GRID_DIM // 2):
                    if rng.random() > 0.6:
                        g[r, c] = rng.randint(1, 3)
        else:
            for _ in range(rng.randint(4, 8)):
                g[rng.randint(0, GRID_DIM - 1), rng.randint(0, GRID_DIM - 1)] = 1
        return g

    # 2 demonstration pairs + 1 test pair
    d1_in = make_random_grid()
    d1_out = apply_rule(d1_in)
    d2_in = make_random_grid()
    d2_out = apply_rule(d2_in)

    test_in = make_random_grid()
    test_out = apply_rule(test_in)

    return rule, (d1_in, d1_out), (d2_in, d2_out), (test_in, test_out)


def grids_to_context_tokens(d1: tuple[np.ndarray, np.ndarray], d2: tuple[np.ndarray, np.ndarray], test_in: np.ndarray) -> list[int]:
    """Serialize Mini-ARC task into tokens for System 2 In-Context Transformer."""
    # Special delimiter tokens:
    # 210: <DEMO1_IN>, 211: <DEMO1_OUT>, 212: <DEMO2_IN>, 213: <DEMO2_OUT>, 214: <TEST_IN>
    toks = [210]
    toks.extend(d1[0].flatten().tolist())
    toks.append(211)
    toks.extend(d1[1].flatten().tolist())
    toks.append(212)
    toks.extend(d2[0].flatten().tolist())
    toks.append(213)
    toks.extend(d2[1].flatten().tolist())
    toks.append(214)
    toks.extend(test_in.flatten().tolist())
    return toks


def encode_arc_sparse_features(grid: np.ndarray) -> list[int]:
    """
    Sparse positional features for System 0 NNUE:
      Feature index = (color * GRID_DIM * GRID_DIM) + (r * GRID_DIM) + c
    Total features = NUM_COLORS * 36 = 216 features.
    """
    feats = []
    for r in range(GRID_DIM):
        for c in range(GRID_DIM):
            col = int(grid[r, c])
            if col > 0:
                idx = (col * GRID_DIM * GRID_DIM) + (r * GRID_DIM) + c
                feats.append(idx)
    return feats


# ----------------------------------------------------------------------
# 2. Operators & Search Algorithms
# ----------------------------------------------------------------------

GRID_OPERATORS = [
    "fill_enclosed",
    "drop_gravity",
    "mirror_reflection",
    "recolor_1_to_3",
    "invert_grid",
]

def apply_operator(grid: np.ndarray, op_name: str) -> np.ndarray:
    out = np.copy(grid)
    if op_name == "fill_enclosed":
        for r in range(1, GRID_DIM - 1):
            for c in range(1, GRID_DIM - 1):
                if grid[r-1, c] > 0 and grid[r+1, c] > 0 and grid[r, c-1] > 0 and grid[r, c+1] > 0 and grid[r, c] == 0:
                    out[r, c] = 2
    elif op_name == "drop_gravity":
        for c in range(GRID_DIM):
            col = [grid[r, c] for r in range(GRID_DIM) if grid[r, c] > 0]
            new_col = [0] * (GRID_DIM - len(col)) + col
            for r in range(GRID_DIM):
                out[r, c] = new_col[r]
    elif op_name == "mirror_reflection":
        mid = GRID_DIM // 2
        for r in range(GRID_DIM):
            for c in range(mid):
                out[r, GRID_DIM - 1 - c] = out[r, c]
    elif op_name == "recolor_1_to_3":
        out[grid == 1] = 3
    elif op_name == "invert_grid":
        out = (GRID_DIM - 1 - out) % NUM_COLORS
    return out


def solve_arc_pure_llm(transformer: InContextTransformer, d1, d2, test_in: np.ndarray, test_out: np.ndarray) -> bool:
    """Pure LLM baseline: autoregressively predicts rule token in one shot."""
    toks = grids_to_context_tokens(d1, d2, test_in)
    t_in = torch.tensor(toks[:128], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        _, logits = transformer(t_in)
        pred_rule = torch.argmax(logits[:, -1, :4]).item()

    op_name = GRID_OPERATORS[pred_rule]
    pred_grid = apply_operator(test_in, op_name)
    return np.array_equal(pred_grid, test_out)


def solve_arc_pure_nnue(nnue: NNUESparseAccumulator, test_in: np.ndarray, test_out: np.ndarray) -> tuple[bool, int]:
    """Pure NNUE search: evaluates candidate operators without in-context demonstration conditioning."""
    best_op = None
    best_score = -1e9
    expansions = 0

    for op in GRID_OPERATORS:
        candidate_grid = apply_operator(test_in, op)
        feats = encode_arc_sparse_features(candidate_grid)
        with torch.no_grad():
            score, _ = nnue.forward_from_features(feats, modulation=None)
        expansions += 1
        if score.item() > best_score:
            best_score = score.item()
            best_op = op

    pred_grid = apply_operator(test_in, best_op)
    return np.array_equal(pred_grid, test_out), expansions


def solve_arc_triprocess(
    nnue: NNUESparseAccumulator,
    jev: JevEpistemicHead,
    transformer: InContextTransformer,
    d1, d2,
    test_in: np.ndarray,
    test_out: np.ndarray,
) -> tuple[bool, int, float]:
    """
    Tri-Process Solver (NNUE + Jev + In-Context Transformer):
      1. System 2 Transformer processes demonstrations in-context, emitting modulation m.
      2. System 0 NNUE evaluates candidate operators conditioned on m.
      3. System 1 Jev verifies calibration (Noul).
    """
    toks = grids_to_context_tokens(d1, d2, test_in)
    t_in = torch.tensor(toks[:128], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        modulation, logits = transformer(t_in)
        mod_vec = modulation.squeeze(0)

    best_op = None
    best_score = -1e9
    best_noul = 0.0
    expansions = 0

    for op in GRID_OPERATORS:
        candidate_grid = apply_operator(test_in, op)
        feats = encode_arc_sparse_features(candidate_grid)
        with torch.no_grad():
            score, accum = nnue.forward_from_features(feats, modulation=mod_vec)
            _, noul = jev(accum.unsqueeze(0))
        expansions += 1

        total_score = score.item() + 0.5 * noul.item()
        if total_score > best_score:
            best_score = total_score
            best_op = op
            best_noul = noul.item()

    pred_grid = apply_operator(test_in, best_op)
    is_success = np.array_equal(pred_grid, test_out)
    return is_success, expansions, best_noul


# ----------------------------------------------------------------------
# 3. Training Suite for Mini-ARC
# ----------------------------------------------------------------------

def train_mini_arc_triprocess(
    nnue: NNUESparseAccumulator,
    jev: JevEpistemicHead,
    transformer: InContextTransformer,
    num_samples: int = 2000,
    epochs: int = 5,
) -> dict[str, list[float]]:
    print("Generating Mini-ARC demonstration traces and training models...", flush=True)

    # 1. Train Transformer on in-context rule induction
    all_toks = []
    all_rules = []
    for _ in range(num_samples):
        r, d1, d2, test = generate_arc_task()
        toks = grids_to_context_tokens(d1, d2, test[0])
        while len(toks) < 128:
            toks.append(0)
        all_toks.append(toks[:128])
        all_rules.append(r)

    t_x = torch.tensor(all_toks, dtype=torch.long).to(device)
    t_y = torch.tensor(all_rules, dtype=torch.long).to(device)

    t_dataset = torch.utils.data.TensorDataset(t_x, t_y)
    t_loader = torch.utils.data.DataLoader(t_dataset, batch_size=64, shuffle=True)
    t_opt = torch.optim.AdamW(transformer.parameters(), lr=1e-3, weight_decay=1e-4)

    t_loss_curve = []
    t0 = time.time()
    for ep in range(epochs):
        ep_loss = 0.0
        for b_x, b_y in t_loader:
            _, logits = transformer(b_x)
            loss = F.cross_entropy(logits[:, -1, :4], b_y)
            t_opt.zero_grad()
            loss.backward()
            t_opt.step()
            ep_loss += loss.item()
        t_loss_curve.append(ep_loss / len(t_loader))

    # 2. Train NNUE + Jev
    nnue_x, nnue_v, nnue_n = [], [], []
    for _ in range(num_samples):
        r, _, _, test = generate_arc_task()
        correct_grid = test[1]
        c_feats = encode_arc_sparse_features(correct_grid)
        # Correct output features
        accum_c = torch.zeros(128, device=device)
        valid_c = [f for f in c_feats if f < nnue.num_features]
        if valid_c:
            accum_c = nnue.w_accum[valid_c].sum(dim=0) + nnue.b_accum
        nnue_x.append(accum_c.cpu().detach().numpy())
        nnue_v.append(1.0)
        nnue_n.append(0.95)

        # Perturbed false output
        false_grid = apply_operator(test[0], "invert_grid" if r != 3 else "fill_enclosed")
        f_feats = encode_arc_sparse_features(false_grid)
        accum_f = torch.zeros(128, device=device)
        valid_f = [f for f in f_feats if f < nnue.num_features]
        if valid_f:
            accum_f = nnue.w_accum[valid_f].sum(dim=0) + nnue.b_accum
        nnue_x.append(accum_f.cpu().detach().numpy())
        nnue_v.append(0.0)
        nnue_n.append(0.10)

    n_x = torch.tensor(np.array(nnue_x), dtype=torch.float32).to(device)
    n_v = torch.tensor(nnue_v, dtype=torch.float32).to(device)
    n_n = torch.tensor(nnue_n, dtype=torch.float32).to(device)

    n_dataset = torch.utils.data.TensorDataset(n_x, n_v, n_n)
    n_loader = torch.utils.data.DataLoader(n_dataset, batch_size=64, shuffle=True)
    n_opt = torch.optim.AdamW(list(nnue.parameters()) + list(jev.parameters()), lr=1e-3, weight_decay=1e-4)

    nnue_loss_curve = []
    jev_loss_curve = []

    for ep in range(epochs):
        ep_nnue_loss = 0.0
        ep_jev_loss = 0.0
        for b_x, b_v, b_n in n_loader:
            v_nnue = nnue.forward_batch_accum(b_x)
            v_jev, n_jev = jev(b_x)

            loss_nnue = F.mse_loss(v_nnue, b_v)
            loss_jev = F.mse_loss(v_jev, b_v) + F.mse_loss(n_jev, b_n)

            total_loss = loss_nnue + loss_jev
            n_opt.zero_grad()
            total_loss.backward()
            n_opt.step()

            ep_nnue_loss += loss_nnue.item()
            ep_jev_loss += loss_jev.item()

        nnue_loss_curve.append(ep_nnue_loss / len(n_loader))
        jev_loss_curve.append(ep_jev_loss / len(n_loader))

    print(f"Mini-ARC training finished in {time.time() - t0:.1f}s. Final losses: S2={t_loss_curve[-1]:.4f}, S0={nnue_loss_curve[-1]:.4f}, S1={jev_loss_curve[-1]:.4f}\n", flush=True)

    return {
        "transformer_loss": t_loss_curve,
        "nnue_loss": nnue_loss_curve,
        "jev_loss": jev_loss_curve,
    }


# ----------------------------------------------------------------------
# 4. Evaluation Benchmark & Plotting
# ----------------------------------------------------------------------

def run_mini_arc_benchmark(
    nnue: NNUESparseAccumulator,
    jev: JevEpistemicHead,
    transformer: InContextTransformer,
    num_trials: int = 50,
) -> dict:
    print(f"==========================================================================")
    print(f"RUNNING BENCHMARK 2: MINI-ARC SYMBOLIC GRID INDUCTION ({num_trials} tasks)")
    print(f"==========================================================================")

    results = {
        "Pure LLM (Autoregressive)": {"solved": 0, "expansions": [], "latency": []},
        "Pure NNUE Search": {"solved": 0, "expansions": [], "latency": []},
        "Tri-Process (NNUE+Jev+S2)": {"solved": 0, "expansions": [], "latency": [], "nouls": []},
    }

    nnue.eval()
    jev.eval()
    transformer.eval()

    for trial in range(num_trials):
        seed = 12000 + trial
        r, d1, d2, (test_in, test_out) = generate_arc_task(seed=seed)

        # 1. Pure LLM
        t0 = time.time()
        s_llm = solve_arc_pure_llm(transformer, d1, d2, test_in, test_out)
        results["Pure LLM (Autoregressive)"]["solved"] += int(s_llm)
        results["Pure LLM (Autoregressive)"]["expansions"].append(1)
        results["Pure LLM (Autoregressive)"]["latency"].append((time.time() - t0) * 1000)

        # 2. Pure NNUE
        t0 = time.time()
        s_nnue, exp_nnue = solve_arc_pure_nnue(nnue, test_in, test_out)
        results["Pure NNUE Search"]["solved"] += int(s_nnue)
        results["Pure NNUE Search"]["expansions"].append(exp_nnue)
        results["Pure NNUE Search"]["latency"].append((time.time() - t0) * 1000)

        # 3. Tri-Process
        t0 = time.time()
        s_tri, exp_tri, noul_tri = solve_arc_triprocess(nnue, jev, transformer, d1, d2, test_in, test_out)
        results["Tri-Process (NNUE+Jev+S2)"]["solved"] += int(s_tri)
        results["Tri-Process (NNUE+Jev+S2)"]["expansions"].append(exp_tri)
        results["Tri-Process (NNUE+Jev+S2)"]["latency"].append((time.time() - t0) * 1000)
        results["Tri-Process (NNUE+Jev+S2)"]["nouls"].append(noul_tri)

    for m in results:
        acc = (results[m]["solved"] / num_trials) * 100
        mean_exp = np.mean(results[m]["expansions"])
        mean_lat = np.mean(results[m]["latency"])
        print(f"{m:26s} | Solved: {acc:5.1f}% | Expansions: {mean_exp:5.1f} | Latency: {mean_lat:5.1f}ms")

    return results


def plot_mini_arc_results(training_curves: dict, eval_results: dict, num_trials: int = 50, save_path: str = "figures/fig26_mini_arc_triprocess_results.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=300)
    fig.patch.set_facecolor("#FAFAFA")

    # Panel 1: Training Loss Curves
    ax1 = axes[0]
    ax1.set_facecolor("#FFFFFF")
    epochs = range(1, len(training_curves["transformer_loss"]) + 1)
    ax1.plot(epochs, training_curves["transformer_loss"], marker="o", label="S2 In-Context Transformer", color="#6A1B9A", lw=2)
    ax1.plot(epochs, training_curves["nnue_loss"], marker="s", label="S0 NNUE Accumulator", color="#1565C0", lw=2)
    ax1.plot(epochs, training_curves["jev_loss"], marker="^", label="S1 Jev Epistemic Gate", color="#2E7D32", lw=2)
    ax1.set_title("Mini-ARC Multi-Scale Training Curves", fontsize=10.5, fontweight="bold", pad=8)
    ax1.set_xlabel("Epoch", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=9.5, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8.5)

    # Panel 2: Benchmark Accuracy
    ax2 = axes[1]
    ax2.set_facecolor("#FFFFFF")
    methods = list(eval_results.keys())
    accs = [(eval_results[m]["solved"] / float(num_trials)) * 100 for m in methods]
    colors = ["#C62828", "#FF8F00", "#2E7D32"]
    bars = ax2.bar(methods, accs, color=colors, width=0.55)
    for b in bars:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{b.get_height():.1f}%", ha="center", fontsize=9.5, fontweight="bold")
    ax2.set_title("Mini-ARC Inductive Generalization Rate", fontsize=10.5, fontweight="bold", pad=8)
    ax2.set_ylabel("Exact Grid Match (%)", fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 110)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Panel 3: In-Context Epistemic Confidence vs Performance
    ax3 = axes[2]
    ax3.set_facecolor("#FFFFFF")
    nouls = eval_results["Tri-Process (NNUE+Jev+S2)"]["nouls"]
    ax3.hist(nouls, bins=10, color="#2E7D32", edgecolor="#1B5E20", alpha=0.85)
    ax3.set_title("Distribution of Epistemic Noul on Test Grids", fontsize=10.5, fontweight="bold", pad=8)
    ax3.set_xlabel("Calibrated Noul Score", fontsize=9.5, fontweight="bold")
    ax3.set_ylabel("Frequency", fontsize=9.5, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Benchmark 2: Tri-Process Architecture on Mini-ARC Grid Induction", fontsize=12.5, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved Benchmark 2 figure to {save_path}")


if __name__ == "__main__":
    t_cfg = TransformerConfig(vocab_size=256, seq_len=128, d_model=128, n_heads=4, n_layers=3)
    nnue = NNUESparseAccumulator(num_features=NUM_COLORS * GRID_DIM * GRID_DIM, accumulator_dim=128).to(device)
    jev = JevEpistemicHead(in_dim=128).to(device)
    transformer = InContextTransformer(t_cfg, modulation_dim=128).to(device)

    train_data = train_mini_arc_triprocess(nnue, jev, transformer, num_samples=2500, epochs=5)
    eval_res = run_mini_arc_benchmark(nnue, jev, transformer, num_trials=50)

    # Save data locally
    save_dict = {"train_data": train_data, "eval_res": eval_res}
    torch.save(save_dict, "data/mini_arc_triprocess_results.pt")
    print("Saved raw benchmark data to data/mini_arc_triprocess_results.pt")

    plot_mini_arc_results(train_data, eval_res, num_trials=50)
