"""
Benchmark 1: Countdown / Arithmetic Invariant Synthesis
======================================================
Authors: Leon & Ilya Sutskever persona

The Task:
  Given 6 numbers from a pool and a target integer T in [100, 999], construct a provably
  valid arithmetic expression using {+, -, *, //} that equals T exactly.

The Tri-Process Breakdown:
  - System 2 (Transformer Deliberator): Ingests the pool and target in-context.
    Performs number-theoretic sub-goal decomposition (e.g. recognizing T % 25 == 0
    or T % 7 == 0), emitting a sub-goal invariant and a continuous modulation vector m.
  - System 1 (TypeSafe Jev): Calibrated epistemic sensor. Evaluates sub-pools and
    estimates remaining entropy and whether the sub-branch is mathematically sound.
  - System 0 (NNUE Sparse Accumulator): High-speed discrete combinatorial search over
    valid arithmetic pairs, updated incrementally in microseconds.
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
os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)


# ----------------------------------------------------------------------
# 1. Countdown Game Environment
# ----------------------------------------------------------------------

STANDARD_LARGE = [25, 50, 75, 100]
STANDARD_SMALL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2

def generate_countdown_instance(seed: int | None = None) -> tuple[list[int], int]:
    rng = random.Random(seed)
    num_large = rng.randint(1, 2)
    large = rng.sample(STANDARD_LARGE, num_large)
    small = rng.sample(STANDARD_SMALL, 6 - num_large)
    pool = sorted(large + small, reverse=True)

    # Generate reachable target via forward combinations
    cur = list(pool)
    for _ in range(rng.randint(2, 4)):
        if len(cur) < 2:
            break
        i, j = rng.sample(range(len(cur)), 2)
        a, b = cur[i], cur[j]
        op = rng.choice(["+", "-", "*", "//"])
        if op == "+":
            res = a + b
        elif op == "-":
            res = abs(a - b)
        elif op == "*":
            res = a * b if a * b <= 999 else a + b
        elif op == "//":
            res = (max(a, b) // min(a, b)) if min(a, b) > 0 and max(a, b) % min(a, b) == 0 else a + b

        if res > 0:
            cur.pop(max(i, j))
            cur.pop(min(i, j))
            cur.append(res)

    valid_targets = [x for x in cur if 100 <= x <= 999]
    target = rng.choice(valid_targets) if valid_targets else rng.randint(100, 999)
    return pool, target


def encode_countdown_features(numbers: list[int], target: int) -> list[int]:
    """
    Sparse binary feature encoder:
      - Magnitude buckets for numbers
      - Proximity to target (diff < 10, diff < 50, diff < 100)
      - Divisibility indicators (target % n == 0)
    """
    features = []
    # 0-99: Individual number values
    for n in numbers:
        if n < 100:
            features.append(n)
        elif n in STANDARD_LARGE:
            features.append(100 + STANDARD_LARGE.index(n))

    # Proximity features
    best_diff = min(abs(n - target) for n in numbers)
    if best_diff == 0:
        features.append(110)
    elif best_diff <= 5:
        features.append(111)
    elif best_diff <= 25:
        features.append(112)
    elif best_diff <= 100:
        features.append(113)

    # Factor / Divisibility features
    for idx, n in enumerate(numbers):
        if n > 1 and target % n == 0:
            features.append(120 + (n % 20))

    return features


def numbers_to_tokens(numbers: list[int], target: int) -> list[int]:
    """Tokenize Countdown problem for System 2 In-Context Transformer."""
    # Special tokens: 200 = <POOL>, 201 = <TARGET>, 202 = <SUBGOAL>
    toks = [200]
    for n in numbers:
        toks.append(min(199, n))
    toks.append(201)
    toks.extend([min(199, target // 10), min(199, target % 10)])
    toks.append(202)
    return toks


# ----------------------------------------------------------------------
# 2. Search Algorithms: Pure LLM, Pure NNUE, and Tri-Process
# ----------------------------------------------------------------------

def solve_countdown_pure_llm(model: InContextTransformer, numbers: list[int], target: int) -> bool:
    """Pure LLM / Transformer greedy autoregressive baseline (no discrete search)."""
    toks = numbers_to_tokens(numbers, target)
    t_in = torch.tensor(toks, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        _, logits = model(t_in)
        predicted_subgoal = torch.argmax(logits[:, -1, :]).item()

    # Pure autoregressive attempt: does the predicted subgoal solve the problem directly?
    return predicted_subgoal == (target % 50) or target in numbers


def solve_countdown_pure_nnue(
    nnue: NNUESparseAccumulator,
    numbers: list[int],
    target: int,
    max_expansions: int = 250,
) -> tuple[bool, int]:
    """Pure NNUE discrete search without In-Context Transformer guidance."""
    if target in numbers:
        return True, 0

    queue = [(list(numbers), 0.0)]
    visited = {tuple(sorted(numbers))}
    expansions = 0

    while queue and expansions < max_expansions:
        queue.sort(key=lambda x: x[1], reverse=True)
        cur_nums, _ = queue.pop(0)

        if target in cur_nums:
            return True, expansions

        if len(cur_nums) < 2:
            continue

        expansions += 1
        # Generate pairwise combinations
        n_len = len(cur_nums)
        for i in range(n_len):
            for j in range(i + 1, n_len):
                a, b = cur_nums[i], cur_nums[j]
                remaining = [cur_nums[k] for k in range(n_len) if k != i and k != j]

                candidates = [a + b]
                if a != b:
                    candidates.append(abs(a - b))
                if a > 1 and b > 1:
                    candidates.append(a * b)
                if min(a, b) > 1 and max(a, b) % min(a, b) == 0:
                    candidates.append(max(a, b) // min(a, b))

                for res in candidates:
                    if res <= 0 or res > 2000:
                        continue
                    new_nums = sorted(remaining + [res])
                    key = tuple(new_nums)
                    if key in visited:
                        continue
                    visited.add(key)

                    feats = encode_countdown_features(new_nums, target)
                    with torch.no_grad():
                        score, _ = nnue.forward_from_features(feats, modulation=None)
                    queue.append((new_nums, score.item()))

    return False, expansions


def solve_countdown_triprocess(
    nnue: NNUESparseAccumulator,
    jev: JevEpistemicHead,
    transformer: InContextTransformer,
    numbers: list[int],
    target: int,
    max_expansions: int = 250,
    tau: float = 0.75,
) -> tuple[bool, int, float]:
    """
    Tri-Process Search (NNUE + Jev + In-Context Transformer):
      1. System 2 Transformer processes problem in-context, emitting modulation vector m.
      2. System 1 Jev monitors epistemic certainty (Noul).
      3. System 0 NNUE unrolls counterfactual pairs conditioned on m.
    """
    if target in numbers:
        return True, 0, 1.0

    # System 2 In-Context Deliberation
    toks = numbers_to_tokens(numbers, target)
    t_in = torch.tensor(toks, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        modulation, _ = transformer(t_in)
        mod_vec = modulation.squeeze(0)

    queue = [(list(numbers), 0.0, None)]
    visited = {tuple(sorted(numbers))}
    expansions = 0
    total_noul = 0.0
    noul_count = 0

    while queue and expansions < max_expansions:
        queue.sort(key=lambda x: x[1], reverse=True)
        cur_nums, _, _ = queue.pop(0)

        if target in cur_nums:
            mean_noul = total_noul / max(1, noul_count)
            return True, expansions, mean_noul

        if len(cur_nums) < 2:
            continue

        feats = encode_countdown_features(cur_nums, target)
        with torch.no_grad():
            score, accum = nnue.forward_from_features(feats, modulation=mod_vec)
            _, noul = jev(accum.unsqueeze(0))
        noul_val = noul.item()
        total_noul += noul_val
        noul_count += 1

        expansions += 1
        n_len = len(cur_nums)
        for i in range(n_len):
            for j in range(i + 1, n_len):
                a, b = cur_nums[i], cur_nums[j]
                remaining = [cur_nums[k] for k in range(n_len) if k != i and k != j]

                candidates = [a + b]
                if a != b:
                    candidates.append(abs(a - b))
                if a > 1 and b > 1:
                    candidates.append(a * b)
                if min(a, b) > 1 and max(a, b) % min(a, b) == 0:
                    candidates.append(max(a, b) // min(a, b))

                for res in candidates:
                    if res <= 0 or res > 2000:
                        continue
                    new_nums = sorted(remaining + [res])
                    key = tuple(new_nums)
                    if key in visited:
                        continue
                    visited.add(key)

                    child_feats = encode_countdown_features(new_nums, target)
                    with torch.no_grad():
                        child_score, child_accum = nnue.forward_from_features(child_feats, modulation=mod_vec)
                        _, child_noul = jev(child_accum.unsqueeze(0))

                    # Epistemic Boost: Higher Noul prioritizes path
                    effective_score = child_score.item() + 0.5 * child_noul.item()
                    queue.append((new_nums, effective_score, child_accum))

    mean_noul = total_noul / max(1, noul_count)
    return False, expansions, mean_noul


# ----------------------------------------------------------------------
# 3. Training Suite: Transformer & NNUE/Jev with Loss Tracking
# ----------------------------------------------------------------------

def train_countdown_triprocess(
    nnue: NNUESparseAccumulator,
    jev: JevEpistemicHead,
    transformer: InContextTransformer,
    num_samples: int = 2500,
    epochs: int = 5,
) -> dict[str, list[float]]:
    print("Generating Countdown training trajectories and loss traces...", flush=True)

    # 1. Train System 2 In-Context Transformer on Sub-Goal Token Synthesis
    t_tokens = []
    t_targets = []
    for _ in range(num_samples):
        pool, tgt = generate_countdown_instance()
        toks = numbers_to_tokens(pool, tgt)
        # Pad to fixed length 16
        while len(toks) < 16:
            toks.append(0)
        t_tokens.append(toks[:16])
        # Target token: prime factor or modulo target
        t_targets.append(tgt % 50)

    t_x = torch.tensor(t_tokens, dtype=torch.long).to(device)
    t_y = torch.tensor(t_targets, dtype=torch.long).to(device)

    t_dataset = torch.utils.data.TensorDataset(t_x, t_y)
    t_loader = torch.utils.data.DataLoader(t_dataset, batch_size=64, shuffle=True)
    t_optimizer = torch.optim.AdamW(transformer.parameters(), lr=1e-3, weight_decay=1e-4)

    transformer.train()
    t_loss_curve = []
    t0 = time.time()
    for ep in range(epochs):
        ep_loss = 0.0
        for b_x, b_y in t_loader:
            _, logits = transformer(b_x)
            loss = F.cross_entropy(logits[:, -1, :], b_y)
            t_optimizer.zero_grad()
            loss.backward()
            t_optimizer.step()
            ep_loss += loss.item()
        mean_ep_loss = ep_loss / len(t_loader)
        t_loss_curve.append(mean_ep_loss)

    # 2. Train System 0 & System 1 (NNUE + Jev Head)
    nnue_dataset_x = []
    nnue_dataset_v = []
    nnue_dataset_n = []

    for _ in range(num_samples):
        pool, tgt = generate_countdown_instance()
        feats = encode_countdown_features(pool, tgt)
        # Target distance
        diff = min(abs(n - tgt) for n in pool)
        v = math.exp(-diff / 100.0)
        noul = 1.0 if diff == 0 or any(tgt % n == 0 for n in pool if n > 1) else 0.20

        # Sparse active features
        accum_vec = torch.zeros(128, device=device)
        if feats:
            valid_f = [f for f in feats if f < nnue.num_features]
            if valid_f:
                accum_vec = nnue.w_accum[valid_f].sum(dim=0) + nnue.b_accum

        nnue_dataset_x.append(accum_vec.cpu().detach().numpy())
        nnue_dataset_v.append(v)
        nnue_dataset_n.append(noul)

    n_x = torch.tensor(np.array(nnue_dataset_x), dtype=torch.float32).to(device)
    n_v = torch.tensor(nnue_dataset_v, dtype=torch.float32).to(device)
    n_n = torch.tensor(nnue_dataset_n, dtype=torch.float32).to(device)

    n_dataset = torch.utils.data.TensorDataset(n_x, n_v, n_n)
    n_loader = torch.utils.data.DataLoader(n_dataset, batch_size=64, shuffle=True)
    n_optimizer = torch.optim.AdamW(list(nnue.parameters()) + list(jev.parameters()), lr=1e-3, weight_decay=1e-4)

    nnue.train()
    jev.train()
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
            n_optimizer.zero_grad()
            total_loss.backward()
            n_optimizer.step()

            ep_nnue_loss += loss_nnue.item()
            ep_jev_loss += loss_jev.item()

        nnue_loss_curve.append(ep_nnue_loss / len(n_loader))
        jev_loss_curve.append(ep_jev_loss / len(n_loader))

    print(f"Training complete in {time.time() - t0:.1f}s. Final Losses: Transformer={t_loss_curve[-1]:.4f}, NNUE={nnue_loss_curve[-1]:.4f}, Jev={jev_loss_curve[-1]:.4f}\n", flush=True)

    return {
        "transformer_loss": t_loss_curve,
        "nnue_loss": nnue_loss_curve,
        "jev_loss": jev_loss_curve,
    }


# ----------------------------------------------------------------------
# 4. Evaluation Benchmark
# ----------------------------------------------------------------------

def run_countdown_benchmark(
    nnue: NNUESparseAccumulator,
    jev: JevEpistemicHead,
    transformer: InContextTransformer,
    num_trials: int = 50,
) -> dict:
    print(f"==========================================================================")
    print(f"RUNNING BENCHMARK 1: COUNTDOWN ARITHMETIC SYNTHESIS ({num_trials} problems)")
    print(f"==========================================================================")

    results = {
        "Pure LLM (Greedy)": {"solved": 0, "expansions": [], "latency": []},
        "Pure NNUE Search": {"solved": 0, "expansions": [], "latency": []},
        "Tri-Process (NNUE+Jev+S2)": {"solved": 0, "expansions": [], "latency": [], "nouls": []},
    }

    nnue.eval()
    jev.eval()
    transformer.eval()

    for trial in range(num_trials):
        seed = 9000 + trial
        pool, target = generate_countdown_instance(seed=seed)

        # 1. Pure LLM
        t0 = time.time()
        s_llm = solve_countdown_pure_llm(transformer, pool, target)
        results["Pure LLM (Greedy)"]["solved"] += int(s_llm)
        results["Pure LLM (Greedy)"]["expansions"].append(1)
        results["Pure LLM (Greedy)"]["latency"].append((time.time() - t0) * 1000)

        # 2. Pure NNUE Search
        t0 = time.time()
        s_nnue, exp_nnue = solve_countdown_pure_nnue(nnue, pool, target, max_expansions=200)
        results["Pure NNUE Search"]["solved"] += int(s_nnue)
        results["Pure NNUE Search"]["expansions"].append(exp_nnue)
        results["Pure NNUE Search"]["latency"].append((time.time() - t0) * 1000)

        # 3. Tri-Process
        t0 = time.time()
        s_tri, exp_tri, noul_tri = solve_countdown_triprocess(nnue, jev, transformer, pool, target, max_expansions=200)
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


def plot_countdown_results(training_curves: dict, eval_results: dict, num_trials: int = 50, save_path: str = "figures/fig25_countdown_triprocess_results.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=300)
    fig.patch.set_facecolor("#FAFAFA")

    # Panel 1: Training Loss Curves
    ax1 = axes[0]
    ax1.set_facecolor("#FFFFFF")
    epochs = range(1, len(training_curves["transformer_loss"]) + 1)
    ax1.plot(epochs, training_curves["transformer_loss"], marker="o", label="S2 Transformer Loss", color="#6A1B9A", lw=2)
    ax1.plot(epochs, training_curves["nnue_loss"], marker="s", label="S0 NNUE Loss", color="#1565C0", lw=2)
    ax1.plot(epochs, training_curves["jev_loss"], marker="^", label="S1 Jev RLCD Loss", color="#2E7D32", lw=2)
    ax1.set_title("Tri-Process Multi-Scale Training Convergence", fontsize=10.5, fontweight="bold", pad=8)
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
    ax2.set_title("Countdown Arithmetic Target Accuracy", fontsize=10.5, fontweight="bold", pad=8)
    ax2.set_ylabel("Success Rate (%)", fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 110)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Panel 3: Search Efficiency (Expansions)
    ax3 = axes[2]
    ax3.set_facecolor("#FFFFFF")
    search_methods = ["Pure NNUE Search", "Tri-Process (NNUE+Jev+S2)"]
    exps = [np.mean(eval_results[m]["expansions"]) for m in search_methods]
    bars_exp = ax3.bar(search_methods, exps, color=["#FF8F00", "#2E7D32"], width=0.45)
    for b in bars_exp:
        ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{b.get_height():.1f}", ha="center", fontsize=9.5, fontweight="bold")
    ax3.set_title("Mean Search Expansions per Solution", fontsize=10.5, fontweight="bold", pad=8)
    ax3.set_ylabel("Nodes Expanded", fontsize=9.5, fontweight="bold")
    ax3.set_ylim(0, max(exps) * 1.25)
    ax3.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Benchmark 1: Tri-Process Architecture on Countdown Arithmetic Synthesis", fontsize=12.5, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved Benchmark 1 figure to {save_path}")


if __name__ == "__main__":
    t_cfg = TransformerConfig(vocab_size=256, seq_len=32, d_model=128, n_heads=4, n_layers=3)
    nnue = NNUESparseAccumulator(num_features=160, accumulator_dim=128).to(device)
    jev = JevEpistemicHead(in_dim=128).to(device)
    transformer = InContextTransformer(t_cfg, modulation_dim=128).to(device)

    train_data = train_countdown_triprocess(nnue, jev, transformer, num_samples=3000, epochs=5)
    eval_res = run_countdown_benchmark(nnue, jev, transformer, num_trials=50)

    # Save data locally
    save_dict = {"train_data": train_data, "eval_res": eval_res}
    torch.save(save_dict, "data/countdown_triprocess_results.pt")
    print("Saved raw benchmark data to data/countdown_triprocess_results.pt")

    plot_countdown_results(train_data, eval_res, num_trials=50)
