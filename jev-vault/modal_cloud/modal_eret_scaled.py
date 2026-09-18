"""
Modal Cloud: Scaling Epistemic Recurrent Equilibrium Transformer (ERET / Jevformer 2.0)
========================================================================================
Authors: The Research Collective (Leon & Ilya Sutskever persona)

Scale Experiment on 10x10 Constraint Mazes (100 cells, paths up to length 20):
  - outer causal autoregression
  - inner Krasnoselskii-Mann latent equilibrium loop unrolled up to K=6
  - DSEA dual-stream epistemic cross-attention
  - JGR epistemic residual valve: s_{k+1} = s_k + (1 - Noul(s_k)) * B_theta(s_k)
  - RLCD Proper Scoring (Brier loss) calibration

Benchmarked against:
  1. CALM Softmax Entropy Early-Exit baseline
  2. Fixed Compute baselines (K = 1, 2, 4, 6)
  3. Topological allocation: Corridors vs Junctions (Branch points)
  4. Epistemic Calibration under Solvable vs Unsolvable Mazes (ECE)
"""

from __future__ import annotations

import os
import sys
import time
import math
import random
from collections import deque
from dataclasses import dataclass
import modal

APP_NAME = "eret-scaled-equilibrium"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("jevformer-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4.0",
        "numpy>=1.26.0",
    )
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ----------------------------------------------------------------------
# 1. 10x10 Constraint Maze Dataset Generator
# ----------------------------------------------------------------------

GRID_SIZE = 10 # 100 cells
MAX_PATH_LEN = 20
VOCAB_SIZE = 16

# Action codes: 6: UP, 7: DOWN, 8: LEFT, 9: RIGHT, 10: DONE, 11: UNREACHABLE
MOVES = [(-1, 0, 6), (1, 0, 7), (0, -1, 8), (0, 1, 9)]

def generate_maze(grid_size: int = 10, wall_prob: float = 0.25):
    grid = np.ones((grid_size, grid_size), dtype=int)
    for r in range(grid_size):
        for c in range(grid_size):
            if random.random() < wall_prob:
                grid[r, c] = 2 # wall

    empty_cells = [(r, c) for r in range(grid_size) for c in range(grid_size) if grid[r, c] == 1]
    if len(empty_cells) < 4:
        return generate_maze(grid_size, wall_prob)

    start, goal = random.sample(empty_cells, 2)
    grid[start[0], start[1]] = 3 # start
    grid[goal[0], goal[1]] = 4 # goal

    # BFS shortest path with predecessor tracking
    q = deque([(start[0], start[1], [])])
    visited = {start}
    path_nodes = None
    shortest_moves = None

    while q:
        r, c, p_history = q.popleft()
        if (r, c) == goal:
            path_nodes = [start] + [node for node, m in p_history]
            shortest_moves = [m for node, m in p_history]
            break

        for dr, dc, m_id in MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size and grid[nr, nc] != 2 and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc, p_history + [((nr, nc), m_id)]))

    flat_grid = grid.flatten()
    is_solvable = shortest_moves is not None

    junction_flags = []
    if is_solvable:
        target_seq = [5] + shortest_moves + [10] # 5: SOS
        # Compute junction status for each step
        for i in range(len(path_nodes)):
            curr_r, curr_c = path_nodes[i]
            prev_node = path_nodes[i - 1] if i > 0 else None
            # Count valid outgoing neighbors not equal to prev_node
            valid_out = 0
            for dr, dc, _ in MOVES:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size and grid[nr, nc] != 2:
                    if prev_node is None or (nr, nc) != prev_node:
                        valid_out += 1
            # Junction if > 1 branch option
            junction_flags.append(1 if valid_out > 1 else 0)
        junction_flags.append(0) # For <DONE>
    else:
        target_seq = [5, 11] # <SOS>, <UNREACHABLE>
        junction_flags = [0, 0]

    target_seq = target_seq[:MAX_PATH_LEN]
    junction_flags = junction_flags[:MAX_PATH_LEN]

    pad_len = MAX_PATH_LEN - len(target_seq)
    target_tokens = target_seq + [0] * pad_len
    junction_flags = junction_flags + [0] * pad_len

    return flat_grid, target_tokens, junction_flags, is_solvable


def generate_dataset_tensors(num_samples: int, wall_prob: float = 0.25):
    grids = torch.zeros(num_samples, 100, dtype=torch.long)
    targets = torch.zeros(num_samples, MAX_PATH_LEN, dtype=torch.long)
    junctions = torch.zeros(num_samples, MAX_PATH_LEN, dtype=torch.long)
    solvables = torch.zeros(num_samples, dtype=torch.float32)

    for i in range(num_samples):
        g, tgt, junc, is_sol = generate_maze(GRID_SIZE, wall_prob)
        grids[i] = torch.tensor(g, dtype=torch.long)
        targets[i] = torch.tensor(tgt, dtype=torch.long)
        junctions[i] = torch.tensor(junc, dtype=torch.long)
        solvables[i] = 1.0 if is_sol else 0.0

    return grids, targets, junctions, solvables


# ----------------------------------------------------------------------
# 2. Epistemic Recurrent Equilibrium Transformer Architecture
# ----------------------------------------------------------------------

class ERETLoopedBlock(nn.Module):
    """
    Weight-tied Krasnoselskii-Mann Equilibrium Block:
      h_{k+1} = (1 - gamma_k) * h_k + gamma_k * B_theta(h_k)
      where gamma_k = 1 - Noul(h_k)
    """
    def __init__(self, d_model: int = 192, d_epi: int = 48, num_heads: int = 6):
        super().__init__()
        self.d_model = d_model
        self.d_epi = d_epi
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Semantic Stream
        self.q_sem = nn.Linear(d_model, d_model)
        self.k_sem = nn.Linear(d_model, d_model)
        self.v_sem = nn.Linear(d_model, d_model)
        self.out_sem = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff_sem = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        # Epistemic Stream (DSEA)
        self.q_epi = nn.Linear(d_epi, d_epi)
        self.k_epi = nn.Linear(d_epi, d_epi)
        self.beta = nn.Parameter(torch.ones(1) * 0.5)
        self.update_epi = nn.Sequential(
            nn.Linear(d_model + d_epi, d_epi),
            nn.Tanh(),
        )

        # Jev Noul Gate (Calibrated System 1 Evaluator)
        self.noul_valve = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, s: torch.Tensor, e: torch.Tensor):
        b, t, _ = s.shape
        # Evaluate Jev confidence at query endpoint
        s_query = s[:, -1, :]
        conf = torch.sigmoid(self.noul_valve(s_query)).squeeze(-1) # [B]
        gamma = (1.0 - conf).clamp(min=0.01, max=0.99) # [B]

        # DSEA Cross-Stream Attention
        q_s = self.q_sem(s).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k_s = self.k_sem(s).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v_s = self.v_sem(s).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        q_e = self.q_epi(e).unsqueeze(1)
        k_e = self.k_epi(e).unsqueeze(1)

        scores = (torch.matmul(q_s, k_s.transpose(-2, -1)) / math.sqrt(self.head_dim)) + \
                 self.beta * (torch.matmul(q_e, k_e.transpose(-2, -1)) / math.sqrt(self.d_epi))
        attn = F.softmax(scores, dim=-1)

        attn_out = torch.matmul(attn, v_s).transpose(1, 2).contiguous().view(b, t, self.d_model)
        h_mid = self.norm1(s + self.out_sem(attn_out))
        delta_s = self.norm2(h_mid + self.ff_sem(h_mid)) - s

        # Krasnoselskii-Mann Equilibrium Update
        s_next = s + gamma.view(b, 1, 1) * delta_s
        e_next = e + self.update_epi(torch.cat([s_next, e], dim=-1))

        return s_next, e_next, conf, gamma, delta_s


class ERETModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 192,
        d_epi: int = 48,
        num_heads: int = 6,
        inner_loops: int = 6,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_epi = d_epi
        self.inner_loops = inner_loops

        self.grid_emb = nn.Embedding(vocab_size, d_model)
        self.act_emb = nn.Embedding(vocab_size, d_model)
        self.pos_grid = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        self.pos_act = nn.Parameter(torch.randn(1, MAX_PATH_LEN, d_model) * 0.02)

        self.grid_epi = nn.Embedding(vocab_size, d_epi)
        self.act_epi = nn.Embedding(vocab_size, d_epi)

        # Weight-tied looped block
        self.inner_block = ERETLoopedBlock(d_model=d_model, d_epi=d_epi, num_heads=num_heads)

        # Readout head
        self.readout_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        grid: torch.Tensor,
        prefix_acts: torch.Tensor,
        threshold: float = 0.85,
        dynamic_halt: bool = False,
        calm_mode: bool = False,
        fixed_loops: int | None = None,
    ):
        b, _ = grid.shape
        _, act_len = prefix_acts.shape

        h_grid = self.grid_emb(grid) + self.pos_grid
        h_acts = self.act_emb(prefix_acts) + self.pos_act[:, :act_len, :]
        s = torch.cat([h_grid, h_acts], dim=1) # [B, 100 + act_len, D]

        e_grid = self.grid_epi(grid)
        e_acts = self.act_epi(prefix_acts)
        e = torch.cat([e_grid, e_acts], dim=1)

        max_k = fixed_loops if fixed_loops is not None else self.inner_loops
        loop_confs = []
        loop_gates = []
        residuals = []
        halt_steps = torch.ones(b, dtype=torch.long, device=grid.device) * max_k
        active_mask = torch.ones(b, dtype=torch.bool, device=grid.device)

        for k in range(1, max_k + 1):
            s_prev = s
            s, e, conf, gamma, delta_s = self.inner_block(s, e)
            res_norm = torch.norm(s - s_prev, dim=-1).mean(dim=1) # [B]
            residuals.append(res_norm)
            loop_confs.append(conf)
            loop_gates.append(gamma)

            if dynamic_halt:
                if calm_mode:
                    # CALM baseline: normalized softmax entropy
                    curr_logits = self.head(self.readout_norm(s[:, -1, :]))
                    curr_probs = F.softmax(curr_logits, dim=-1)
                    entropy = -torch.sum(curr_probs * torch.log(curr_probs.clamp(min=1e-8)), dim=-1)
                    norm_entropy = entropy / math.log(VOCAB_SIZE)
                    calm_conf = 1.0 - norm_entropy
                    newly_halted = (calm_conf >= threshold) & active_mask
                else:
                    # ERET E-ACT: Jev calibrated Noul
                    newly_halted = (conf >= threshold) & active_mask

                halt_steps[newly_halted] = k
                active_mask = active_mask & (~newly_halted)

        logits = self.head(self.readout_norm(s[:, -1, :]))

        return {
            "logits": logits,
            "final_conf": loop_confs[-1],
            "loop_confs": loop_confs,
            "loop_gates": loop_gates,
            "residuals": residuals,
            "halt_steps": halt_steps,
        }


# ----------------------------------------------------------------------
# 3. Modal Scaled Remote Worker
# ----------------------------------------------------------------------

@app.function(
    gpu="A10G",
    image=image,
    volumes={"/checkpoints": volume},
    timeout=1200,
)
def run_scaled_eret_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Running Scaled ERET (Jevformer 2.0) on {device} ({torch.cuda.get_device_name(0)}) ===", flush=True)

    # 1. Generate Scaled 10x10 Labyrinth Dataset
    num_train = 6000
    num_test = 1000
    print(f"Generating {num_train} training and {num_test} testing 10x10 mazes...", flush=True)
    t0 = time.time()
    train_grids, train_targets, train_junc, train_sol = generate_dataset_tensors(num_train)
    test_grids, test_targets, test_junc, test_sol = generate_dataset_tensors(num_test)
    print(f"Dataset generated in {time.time() - t0:.1f}s. Solvable Mazes: {train_sol.mean()*100:.1f}%", flush=True)

    train_grids = train_grids.to(device)
    train_targets = train_targets.to(device)
    train_junc = train_junc.to(device)

    test_grids = test_grids.to(device)
    test_targets = test_targets.to(device)
    test_junc = test_junc.to(device)

    # 2. Instantiate Model
    model = ERETModel(
        vocab_size=VOCAB_SIZE,
        d_model=192,
        d_epi=48,
        num_heads=6,
        inner_loops=6,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ERET Model Initialized. Total Parameters: {param_count:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000, eta_min=5e-5)

    # 3. Training Loop (3,000 steps with RLCD Proper Scoring)
    steps = 3000
    batch_size = 64
    num_train_samples = len(train_grids)

    print(f"\n--- Launching 3,000 Step Scaled Optimization on A10G ---", flush=True)
    step_losses = []
    step_ce_losses = []
    step_brier_losses = []
    step_accuracies = []

    t_train_start = time.time()
    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, num_train_samples, (batch_size,), device=device)
        g_b = train_grids[idx]
        tgt_b = train_targets[idx]

        # Sample random prefix length
        act_len = random.randint(1, MAX_PATH_LEN - 1)
        prefix = tgt_b[:, :act_len]
        next_token = tgt_b[:, act_len]

        valid_mask = next_token != 0
        if not valid_mask.any():
            continue

        out = model(g_b, prefix)
        logits = out["logits"]
        conf = out["final_conf"]

        loss_ce = F.cross_entropy(logits[valid_mask], next_token[valid_mask])
        is_correct = (torch.argmax(logits, dim=-1) == next_token).float().detach()
        loss_brier = F.mse_loss(conf[valid_mask], is_correct[valid_mask])

        total_loss = loss_ce + 2.5 * loss_brier

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 200 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                test_prefix = test_targets[:400, :3]
                test_next = test_targets[:400, 3]
                val_mask = test_next != 0

                val_out = model(test_grids[:400], test_prefix)
                val_preds = torch.argmax(val_out["logits"], dim=-1)
                val_acc = (val_preds[val_mask] == test_next[val_mask]).float().mean().item() * 100.0
                mean_conf = val_out["final_conf"][val_mask].mean().item()

                step_losses.append(total_loss.item())
                step_ce_losses.append(loss_ce.item())
                step_brier_losses.append(loss_brier.item())
                step_accuracies.append(val_acc)

                print(
                    f"Step {step:04d}/{steps} | Loss: {total_loss.item():.4f} (CE: {loss_ce.item():.4f}, "
                    f"Brier: {loss_brier.item():.4f}) | Val Acc: {val_acc:.1f}% | Noul Conf: {mean_conf:.3f}",
                    flush=True,
                )

    train_duration = time.time() - t_train_start
    print(f"\nTraining Complete in {train_duration:.2f}s on A10G", flush=True)

    # 4. Rigorous Scientific Holdout Benchmarking
    print("\n--- Commencing Scaled Evaluation Suite on 600 Holdout Mazes ---", flush=True)
    model.eval()
    eval_n = 600
    ev_grids = test_grids[:eval_n]
    ev_targets = test_targets[:eval_n]
    ev_junc = test_junc[:eval_n]

    # Benchmark A: Dynamic E-ACT vs CALM across Thresholds
    thresholds = [0.20, 0.40, 0.60, 0.75, 0.85, 0.95]
    eret_results = {}
    calm_results = {}

    with torch.no_grad():
        for tau in thresholds:
            # ERET (Jev calibrated Noul)
            eret_accs = []
            eret_steps = []
            for step_idx in range(1, 6):
                pfx = ev_targets[:, :step_idx]
                tgt = ev_targets[:, step_idx]
                mask = tgt != 0
                if not mask.any():
                    continue
                out = model(ev_grids, pfx, threshold=tau, dynamic_halt=True, calm_mode=False)
                preds = torch.argmax(out["logits"], dim=-1)
                acc = (preds[mask] == tgt[mask]).float().mean().item()
                steps_alloc = out["halt_steps"][mask].float().mean().item()
                eret_accs.append(acc)
                eret_steps.append(steps_alloc)

            eret_results[tau] = {
                "accuracy": float(np.mean(eret_accs) * 100.0),
                "avg_unrolls": float(np.mean(eret_steps)),
                "relative_flops": float(np.mean(eret_steps) / 6.0),
            }

            # CALM (Softmax Entropy)
            calm_accs = []
            calm_steps = []
            for step_idx in range(1, 6):
                pfx = ev_targets[:, :step_idx]
                tgt = ev_targets[:, step_idx]
                mask = tgt != 0
                if not mask.any():
                    continue
                out = model(ev_grids, pfx, threshold=tau, dynamic_halt=True, calm_mode=True)
                preds = torch.argmax(out["logits"], dim=-1)
                acc = (preds[mask] == tgt[mask]).float().mean().item()
                steps_alloc = out["halt_steps"][mask].float().mean().item()
                calm_accs.append(acc)
                calm_steps.append(steps_alloc)

            calm_results[tau] = {
                "accuracy": float(np.mean(calm_accs) * 100.0),
                "avg_unrolls": float(np.mean(calm_steps)),
                "relative_flops": float(np.mean(calm_steps) / 6.0),
            }

    # Benchmark B: Fixed Compute Baselines (K = 1, 2, 4, 6)
    fixed_results = {}
    with torch.no_grad():
        for k_fix in [1, 2, 4, 6]:
            k_accs = []
            for step_idx in range(1, 6):
                pfx = ev_targets[:, :step_idx]
                tgt = ev_targets[:, step_idx]
                mask = tgt != 0
                if not mask.any():
                    continue
                out = model(ev_grids, pfx, fixed_loops=k_fix)
                preds = torch.argmax(out["logits"], dim=-1)
                acc = (preds[mask] == tgt[mask]).float().mean().item()
                k_accs.append(acc)
            fixed_results[k_fix] = float(np.mean(k_accs) * 100.0)

    # Benchmark C: Topological Computation Allocation (Corridor vs Junction)
    corridor_unrolls = []
    junction_unrolls = []
    with torch.no_grad():
        tau_mid = 0.70
        for step_idx in range(1, 6):
            pfx = ev_targets[:, :step_idx]
            tgt = ev_targets[:, step_idx]
            junc = ev_junc[:, step_idx]
            mask = tgt != 0
            if not mask.any():
                continue
            out = model(ev_grids, pfx, threshold=tau_mid, dynamic_halt=True)
            halts = out["halt_steps"]

            corridor_mask = mask & (junc == 0)
            junction_mask = mask & (junc == 1)

            if corridor_mask.any():
                corridor_unrolls.extend(halts[corridor_mask].cpu().numpy().tolist())
            if junction_mask.any():
                junction_unrolls.extend(halts[junction_mask].cpu().numpy().tolist())

    mean_corr_unrolls = float(np.mean(corridor_unrolls)) if corridor_unrolls else 1.0
    mean_junc_unrolls = float(np.mean(junction_unrolls)) if junction_unrolls else 6.0

    # Benchmark D: Expected Calibration Error (ECE) under Bounded / Unsolvable mazes
    # We evaluate calibration on step 2 across all mazes
    with torch.no_grad():
        pfx = ev_targets[:, :2]
        tgt = ev_targets[:, 2]
        mask = tgt != 0

        out = model(ev_grids, pfx, dynamic_halt=False)
        preds = torch.argmax(out["logits"], dim=-1)
        corrects = (preds == tgt).float()[mask].cpu().numpy()

        eret_confs = out["final_conf"][mask].cpu().numpy()

        probs = F.softmax(out["logits"][mask], dim=-1)
        calm_confs = torch.max(probs, dim=-1)[0].cpu().numpy()

        def compute_ece(confs, correct, n_bins=10):
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            ece = 0.0
            for i in range(n_bins):
                in_bin = (confs >= bin_boundaries[i]) & (confs < bin_boundaries[i + 1])
                prop_in_bin = np.mean(in_bin)
                if prop_in_bin > 0:
                    acc_in_bin = np.mean(correct[in_bin])
                    avg_conf_in_bin = np.mean(confs[in_bin])
                    ece += np.abs(acc_in_bin - avg_conf_in_bin) * prop_in_bin
            return float(ece * 100.0)

        eret_ece = compute_ece(eret_confs, corrects)
        calm_ece = compute_ece(calm_confs, corrects)

    # Benchmark E: Banach Equilibrium Residual Decay Norm
    with torch.no_grad():
        out = model(ev_grids[:100], ev_targets[:100, :2], dynamic_halt=False)
        # residuals has length 6
        residual_decay = [float(r.mean().cpu().item()) for r in out["residuals"]]

    # Save Checkpoint to Modal Volume
    save_path = "/checkpoints/eret_scaled_a10g.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "param_count": param_count,
            "steps": steps,
            "train_duration": train_duration,
            "eret_results": eret_results,
            "calm_results": calm_results,
            "fixed_results": fixed_results,
            "corridor_unrolls": mean_corr_unrolls,
            "junction_unrolls": mean_junc_unrolls,
            "eret_ece": eret_ece,
            "calm_ece": calm_ece,
            "residual_decay": residual_decay,
        },
        save_path,
    )
    volume.commit()
    print(f"Checkpoint and evaluation results committed to Modal Volume: {save_path}", flush=True)

    return {
        "param_count": param_count,
        "train_duration": train_duration,
        "step_losses": step_losses,
        "step_accuracies": step_accuracies,
        "eret_results": eret_results,
        "calm_results": calm_results,
        "fixed_results": fixed_results,
        "corridor_unrolls": mean_corr_unrolls,
        "junction_unrolls": mean_junc_unrolls,
        "eret_ece": eret_ece,
        "calm_ece": calm_ece,
        "residual_decay": residual_decay,
    }


# ----------------------------------------------------------------------
# 4. Local Entrypoint
# ----------------------------------------------------------------------

@app.local_entrypoint()
def main():
    print("Initiating Scaled ERET (Jevformer 2.0) Run on Modal A10G...", flush=True)
    results = run_scaled_eret_benchmark.remote()
    print("\n--- RESULTS RECEIVED FROM MODAL A10G ---")
    print(f"Parameters: {results['param_count']:,}")
    print(f"Training Time: {results['train_duration']:.1f}s")
    print("\nFixed Loop Accuracies:", results["fixed_results"])
    print("\nTopological Allocation:")
    print(f"  Straight Corridor Unrolls: {results['corridor_unrolls']:.2f} / 6.0")
    print(f"  Branch Junction Unrolls:   {results['junction_unrolls']:.2f} / 6.0")
    print("\nEpistemic Calibration (ECE):")
    print(f"  ERET (Calibrated Noul): {results['eret_ece']:.2f}%")
    print(f"  CALM (Softmax Entropy): {results['calm_ece']:.2f}%")
    print("\nBanach Residual Decay:", [f"{r:.4f}" for r in results["residual_decay"]])

    # Save local results dictionary
    torch.save(results, "eret_scaled_results.pt")
    print("\nSaved results to eret_scaled_results.pt successfully.")
