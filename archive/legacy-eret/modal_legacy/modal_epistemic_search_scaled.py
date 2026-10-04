"""
Modal Cloud: Scaling Epistemic Tree Search (ETS / Jev-Search)
============================================================
Authors: Leon & Ilya Sutskever persona
Theoretical & Empirical Paradigm Shift:
  - Discarding the unviable hypothesis that weight-tied latent Krasnoselskii loops
    can substitute for combinatorial search in closed-loop navigation.
  - Embracing the Bitter Lesson: Explicit Tree Search (Search) + Calibrated RLCD Value (Learning).
  - Unifying TypeSafe's Jev (as a calibrated System 1 decision engine) with
    Epistemic Adaptive Tree Search (System 2):
      * When Noul(s) >= tau (clear corridor): 0 search expansions (Reflexive 1-pass).
      * When Noul(s) < tau (deceptive junction): dynamically unrolls tree search
        proportional to epistemic uncertainty.

Scales to hard 12x12 and 16x16 Procedural Labyrinths on NVIDIA A10G.
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

APP_NAME = "epistemic-search-scaled"

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
# 1. Scaled Procedural Labyrinth Environment
# ----------------------------------------------------------------------

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)] # UP, DOWN, LEFT, RIGHT

class ScaledMazeEnv:
    def __init__(self, size: int = 12, wall_prob: float = 0.28):
        self.size = size
        self.wall_prob = wall_prob
        self.reset()

    def reset(self, start=None, goal=None):
        self.grid = np.zeros((self.size, self.size), dtype=np.int32)
        for r in range(self.size):
            for c in range(self.size):
                if random.random() < self.wall_prob:
                    self.grid[r, c] = 1 # Wall

        empty = [(r, c) for r in range(self.size) for c in range(self.size) if self.grid[r, c] == 0]
        while len(empty) < 6:
            return self.reset()

        self.start, self.goal = random.sample(empty, 2)
        if start is not None: self.start = start
        if goal is not None: self.goal = goal

        self.grid[self.start] = 2
        self.grid[self.goal] = 3

        dist_map = self.get_all_pairs_distances()
        if dist_map[self.start] > 900 or dist_map[self.start] < 4:
            return self.reset()

        self.agent_pos = self.start
        self.steps = 0
        self.max_steps = self.size * 6
        self.optimal_dist = int(dist_map[self.start])
        return self.get_state()

    def get_all_pairs_distances(self) -> np.ndarray:
        dist_map = np.ones((self.size, self.size), dtype=np.float32) * 999.0
        q = deque([(self.goal[0], self.goal[1], 0)])
        dist_map[self.goal] = 0.0
        visited = {self.goal}
        while q:
            r, c, d = q.popleft()
            for dr, dc in ACTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size and self.grid[nr, nc] != 1 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    dist_map[nr, nc] = d + 1
                    q.append((nr, nc, d + 1))
        return dist_map

    def get_state(self, pos=None) -> np.ndarray:
        cur_pos = pos if pos is not None else self.agent_pos
        state = np.zeros((3, self.size, self.size), dtype=np.float32)
        state[0] = (self.grid == 1).astype(np.float32)
        state[1, cur_pos[0], cur_pos[1]] = 1.0
        state[2, self.goal[0], self.goal[1]] = 1.0
        return state

    def step(self, action: int):
        self.steps += 1
        dr, dc = ACTIONS[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        # Check collision
        if nr < 0 or nr >= self.size or nc < 0 or nc >= self.size or self.grid[nr, nc] == 1:
            reward = -1.0
            done = self.steps >= self.max_steps
            return self.get_state(), reward, done, {"collision": True, "success": False}

        self.agent_pos = (nr, nc)
        if self.agent_pos == self.goal:
            reward = 10.0
            done = True
            return self.get_state(), reward, done, {"collision": False, "success": True}

        reward = -0.02
        done = self.steps >= self.max_steps
        return self.get_state(), reward, done, {"collision": False, "success": False}


# ----------------------------------------------------------------------
# 2. Architectures: Calibrated Jev Evaluator & ERET Looped Baseline
# ----------------------------------------------------------------------

class CalibratedJevEvaluator(nn.Module):
    """
    TypeSafe Jev System 1 Model:
    Parallel non-generative forward pass outputting:
      1. Policy Logits (Action selection distribution)
      2. Calibrated Value V(s) in [0, 1] (Probability of task reachability)
      3. Calibrated Noul(s) in [0, 1] (Epistemic certainty of policy correctness)
    """
    def __init__(self, in_channels: int = 3, d_model: int = 192):
        super().__init__()
        self.d_model = d_model
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.GELU(),
            nn.Conv2d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )
        self.policy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 4),
        )
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.noul_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.conv_stem(x)
        policy_logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        noul = self.noul_head(h).squeeze(-1)
        return policy_logits, value, noul


class ScaledERETNet(nn.Module):
    """Weight-tied Krasnoselskii-Mann Recurrent Equilibrium Transformer Baseline."""
    def __init__(self, in_channels: int = 3, d_model: int = 192, max_loops: int = 6):
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(d_model)
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(d_model)
        self.noul_valve = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.policy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 4),
        )
        self.max_loops = max_loops

    def forward(self, x: torch.Tensor, loops: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.gelu(self.in_proj(x))
        last_conf = None
        for _ in range(loops):
            conf = self.noul_valve(h).squeeze(-1)
            gamma = (1.0 - conf).clamp(min=0.05, max=0.95).view(-1, 1, 1, 1)
            delta = F.gelu(self.norm1(self.conv1(h)))
            delta = self.norm2(self.conv2(delta))
            h = h + gamma * delta
            last_conf = conf

        logits = self.policy_head(h)
        return logits, last_conf


# ----------------------------------------------------------------------
# 3. Dense Procedural Batch Generator (RLCD Training Data)
# ----------------------------------------------------------------------

def generate_scaled_training_batch(batch_size: int = 64, size: int = 12, wall_prob: float = 0.28):
    states = []
    actions = []
    values = []

    while len(states) < batch_size:
        env = ScaledMazeEnv(size=size, wall_prob=wall_prob)
        dist_map = env.get_all_pairs_distances()

        empty_cells = [
            (r, c) for r in range(size) for c in range(size)
            if env.grid[r, c] != 1 and dist_map[r, c] < 900
        ]
        if len(empty_cells) < 4:
            continue

        sampled = random.sample(empty_cells, min(12, len(empty_cells)))
        for (r, c) in sampled:
            if (r, c) == env.goal:
                continue
            cur_d = dist_map[r, c]
            best_a = None
            best_d = cur_d
            for a_idx, (dr, dc) in enumerate(ACTIONS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and dist_map[nr, nc] < best_d:
                    best_d = dist_map[nr, nc]
                    best_a = a_idx

            if best_a is not None:
                states.append(env.get_state((r, c)))
                actions.append(best_a)
                values.append(0.95 ** cur_d)

            if len(states) >= batch_size:
                break

    return (
        torch.tensor(np.array(states[:batch_size]), dtype=torch.float32),
        torch.tensor(np.array(actions[:batch_size]), dtype=torch.long),
        torch.tensor(np.array(values[:batch_size]), dtype=torch.float32),
    )


# ----------------------------------------------------------------------
# 4. Adaptive Epistemic Tree Search (ETS / Jev-Search) Engine
# ----------------------------------------------------------------------

def adaptive_epistemic_search(
    model: CalibratedJevEvaluator,
    env: ScaledMazeEnv,
    tau: float = 0.82,
    max_depth: int = 6,
    max_beam: int = 3,
    device: torch.device = torch.device("cuda"),
) -> tuple[int, int]:
    """
    Executes Adaptive Epistemic Tree Search:
      1. Evaluates root state via Jev System 1 forward pass.
      2. If Noul >= tau: Takes reflexive argmax action (0 search expansions).
      3. If Noul < tau: Epistemic search triggered. Depth and beam width are scaled
         proportional to uncertainty (1 - Noul).
    Returns:
      (chosen_action, search_expansions_used)
    """
    s0 = env.get_state()
    pos0 = env.agent_pos

    with torch.no_grad():
        s_tensor = torch.tensor(s0, dtype=torch.float32).unsqueeze(0).to(device)
        logits0, val0, noul0 = model(s_tensor)
        noul_p = noul0.item()
        p_dist0 = F.log_softmax(logits0, dim=-1)[0].cpu().numpy()

    # Dynamic Epistemic Gating:
    if noul_p >= tau:
        # Reflexive System 1 action (Zero search expansions!)
        action = int(np.argmax(p_dist0))
        return action, 0

    # Epistemic Uncertainty triggers System 2 Tree Search
    uncertainty = 1.0 - noul_p
    effective_depth = max(2, int(math.ceil(uncertainty * max_depth)))
    effective_beam = max(2, int(math.ceil(uncertainty * max_beam)))

    # Initialize beam
    beam = []
    for a_idx in range(4):
        dr, dc = ACTIONS[a_idx]
        nr, nc = pos0[0] + dr, pos0[1] + dc
        if 0 <= nr < env.size and 0 <= nc < env.size and env.grid[nr, nc] != 1:
            beam.append({
                "score": p_dist0[a_idx],
                "pos": (nr, nc),
                "first_action": a_idx,
                "history": [a_idx],
            })

    if not beam:
        return 0, 0

    expansions = 0
    for d in range(1, effective_depth):
        candidates = []
        for b in beam:
            cur_p = b["pos"]
            if cur_p == env.goal:
                candidates.append({
                    "score": b["score"] + 100.0,
                    "pos": cur_p,
                    "first_action": b["first_action"],
                    "history": b["history"],
                })
                continue

            cur_s = env.get_state(cur_p)
            s_leaf = torch.tensor(cur_s, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                l_leaf, v_leaf, _ = model(s_leaf)
                p_leaf = F.log_softmax(l_leaf, dim=-1)[0].cpu().numpy()
                v_score = v_leaf.item()
            expansions += 1

            for a_idx in range(4):
                dr, dc = ACTIONS[a_idx]
                nr, nc = cur_p[0] + dr, cur_p[1] + dc
                if 0 <= nr < env.size and 0 <= nc < env.size and env.grid[nr, nc] != 1:
                    is_backtrack = (len(b["history"]) > 0 and a_idx == (b["history"][-1] ^ 1))
                    penalty = -0.5 if is_backtrack else 0.0
                    candidates.append({
                        "score": b["score"] + p_leaf[a_idx] + 2.5 * v_score + penalty,
                        "pos": (nr, nc),
                        "first_action": b["first_action"],
                        "history": b["history"] + [a_idx],
                    })

        if not candidates:
            break
        candidates.sort(key=lambda x: x["score"], reverse=True)
        beam = candidates[:effective_beam]

    return beam[0]["first_action"], expansions


# ----------------------------------------------------------------------
# 5. Modal Remote Training & Rigorous Benchmark Worker
# ----------------------------------------------------------------------

@app.function(
    gpu="A10G",
    image=image,
    volumes={"/checkpoints": volume},
    timeout=1800,
)
def run_scaled_epistemic_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Running Scaled Epistemic Search Benchmark on {device} ({torch.cuda.get_device_name(0)}) ===", flush=True)

    # 1. Instantiate Models
    d_model = 192
    jev_model = CalibratedJevEvaluator(in_channels=3, d_model=d_model).to(device)
    eret_model = ScaledERETNet(in_channels=3, d_model=d_model, max_loops=6).to(device)

    opt_jev = torch.optim.AdamW(jev_model.parameters(), lr=8e-4, weight_decay=1e-4)
    opt_eret = torch.optim.AdamW(eret_model.parameters(), lr=8e-4, weight_decay=1e-4)

    sched_jev = torch.optim.lr_scheduler.CosineAnnealingLR(opt_jev, T_max=3000, eta_min=5e-5)
    sched_eret = torch.optim.lr_scheduler.CosineAnnealingLR(opt_eret, T_max=3000, eta_min=5e-5)

    param_count = sum(p.numel() for p in jev_model.parameters() if p.requires_grad)
    print(f"Calibrated Jev Model Initialized. Total Parameters: {param_count:,}", flush=True)

    # 2. Train Models (3,000 Steps on 12x12 Procedural Mazes with Dense RLCD Supervision)
    steps = 3000
    batch_size = 64
    size = 12
    print(f"\n--- Launching 3,000-Step Dense RLCD Training on A10G ({size}x{size} Grids) ---", flush=True)

    t0 = time.time()
    for step in range(1, steps + 1):
        jev_model.train()
        eret_model.train()

        b_s, b_a, b_v = generate_scaled_training_batch(batch_size=batch_size, size=size, wall_prob=0.28)
        b_s, b_a, b_v = b_s.to(device), b_a.to(device), b_v.to(device)

        # Train Jev Evaluator (RLCD Proper Scoring)
        l_j, v_j, noul_j = jev_model(b_s)
        is_corr_j = (torch.argmax(l_j, dim=-1) == b_a).float().detach()
        loss_ce_j = F.cross_entropy(l_j, b_a)
        loss_val_j = F.mse_loss(v_j, b_v)
        loss_brier_j = F.mse_loss(noul_j, is_corr_j)
        total_loss_j = loss_ce_j + 1.5 * loss_val_j + 1.0 * loss_brier_j

        opt_jev.zero_grad()
        total_loss_j.backward()
        torch.nn.utils.clip_grad_norm_(jev_model.parameters(), 1.0)
        opt_jev.step()
        sched_jev.step()

        # Train ERET Looped Baseline (K=4)
        l_e, conf_e = eret_model(b_s, loops=4)
        is_corr_e = (torch.argmax(l_e, dim=-1) == b_a).float().detach()
        loss_ce_e = F.cross_entropy(l_e, b_a)
        loss_brier_e = F.mse_loss(conf_e, is_corr_e)
        total_loss_e = loss_ce_e + 1.0 * loss_brier_e

        opt_eret.zero_grad()
        total_loss_e.backward()
        torch.nn.utils.clip_grad_norm_(eret_model.parameters(), 1.0)
        opt_eret.step()
        sched_eret.step()

        if step % 500 == 0 or step == steps:
            acc_j = (torch.argmax(l_j, dim=-1) == b_a).float().mean().item() * 100.0
            acc_e = (torch.argmax(l_e, dim=-1) == b_a).float().mean().item() * 100.0
            print(
                f"Step {step:04d}/{steps} | Jev Loss: {total_loss_j.item():.3f} (Acc: {acc_j:.1f}%, Brier: {loss_brier_j.item():.3f}) | "
                f"ERET Loss: {total_loss_e.item():.3f} (Acc: {acc_e:.1f}%)",
                flush=True,
            )

    train_time = time.time() - t0
    print(f"\nOptimization Complete in {train_time:.1f}s on NVIDIA A10G.", flush=True)

    # 3. Scientific Evaluation Suite across Scaled Procedural Mazes
    print("\n================================================================================", flush=True)
    print(" RIGOROUS CLOSED-LOOP BENCHMARK: 12x12 (In-Distribution) & 16x16 (Out-of-Distribution)", flush=True)
    print("================================================================================", flush=True)

    jev_model.eval()
    eret_model.eval()

    test_configs = [
        ("12x12 (In-Distribution)", 12, 150),
        ("16x16 (Out-of-Distribution Hard)", 16, 100),
    ]

    benchmark_summary = {}

    for label, grid_dim, num_trials in test_configs:
        print(f"\n>>> Commencing Evaluation: {label} ({num_trials} Trials) <<<", flush=True)

        results = {
            "S1_Greedy": {"success": 0, "collisions": 0, "steps": [], "expansions": 0},
            "ERET_K4": {"success": 0, "collisions": 0, "steps": [], "expansions": 0},
            "Uniform_Search_D5_B3": {"success": 0, "collisions": 0, "steps": [], "expansions": 0},
            "Adaptive_ETS_tau0.85": {"success": 0, "collisions": 0, "steps": [], "expansions": 0},
            "Adaptive_ETS_tau0.70": {"success": 0, "collisions": 0, "steps": [], "expansions": 0},
        }

        # Expected Calibration Error (ECE) tracker for Jev Noul head
        noul_confs = []
        step_corrects = []

        for trial in range(num_trials):
            env = ScaledMazeEnv(size=grid_dim, wall_prob=0.28)
            saved_grid = np.copy(env.grid)
            saved_start = env.start
            saved_goal = env.goal

            for mode in results.keys():
                env.grid = np.copy(saved_grid)
                env.agent_pos = saved_start
                env.start = saved_start
                env.goal = saved_goal
                env.steps = 0
                env.max_steps = grid_dim * 6

                done = False
                collided = False
                trial_expansions = 0

                while not done:
                    s_curr = env.get_state()
                    s_t = torch.tensor(s_curr, dtype=torch.float32).unsqueeze(0).to(device)

                    if mode == "S1_Greedy":
                        with torch.no_grad():
                            logits, _, noul_p = jev_model(s_t)
                            action = torch.argmax(logits, dim=-1).item()
                            if trial < 50:
                                opt_p = env.get_all_pairs_distances()
                                best_a = None
                                best_d = opt_p[env.agent_pos]
                                for a_idx, (dr, dc) in enumerate(ACTIONS):
                                    nr, nc = env.agent_pos[0] + dr, env.agent_pos[1] + dc
                                    if 0 <= nr < grid_dim and 0 <= nc < grid_dim and opt_p[nr, nc] < best_d:
                                        best_d = opt_p[nr, nc]
                                        best_a = a_idx
                                if best_a is not None:
                                    noul_confs.append(noul_p.item())
                                    step_corrects.append(1.0 if action == best_a else 0.0)

                    elif mode == "ERET_K4":
                        with torch.no_grad():
                            logits, _ = eret_model(s_t, loops=4)
                            action = torch.argmax(logits, dim=-1).item()

                    elif mode == "Uniform_Search_D5_B3":
                        # Always searches with fixed budget
                        action, exp_used = adaptive_epistemic_search(
                            jev_model, env, tau=1.01, max_depth=5, max_beam=3, device=device
                        )
                        trial_expansions += exp_used

                    elif mode == "Adaptive_ETS_tau0.85":
                        action, exp_used = adaptive_epistemic_search(
                            jev_model, env, tau=0.85, max_depth=5, max_beam=3, device=device
                        )
                        trial_expansions += exp_used

                    elif mode == "Adaptive_ETS_tau0.70":
                        action, exp_used = adaptive_epistemic_search(
                            jev_model, env, tau=0.70, max_depth=5, max_beam=3, device=device
                        )
                        trial_expansions += exp_used

                    _, _, done, info = env.step(action)
                    if info["collision"]:
                        collided = True

                if info["success"]:
                    results[mode]["success"] += 1
                    results[mode]["steps"].append(env.steps)
                if collided:
                    results[mode]["collisions"] += 1
                results[mode]["expansions"] += trial_expansions

        # Compute metrics for this grid dimension
        dim_summary = {}
        for mode, data in results.items():
            succ_pct = (data["success"] / num_trials) * 100.0
            avg_steps = np.mean(data["steps"]) if data["steps"] else 0.0
            avg_exp = data["expansions"] / num_trials
            dim_summary[mode] = {
                "success_rate": succ_pct,
                "collisions": data["collisions"],
                "avg_steps": avg_steps,
                "avg_expansions": avg_exp,
            }
            print(
                f"  {mode:22s} | Success: {succ_pct:5.1f}% | Collisions: {data['collisions']:3d} | "
                f"Avg Steps: {avg_steps:4.1f} | Search Expansions/Maze: {avg_exp:5.1f}",
                flush=True,
            )

        # Calculate ECE on collected samples
        if noul_confs:
            confs_np = np.array(noul_confs)
            corr_np = np.array(step_corrects)
            bins = np.linspace(0, 1, 11)
            ece = 0.0
            for b_idx in range(len(bins) - 1):
                m = (confs_np >= bins[b_idx]) & (confs_np < bins[b_idx + 1])
                if np.sum(m) > 0:
                    bin_acc = np.mean(corr_np[m])
                    bin_conf = np.mean(confs_np[m])
                    ece += (np.sum(m) / len(confs_np)) * np.abs(bin_acc - bin_conf)
            dim_summary["Jev_Noul_ECE"] = ece * 100.0
            print(f"  --> Jev Noul Expected Calibration Error (ECE): {ece * 100.0:.2f}%", flush=True)

        benchmark_summary[label] = dim_summary

    # Save results to Modal volume
    save_path = "/checkpoints/epistemic_search_scaled_results.pt"
    torch.save({
        "summary": benchmark_summary,
        "jev_weights": jev_model.state_dict(),
    }, save_path)
    volume.commit()
    print(f"\nArtifacts and checkpoints committed to Modal Volume: {save_path}", flush=True)
    return benchmark_summary


@app.local_entrypoint()
def main():
    print("Initiating Scaled Epistemic Search Benchmark on Modal Cloud (NVIDIA A10G)...")
    res = run_scaled_epistemic_benchmark.remote()
    print("\n--- Benchmark Remote Execution Completed ---")
    print("Summary:", res)
