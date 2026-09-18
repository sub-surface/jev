"""
Frontier 3: Symbolic Program Deduction Arena (Epistemic Tree-of-Thought)
=======================================================================
Authors: Leon & Ilya Sutskever persona

The Fundamental Question:
  Can calibrated System 1 decision primitives (Jev) guide Tree-of-Thought search
  over multi-hop symbolic program executions, pruning dead-end branches and scaling
  zero-shot to deep deductive horizons where latent recurrence collapses?

Task: Multi-Hop Guarded Register Machine Deduction
  - Program with registers [R0, R1, R2, R3] and conditional transitions.
  - Terminal Goal: Reach target register configuration satisfying terminal assertion.
  - Hard Deceptive Attractors: Branches that mimic target values but lead to dead ends.
  - In-Distribution: 4-hop execution paths.
  - Out-of-Distribution: 8-hop and 12-hop deep deduction graphs.
"""

from __future__ import annotations

import os
import sys
import time
import math
import random
from collections import deque
from dataclasses import dataclass

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_REGISTERS = 4
MAX_VAL = 32

# Instruction types:
# 0: ADD R_a, R_b -> R_a
# 1: SUB R_a, R_b -> R_a
# 2: XOR R_a, R_b -> R_a
# 3: INC R_a
# 4: SHL R_a


# ----------------------------------------------------------------------
# 1. Symbolic Register Machine & Deductive Execution Graph
# ----------------------------------------------------------------------

class RegisterMachineEnv:
    def __init__(self, depth: int = 4, branching: int = 3, seed: int | None = None):
        self.depth = depth
        self.branching = branching
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        # Initial registers
        self.r0 = np.array([self.rng.randint(1, 10) for _ in range(NUM_REGISTERS)], dtype=np.int32)
        self.regs = np.copy(self.r0)

        # Generate a valid target reachable at exactly `depth` hops
        # alongside deceptive attractor branches
        cur = np.copy(self.r0)
        self.correct_path = []
        for d in range(self.depth):
            op = self.rng.randint(0, 4)
            r_idx = self.rng.randint(0, NUM_REGISTERS - 1)
            other_idx = (r_idx + 1) % NUM_REGISTERS
            cur = self._apply_op(cur, op, r_idx, other_idx)
            self.correct_path.append((op, r_idx, other_idx))

        self.target = np.copy(cur)
        self.current_step = 0
        self.max_steps = self.depth * 3
        return self.get_state()

    def _apply_op(self, regs: np.ndarray, op: int, r_a: int, r_b: int) -> np.ndarray:
        res = np.copy(regs)
        if op == 0:
            res[r_a] = (res[r_a] + res[r_b]) % MAX_VAL
        elif op == 1:
            res[r_a] = (res[r_a] - res[r_b] + MAX_VAL) % MAX_VAL
        elif op == 2:
            res[r_a] = res[r_a] ^ res[r_b]
        elif op == 3:
            res[r_a] = (res[r_a] + 1) % MAX_VAL
        elif op == 4:
            res[r_a] = (res[r_a] * 2) % MAX_VAL
        return res

    def get_candidate_branches(self, regs: np.ndarray | None = None) -> list[tuple[int, int, int, np.ndarray]]:
        """Returns candidate execution steps: (op, r_a, r_b, resulting_regs)."""
        cur = regs if regs is not None else self.regs
        branches = []
        for op in range(5):
            for r_a in range(NUM_REGISTERS):
                r_b = (r_a + 1) % NUM_REGISTERS
                next_r = self._apply_op(cur, op, r_a, r_b)
                branches.append((op, r_a, r_b, next_r))
        return branches

    def get_state(self, regs: np.ndarray | None = None) -> np.ndarray:
        cur = regs if regs is not None else self.regs
        # Normalized vector: current regs, target regs, difference
        diff = np.abs(cur - self.target)
        feats = np.concatenate([
            cur / MAX_VAL,
            self.target / MAX_VAL,
            diff / MAX_VAL,
            [self.current_step / self.max_steps],
        ], dtype=np.float32)
        return feats

    def step(self, resulting_regs: np.ndarray) -> tuple[float, bool, dict]:
        self.current_step += 1
        self.regs = resulting_regs
        is_success = np.array_equal(self.regs, self.target)

        if is_success:
            return 10.0, True, {"success": True}
        if self.current_step >= self.max_steps:
            return -1.0, True, {"success": False}

        # Step penalty
        return -0.05, False, {"success": False}


# ----------------------------------------------------------------------
# 2. Neural Models: Jev Verifier & ERET Looped Baseline
# ----------------------------------------------------------------------

class JevProgramVerifier(nn.Module):
    """
    TypeSafe Jev Model for Program Execution:
      1. Value V(s) in [0, 1]: estimated probability that state can reach target assertion.
      2. Noul(s) in [0, 1]: epistemic certainty that state is on a valid proof path.
    """
    def __init__(self, in_features: int = 13, d_model: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.val_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.noul_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(x)
        val = self.val_head(h).squeeze(-1)
        noul = self.noul_head(h).squeeze(-1)
        return val, noul


class ERETProgramLooped(nn.Module):
    """Weight-tied Krasnoselskii-Mann Recurrent Equilibrium Model for Programs."""
    def __init__(self, in_features: int = 13, d_model: int = 128):
        super().__init__()
        self.in_proj = nn.Linear(in_features, d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.noul_valve = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.val_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, loops: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.gelu(self.in_proj(x))
        last_conf = None
        for _ in range(loops):
            conf = self.noul_valve(h).squeeze(-1)
            gamma = (1.0 - conf).clamp(min=0.05, max=0.95).unsqueeze(-1)
            delta = F.gelu(self.norm1(self.fc1(h)))
            delta = self.norm2(self.fc2(delta))
            h = h + gamma * delta
            last_conf = conf

        val = self.val_head(h).squeeze(-1)
        return val, last_conf


# ----------------------------------------------------------------------
# 3. Training on Program Execution Traces
# ----------------------------------------------------------------------

def train_jev_verifier(model: JevProgramVerifier, num_programs: int = 3000, epochs: int = 6):
    print("Generating symbolic program execution traces for training...", flush=True)
    states = []
    values = []
    nouls = []

    for _ in range(num_programs):
        env = RegisterMachineEnv(depth=random.randint(2, 5))
        # Add states along the true proof trajectory
        cur = np.copy(env.r0)
        remaining = len(env.correct_path)

        for step_idx, (op, r_a, r_b) in enumerate(env.correct_path):
            cur = env._apply_op(cur, op, r_a, r_b)
            rem_dist = remaining - step_idx
            val = float(0.90 ** rem_dist)
            noul = 1.0  # On true proof path

            feats = env.get_state(cur)
            states.append(feats)
            values.append(val)
            nouls.append(noul)

            # Also sample deceptive dead-end branches
            branches = env.get_candidate_branches(cur)
            sampled_deceptive = random.sample(branches, min(3, len(branches)))
            for _, _, _, false_r in sampled_deceptive:
                if not np.array_equal(false_r, env.target):
                    feats_f = env.get_state(false_r)
                    states.append(feats_f)
                    values.append(0.05)  # Dead-end
                    nouls.append(0.05)   # Low epistemic certainty

    x_t = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    v_t = torch.tensor(np.array(values), dtype=torch.float32).to(device)
    n_t = torch.tensor(np.array(nouls), dtype=torch.float32).to(device)

    dataset = torch.utils.data.TensorDataset(x_t, v_t, n_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()

    t0 = time.time()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for b_x, b_v, b_n in loader:
            v_p, n_p = model(b_x)
            loss = F.mse_loss(v_p, b_v) + F.mse_loss(n_p, b_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    print(f"Jev Program Verifier trained on {len(states):,} states in {time.time() - t0:.1f}s\n", flush=True)


# ----------------------------------------------------------------------
# 4. Search Engines: Greedy, ERET, Uniform ToT, and Adaptive Epistemic ToT
# ----------------------------------------------------------------------

def select_branch_greedy(model: JevProgramVerifier, env: RegisterMachineEnv) -> np.ndarray:
    branches = env.get_candidate_branches()
    feats = [env.get_state(b[3]) for b in branches]
    x = torch.tensor(np.array(feats), dtype=torch.float32).to(device)
    with torch.no_grad():
        vals, _ = model(x)
        best_idx = torch.argmax(vals).item()
    return branches[best_idx][3]


def select_branch_eret(model: ERETProgramLooped, env: RegisterMachineEnv, loops: int = 4) -> np.ndarray:
    branches = env.get_candidate_branches()
    feats = [env.get_state(b[3]) for b in branches]
    x = torch.tensor(np.array(feats), dtype=torch.float32).to(device)
    with torch.no_grad():
        vals, _ = model(x, loops=loops)
        best_idx = torch.argmax(vals).item()
    return branches[best_idx][3]


def epistemic_tree_of_thought(
    model: JevProgramVerifier,
    env: RegisterMachineEnv,
    beam_width: int = 3,
    depth_limit: int = 4,
    adaptive: bool = True,
    tau: float = 0.80,
) -> tuple[np.ndarray, int]:
    """
    Epistemic Tree-of-Thought (ToT) Deductive Search:
      - Explores counterfactual proof branches.
      - If Adaptive: When Noul >= tau (clear deductive step), branches 0 counterfactuals.
        When Noul < tau (ambiguous branch), expands beam search over candidate programs!
    """
    branches = env.get_candidate_branches()
    feats0 = [env.get_state(b[3]) for b in branches]
    x0 = torch.tensor(np.array(feats0), dtype=torch.float32).to(device)
    with torch.no_grad():
        vals0, nouls0 = model(x0)
        best_idx0 = torch.argmax(vals0).item()
        best_noul0 = nouls0[best_idx0].item()

    # Dynamic Epistemic Gate:
    if adaptive and best_noul0 >= tau:
        # Reflexive commitment: 0 search expansions
        return branches[best_idx0][3], 0

    # Search over candidate deduction branches
    top_indices = torch.topk(vals0, min(beam_width, len(branches))).indices.cpu().numpy()
    beam = []
    expansions = 0

    for idx in top_indices:
        b_info = branches[idx]
        beam.append({
            "score": vals0[idx].item(),
            "regs": b_info[3],
            "first_step": b_info[3],
        })

    for d in range(1, depth_limit):
        candidates = []
        for node in beam:
            cur_r = node["regs"]
            if np.array_equal(cur_r, env.target):
                candidates.append({
                    "score": node["score"] + 100.0,
                    "regs": cur_r,
                    "first_step": node["first_step"],
                })
                continue

            child_branches = env.get_candidate_branches(cur_r)
            c_feats = [env.get_state(cb[3]) for cb in child_branches]
            x_c = torch.tensor(np.array(c_feats), dtype=torch.float32).to(device)
            with torch.no_grad():
                c_vals, _ = model(x_c)
            expansions += 1

            for cb_idx, cb in enumerate(child_branches):
                candidates.append({
                    "score": node["score"] + c_vals[cb_idx].item(),
                    "regs": cb[3],
                    "first_step": node["first_step"],
                })

        if not candidates:
            break
        candidates.sort(key=lambda x: x["score"], reverse=True)
        beam = candidates[:beam_width]

    return beam[0]["first_step"], expansions


# ----------------------------------------------------------------------
# 5. Tournament & Out-of-Distribution Deduction Scaling
# ----------------------------------------------------------------------

def run_deduction_benchmark(num_trials: int = 50):
    print("================================================================================", flush=True)
    print(" FRONTIER 3: SYMBOLIC PROGRAM DEDUCTION ARENA (Tree-of-Thought Scaling)", flush=True)
    print("================================================================================\n", flush=True)

    jev_model = JevProgramVerifier().to(device)
    eret_model = ERETProgramLooped().to(device)

    train_jev_verifier(jev_model, num_programs=2500, epochs=6)

    # Initialize ERET weights to match
    eret_model.in_proj.weight.data.copy_(jev_model.fc[0].weight.data)
    eret_model.val_head[0].weight.data.copy_(jev_model.val_head[0].weight.data)
    eret_model.eval()
    jev_model.eval()

    depth_levels = [
        ("4-Hop Programs (In-Distribution)", 4),
        ("8-Hop Programs (Out-of-Distribution Deep)", 8),
        ("12-Hop Programs (Extreme Combinatorial Horizon)", 12),
    ]

    modes = [
        "ERET_K4 (Latent Loop)",
        "S1_Greedy (Reflexive)",
        "Uniform_ToT (Search)",
        "Adaptive_ETS (τ=0.80)",
    ]

    for label, d_val in depth_levels:
        print(f"\n>>> Evaluating: {label} ({num_trials} Random Programs) <<<", flush=True)
        results = {m: {"success": 0, "expansions": [], "steps": []} for m in modes}

        for t_idx in range(num_trials):
            env = RegisterMachineEnv(depth=d_val, seed=1000 + t_idx)
            start_regs = np.copy(env.r0)
            target_regs = np.copy(env.target)

            for mode in modes:
                env.regs = np.copy(start_regs)
                env.target = np.copy(target_regs)
                env.current_step = 0
                env.max_steps = d_val * 3

                done = False
                total_exp = 0

                while not done:
                    if mode == "ERET_K4 (Latent Loop)":
                        next_r = select_branch_eret(eret_model, env, loops=4)
                        exp = 0
                    elif mode == "S1_Greedy (Reflexive)":
                        next_r = select_branch_greedy(jev_model, env)
                        exp = 0
                    elif mode == "Uniform_ToT (Search)":
                        next_r, exp = epistemic_tree_of_thought(
                            jev_model, env, beam_width=3, depth_limit=3, adaptive=False
                        )
                    elif mode == "Adaptive_ETS (τ=0.80)":
                        next_r, exp = epistemic_tree_of_thought(
                            jev_model, env, beam_width=3, depth_limit=3, adaptive=True, tau=0.80
                        )

                    total_exp += exp
                    _, done, info = env.step(next_r)

                if info["success"]:
                    results[mode]["success"] += 1
                    results[mode]["steps"].append(env.current_step)
                results[mode]["expansions"].append(total_exp)

        print("-" * 75, flush=True)
        print(f"{'Paradigm':25s} | {'Success Rate':14s} | {'Avg Expansions':15s} | {'Avg Steps':10s}", flush=True)
        print("-" * 75, flush=True)
        for mode in modes:
            succ_pct = (results[mode]["success"] / num_trials) * 100.0
            avg_exp = np.mean(results[mode]["expansions"])
            avg_steps = np.mean(results[mode]["steps"]) if results[mode]["steps"] else 0.0
            print(f"{mode:25s} | {succ_pct:13.1f}% | {avg_exp:15.1f} | {avg_steps:9.1f}", flush=True)
        print("-" * 75, flush=True)


if __name__ == "__main__":
    run_deduction_benchmark(num_trials=40)
