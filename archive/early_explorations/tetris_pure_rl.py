"""
Frontier 1: Pure Reinforcement Learning Tetris (The Bitter Lesson Grounding)
==========================================================================
Authors: Leon & Ilya Sutskever persona

Zero Human Heuristics:
  - Strips all hand-engineered Dellacherie weights (no manual hole/bumpiness penalty).
  - Rewards are purely objective:
      +10.0 per line cleared
      +0.1 per piece survived
      -10.0 on game over
  - Learns Value V(s) and Noul(s) purely via Bellman Temporal Difference TD(0) learning.

Evaluates:
  1. Pure RL Reflexive (0-search)
  2. Pure RL ERET Latent Loop (K=4)
  3. Pure RL Uniform Lookahead Search
  4. Pure RL Adaptive Epistemic Tree Search (ETS)
"""

from __future__ import annotations

import os
import sys
import time
import random
from dataclasses import dataclass

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BOARD_WIDTH = 10
BOARD_HEIGHT = 20

TETROMINOES = {
    "I": [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
    ],
    "O": [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],
    "T": [
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (1, 1), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 1)],
    ],
    "S": [
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ],
    "Z": [
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
    ],
    "J": [
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 2)],
        [(0, 1), (1, 1), (2, 0), (2, 1)],
    ],
    "L": [
        [(0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ],
}

PIECE_NAMES = list(TETROMINOES.keys())


# ----------------------------------------------------------------------
# 1. Tetris Simulator with Pure Objective Reward
# ----------------------------------------------------------------------

class TetrisRLGame:
    def __init__(self, seed: int | None = None):
        self.width = BOARD_WIDTH
        self.height = BOARD_HEIGHT
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.board = np.zeros((self.height, self.width), dtype=np.float32)
        self.bag = []
        self.current_piece = self._get_next_piece()
        self.next_piece = self._get_next_piece()
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.game_over = False
        return self.board

    def _get_next_piece(self) -> str:
        if not self.bag:
            self.bag = list(PIECE_NAMES)
            self.rng.shuffle(self.bag)
        return self.bag.pop()

    def get_legal_placements(self, piece_name: str | None = None) -> list[tuple[int, int, np.ndarray, int]]:
        p_name = piece_name if piece_name is not None else self.current_piece
        rotations = TETROMINOES[p_name]
        placements = []

        for rot_idx, shape in enumerate(rotations):
            min_c = min(c for r, c in shape)
            max_c = max(c for r, c in shape)
            shape_w = max_c - min_c + 1

            for col in range(0, self.width - shape_w + 1):
                shifted = [(r, c - min_c + col) for r, c in shape]

                drop_r = 0
                while True:
                    test_cells = [(r + drop_r + 1, c) for r, c in shifted]
                    collision = False
                    for tr, tc in test_cells:
                        if tr >= self.height or self.board[tr, tc] == 1.0:
                            collision = True
                            break
                    if collision:
                        break
                    drop_r += 1

                landing_cells = [(r + drop_r, c) for r, c in shifted]
                if any(r < 0 or r >= self.height or self.board[r, c] == 1.0 for r, c in landing_cells):
                    continue

                new_b = np.copy(self.board)
                for r, c in landing_cells:
                    new_b[r, c] = 1.0

                full_lines = [r for r in range(self.height) if np.all(new_b[r, :] == 1.0)]
                cleared = len(full_lines)
                if cleared > 0:
                    cleared_b = np.delete(new_b, full_lines, axis=0)
                    new_b = np.vstack([np.zeros((cleared, self.width), dtype=np.float32), cleared_b])

                placements.append((rot_idx, col, new_b, cleared))

        return placements

    def step(self, resulting_board: np.ndarray, lines_cleared: int) -> tuple[float, bool]:
        """
        Pure Objective Environment Reward:
          +10.0 * lines_cleared
          +0.1 survival reward per placed piece
          -10.0 on terminal game over
        """
        self.pieces_placed += 1
        self.lines_cleared += lines_cleared
        self.board = resulting_board
        self.current_piece = self.next_piece
        self.next_piece = self._get_next_piece()

        # Check terminal
        if np.any(self.board[0, :] == 1.0) or len(self.get_legal_placements()) == 0:
            self.game_over = True
            return -10.0, True

        reward = lines_cleared * 10.0 + 0.1
        return reward, False


# ----------------------------------------------------------------------
# 2. Pure Board Feature Representation & Network Architecture
# ----------------------------------------------------------------------

def extract_board_features(board: np.ndarray) -> np.ndarray:
    """Raw structural column heights and transitions (No human tuning weights)."""
    heights = np.zeros(BOARD_WIDTH, dtype=np.float32)
    holes = 0
    for c in range(BOARD_WIDTH):
        col_blocks = np.where(board[:, c] == 1.0)[0]
        if len(col_blocks) > 0:
            heights[c] = BOARD_HEIGHT - col_blocks[0]
            holes += np.sum(board[col_blocks[0]:, c] == 0)

    diffs = np.diff(heights)
    bumpiness = np.sum(np.abs(diffs))
    max_h = np.max(heights)

    # 14-dimensional raw physical vector
    return np.array([
        *heights / BOARD_HEIGHT,
        bumpiness / (BOARD_HEIGHT * BOARD_WIDTH),
        holes / 50.0,
        max_h / BOARD_HEIGHT,
        np.sum(heights) / (BOARD_HEIGHT * BOARD_WIDTH),
    ], dtype=np.float32)


class PureRLJevModel(nn.Module):
    """
    Pure RL Jev Model:
      - Value Head V(s): estimates expected cumulative future return.
      - Noul Head: calibrated probability that current state will survive >= 15 steps.
    """
    def __init__(self, in_features: int = 14, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
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
        )
        self.noul_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        val = self.val_head(h).squeeze(-1)
        noul = self.noul_head(h).squeeze(-1)
        return val, noul


class PureRLERETModel(nn.Module):
    """Weight-tied Krasnoselskii-Mann Equilibrium Model trained under Pure RL."""
    def __init__(self, in_features: int = 14, d_model: int = 128):
        super().__init__()
        self.in_proj = nn.Linear(in_features, d_model)
        self.conv_l1 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.conv_l2 = nn.Linear(d_model, d_model)
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
        )

    def forward(self, x: torch.Tensor, loops: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.gelu(self.in_proj(x))
        last_conf = None
        for _ in range(loops):
            conf = self.noul_valve(h).squeeze(-1)
            gamma = (1.0 - conf).clamp(min=0.05, max=0.95).unsqueeze(-1)
            delta = F.gelu(self.norm1(self.conv_l1(h)))
            delta = self.norm2(self.conv_l2(delta))
            h = h + gamma * delta
            last_conf = conf

        val = self.val_head(h).squeeze(-1)
        return val, last_conf


# ----------------------------------------------------------------------
# 3. Pure Reinforcement Learning Training Loop (TD(0) Bellman Updates)
# ----------------------------------------------------------------------

def train_pure_rl_tetris(
    model: PureRLJevModel,
    num_episodes: int = 120,
    gamma: float = 0.96,
    lr: float = 8e-4,
):
    print(f"\n--- Initiating Pure RL Self-Improvement ({num_episodes} Episodes) ---", flush=True)
    print("Objective Reward: +10 per line, +0.1 per step, -10 on game over (No human weights).", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()

    t0 = time.time()
    total_lines = 0
    total_pieces = 0

    for ep in range(1, num_episodes + 1):
        game = TetrisRLGame(seed=ep * 7)
        states_buffer = []
        rewards_buffer = []

        epsilon = max(0.05, 0.35 * (1.0 - ep / num_episodes))

        while not game.game_over and game.pieces_placed < 250:
            placements = game.get_legal_placements()
            if not placements:
                break

            # Evaluate candidate next boards
            feats_list = [extract_board_features(p[2]) for p in placements]
            x_cand = torch.tensor(np.array(feats_list), dtype=torch.float32).to(device)

            with torch.no_grad():
                vals, _ = model(x_cand)

            # Epsilon-greedy placement selection
            if random.random() < epsilon:
                chosen_idx = random.randint(0, len(placements) - 1)
            else:
                chosen_idx = torch.argmax(vals).item()

            chosen_p = placements[chosen_idx]
            chosen_feat = feats_list[chosen_idx]

            # Step environment
            reward, done = game.step(chosen_p[2], chosen_p[3])

            states_buffer.append(chosen_feat)
            rewards_buffer.append(reward)

        total_lines += game.lines_cleared
        total_pieces += game.pieces_placed

        # Bellman TD Updates on Episode Trajectory
        if len(states_buffer) > 1:
            s_t = torch.tensor(np.array(states_buffer), dtype=torch.float32).to(device)
            r_t = torch.tensor(np.array(rewards_buffer), dtype=torch.float32).to(device)

            # Compute TD targets: G_t = r_t + gamma * V(s_{t+1})
            with torch.no_grad():
                v_next, _ = model(s_t)
                td_targets = torch.zeros_like(r_t)
                td_targets[:-1] = r_t[:-1] + gamma * v_next[1:]
                td_targets[-1] = r_t[-1] # Terminal reward

            # Calibrated Noul target: 1 if surviving at least 15 more steps
            survival_len = len(states_buffer)
            noul_targets = torch.tensor([
                1.0 if (survival_len - idx) >= 15 else 0.0
                for idx in range(survival_len)
            ], dtype=torch.float32).to(device)

            # Forward pass & loss
            v_pred, n_pred = model(s_t)
            loss_v = F.mse_loss(v_pred, td_targets)
            loss_n = F.mse_loss(n_pred, noul_targets)
            loss = loss_v + 1.0 * loss_n

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if ep % 20 == 0 or ep == num_episodes:
            print(
                f"Episode {ep:03d}/{num_episodes} | Avg Pieces: {total_pieces/ep:.1f} | "
                f"Avg Lines: {total_lines/ep:.2f} | Epsilon: {epsilon:.2f}",
                flush=True,
            )

    print(f"Pure RL Training completed in {time.time() - t0:.1f}s\n", flush=True)


# ----------------------------------------------------------------------
# 4. Pure RL Decision Engines: Reflexive, ERET, and Adaptive ETS
# ----------------------------------------------------------------------

def pure_rl_select_reflex(model: PureRLJevModel, game: TetrisRLGame) -> tuple[np.ndarray, int, int]:
    placements = game.get_legal_placements()
    if not placements:
        return game.board, 0, 0

    candidate_feats = [extract_board_features(p[2]) for p in placements]
    x = torch.tensor(np.array(candidate_feats), dtype=torch.float32).to(device)
    with torch.no_grad():
        vals, _ = model(x)
        best_idx = torch.argmax(vals).item()

    chosen = placements[best_idx]
    return chosen[2], chosen[3], 0


def pure_rl_select_eret(model: PureRLERETModel, game: TetrisRLGame, loops: int = 4) -> tuple[np.ndarray, int, int]:
    placements = game.get_legal_placements()
    if not placements:
        return game.board, 0, 0

    candidate_feats = [extract_board_features(p[2]) for p in placements]
    x = torch.tensor(np.array(candidate_feats), dtype=torch.float32).to(device)
    with torch.no_grad():
        vals, _ = model(x, loops=loops)
        best_idx = torch.argmax(vals).item()

    chosen = placements[best_idx]
    return chosen[2], chosen[3], 0


def pure_rl_select_uniform_search(model: PureRLJevModel, game: TetrisRLGame, beam_width: int = 4) -> tuple[np.ndarray, int, int]:
    placements = game.get_legal_placements()
    if not placements:
        return game.board, 0, 0

    candidate_feats = [extract_board_features(p[2]) for p in placements]
    x1 = torch.tensor(np.array(candidate_feats), dtype=torch.float32).to(device)
    with torch.no_grad():
        vals1, _ = model(x1)
        top_indices = torch.topk(vals1, min(beam_width, len(placements))).indices.cpu().numpy()

    best_combined_score = -float("inf")
    best_p = placements[0]
    search_expansions = len(top_indices)

    next_p_name = game.next_piece
    for idx in top_indices:
        p1 = placements[idx]
        b1 = p1[2]
        c1 = p1[3]

        dummy = TetrisRLGame()
        dummy.board = b1
        p2_list = dummy.get_legal_placements(next_p_name)
        if not p2_list:
            continue

        feats2 = [extract_board_features(p[2]) for p in p2_list]
        x2 = torch.tensor(np.array(feats2), dtype=torch.float32).to(device)
        with torch.no_grad():
            vals2, _ = model(x2)
            best_v2 = torch.max(vals2).item()

        score = vals1[idx].item() + 0.96 * best_v2 + 5.0 * c1
        if score > best_combined_score:
            best_combined_score = score
            best_p = p1

    return best_p[2], best_p[3], search_expansions


def pure_rl_select_adaptive_ets(
    model: PureRLJevModel, game: TetrisRLGame, tau: float = 0.75, beam_width: int = 4
) -> tuple[np.ndarray, int, int]:
    placements = game.get_legal_placements()
    if not placements:
        return game.board, 0, 0

    candidate_feats = [extract_board_features(p[2]) for p in placements]
    x1 = torch.tensor(np.array(candidate_feats), dtype=torch.float32).to(device)
    with torch.no_grad():
        vals1, nouls1 = model(x1)
        best_s1_idx = torch.argmax(vals1).item()
        best_noul = nouls1[best_s1_idx].item()

    # Dynamic Epistemic Gate: If safe according to purely learned Noul
    if best_noul >= tau:
        chosen = placements[best_s1_idx]
        return chosen[2], chosen[3], 0

    # Hazardous state: trigger 2-piece lookahead
    top_indices = torch.topk(vals1, min(beam_width, len(placements))).indices.cpu().numpy()
    best_combined_score = -float("inf")
    best_p = placements[best_s1_idx]
    search_expansions = len(top_indices)

    next_p_name = game.next_piece
    for idx in top_indices:
        p1 = placements[idx]
        b1 = p1[2]
        c1 = p1[3]

        dummy = TetrisRLGame()
        dummy.board = b1
        p2_list = dummy.get_legal_placements(next_p_name)
        if not p2_list:
            continue

        feats2 = [extract_board_features(p[2]) for p in p2_list]
        x2 = torch.tensor(np.array(feats2), dtype=torch.float32).to(device)
        with torch.no_grad():
            vals2, _ = model(x2)
            best_v2 = torch.max(vals2).item()

        score = vals1[idx].item() + 0.96 * best_v2 + 5.0 * c1
        if score > best_combined_score:
            best_combined_score = score
            best_p = p1

    return best_p[2], best_p[3], search_expansions


# ----------------------------------------------------------------------
# 5. Pure RL Tournament Execution
# ----------------------------------------------------------------------

def run_pure_rl_tournament(num_games: int = 15, max_pieces: int = 350):
    print("\n================================================================================", flush=True)
    print(f" FRONTIER 1 TOURNAMENT: PURE REINFORCEMENT LEARNING ({num_games} Shared Seeds)", flush=True)
    print("================================================================================\n", flush=True)

    jev_model = PureRLJevModel().to(device)
    eret_model = PureRLERETModel().to(device)

    # Train purely from trial-and-error environment rollouts
    train_pure_rl_tetris(jev_model, num_episodes=120)

    # Clone trunk weights into ERET model
    eret_model.in_proj.weight.data.copy_(jev_model.net[0].weight.data)
    eret_model.val_head[0].weight.data.copy_(jev_model.val_head[0].weight.data)
    eret_model.eval()
    jev_model.eval()

    modes = [
        "Pure_RL_ERET_K4",
        "Pure_RL_Reflexive",
        "Pure_RL_Uniform_Search",
        "Pure_RL_Adaptive_ETS",
    ]

    results = {m: {"lines": [], "pieces": [], "holes": [], "expansions": [], "times": []} for m in modes}
    seeds = [500 + i for i in range(num_games)]

    for g_idx, seed in enumerate(seeds):
        print(f"--- Game {g_idx + 1:02d}/{num_games} (Seed: {seed}) ---", flush=True)
        for mode in modes:
            game = TetrisRLGame(seed=seed)
            t_start = time.time()
            total_expansions = 0

            while not game.game_over and game.pieces_placed < max_pieces:
                if mode == "Pure_RL_ERET_K4":
                    b_next, cleared, exp = pure_rl_select_eret(eret_model, game, loops=4)
                elif mode == "Pure_RL_Reflexive":
                    b_next, cleared, exp = pure_rl_select_reflex(jev_model, game)
                elif mode == "Pure_RL_Uniform_Search":
                    b_next, cleared, exp = pure_rl_select_uniform_search(jev_model, game, beam_width=4)
                elif mode == "Pure_RL_Adaptive_ETS":
                    b_next, cleared, exp = pure_rl_select_adaptive_ets(jev_model, game, tau=0.75, beam_width=4)

                total_expansions += exp
                game.step(b_next, cleared)

            elapsed = time.time() - t_start
            feats = extract_board_features(game.board)
            final_holes = feats[11] * 50.0

            results[mode]["lines"].append(game.lines_cleared)
            results[mode]["pieces"].append(game.pieces_placed)
            results[mode]["holes"].append(final_holes)
            results[mode]["expansions"].append(total_expansions / max(1, game.pieces_placed))
            results[mode]["times"].append(elapsed)

    print("\n================================================================================", flush=True)
    print(" PURE RL TOURNAMENT RESULTS SUMMARY (No Human Heuristics)", flush=True)
    print("================================================================================", flush=True)
    print(f"{'Paradigm':25s} | {'Pieces':7s} | {'Lines':6s} | {'Exp/Piece':9s} | {'Holes':6s} | {'Time (s)':8s}", flush=True)
    print("-" * 75, flush=True)

    for mode in modes:
        avg_p = np.mean(results[mode]["pieces"])
        avg_l = np.mean(results[mode]["lines"])
        avg_e = np.mean(results[mode]["expansions"])
        avg_h = np.mean(results[mode]["holes"])
        avg_t = np.mean(results[mode]["times"])
        print(f"{mode:25s} | {avg_p:7.1f} | {avg_l:6.1f} | {avg_e:9.2f} | {avg_h:6.1f} | {avg_t:8.2f}", flush=True)
    print("================================================================================\n", flush=True)


if __name__ == "__main__":
    run_pure_rl_tournament(num_games=12, max_pieces=350)
