"""
Tetris Epistemic Arena: The Compute-Bounded Survival Benchmark
=============================================================
Authors: Leon & Ilya Sutskever persona

The Fundamental Question:
  In a fast, time-sensitive environment with combinatorial branching,
  does an unrolled latent equilibrium (ERET) survive, or does Epistemic Tree Search
  (System 1 Jev Evaluator + System 2 Adaptive Lookahead) dominate?

Architectures Evaluated:
  1. ERET Latent Loop: Unrolls K=4 Krasnoselskii-Mann latent iterations.
  2. S1 Reflexive Jev: Single-pass parallel placement evaluation (0 search expansions).
  3. Uniform Search: Fixed 2-piece lookahead search (evaluates current x next piece combinations).
  4. Adaptive Epistemic Search (ETS): Reflexive on safe drops (Noul >= tau), expands 2-piece
     search only when placement risk is high (Noul < tau).
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

# Standard 7 Tetromino Shapes (represented as sets of (r, c) offsets)
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
# 1. Fast Vectorized Tetris Simulator
# ----------------------------------------------------------------------

class TetrisGame:
    def __init__(self, seed: int | None = None):
        self.width = BOARD_WIDTH
        self.height = BOARD_HEIGHT
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.board = np.zeros((self.height, self.width), dtype=np.int32)
        self.bag = []
        self.current_piece = self._get_next_piece()
        self.next_piece = self._get_next_piece()
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.game_over = False
        return self.board

    def _get_next_piece(self) -> str:
        if not self.bag:
            self.bag = list(PIECE_NAMES)
            self.rng.shuffle(self.bag)
        return self.bag.pop()

    def get_legal_placements(self, piece_name: str | None = None) -> list[tuple[int, int, np.ndarray, int, int]]:
        """
        Returns all legal landing positions for the given piece:
        Each entry: (rot_idx, col_offset, resulting_board, lines_cleared, holes_created)
        """
        p_name = piece_name if piece_name is not None else self.current_piece
        rotations = TETROMINOES[p_name]
        placements = []

        for rot_idx, shape in enumerate(rotations):
            min_c = min(c for r, c in shape)
            max_c = max(c for r, c in shape)
            shape_w = max_c - min_c + 1

            for col in range(0, self.width - shape_w + 1):
                # Shift shape so min_c aligns with col
                shifted = [(r, c - min_c + col) for r, c in shape]

                # Drop shape to lowest valid row
                drop_r = 0
                while True:
                    test_cells = [(r + drop_r + 1, c) for r, c in shifted]
                    # Check if any test cell collides with floor or existing block
                    collision = False
                    for tr, tc in test_cells:
                        if tr >= self.height or self.board[tr, tc] == 1:
                            collision = True
                            break
                    if collision:
                        break
                    drop_r += 1

                # If drop_r is at top, game might be blocked
                landing_cells = [(r + drop_r, c) for r, c in shifted]
                if any(r < 0 or r >= self.height or self.board[r, c] == 1 for r, c in landing_cells):
                    continue

                # Simulate resulting board
                new_b = np.copy(self.board)
                for r, c in landing_cells:
                    new_b[r, c] = 1

                # Count and clear full lines
                full_lines = [r for r in range(self.height) if np.all(new_b[r, :] == 1)]
                cleared = len(full_lines)
                if cleared > 0:
                    cleared_b = np.delete(new_b, full_lines, axis=0)
                    new_b = np.vstack([np.zeros((cleared, self.width), dtype=np.int32), cleared_b])

                # Count holes
                holes = 0
                for c in range(self.width):
                    col_blocks = np.where(new_b[:, c] == 1)[0]
                    if len(col_blocks) > 0:
                        top_r = col_blocks[0]
                        holes += np.sum(new_b[top_r:, c] == 0)

                placements.append((rot_idx, col, new_b, cleared, holes))

        return placements

    def step(self, resulting_board: np.ndarray, lines_cleared: int):
        self.pieces_placed += 1
        self.lines_cleared += lines_cleared
        self.board = resulting_board
        self.current_piece = self.next_piece
        self.next_piece = self._get_next_piece()

        # Check game over: if top row has any block
        if np.any(self.board[0, :] == 1):
            self.game_over = True

        # Check if next piece has legal placements
        if not self.game_over and len(self.get_legal_placements()) == 0:
            self.game_over = True


# ----------------------------------------------------------------------
# 2. Neural System 1 Evaluator & ERET Looped Baseline
# ----------------------------------------------------------------------

def extract_board_features(board: np.ndarray) -> np.ndarray:
    """Extracts structural features: column heights, bumpiness, holes, max height."""
    heights = np.zeros(BOARD_WIDTH, dtype=np.float32)
    holes = 0
    for c in range(BOARD_WIDTH):
        col_blocks = np.where(board[:, c] == 1)[0]
        if len(col_blocks) > 0:
            heights[c] = BOARD_HEIGHT - col_blocks[0]
            holes += np.sum(board[col_blocks[0]:, c] == 0)

    bumpiness = np.sum(np.abs(np.diff(heights)))
    max_h = np.max(heights)
    agg_h = np.sum(heights)
    # Normalized feature vector
    feats = np.array([
        *heights / BOARD_HEIGHT,
        bumpiness / (BOARD_HEIGHT * BOARD_WIDTH),
        holes / 50.0,
        max_h / BOARD_HEIGHT,
        agg_h / (BOARD_HEIGHT * BOARD_WIDTH),
    ], dtype=np.float32)
    return feats


class JevTetrisEvaluator(nn.Module):
    """
    TypeSafe Jev System 1 Model for Tetris:
    Parallel evaluator mapping candidate board state to:
      1. Score / Quality V(s) in [0, 1] (Higher is better board survival)
      2. Calibrated Noul(s) in [0, 1] (Probability of zero hole formation / safety)
    """
    def __init__(self, in_features: int = 14, d_model: int = 128):
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


class ERETTetrisLooped(nn.Module):
    """Weight-tied Krasnoselskii-Mann Recurrent Equilibrium Model for Tetris."""
    def __init__(self, in_features: int = 14, d_model: int = 128, max_loops: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(in_features, d_model)
        self.loop_fc1 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.loop_fc2 = nn.Linear(d_model, d_model)
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
        self.max_loops = max_loops

    def forward(self, x: torch.Tensor, loops: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.gelu(self.in_proj(x))
        last_conf = None
        for _ in range(loops):
            conf = self.noul_valve(h).squeeze(-1)
            gamma = (1.0 - conf).clamp(min=0.05, max=0.95).unsqueeze(-1)
            delta = F.gelu(self.norm1(self.loop_fc1(h)))
            delta = self.norm2(self.loop_fc2(delta))
            h = h + gamma * delta
            last_conf = conf

        val = self.val_head(h).squeeze(-1)
        return val, last_conf


# ----------------------------------------------------------------------
# 3. Training the Evaluator on Heuristic Expert Placements
# ----------------------------------------------------------------------

def train_jev_tetris_model(model: JevTetrisEvaluator, num_samples: int = 5000, epochs: int = 8):
    """Trains the Jev evaluator on simulated board states with proper scoring rules."""
    print("Generating Tetris state-evaluation training set...", flush=True)
    states = []
    values = []
    nouls = []

    game = TetrisGame(seed=42)
    while len(states) < num_samples:
        placements = game.get_legal_placements()
        if not placements or game.game_over:
            game.reset()
            continue

        for _, _, b_res, cleared, holes in placements:
            feats = extract_board_features(b_res)
            # True heuristic score (Dellacherie-inspired ground truth):
            # Penalize holes, bumpiness, max height; reward cleared lines
            max_h = feats[12] * BOARD_HEIGHT
            bump = feats[10] * (BOARD_HEIGHT * BOARD_WIDTH)
            hole_cnt = feats[11] * 50.0

            # Score in [0, 1]
            raw_score = 1.0 - (0.04 * max_h + 0.08 * bump + 0.25 * hole_cnt) + 0.20 * cleared
            val = float(np.clip(raw_score, 0.01, 0.99))
            # Noul = binary probability that hole count is 0
            noul = 1.0 if hole_cnt == 0 else 0.0

            states.append(feats)
            values.append(val)
            nouls.append(noul)

            if len(states) >= num_samples:
                break

        # Step random placement
        chosen = random.choice(placements)
        game.step(chosen[2], chosen[3])

    s_t = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    v_t = torch.tensor(np.array(values), dtype=torch.float32).to(device)
    n_t = torch.tensor(np.array(nouls), dtype=torch.float32).to(device)

    dataset = torch.utils.data.TensorDataset(s_t, v_t, n_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()

    print(f"Training Jev Tetris Model on {len(states):,} states...", flush=True)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for b_s, b_v, b_n in loader:
            v_pred, n_pred = model(b_s)
            loss_v = F.mse_loss(v_pred, b_v)
            loss_brier = F.mse_loss(n_pred, b_n)
            loss = loss_v + 1.0 * loss_brier

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 2 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs} | Training Loss: {total_loss/len(loader):.4f}", flush=True)
    print(f"Training complete in {time.time() - t0:.1f}s\n", flush=True)


# ----------------------------------------------------------------------
# 4. Decision Paradigms Under Comparison
# ----------------------------------------------------------------------

def select_action_s1_greedy(model: JevTetrisEvaluator, game: TetrisGame) -> tuple[np.ndarray, int, int]:
    """Reflexive System 1: Evaluates all 1st-piece placements in a single parallel pass."""
    placements = game.get_legal_placements()
    if not placements:
        return game.board, 0, 0

    candidate_feats = [extract_board_features(p[2]) for p in placements]
    x = torch.tensor(np.array(candidate_feats), dtype=torch.float32).to(device)
    with torch.no_grad():
        vals, _ = model(x)
        best_idx = torch.argmax(vals).item()

    chosen = placements[best_idx]
    return chosen[2], chosen[3], 0 # 0 search expansions


def select_action_eret(model: ERETTetrisLooped, game: TetrisGame, loops: int = 4) -> tuple[np.ndarray, int, int]:
    """ERET Latent Equilibrium: Unrolls Krasnoselskii-Mann loop on candidates."""
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


def select_action_uniform_search(model: JevTetrisEvaluator, game: TetrisGame, beam_width: int = 4) -> tuple[np.ndarray, int, int]:
    """Uniform System 2 Search: Explores current piece x next piece lookahead tree."""
    placements = game.get_legal_placements()
    if not placements:
        return game.board, 0, 0

    # 1. Evaluate 1st piece candidates
    candidate_feats = [extract_board_features(p[2]) for p in placements]
    x1 = torch.tensor(np.array(candidate_feats), dtype=torch.float32).to(device)
    with torch.no_grad():
        vals1, _ = model(x1)
        top_indices = torch.topk(vals1, min(beam_width, len(placements))).indices.cpu().numpy()

    best_combined_score = -float("inf")
    best_p = placements[0]
    search_expansions = len(top_indices)

    # 2. Lookahead into next piece
    next_p_name = game.next_piece
    for idx in top_indices:
        p1 = placements[idx]
        b1 = p1[2]
        c1 = p1[3]

        # Get legal placements for next piece on board b1
        dummy_game = TetrisGame()
        dummy_game.board = b1
        p2_list = dummy_game.get_legal_placements(next_p_name)
        if not p2_list:
            continue

        feats2 = [extract_board_features(p[2]) for p in p2_list]
        x2 = torch.tensor(np.array(feats2), dtype=torch.float32).to(device)
        with torch.no_grad():
            vals2, _ = model(x2)
            best_v2 = torch.max(vals2).item()

        score = vals1[idx].item() + 1.2 * best_v2 + 0.5 * c1
        if score > best_combined_score:
            best_combined_score = score
            best_p = p1

    return best_p[2], best_p[3], search_expansions


def select_action_adaptive_ets(
    model: JevTetrisEvaluator, game: TetrisGame, tau: float = 0.85, beam_width: int = 4
) -> tuple[np.ndarray, int, int]:
    """
    Adaptive Epistemic Tree Search (ETS / Jev-Search):
      1. Evaluates 1st piece candidates.
      2. If best placement has Noul >= tau (safe, 0 holes, clean): Takes reflexively (0 expansions).
      3. If Noul < tau (hazard detected): Unrolls next-piece lookahead tree!
    """
    placements = game.get_legal_placements()
    if not placements:
        return game.board, 0, 0

    candidate_feats = [extract_board_features(p[2]) for p in placements]
    x1 = torch.tensor(np.array(candidate_feats), dtype=torch.float32).to(device)
    with torch.no_grad():
        vals1, nouls1 = model(x1)
        best_s1_idx = torch.argmax(vals1).item()
        best_noul = nouls1[best_s1_idx].item()

    # Dynamic Epistemic Gate:
    if best_noul >= tau:
        # Safe placement: zero search expansions
        chosen = placements[best_s1_idx]
        return chosen[2], chosen[3], 0

    # Epistemic Hazard: Trigger System 2 Tree Search
    top_indices = torch.topk(vals1, min(beam_width, len(placements))).indices.cpu().numpy()
    best_combined_score = -float("inf")
    best_p = placements[best_s1_idx]
    search_expansions = len(top_indices)

    next_p_name = game.next_piece
    for idx in top_indices:
        p1 = placements[idx]
        b1 = p1[2]
        c1 = p1[3]

        dummy_game = TetrisGame()
        dummy_game.board = b1
        p2_list = dummy_game.get_legal_placements(next_p_name)
        if not p2_list:
            continue

        feats2 = [extract_board_features(p[2]) for p in p2_list]
        x2 = torch.tensor(np.array(feats2), dtype=torch.float32).to(device)
        with torch.no_grad():
            vals2, _ = model(x2)
            best_v2 = torch.max(vals2).item()

        score = vals1[idx].item() + 1.2 * best_v2 + 0.5 * c1
        if score > best_combined_score:
            best_combined_score = score
            best_p = p1

    return best_p[2], best_p[3], search_expansions


# ----------------------------------------------------------------------
# 5. Tournament Runner
# ----------------------------------------------------------------------

def run_tetris_tournament(num_games: int = 15, max_pieces_per_game: int = 400):
    print(f"\n================================================================================", flush=True)
    print(f" TETRIS ARENA: TIME-BOUNDED SURVIVAL TOURNAMENT ({num_games} Shared Seeds)", flush=True)
    print(f"================================================================================\n", flush=True)

    jev_model = JevTetrisEvaluator().to(device)
    eret_model = ERETTetrisLooped().to(device)

    train_jev_tetris_model(jev_model, num_samples=6000, epochs=8)

    # Initialize ERET weights to match
    eret_model.in_proj.weight.data.copy_(jev_model.fc[0].weight.data)
    eret_model.val_head[0].weight.data.copy_(jev_model.val_head[0].weight.data)
    eret_model.eval()
    jev_model.eval()

    modes = [
        "ERET_K4 (Latent Loop)",
        "S1_Greedy (Reflexive)",
        "Uniform_Search (Lookahead)",
        "Adaptive_ETS (tau=0.80)",
        "Adaptive_ETS (tau=0.90)",
    ]

    results = {m: {"lines": [], "pieces": [], "holes": [], "expansions": [], "times": []} for m in modes}

    seeds = [100 + i for i in range(num_games)]

    for g_idx, seed in enumerate(seeds):
        print(f"--- Game {g_idx + 1:02d}/{num_games} (Seed: {seed}) ---", flush=True)

        for mode in modes:
            game = TetrisGame(seed=seed)
            t_start = time.time()
            total_expansions = 0

            while not game.game_over and game.pieces_placed < max_pieces_per_game:
                if mode == "ERET_K4 (Latent Loop)":
                    b_next, cleared, exp = select_action_eret(eret_model, game, loops=4)
                elif mode == "S1_Greedy (Reflexive)":
                    b_next, cleared, exp = select_action_s1_greedy(jev_model, game)
                elif mode == "Uniform_Search (Lookahead)":
                    b_next, cleared, exp = select_action_uniform_search(jev_model, game, beam_width=4)
                elif mode == "Adaptive_ETS (tau=0.80)":
                    b_next, cleared, exp = select_action_adaptive_ets(jev_model, game, tau=0.80, beam_width=4)
                elif mode == "Adaptive_ETS (tau=0.90)":
                    b_next, cleared, exp = select_action_adaptive_ets(jev_model, game, tau=0.90, beam_width=4)

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
    print(" TOURNAMENT RESULTS SUMMARY (Averaged across games)", flush=True)
    print("================================================================================", flush=True)
    print(f"{'Paradigm':28s} | {'Pieces':7s} | {'Lines':6s} | {'Exp/Piece':9s} | {'Holes':6s} | {'Time (s)':8s}", flush=True)
    print("-" * 75, flush=True)

    for mode in modes:
        avg_p = np.mean(results[mode]["pieces"])
        avg_l = np.mean(results[mode]["lines"])
        avg_e = np.mean(results[mode]["expansions"])
        avg_h = np.mean(results[mode]["holes"])
        avg_t = np.mean(results[mode]["times"])
        print(f"{mode:28s} | {avg_p:7.1f} | {avg_l:6.1f} | {avg_e:9.2f} | {avg_h:6.1f} | {avg_t:8.2f}", flush=True)
    print("================================================================================\n", flush=True)
    return results


if __name__ == "__main__":
    run_tetris_tournament(num_games=12, max_pieces_per_game=350)
