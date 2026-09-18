"""
Frontier 2+: Harder 6x6 Bullet Chess Arena (Strict 1.5s Clock)
=============================================================
Authors: Leon & Ilya Sutskever persona

The Fundamental Question:
  In an expanded 6x6 board with 36 squares, full piece complement (R, N, B, Q, K, R),
  and a punishing 1.5-second total chess clock, can Adaptive Epistemic Tree Search (ETS)
  outplay both fixed-depth minimax searchers (who collapse to time flags) and pure
  reflexive heuristics (who blunder tactically)?

Board Layout (6x6):
  Rank 0 (Black): r, n, b, q, k, r
  Rank 1 (Black): p, p, p, p, p, p
  Rank 2: . . . . . .
  Rank 3: . . . . . .
  Rank 4 (White): P, P, P, P, P, P
  Rank 5 (White): R, N, B, Q, K, R
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BOARD_SIZE = 6

EMPTY = 0
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6

PIECE_VALUES = {
    PAWN: 100,
    KNIGHT: 300,
    BISHOP: 325,
    ROOK: 500,
    QUEEN: 900,
    KING: 10000,
}

# Positional piece-square tables for 6x6
PST_PAWN_WHITE = np.array([
    [90, 90, 90, 90, 90, 90],
    [50, 50, 50, 50, 50, 50],
    [10, 15, 25, 25, 15, 10],
    [ 5, 10, 20, 20, 10,  5],
    [ 0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0],
])

PST_KNIGHT = np.array([
    [-10, -5, -5, -5, -5, -10],
    [ -5,  5, 10, 10,  5,  -5],
    [ -5, 10, 20, 20, 10,  -5],
    [ -5, 10, 20, 20, 10,  -5],
    [ -5,  5, 10, 10,  5,  -5],
    [-10, -5, -5, -5, -5, -10],
])


# ----------------------------------------------------------------------
# 1. 6x6 Chess Engine
# ----------------------------------------------------------------------

class Chess6x6:
    def __init__(self):
        self.reset()

    def reset(self):
        # Starting layout (White at row 4-5, Black at row 0-1)
        self.board = np.array([
            [-ROOK, -KNIGHT, -BISHOP, -QUEEN, -KING, -ROOK],
            [-PAWN, -PAWN,   -PAWN,   -PAWN,  -PAWN, -PAWN],
            [EMPTY,  EMPTY,   EMPTY,   EMPTY,  EMPTY, EMPTY],
            [EMPTY,  EMPTY,   EMPTY,   EMPTY,  EMPTY, EMPTY],
            [PAWN,   PAWN,    PAWN,    PAWN,   PAWN,  PAWN],
            [ROOK,   KNIGHT,  BISHOP,  QUEEN,  KING,  ROOK],
        ], dtype=np.int32)
        self.turn = 1  # 1: White, -1: Black
        self.move_count = 0
        self.game_over = False
        self.winner = 0  # 1: White, -1: Black, 0: Draw
        return self.board

    def clone(self) -> Chess6x6:
        other = Chess6x6()
        other.board = np.copy(self.board)
        other.turn = self.turn
        other.move_count = self.move_count
        other.game_over = self.game_over
        other.winner = self.winner
        return other

    def get_legal_moves(self, side: int | None = None) -> list[tuple[int, int, int, int]]:
        current_side = side if side is not None else self.turn
        moves = []

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = self.board[r, c]
                if p * current_side <= 0:
                    continue

                piece_type = abs(p)

                # 1. Pawns
                if piece_type == PAWN:
                    fwd = -1 if current_side == 1 else 1
                    nr = r + fwd
                    # Single step
                    if 0 <= nr < BOARD_SIZE and self.board[nr, c] == EMPTY:
                        moves.append((r, c, nr, c))
                        # Double step from initial row
                        start_row = 4 if current_side == 1 else 1
                        nnr = r + 2 * fwd
                        if r == start_row and 0 <= nnr < BOARD_SIZE and self.board[nnr, c] == EMPTY:
                            moves.append((r, c, nnr, c))
                    # Diagonal captures
                    for dc in [-1, 1]:
                        nc = c + dc
                        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                            tgt = self.board[nr, nc]
                            if tgt * current_side < 0:
                                moves.append((r, c, nr, nc))

                # 2. Knights
                elif piece_type == KNIGHT:
                    jumps = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                             (1, -2), (1, 2), (2, -1), (2, 1)]
                    for dr, dc in jumps:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                            if self.board[nr, nc] * current_side <= 0:
                                moves.append((r, c, nr, nc))

                # 3. Bishops
                elif piece_type == BISHOP:
                    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        nr, nc = r + dr, c + dc
                        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                            tgt = self.board[nr, nc]
                            if tgt == EMPTY:
                                moves.append((r, c, nr, nc))
                            elif tgt * current_side < 0:
                                moves.append((r, c, nr, nc))
                                break
                            else:
                                break
                            nr += dr
                            nc += dc

                # 4. Rooks
                elif piece_type == ROOK:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                            tgt = self.board[nr, nc]
                            if tgt == EMPTY:
                                moves.append((r, c, nr, nc))
                            elif tgt * current_side < 0:
                                moves.append((r, c, nr, nc))
                                break
                            else:
                                break
                            nr += dr
                            nc += dc

                # 5. Queen
                elif piece_type == QUEEN:
                    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                            tgt = self.board[nr, nc]
                            if tgt == EMPTY:
                                moves.append((r, c, nr, nc))
                            elif tgt * current_side < 0:
                                moves.append((r, c, nr, nc))
                                break
                            else:
                                break
                            nr += dr
                            nc += dc

                # 6. King
                elif piece_type == KING:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                                if self.board[nr, nc] * current_side <= 0:
                                    moves.append((r, c, nr, nc))

        return moves

    def make_move(self, move: tuple[int, int, int, int]) -> int:
        r1, c1, r2, c2 = move
        piece = self.board[r1, c1]
        captured = self.board[r2, c2]

        self.board[r1, c1] = EMPTY
        # Promotion to Queen
        if abs(piece) == PAWN and ((self.turn == 1 and r2 == 0) or (self.turn == -1 and r2 == BOARD_SIZE - 1)):
            self.board[r2, c2] = QUEEN if self.turn == 1 else -QUEEN
        else:
            self.board[r2, c2] = piece

        self.move_count += 1

        # Check for King capture (instant terminal win)
        if abs(captured) == KING:
            self.game_over = True
            self.winner = self.turn
            return captured

        # Draw by move limit
        if self.move_count >= 50:
            self.game_over = True
            self.winner = 0
            return captured

        self.turn = -self.turn
        return captured

    def evaluate_static(self) -> float:
        """Returns heuristic score from perspective of White (+ favors White, - favors Black)."""
        white_score = 0.0
        black_score = 0.0

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = self.board[r, c]
                if p == EMPTY:
                    continue
                val = PIECE_VALUES[abs(p)]
                pos = 0.0
                if abs(p) == PAWN:
                    pos = PST_PAWN_WHITE[r, c] if p > 0 else PST_PAWN_WHITE[BOARD_SIZE - 1 - r, c]
                elif abs(p) == KNIGHT:
                    pos = PST_KNIGHT[r, c]

                if p > 0:
                    white_score += (val + pos)
                else:
                    black_score += (val + pos)

        return (white_score - black_score) / 100.0


# ----------------------------------------------------------------------
# 2. Neural Architecture: 6x6 Jev Calibrated Evaluator
# ----------------------------------------------------------------------

def encode_board_tensor(board: np.ndarray, turn: int) -> torch.Tensor:
    """13 x 6 x 6 tensor representation: 6 piece types * 2 colors + 1 turn plane."""
    t = np.zeros((13, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = board[r, c]
            if p > 0:
                t[p - 1, r, c] = 1.0
            elif p < 0:
                t[abs(p) + 5, r, c] = 1.0
    if turn == 1:
        t[12, :, :] = 1.0
    return torch.tensor(t, dtype=torch.float32)


class JevChess6x6Evaluator(nn.Module):
    """
    TypeSafe Jev Evaluator for 6x6 Chess:
      Outputs:
        - Value V(s) in [-1, 1]: estimated win probability advantage.
        - Noul(s) in [0, 1]: epistemic certainty / tactical quietness.
          (Low Noul = tactical crisis, piece hangs, search required).
    """
    def __init__(self, in_channels: int = 13, num_filters: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.GELU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.GELU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(num_filters * BOARD_SIZE * BOARD_SIZE, 128),
            nn.GELU(),
        )
        self.val_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
        self.noul_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        val = self.val_head(h).squeeze(-1)
        noul = self.noul_head(h).squeeze(-1)
        return val, noul


def train_jev_chess6x6_model(model: JevChess6x6Evaluator, num_samples: int = 3500, epochs: int = 5):
    print("Generating 6x6 chess positions for Jev training...", flush=True)
    tensors, values, nouls = [], [], []

    for _ in range(num_samples):
        game = Chess6x6()
        # Random playout of 1-15 plies
        for _ in range(random.randint(1, 15)):
            if game.game_over:
                break
            moves = game.get_legal_moves()
            if not moves:
                break
            game.make_move(random.choice(moves))

        score = game.evaluate_static()
        val = float(math.tanh(score / 5.0))

        # Tactical volatility: check if high value captures are immediately available
        moves = game.get_legal_moves()
        tactical = False
        for m in moves:
            captured = abs(game.board[m[2], m[3]])
            if captured in [KNIGHT, BISHOP, ROOK, QUEEN, KING]:
                tactical = True
                break

        # Calm position = high Noul; Tactical crisis = low Noul
        noul = 0.25 if tactical else 0.88

        t = encode_board_tensor(game.board, game.turn)
        tensors.append(t)
        values.append(val)
        nouls.append(noul)

    x_t = torch.stack(tensors).to(device)
    v_t = torch.tensor(values, dtype=torch.float32).to(device)
    n_t = torch.tensor(nouls, dtype=torch.float32).to(device)

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

    print(f"Jev 6x6 Chess Evaluator trained on {len(tensors):,} positions in {time.time() - t0:.1f}s\n", flush=True)


# ----------------------------------------------------------------------
# 3. Negamax Search Engine & Players
# ----------------------------------------------------------------------

def negamax(game: Chess6x6, depth: int, alpha: float, beta: float, side: int) -> float:
    if depth == 0 or game.game_over:
        if game.game_over:
            return 1000.0 if game.winner == side else -1000.0
        return side * game.evaluate_static()

    max_eval = -10000.0
    moves = game.get_legal_moves()
    if not moves:
        return 0.0

    for m in moves:
        child = game.clone()
        child.make_move(m)
        score = -negamax(child, depth - 1, -beta, -alpha, -side)
        max_eval = max(max_eval, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return max_eval


def select_move_reflexive_jev(game: Chess6x6, model: JevChess6x6Evaluator) -> tuple[int, int, int, int]:
    """Reflexive System 1 Jev: 1-ply evaluation in ~0.5ms."""
    legal = game.get_legal_moves()
    if not legal:
        return (0, 0, 0, 0)

    tensors = []
    for m in legal:
        child = game.clone()
        child.make_move(m)
        tensors.append(encode_board_tensor(child.board, child.turn))

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        vals, _ = model(batch)
        if game.turn == 1:
            best_idx = torch.argmax(vals).item()
        else:
            best_idx = torch.argmin(vals).item()
    return legal[best_idx]


def select_move_fixed_minimax(game: Chess6x6, depth: int) -> tuple[int, int, int, int]:
    """Fixed-Depth Minimax Searcher."""
    legal = game.get_legal_moves()
    if not legal:
        return (0, 0, 0, 0)

    best_score = -10000.0
    best_move = legal[0]
    alpha = -10000.0
    beta = 10000.0

    for m in legal:
        child = game.clone()
        child.make_move(m)
        score = -negamax(child, depth - 1, -beta, -alpha, -game.turn)
        if score > best_score:
            best_score = score
            best_move = m
        alpha = max(alpha, score)
    return best_move


def select_move_adaptive_ets(
    game: Chess6x6,
    model: JevChess6x6Evaluator,
    clock_remaining: float,
    tau: float = 0.70,
) -> tuple[tuple[int, int, int, int], int]:
    """
    Adaptive Epistemic Tree Search (ETS) for 6x6 Bullet Chess:
      1. S1 Jev assesses position (V, Noul).
      2. If Noul >= tau (quiet) OR clock < 0.35s (time pressure):
         -> Reflexive 1-ply move in 0.5ms. Saves clock!
      3. If Noul < tau (tactical crisis) AND clock >= 0.35s:
         -> Dynamic search:
            - If clock > 0.8s: Depth 3 Minimax search
            - Else: Depth 2 Minimax search
    """
    t_s = encode_board_tensor(game.board, game.turn).unsqueeze(0).to(device)
    with torch.no_grad():
        val, noul = model(t_s)
    noul_score = noul.item()

    if noul_score >= tau or clock_remaining < 0.35:
        # Move reflexively
        m = select_move_reflexive_jev(game, model)
        return m, 1

    # Tactical Crisis: allocate search depth dynamically
    depth = 3 if clock_remaining > 0.80 else 2
    m = select_move_fixed_minimax(game, depth=depth)
    return m, depth


# ----------------------------------------------------------------------
# 4. Tournament Execution
# ----------------------------------------------------------------------

def play_bullet_game_6x6(white_type: str, black_type: str, model: JevChess6x6Evaluator, clock_limit: float = 1.5):
    game = Chess6x6()
    white_clock = clock_limit
    black_clock = clock_limit

    while not game.game_over:
        legal = game.get_legal_moves()
        if not legal:
            break

        is_white = (game.turn == 1)
        cur_type = white_type if is_white else black_type
        clk = white_clock if is_white else black_clock

        t0 = time.time()
        if cur_type == "Reflexive Jev":
            move = select_move_reflexive_jev(game, model)
        elif cur_type == "Fixed Depth 2":
            move = select_move_fixed_minimax(game, depth=2)
        elif cur_type == "Fixed Depth 3":
            move = select_move_fixed_minimax(game, depth=3)
        elif cur_type == "Adaptive ETS":
            move, _ = select_move_adaptive_ets(game, model, clk, tau=0.70)
        else:
            move = random.choice(legal)
        elapsed = time.time() - t0

        if is_white:
            white_clock -= elapsed
            if white_clock <= 0.0:
                return -1, "Black (Time Forfeit)", white_clock, black_clock
        else:
            black_clock -= elapsed
            if black_clock <= 0.0:
                return 1, "White (Time Forfeit)", white_clock, black_clock

        game.make_move(move)

    # Game completed within clock
    if game.winner == 1:
        res_str = "White (Checkmate)"
    elif game.winner == -1:
        res_str = "Black (Checkmate)"
    else:
        res_str = "Draw"
    return game.winner, res_str, white_clock, black_clock


def run_6x6_chess_tournament(model: JevChess6x6Evaluator, games_per_pair: int = 10):
    print(f"==========================================================================")
    print(f"6x6 BULLET CHESS TOURNAMENT (Clock: 1.5s per player | {games_per_pair} games/matchup)")
    print(f"==========================================================================")

    competitors = ["Reflexive Jev", "Fixed Depth 2", "Fixed Depth 3", "Adaptive ETS"]
    matchups = [
        ("Adaptive ETS", "Fixed Depth 3"),
        ("Adaptive ETS", "Fixed Depth 2"),
        ("Adaptive ETS", "Reflexive Jev"),
        ("Fixed Depth 3", "Reflexive Jev"),
        ("Fixed Depth 2", "Reflexive Jev"),
    ]

    results = {}

    for p1, p2 in matchups:
        print(f"\n--- Matchup: {p1} vs {p2} ---")
        p1_wins = 0
        p2_wins = 0
        draws = 0
        p1_flags = 0
        p2_flags = 0

        for g in range(games_per_pair):
            # Alternate colors
            if g % 2 == 0:
                w_agent, b_agent = p1, p2
                winner, res_str, w_clk, b_clk = play_bullet_game_6x6(w_agent, b_agent, model, clock_limit=1.5)
                if winner == 1:
                    p1_wins += 1
                elif winner == -1:
                    p2_wins += 1
                else:
                    draws += 1
                if "Time Forfeit" in res_str:
                    if winner == 1:
                        p2_flags += 1
                    else:
                        p1_flags += 1
            else:
                w_agent, b_agent = p2, p1
                winner, res_str, w_clk, b_clk = play_bullet_game_6x6(w_agent, b_agent, model, clock_limit=1.5)
                if winner == 1:
                    p2_wins += 1
                elif winner == -1:
                    p1_wins += 1
                else:
                    draws += 1
                if "Time Forfeit" in res_str:
                    if winner == 1:
                        p1_flags += 1
                    else:
                        p2_flags += 1

        results[f"{p1} vs {p2}"] = {
            "p1_wins": p1_wins,
            "p2_wins": p2_wins,
            "draws": draws,
            "p1_flags": p1_flags,
            "p2_flags": p2_flags,
        }
        print(f"Outcome: {p1} {p1_wins} - {p2_wins} {p2} (Draws: {draws}) | Flags: {p1}({p1_flags}) vs {p2}({p2_flags})")

    return results


def plot_6x6_chess_results(results: dict, save_path: str = "figures/fig22_bullet_chess_6x6_tournament.png"):
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")

    matchup_names = list(results.keys())
    p1_scores = [results[m]["p1_wins"] for m in matchup_names]
    p2_scores = [results[m]["p2_wins"] for m in matchup_names]
    draw_scores = [results[m]["draws"] for m in matchup_names]

    x = np.arange(len(matchup_names))
    width = 0.35

    ax.bar(x - width/2, p1_scores, width, label="Player 1 Wins", color="#2E7D32")
    ax.bar(x + width/2, p2_scores, width, label="Player 2 Wins", color="#C62828")
    if any(d > 0 for d in draw_scores):
        ax.bar(x, draw_scores, width * 0.5, label="Draws", color="#757575")

    # Annotate flag forfeits
    for i, m in enumerate(matchup_names):
        f1 = results[m]["p1_flags"]
        f2 = results[m]["p2_flags"]
        note = f"Flags: P1({f1}) P2({f2})"
        ax.text(i, max(p1_scores[i], p2_scores[i]) + 0.3, note, ha="center", fontsize=8.5, fontweight="bold", color="#333333")

    ax.set_ylabel("Games Won", fontsize=11, fontweight="bold")
    ax.set_title("Harder 6x6 Bullet Chess Tournament (1.5s Clock): Adaptive ETS vs. Fixed Searchers",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(matchup_names, fontsize=9.5, fontweight="bold", rotation=10)
    ax.set_ylim(0, max(p1_scores + p2_scores) + 2)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"\nSaved 6x6 bullet chess tournament figure to {save_path}")


if __name__ == "__main__":
    jev_chess = JevChess6x6Evaluator().to(device)
    train_jev_chess6x6_model(jev_chess, num_samples=3000, epochs=5)
    results = run_6x6_chess_tournament(jev_chess, games_per_pair=10)
    plot_6x6_chess_results(results)
