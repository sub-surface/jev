"""
Frontier 2: Bullet Micro-Chess Arena (5x5 Gardner Chess under Clock Pressure)
===========================================================================
Authors: Leon & Ilya Sutskever persona

The Fundamental Question:
  In an adversarial two-player zero-sum game with strict clock constraints (bullet chess),
  can an agent using calibrated epistemic gating (Jev System 1) dynamically manage its
  time budget—moving reflexively on quiet moves and unrolling deep minimax search during
  tactical crises—to defeat fixed-depth and fixed-time opponents?

Game: 5x5 Gardner Micro-Chess (R, N, B, Q, K, 5 Pawns)
Clock: 3.0 seconds total per player (Flag falls = instant loss).
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

# 5x5 Board Constants
BOARD_SIZE = 5

# Pieces:
# Empty: 0
# White: P=1, N=2, B=3, R=4, Q=5, K=6
# Black: p=-1, n=-2, b=-3, r=-4, q=-5, k=-6
EMPTY = 0
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6

PIECE_SYMBOLS = {
    0: ".",
    1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K",
    -1: "p", -2: "n", -3: "b", -4: "r", -5: "q", -6: "k",
}

PIECE_VALUES = {
    PAWN: 100,
    KNIGHT: 300,
    BISHOP: 320,
    ROOK: 500,
    QUEEN: 900,
    KING: 10000,
}


# ----------------------------------------------------------------------
# 1. 5x5 Gardner Chess Engine
# ----------------------------------------------------------------------

class GardnerChess:
    def __init__(self):
        self.reset()

    def reset(self):
        # Gardner starting layout (White at bottom, Black at top)
        # Row 0: Black major pieces: r, n, b, q, k
        # Row 1: Black pawns: p, p, p, p, p
        # Row 2: Empty
        # Row 3: White pawns: P, P, P, P, P
        # Row 4: White major pieces: R, N, B, Q, K
        self.board = np.array([
            [-ROOK, -KNIGHT, -BISHOP, -QUEEN, -KING],
            [-PAWN, -PAWN,   -PAWN,   -PAWN,  -PAWN],
            [EMPTY,  EMPTY,   EMPTY,   EMPTY,  EMPTY],
            [PAWN,   PAWN,    PAWN,    PAWN,   PAWN],
            [ROOK,   KNIGHT,  BISHOP,  QUEEN,  KING],
        ], dtype=np.int32)
        self.turn = 1  # 1: White, -1: Black
        self.move_count = 0
        self.game_over = False
        self.winner = 0  # 1: White, -1: Black, 0: Draw
        return self.board

    def clone(self) -> GardnerChess:
        other = GardnerChess()
        other.board = np.copy(self.board)
        other.turn = self.turn
        other.move_count = self.move_count
        other.game_over = self.game_over
        other.winner = self.winner
        return other

    def get_legal_moves(self, side: int | None = None) -> list[tuple[int, int, int, int]]:
        """Returns list of (r1, c1, r2, c2) moves for current player."""
        current_side = side if side is not None else self.turn
        moves = []

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = self.board[r, c]
                if p * current_side <= 0:
                    continue  # Empty or opponent piece

                piece_type = abs(p)

                # 1. Pawns
                if piece_type == PAWN:
                    forward = -1 if current_side == 1 else 1
                    nr = r + forward
                    # Step forward
                    if 0 <= nr < BOARD_SIZE and self.board[nr, c] == EMPTY:
                        moves.append((r, c, nr, c))
                    # Diagonal captures
                    for dc in [-1, 1]:
                        nc = c + dc
                        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                            if self.board[nr, nc] * current_side < 0:
                                moves.append((r, c, nr, nc))

                # 2. Knights
                elif piece_type == KNIGHT:
                    offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
                    for dr, dc in offsets:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                            if self.board[nr, nc] * current_side <= 0:
                                moves.append((r, c, nr, nc))

                # 3. Bishops
                elif piece_type == BISHOP:
                    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        for step in range(1, BOARD_SIZE):
                            nr, nc = r + dr * step, c + dc * step
                            if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                                break
                            target = self.board[nr, nc]
                            if target == EMPTY:
                                moves.append((r, c, nr, nc))
                            elif target * current_side < 0:
                                moves.append((r, c, nr, nc))
                                break
                            else:
                                break

                # 4. Rooks
                elif piece_type == ROOK:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        for step in range(1, BOARD_SIZE):
                            nr, nc = r + dr * step, c + dc * step
                            if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                                break
                            target = self.board[nr, nc]
                            if target == EMPTY:
                                moves.append((r, c, nr, nc))
                            elif target * current_side < 0:
                                moves.append((r, c, nr, nc))
                                break
                            else:
                                break

                # 5. Queens
                elif piece_type == QUEEN:
                    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                        for step in range(1, BOARD_SIZE):
                            nr, nc = r + dr * step, c + dc * step
                            if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                                break
                            target = self.board[nr, nc]
                            if target == EMPTY:
                                moves.append((r, c, nr, nc))
                            elif target * current_side < 0:
                                moves.append((r, c, nr, nc))
                                break
                            else:
                                break

                # 6. Kings
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

    def make_move(self, move: tuple[int, int, int, int]):
        r1, c1, r2, c2 = move
        piece = self.board[r1, c1]
        target = self.board[r2, c2]

        # Check King capture (Terminal win)
        if abs(target) == KING:
            self.game_over = True
            self.winner = self.turn

        self.board[r2, c2] = piece
        self.board[r1, c1] = EMPTY

        # Pawn Promotion to Queen on final rank
        if abs(piece) == PAWN:
            if (self.turn == 1 and r2 == 0) or (self.turn == -1 and r2 == BOARD_SIZE - 1):
                self.board[r2, c2] = QUEEN * self.turn

        self.turn = -self.turn
        self.move_count += 1

        # Draw limit at 60 moves
        if self.move_count >= 60 and not self.game_over:
            self.game_over = True
            self.winner = 0

    def evaluate_material(self) -> int:
        """Heuristic material evaluation from White's perspective."""
        score = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = self.board[r, c]
                if p != EMPTY:
                    val = PIECE_VALUES[abs(p)]
                    score += val if p > 0 else -val
        return score


# ----------------------------------------------------------------------
# 2. Neural System 1 Jev Position Evaluator & Tactical Volatility Noul
# ----------------------------------------------------------------------

def encode_board_tensor(board: np.ndarray, turn: int) -> torch.Tensor:
    """
    13-channel binary tensor representation of 5x5 board:
      Channels 0..5: White P, N, B, R, Q, K
      Channels 6..11: Black p, n, b, r, q, k
      Channel 12: Turn indicator
    """
    t = torch.zeros(13, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = board[r, c]
            if p > 0:
                t[p - 1, r, c] = 1.0
            elif p < 0:
                t[abs(p) + 5, r, c] = 1.0
    t[12, :, :] = 1.0 if turn == 1 else 0.0
    return t


class JevChessEvaluator(nn.Module):
    """
    Jev System 1 Model for Gardner Chess:
      1. Position Value V(s) in [-1.0, 1.0]: winning probability for current turn.
      2. Noul(s) in [0.0, 1.0]: Epistemic Certainty / Quietness.
         Noul ~ 1.0: Peaceful, quiet position (Reflexive move safe).
         Noul < tau: Tactical crisis, King in check, or piece hanging (Search required!).
    """
    def __init__(self, in_channels: int = 13, d_model: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )
        self.val_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh(),  # Value in [-1.0, 1.0]
        )
        self.noul_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Epistemic certainty in [0.0, 1.0]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        val = self.val_head(h).squeeze(-1)
        noul = self.noul_head(h).squeeze(-1)
        return val, noul


def train_jev_chess_model(model: JevChessEvaluator, num_samples: int = 4000, epochs: int = 6):
    print("Simulating self-play tactical positions for Jev Chess Evaluator...", flush=True)
    states = []
    values = []
    nouls = []

    for s_idx in range(num_samples):
        game = GardnerChess()
        # Play random 4-16 moves to reach diverse positions
        num_plies = random.randint(4, 20)
        for _ in range(num_plies):
            legal = game.get_legal_moves()
            if not legal or game.game_over:
                break
            # Prefer captures slightly for richer tactics
            captures = [m for m in legal if game.board[m[2], m[3]] != 0]
            m = random.choice(captures) if captures and random.random() < 0.4 else random.choice(legal)
            game.make_move(m)

        mat = game.evaluate_material() * game.turn
        norm_val = float(np.tanh(mat / 600.0))

        # Check if immediate tactical crisis: can any piece capture high value?
        legal = game.get_legal_moves()
        tactical_crisis = any(abs(game.board[m[2], m[3]]) >= KNIGHT for m in legal)
        # Noul = 1 if quiet, 0 if tactical threat
        noul = 0.1 if tactical_crisis else 0.9

        t_state = encode_board_tensor(game.board, game.turn)
        states.append(t_state)
        values.append(norm_val)
        nouls.append(noul)

    x_t = torch.stack(states).to(device)
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

    print(f"Jev Chess Model trained in {time.time() - t0:.1f}s on {device}\n", flush=True)


# ----------------------------------------------------------------------
# 3. Decision Engines: Fixed-Depth, Fixed-Time, and Adaptive ETS
# ----------------------------------------------------------------------

def negamax(game: GardnerChess, model: JevChessEvaluator, depth: int, alpha: float, beta: float) -> float:
    if game.game_over:
        return -100.0  # Current side has lost (King captured)
    if depth == 0:
        t_s = encode_board_tensor(game.board, game.turn).unsqueeze(0).to(device)
        with torch.no_grad():
            v, _ = model(t_s)
        return v.item()

    legal = game.get_legal_moves()
    if not legal:
        return 0.0  # Draw

    max_eval = -float("inf")
    for m in legal:
        child = game.clone()
        child.make_move(m)
        ev = -negamax(child, model, depth - 1, -beta, -alpha)
        max_eval = max(max_eval, ev)
        alpha = max(alpha, ev)
        if alpha >= beta:
            break
    return max_eval


def select_move_fixed_depth(game: GardnerChess, model: JevChessEvaluator, depth: int = 2) -> tuple[int, int, int, int]:
    """Exhaustively searches fixed negamax depth regardless of clock."""
    legal = game.get_legal_moves()
    best_move = legal[0]
    best_val = -float("inf")
    for m in legal:
        child = game.clone()
        child.make_move(m)
        ev = -negamax(child, model, depth - 1, -float("inf"), float("inf"))
        if ev > best_val:
            best_val = ev
            best_move = m
    return best_move


def select_move_reflexive(game: GardnerChess, model: JevChessEvaluator) -> tuple[int, int, int, int]:
    """Reflexive 1-step move: evaluates candidate board states in single forward pass."""
    legal = game.get_legal_moves()
    tensors = []
    for m in legal:
        child = game.clone()
        child.make_move(m)
        tensors.append(encode_board_tensor(child.board, child.turn))

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        vals, _ = model(batch)
    # Minimizing opponent's value is maximizing current player's return
    best_idx = torch.argmin(vals).item()
    return legal[best_idx]


def select_move_adaptive_ets(
    game: GardnerChess, model: JevChessEvaluator, clock_remaining: float, tau: float = 0.75
) -> tuple[tuple[int, int, int, int], int]:
    """
    Adaptive Epistemic Bullet Agent:
      1. Evaluates current position volatility via Jev Noul.
      2. If Noul >= tau (quiet) or clock is low (< 0.4s): Move reflexively (0ms).
      3. If Noul < tau (tactical crisis): Unrolls depth-2 negamax search!
    Returns:
      (chosen_move, search_depth_used)
    """
    legal = game.get_legal_moves()
    t_s = encode_board_tensor(game.board, game.turn).unsqueeze(0).to(device)
    with torch.no_grad():
        _, noul = model(t_s)
    noul_p = noul.item()

    # Time/Epistemic Management:
    if clock_remaining < 0.35 or noul_p >= tau:
        return select_move_reflexive(game, model), 0

    # Tactical Crisis: allocate search depth
    search_depth = 2 if clock_remaining > 0.8 else 1
    best_move = legal[0]
    best_val = -float("inf")
    for m in legal:
        child = game.clone()
        child.make_move(m)
        ev = -negamax(child, model, search_depth - 1, -float("inf"), float("inf"))
        if ev > best_val:
            best_val = ev
            best_move = m

    return best_move, search_depth


# ----------------------------------------------------------------------
# 4. Bullet Chess Tournament Runner (Strict 3.0-Second Clock)
# ----------------------------------------------------------------------

def play_bullet_game(
    white_agent_type: str,
    black_agent_type: str,
    model: JevChessEvaluator,
    time_limit_sec: float = 2.5,
) -> tuple[int, str, float, float]:
    game = GardnerChess()
    white_clock = time_limit_sec
    black_clock = time_limit_sec

    while not game.game_over:
        legal = game.get_legal_moves()
        if not legal:
            # Stalemate / Draw
            return 0, "No legal moves", white_clock, black_clock

        current_agent = white_agent_type if game.turn == 1 else black_agent_type
        current_clock = white_clock if game.turn == 1 else black_clock

        t0 = time.time()
        if current_agent == "Reflexive":
            move = select_move_reflexive(game, model)
        elif current_agent == "Fixed_Depth_2":
            move = select_move_fixed_depth(game, model, depth=2)
        elif current_agent == "Adaptive_ETS":
            move, _ = select_move_adaptive_ets(game, model, current_clock, tau=0.75)
        else:
            raise ValueError(f"Unknown agent: {current_agent}")

        elapsed = time.time() - t0

        if game.turn == 1:
            white_clock -= elapsed
            if white_clock <= 0:
                # White flags!
                return -1, "White Flag Fall", 0.0, black_clock
        else:
            black_clock -= elapsed
            if black_clock <= 0:
                # Black flags!
                return 1, "Black Flag Fall", white_clock, 0.0

        game.make_move(move)

    reason = "Checkmate/King Capture" if game.winner != 0 else "Draw"
    return game.winner, reason, white_clock, black_clock


def run_bullet_tournament(num_matches: int = 16):
    print("================================================================================", flush=True)
    print(f" FRONTIER 2: BULLET MICRO-CHESS TOURNAMENT (3.0s Clock — {num_matches} Matches)", flush=True)
    print("================================================================================\n", flush=True)

    model = JevChessEvaluator().to(device)
    train_jev_chess_model(model, num_samples=3500, epochs=6)
    model.eval()

    # Head-to-Head Matchups:
    # Matchup 1: Adaptive ETS vs Fixed_Depth_2 (Does Fixed Depth flag?)
    # Matchup 2: Adaptive ETS vs Reflexive (Does Reflexive get tactically blundered?)
    matchups = [
        ("Adaptive_ETS", "Fixed_Depth_2"),
        ("Adaptive_ETS", "Reflexive"),
        ("Fixed_Depth_2", "Reflexive"),
    ]

    for agent_A, agent_B in matchups:
        print(f"\n--- Contest: {agent_A} vs. {agent_B} ({num_matches} Games, Swapping Colors) ---", flush=True)
        wins_A = 0
        wins_B = 0
        draws = 0
        flags_A = 0
        flags_B = 0
        remaining_clocks_A = []
        remaining_clocks_B = []

        for i in range(num_matches):
            # Alternate colors
            white = agent_A if i % 2 == 0 else agent_B
            black = agent_B if i % 2 == 0 else agent_A

            winner, reason, w_clk, b_clk = play_bullet_game(white, black, model, time_limit_sec=2.5)

            if i % 2 == 0:
                # Agent A was White
                winner_agent = agent_A if winner == 1 else (agent_B if winner == -1 else "Draw")
                remaining_clocks_A.append(w_clk)
                remaining_clocks_B.append(b_clk)
                if "White Flag" in reason: flags_A += 1
                if "Black Flag" in reason: flags_B += 1
            else:
                # Agent A was Black
                winner_agent = agent_B if winner == 1 else (agent_A if winner == -1 else "Draw")
                remaining_clocks_A.append(b_clk)
                remaining_clocks_B.append(w_clk)
                if "Black Flag" in reason: flags_A += 1
                if "White Flag" in reason: flags_B += 1

            if winner_agent == agent_A:
                wins_A += 1
            elif winner_agent == agent_B:
                wins_B += 1
            else:
                draws += 1

        print(f"Result: {agent_A}: {wins_A} wins | {agent_B}: {wins_B} wins | Draws: {draws}", flush=True)
        print(f"Clock Flags: {agent_A} flagged {flags_A} times | {agent_B} flagged {flags_B} times", flush=True)
        print(f"Average Clock Remaining: {agent_A}: {np.mean(remaining_clocks_A):.2f}s | {agent_B}: {np.mean(remaining_clocks_B):.2f}s", flush=True)


if __name__ == "__main__":
    run_bullet_tournament(num_matches=16)
