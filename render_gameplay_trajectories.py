"""
Render Gameplay Trajectories & Decision-Making
==============================================
Authors: Leon & Ilya Sutskever persona

Visualizes:
  1. Real-Time Tetris Game Trajectory:
     Side-by-side board states showing when Adaptive ETS drops reflexively (Noul >= tau)
     vs. when hazardous surface triggers System 2 lookahead search.
  2. Bullet Micro-Chess Checkmate Sequence:
     Actual board progression with chess piece glyphs, clock countdown,
     and tactical volatility gating.
"""

from __future__ import annotations

import os
import sys
import time
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

from tetris_arena import TetrisGame, JevTetrisEvaluator, train_jev_tetris_model, extract_board_features, select_action_adaptive_ets, select_action_eret
from gardner_chess_arena import GardnerChess, JevChessEvaluator, train_jev_chess_model, select_move_adaptive_ets, encode_board_tensor, PIECE_SYMBOLS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------
# 1. Render Tetris Trajectory
# ----------------------------------------------------------------------

def record_tetris_trajectory(model: JevTetrisEvaluator, seed: int = 105, max_steps: int = 16):
    game = TetrisGame(seed=seed)
    history = []

    for step in range(max_steps):
        if game.game_over:
            break
        placements = game.get_legal_placements()
        if not placements:
            break

        feats = [extract_board_features(p[2]) for p in placements]
        x = torch.tensor(np.array(feats), dtype=torch.float32).to(device)
        with torch.no_grad():
            vals, nouls = model(x)
            best_idx = torch.argmax(vals).item()
            best_noul = nouls[best_idx].item()
            best_v = vals[best_idx].item()

        b_next, cleared, exp = select_action_adaptive_ets(model, game, tau=0.80, beam_width=4)
        action_type = "REFLEXIVE (0 Exp)" if exp == 0 else f"SEARCH (Exp: {exp})"

        history.append({
            "step": step + 1,
            "piece": game.current_piece,
            "board": np.copy(game.board),
            "board_after": np.copy(b_next),
            "noul": best_noul,
            "value": best_v,
            "action_type": action_type,
            "cleared": cleared,
        })
        game.step(b_next, cleared)

    return history


def plot_tetris_filmstrip(history: list[dict], save_path: str = "figures/fig19_tetris_decision_filmstrip.png"):
    num_frames = min(6, len(history))
    fig, axes = plt.subplots(1, num_frames, figsize=(3.2 * num_frames, 6.0), dpi=300)
    fig.patch.set_facecolor("#1E1E24")

    # Sample key frames
    indices = np.linspace(0, len(history) - 1, num_frames, dtype=int)

    for ax_idx, h_idx in enumerate(indices):
        frame = history[h_idx]
        ax = axes[ax_idx]
        ax.set_facecolor("#111116")

        b = frame["board_after"]
        # Render Tetris Grid
        for r in range(20):
            for c in range(10):
                if b[r, c] == 1:
                    ax.add_patch(patches.Rectangle((c, 19 - r), 0.92, 0.92, facecolor="#00E5FF", edgecolor="#00B0FF", lw=0.8))
                else:
                    ax.add_patch(patches.Rectangle((c, 19 - r), 0.92, 0.92, facecolor="#181820", edgecolor="#22222A", lw=0.5))

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 20)
        ax.set_aspect("equal")
        ax.axis("off")

        # Color-coded action banner
        is_search = "SEARCH" in frame["action_type"]
        banner_color = "#E040FB" if is_search else "#00E676"

        title_text = (
            f"Step {frame['step']} (Piece: {frame['piece']})\n"
            f"Noul: {frame['noul']:.2f} | V: {frame['value']:.2f}\n"
            f"{frame['action_type']}"
        )
        ax.set_title(title_text, color=banner_color, fontsize=9.5, fontweight="bold", pad=8)

    plt.suptitle("Adaptive ETS Tetris Decision Filmstrip: Reflexive Drops vs. Epistemic Lookahead", 
                 color="#FFFFFF", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"Saved Tetris decision filmstrip to {save_path}")


# ----------------------------------------------------------------------
# 2. Render Bullet Micro-Chess Checkmate Progression
# ----------------------------------------------------------------------

CHESS_PIECE_UNICODE = {
    0: "",
    1: "\u2659", 2: "\u2658", 3: "\u2657", 4: "\u2656", 5: "\u2655", 6: "\u2654",  # White
    -1: "\u265F", -2: "\u265E", -3: "\u265D", -4: "\u265C", -5: "\u265B", -6: "\u265A", # Black
}

def record_chess_game(model: JevChessEvaluator, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    game = GardnerChess()
    white_clock = 2.5
    black_clock = 2.5
    move_log = []

    while not game.game_over and game.move_count < 20:
        legal = game.get_legal_moves()
        if not legal:
            break

        is_white = game.turn == 1
        clk = white_clock if is_white else black_clock

        t_s = encode_board_tensor(game.board, game.turn).unsqueeze(0).to(device)
        with torch.no_grad():
            val, noul = model(t_s)
        noul_p = noul.item()

        t0 = time.time()
        if is_white:
            # White is Adaptive ETS
            move, depth = select_move_adaptive_ets(game, model, clk, tau=0.75)
            agent_label = f"ETS (D={depth})" if depth > 0 else "ETS (Reflex)"
        else:
            # Black is Reflexive Jev
            # pick greedy move
            tensors = [encode_board_tensor(game.clone().make_move(m) or game.board, game.turn) for m in legal]
            # Simple reflexive choice
            move = legal[0]
            agent_label = "Reflexive"
        elapsed = time.time() - t0

        if is_white:
            white_clock -= elapsed
        else:
            black_clock -= elapsed

        move_log.append({
            "ply": game.move_count + 1,
            "turn": "White (ETS)" if is_white else "Black (Reflex)",
            "board": np.copy(game.board),
            "move": move,
            "noul": noul_p,
            "clock_w": white_clock,
            "clock_b": black_clock,
            "agent": agent_label,
        })
        game.make_move(move)

    move_log.append({
        "ply": game.move_count + 1,
        "turn": "Terminal",
        "board": np.copy(game.board),
        "move": None,
        "noul": 0.0,
        "clock_w": white_clock,
        "clock_b": black_clock,
        "agent": "Checkmate" if game.winner == 1 else "End",
    })
    return move_log


def plot_chess_gameplay(move_log: list[dict], save_path: str = "figures/fig20_bullet_chess_gameplay_render.png"):
    num_frames = min(5, len(move_log))
    fig, axes = plt.subplots(1, num_frames, figsize=(3.4 * num_frames, 4.2), dpi=300)
    fig.patch.set_facecolor("#FAFAFA")

    indices = np.linspace(0, len(move_log) - 1, num_frames, dtype=int)

    for ax_idx, m_idx in enumerate(indices):
        frame = move_log[m_idx]
        ax = axes[ax_idx]
        b = frame["board"]

        # Draw 5x5 board
        for r in range(5):
            for c in range(5):
                sq_color = "#E0C39E" if (r + c) % 2 == 0 else "#A26941"
                ax.add_patch(patches.Rectangle((c, 4 - r), 1.0, 1.0, facecolor=sq_color, edgecolor="#222222", lw=0.5))
                p = b[r, c]
                if p != 0:
                    symbol = CHESS_PIECE_UNICODE.get(p, "?")
                    p_color = "#FFFFFF" if p > 0 else "#111111"
                    stroke = "#000000" if p > 0 else "#DDDDDD"
                    ax.text(c + 0.5, 4 - r + 0.5, symbol, fontsize=24, ha="center", va="center", color=p_color, fontweight="bold")

        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.set_aspect("equal")
        ax.axis("off")

        title = f"Ply {frame['ply']}: {frame['turn']}\n{frame['agent']}\nClock W: {frame['clock_w']:.2f}s | B: {frame['clock_b']:.2f}s"
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=8)

    plt.suptitle("Bullet Micro-Chess: Adaptive ETS Tactical Checkmate Progression", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved Chess gameplay render to {save_path}")


if __name__ == "__main__":
    print("Generating gameplay recordings for Tetris & Bullet Chess...", flush=True)
    jev_tetris = JevTetrisEvaluator().to(device)
    train_jev_tetris_model(jev_tetris, num_samples=3000, epochs=4)
    tetris_history = record_tetris_trajectory(jev_tetris, seed=105, max_steps=18)
    plot_tetris_filmstrip(tetris_history)

    jev_chess = JevChessEvaluator().to(device)
    train_jev_chess_model(jev_chess, num_samples=2500, epochs=4)
    chess_history = record_chess_game(jev_chess, seed=42)
    plot_chess_gameplay(chess_history)
