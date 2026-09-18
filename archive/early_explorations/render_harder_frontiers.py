"""
Render Decision-Making Trajectories for Harder Challenges
=========================================================
Authors: Leon & Ilya Sutskever persona

Visualizes:
  1. Harder 6x6 Bullet Chess (1.5s Clock):
     Board progression showing tactical crises, millisecond clock countdowns,
     and Adaptive ETS switching between 1ms reflexive moves and deep calculation.
  2. 20-Hop Deep Symbolic Deduction Graph:
     Visualizes the proof search trajectory, showing deceptive attractor branches
     pruned by calibrated Noul gating vs. ERET trapped in cyclic loops.
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

from bullet_chess_6x6 import (
    Chess6x6, JevChess6x6Evaluator, train_jev_chess6x6_model,
    select_move_adaptive_ets, select_move_reflexive_jev, select_move_fixed_minimax,
    encode_board_tensor, BOARD_SIZE, EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
)
from symbolic_proof_deep import (
    DeepRegisterMachineEnv, DeepJevVerifier, DeepERETLooped,
    deep_epistemic_tree_of_thought, select_branch_eret, select_branch_greedy
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHESS_PIECE_UNICODE_6X6 = {
    0: "",
    PAWN: "\u2659", KNIGHT: "\u2658", BISHOP: "\u2657", ROOK: "\u2656", QUEEN: "\u2655", KING: "\u2654",
    -PAWN: "\u265F", -KNIGHT: "\u265E", -BISHOP: "\u265D", -ROOK: "\u265C", -QUEEN: "\u265B", -KING: "\u265A",
}


# ----------------------------------------------------------------------
# 1. 6x6 Bullet Chess Trajectory Recording
# ----------------------------------------------------------------------

def record_6x6_chess_game(model: JevChess6x6Evaluator, seed: int = 101, clock_limit: float = 1.5):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    game = Chess6x6()
    white_clock = clock_limit
    black_clock = clock_limit
    move_log = []

    # White is Adaptive ETS, Black is Fixed Depth 2
    while not game.game_over and game.move_count < 22:
        legal = game.get_legal_moves()
        if not legal:
            break

        is_white = (game.turn == 1)
        clk = white_clock if is_white else black_clock

        t_s = encode_board_tensor(game.board, game.turn).unsqueeze(0).to(device)
        with torch.no_grad():
            val, noul = model(t_s)
        noul_val = noul.item()

        t0 = time.time()
        if is_white:
            move, depth = select_move_adaptive_ets(game, model, clk, tau=0.70)
            agent_str = f"ETS (D={depth})" if depth > 1 else "ETS (Reflex 1ms)"
        else:
            move = select_move_fixed_minimax(game, depth=2)
            agent_str = "Fixed D=2"
        elapsed = time.time() - t0

        if is_white:
            white_clock -= elapsed
        else:
            black_clock -= elapsed

        move_log.append({
            "ply": game.move_count + 1,
            "turn": "White (ETS)" if is_white else "Black (Fixed D=2)",
            "board": np.copy(game.board),
            "move": move,
            "noul": noul_val,
            "clock_w": max(0.0, white_clock),
            "clock_b": max(0.0, black_clock),
            "agent": agent_str,
        })

        if white_clock <= 0.0 or black_clock <= 0.0:
            break

        game.make_move(move)

    move_log.append({
        "ply": game.move_count + 1,
        "turn": "Terminal",
        "board": np.copy(game.board),
        "move": None,
        "noul": 0.0,
        "clock_w": max(0.0, white_clock),
        "clock_b": max(0.0, black_clock),
        "agent": "White Won (Clock Forfeit)" if black_clock <= 0 else ("White Won (Mate)" if game.winner == 1 else "End"),
    })
    return move_log


def plot_6x6_chess_gameplay(move_log: list[dict], save_path: str = "figures/fig23_bullet_chess_6x6_filmstrip.png"):
    num_frames = min(6, len(move_log))
    fig, axes = plt.subplots(1, num_frames, figsize=(3.3 * num_frames, 4.4), dpi=300)
    fig.patch.set_facecolor("#FAFAFA")

    indices = np.linspace(0, len(move_log) - 1, num_frames, dtype=int)

    for ax_idx, m_idx in enumerate(indices):
        frame = move_log[m_idx]
        ax = axes[ax_idx]
        b = frame["board"]

        # Draw 6x6 board
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                sq_color = "#E0C39E" if (r + c) % 2 == 0 else "#A26941"
                ax.add_patch(patches.Rectangle((c, 5 - r), 1.0, 1.0, facecolor=sq_color, edgecolor="#222222", lw=0.5))
                p = b[r, c]
                if p != 0:
                    symbol = CHESS_PIECE_UNICODE_6X6.get(p, "?")
                    p_color = "#FFFFFF" if p > 0 else "#111111"
                    ax.text(c + 0.5, 5 - r + 0.5, symbol, fontsize=20, ha="center", va="center", color=p_color, fontweight="bold")

        ax.set_xlim(0, 6)
        ax.set_ylim(0, 6)
        ax.set_aspect("equal")
        ax.axis("off")

        # Title
        color_tag = "#1B5E20" if "White" in frame["turn"] else "#B71C1C"
        title = (
            f"Ply {frame['ply']}: {frame['turn']}\n"
            f"{frame['agent']}\n"
            f"W: {frame['clock_w']:.2f}s | B: {frame['clock_b']:.2f}s"
        )
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=8, color=color_tag)

    plt.suptitle("6x6 Bullet Chess: Adaptive ETS Clock Management vs Fixed Depth Searcher",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Saved 6x6 Bullet Chess filmstrip to {save_path}")


# ----------------------------------------------------------------------
# 2. 20-Hop Symbolic Deduction Trajectory Visualization
# ----------------------------------------------------------------------

def plot_deduction_trajectory(env: DeepRegisterMachineEnv, jev_path: list[np.ndarray], eret_path: list[np.ndarray], save_path: str = "figures/fig24_deep_deduction_path_render.png"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6.5), dpi=300)
    fig.patch.set_facecolor("#1E1E24")

    # Heatmap of register values over time
    jev_mat = np.array(jev_path)  # (T, 6)
    eret_mat = np.array(eret_path) # (T, 6)
    target = env.target

    # Panel 1: Adaptive Epistemic ToT
    ax1.set_facecolor("#111116")
    im1 = ax1.imshow(jev_mat.T, aspect="auto", cmap="viridis", vmin=0, vmax=32)
    ax1.set_title("Adaptive Epistemic ToT: Coherent Convergence to 20-Hop Target Assertion", color="#00E676", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Registers (R0-R5)", color="#FFFFFF", fontsize=9.5)
    ax1.set_yticks(range(6))
    ax1.set_yticklabels([f"R{i} (tgt={target[i]})" for i in range(6)], color="#FFFFFF", fontsize=8.5)
    ax1.tick_params(colors="#FFFFFF")
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.ax.yaxis.set_tick_params(color="#FFFFFF")
    plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color='#FFFFFF')

    # Panel 2: ERET Looped Recurrence
    ax2.set_facecolor("#111116")
    im2 = ax2.imshow(eret_mat.T, aspect="auto", cmap="magma", vmin=0, vmax=32)
    ax2.set_title("ERET Latent Recurrence: Divergence & Entrapment in Cyclic Attractors", color="#FF5252", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Execution Step (Hop)", color="#FFFFFF", fontsize=9.5)
    ax2.set_ylabel("Registers (R0-R5)", color="#FFFFFF", fontsize=9.5)
    ax2.set_yticks(range(6))
    ax2.set_yticklabels([f"R{i} (tgt={target[i]})" for i in range(6)], color="#FFFFFF", fontsize=8.5)
    ax2.tick_params(colors="#FFFFFF")
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.ax.yaxis.set_tick_params(color="#FFFFFF")
    plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='#FFFFFF')

    plt.suptitle("Deep Symbolic Deduction (20-Hop Proof): Counterfactual Search vs. Latent Drift",
                 color="#FFFFFF", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"Saved Deep Deduction trajectory render to {save_path}")


def record_deduction_game(jev_model: DeepJevVerifier, eret_model: DeepERETLooped, seed: int = 7025, depth: int = 20):
    env_jev = DeepRegisterMachineEnv(depth=depth, seed=seed)
    env_eret = DeepRegisterMachineEnv(depth=depth, seed=seed)

    jev_path = [np.copy(env_jev.regs)]
    eret_path = [np.copy(env_eret.regs)]

    # Run Adaptive ToT
    for _ in range(depth + 2):
        if np.array_equal(env_jev.regs, env_jev.target):
            break
        next_r, _ = deep_epistemic_tree_of_thought(jev_model, env_jev, beam_width=3, depth_limit=3, adaptive=True, tau=0.82)
        env_jev.step(next_r)
        jev_path.append(np.copy(env_jev.regs))

    # Run ERET
    for _ in range(depth + 2):
        if np.array_equal(env_eret.regs, env_eret.target):
            break
        next_r = select_branch_eret(eret_model, env_eret, loops=4)
        env_eret.step(next_r)
        eret_path.append(np.copy(env_eret.regs))

    max_len = max(len(jev_path), len(eret_path))
    while len(jev_path) < max_len:
        jev_path.append(np.copy(jev_path[-1]))
    while len(eret_path) < max_len:
        eret_path.append(np.copy(eret_path[-1]))

    return env_jev, jev_path, eret_path


if __name__ == "__main__":
    from symbolic_proof_deep import train_deep_jev_verifier, train_deep_eret_model

    print("Generating renders for Harder 6x6 Bullet Chess and 20-Hop Symbolic Deduction...", flush=True)

    # 1. 6x6 Bullet Chess
    jev_chess = JevChess6x6Evaluator().to(device)
    train_jev_chess6x6_model(jev_chess, num_samples=2500, epochs=4)
    chess_log = record_6x6_chess_game(jev_chess, seed=105, clock_limit=1.5)
    plot_6x6_chess_gameplay(chess_log)

    # 2. 20-Hop Symbolic Deduction
    jev_deep = DeepJevVerifier().to(device)
    train_deep_jev_verifier(jev_deep, num_programs=2500, epochs=4)

    eret_deep = DeepERETLooped().to(device)
    train_deep_eret_model(eret_deep, num_programs=2500, epochs=4)

    env_sim, j_path, e_path = record_deduction_game(jev_deep, eret_deep, seed=7025, depth=20)
    plot_deduction_trajectory(env_sim, j_path, e_path)

