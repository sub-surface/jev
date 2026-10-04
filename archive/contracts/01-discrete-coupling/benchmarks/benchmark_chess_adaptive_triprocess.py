"""
Benchmark 3: 6x6 Bullet Chess with In-Context Opponent Adaptation
================================================================
Authors: Leon & Ilya Sutskever persona

The Task:
  Adversarial multi-game tournament in 6x6 Los Alamos Mini-Chess under a strict 1.5s clock.
  An opponent employs specialized asymmetric styles:
    - Style A: Aggressive Kingside Pawn Storm (sacrifices safety for rapid kingside attack)
    - Style B: Defensive Fortress Turtle (tucks king, locks center pawns)

The Tri-Process Breakdown:
  - System 2 (In-Context Transformer): Ingests the multi-game history (past game results,
    board snapshots, opponent move tendencies) in-context. Detects opponent stylistic bias
    and emits a modulation vector m that dynamically modifies System 0's positional evaluation.
  - System 1 (TypeSafe Jev): Manages the 1.5s clock and monitors tactical volatility (Noul).
  - System 0 (NNUE Accumulator): High-speed HalfKP sparse accumulator running discrete
    alpha-beta search, conditioned on m.
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

from bullet_chess_6x6 import (
    Chess6x6, BOARD_SIZE, EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    PIECE_VALUES, PST_PAWN_WHITE, PST_KNIGHT
)
from triprocess_engine import NNUESparseAccumulator, JevEpistemicHead, InContextTransformer, TransformerConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)


# ----------------------------------------------------------------------
# 1. HalfKP-style Sparse Feature Encoder for 6x6 Chess
# ----------------------------------------------------------------------

# Total features: 2 king positions (36) * 10 non-king piece types * 36 squares = 720 features
NUM_CHESS_FEATURES = 720

def encode_halfkp_sparse_features(board: np.ndarray, turn: int) -> list[int]:
    """
    HalfKP Sparse Encoding for 6x6:
      Indices map (piece_type, square) relative to White / Black king.
    """
    feats = []
    # Find kings
    w_king_sq = 0
    b_king_sq = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] == KING:
                w_king_sq = r * BOARD_SIZE + c
            elif board[r, c] == -KING:
                b_king_sq = r * BOARD_SIZE + c

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = board[r, c]
            if p == EMPTY or abs(p) == KING:
                continue
            sq = r * BOARD_SIZE + c
            p_idx = (abs(p) - 1) * 2 + (0 if p > 0 else 1)
            # Feature index
            feat_idx = (p_idx * 36 + sq) % NUM_CHESS_FEATURES
            feats.append(feat_idx)

    # Active turn feature
    feats.append((700 + (0 if turn == 1 else 1)) % NUM_CHESS_FEATURES)
    return feats


# ----------------------------------------------------------------------
# 2. Asymmetric Opponent Engines (Aggressive Storm vs Fortress)
# ----------------------------------------------------------------------

def select_opponent_move(game: Chess6x6, style: str) -> tuple[int, int, int, int]:
    legal = game.get_legal_moves()
    if not legal:
        return (0, 0, 0, 0)

    best_m = legal[0]
    best_eval = -1e9

    for m in legal:
        child = game.clone()
        child.make_move(m)
        base_score = -child.evaluate_static()

        # Style-specific bias
        bias = 0.0
        r1, c1, r2, c2 = m
        if style == "aggressive_storm":
            # Strongly prefers advancing kingside pawns (files 3, 4, 5) and attacking
            if abs(game.board[r1, c1]) == PAWN and c2 >= 3:
                bias += 1.5
            if abs(game.board[r2, c2]) > 0:  # Captures
                bias += 2.0
        elif style == "defensive_fortress":
            # Prefers keeping king safe on back ranks and clustering pawns
            if abs(game.board[r1, c1]) == KING:
                bias -= 1.0  # Avoid moving king out
            if r2 in [1, 2]:  # Solid rank positioning
                bias += 0.8

        total = base_score + bias
        if total > best_eval:
            best_eval = total
            best_m = m

    return best_m


# ----------------------------------------------------------------------
# 3. In-Context Match History Tokenizer
# ----------------------------------------------------------------------

def serialize_match_history_tokens(past_games: list[dict]) -> list[int]:
    """
    Serializes past game results into tokens for System 2:
      Token 220: <GAME_START>, 221: <RESULT_WIN>, 222: <RESULT_LOSS>, 223: <OPP_STYLE_TOKEN>
    """
    toks = [220]
    for g in past_games:
        toks.append(221 if g["winner"] == 1 else 222)
        # Add opponent move signature (e.g. mean file of opponent moves)
        toks.append(min(199, int(g["opp_kingside_bias"] * 50)))
        toks.append(min(199, g["num_plies"]))
        toks.append(220)
    return toks[:64]


# ----------------------------------------------------------------------
# 4. Search Engines: Static NNUE vs Adaptive Tri-Process
# ----------------------------------------------------------------------

def alphabeta_nnue(
    game: Chess6x6,
    nnue: NNUESparseAccumulator,
    depth: int,
    alpha: float,
    beta: float,
    side: int,
    modulation: torch.Tensor | None = None,
) -> float:
    if depth == 0 or game.game_over:
        if game.game_over:
            return 1000.0 if game.winner == side else -1000.0
        feats = encode_halfkp_sparse_features(game.board, game.turn)
        with torch.no_grad():
            score, _ = nnue.forward_from_features(feats, modulation=modulation)
        return float(side * score.item())

    legal = game.get_legal_moves()
    if not legal:
        return 0.0

    max_eval = -10000.0
    for m in legal:
        child = game.clone()
        child.make_move(m)
        score = -alphabeta_nnue(child, nnue, depth - 1, -beta, -alpha, -side, modulation)
        max_eval = max(max_eval, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return max_eval


def select_move_nnue_search(
    game: Chess6x6,
    nnue: NNUESparseAccumulator,
    depth: int = 2,
    modulation: torch.Tensor | None = None,
) -> tuple[int, int, int, int]:
    legal = game.get_legal_moves()
    if not legal:
        return (0, 0, 0, 0)

    best_m = legal[0]
    best_score = -10000.0
    alpha = -10000.0
    beta = 10000.0

    for m in legal:
        child = game.clone()
        child.make_move(m)
        score = -alphabeta_nnue(child, nnue, depth - 1, -beta, -alpha, -game.turn, modulation)
        if score > best_score:
            best_score = score
            best_m = m
        alpha = max(alpha, score)
    return best_m


# ----------------------------------------------------------------------
# 5. Multi-Game Tournament with In-Context Adaptation
# ----------------------------------------------------------------------

def play_match_series(
    agent_type: str,
    nnue: NNUESparseAccumulator,
    jev: JevEpistemicHead,
    transformer: InContextTransformer,
    opponent_style: str,
    num_games: int = 6,
    clock_limit: float = 1.5,
) -> list[dict]:
    history = []
    game_results = []

    for g_idx in range(num_games):
        game = Chess6x6()
        white_clock = clock_limit
        black_clock = clock_limit
        opp_moves = []

        # System 2 In-Context Deliberation: derive modulation vector from history
        mod_vec = None
        if agent_type == "Tri-Process (Adaptive)" and history:
            toks = serialize_match_history_tokens(history)
            while len(toks) < 32:
                toks.append(0)
            t_in = torch.tensor(toks, dtype=torch.long).unsqueeze(0).to(device)
            with torch.no_grad():
                mod_vec, _ = transformer(t_in)
                mod_vec = mod_vec.squeeze(0)

        while not game.game_over:
            is_white = (game.turn == 1)
            clk = white_clock if is_white else black_clock

            t0 = time.time()
            if is_white:
                # White Agent (Tri-Process or Static NNUE)
                feats = encode_halfkp_sparse_features(game.board, game.turn)
                with torch.no_grad():
                    _, accum = nnue.forward_from_features(feats, modulation=mod_vec)
                    _, noul = jev(accum.unsqueeze(0))

                # Gating: if Noul >= 0.80 or low clock, depth 1; else depth 2
                depth = 1 if (noul.item() >= 0.80 or clk < 0.35) else 2
                move = select_move_nnue_search(game, nnue, depth=depth, modulation=mod_vec)
            else:
                # Black Opponent (Specialized Asymmetric Style)
                move = select_opponent_move(game, style=opponent_style)
                opp_moves.append(move)

            elapsed = time.time() - t0
            if is_white:
                white_clock -= elapsed
                if white_clock <= 0.0:
                    game.game_over = True
                    game.winner = -1
                    break
            else:
                black_clock -= elapsed
                if black_clock <= 0.0:
                    game.game_over = True
                    game.winner = 1
                    break

            game.make_move(move)

        # Record game metadata
        opp_kingside = np.mean([m[3] >= 3 for m in opp_moves]) if opp_moves else 0.5
        g_data = {
            "game_idx": g_idx + 1,
            "winner": game.winner,
            "num_plies": game.move_count,
            "opp_kingside_bias": float(opp_kingside),
            "clock_w": white_clock,
            "clock_b": black_clock,
        }
        history.append(g_data)
        game_results.append(g_data)

    return game_results


# ----------------------------------------------------------------------
# 6. Benchmark Execution & Plotting
# ----------------------------------------------------------------------

def run_chess_adaptive_benchmark(
    nnue: NNUESparseAccumulator,
    jev: JevEpistemicHead,
    transformer: InContextTransformer,
    num_series: int = 8,
) -> dict:
    print(f"==========================================================================")
    print(f"RUNNING BENCHMARK 3: 6x6 CHESS IN-CONTEXT ADAPTATION ({num_series} matches)")
    print(f"==========================================================================")

    results = {
        "Static NNUE": {"game1_wins": 0, "game3_wins": 0, "game6_wins": 0, "total_wins": 0, "total_games": 0},
        "Tri-Process (Adaptive)": {"game1_wins": 0, "game3_wins": 0, "game6_wins": 0, "total_wins": 0, "total_games": 0},
    }

    nnue.eval()
    jev.eval()
    transformer.eval()

    styles = ["aggressive_storm", "defensive_fortress"]

    for s_idx in range(num_series):
        style = styles[s_idx % len(styles)]

        # 1. Static NNUE series
        res_static = play_match_series("Static NNUE", nnue, jev, transformer, opponent_style=style, num_games=6)
        for g in res_static:
            results["Static NNUE"]["total_games"] += 1
            if g["winner"] == 1:
                results["Static NNUE"]["total_wins"] += 1
                if g["game_idx"] == 1:
                    results["Static NNUE"]["game1_wins"] += 1
                elif g["game_idx"] == 3:
                    results["Static NNUE"]["game3_wins"] += 1
                elif g["game_idx"] == 6:
                    results["Static NNUE"]["game6_wins"] += 1

        # 2. Tri-Process Adaptive series
        res_tri = play_match_series("Tri-Process (Adaptive)", nnue, jev, transformer, opponent_style=style, num_games=6)
        for g in res_tri:
            results["Tri-Process (Adaptive)"]["total_games"] += 1
            if g["winner"] == 1:
                results["Tri-Process (Adaptive)"]["total_wins"] += 1
                if g["game_idx"] == 1:
                    results["Tri-Process (Adaptive)"]["game1_wins"] += 1
                elif g["game_idx"] == 3:
                    results["Tri-Process (Adaptive)"]["game3_wins"] += 1
                elif g["game_idx"] == 6:
                    results["Tri-Process (Adaptive)"]["game6_wins"] += 1

    for m in results:
        tot_win_rate = (results[m]["total_wins"] / results[m]["total_games"]) * 100
        g1_wr = (results[m]["game1_wins"] / num_series) * 100
        g6_wr = (results[m]["game6_wins"] / num_series) * 100
        print(f"{m:26s} | Total Win Rate: {tot_win_rate:5.1f}% | Game 1: {g1_wr:5.1f}% -> Game 6: {g6_wr:5.1f}%")

    return results


def plot_chess_adaptive_results(results: dict, num_series: int = 8, save_path: str = "figures/fig27_chess_adaptive_triprocess.png"):
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")

    games = [1, 3, 6]
    static_wr = [
        (results["Static NNUE"]["game1_wins"] / float(num_series)) * 100,
        (results["Static NNUE"]["game3_wins"] / float(num_series)) * 100,
        (results["Static NNUE"]["game6_wins"] / float(num_series)) * 100,
    ]
    tri_wr = [
        (results["Tri-Process (Adaptive)"]["game1_wins"] / float(num_series)) * 100,
        (results["Tri-Process (Adaptive)"]["game3_wins"] / float(num_series)) * 100,
        (results["Tri-Process (Adaptive)"]["game6_wins"] / float(num_series)) * 100,
    ]

    ax.plot(games, static_wr, marker="s", lw=2.5, color="#FF8F00", label="Static NNUE (No In-Context Adaptation)")
    ax.plot(games, tri_wr, marker="D", lw=3.0, color="#2E7D32", label="Tri-Process (In-Context S2 Adaptation)")

    for i, g in enumerate(games):
        ax.annotate(f"{static_wr[i]:.1f}%", (g, static_wr[i] - 4), ha="center", fontsize=9, fontweight="bold", color="#C67100")
        ax.annotate(f"{tri_wr[i]:.1f}%", (g, tri_wr[i] + 3), ha="center", fontsize=9, fontweight="bold", color="#1B5E20")

    ax.set_title("Benchmark 3: In-Context Opponent Adaptation across Multi-Game Matches", fontsize=11.5, fontweight="bold", pad=10)
    ax.set_xlabel("Match Game Number (Context Length Increasing)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Win Rate against Asymmetric Opponent (%)", fontsize=10, fontweight="bold")
    ax.set_xticks(games)
    ax.set_ylim(0, 115)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved Benchmark 3 figure to {save_path}")


if __name__ == "__main__":
    t_cfg = TransformerConfig(vocab_size=256, seq_len=64, d_model=128, n_heads=4, n_layers=3)
    nnue = NNUESparseAccumulator(num_features=NUM_CHESS_FEATURES, accumulator_dim=128).to(device)
    jev = JevEpistemicHead(in_dim=128).to(device)
    transformer = InContextTransformer(t_cfg, modulation_dim=128).to(device)

    # Fast initial weights initialization
    eval_res = run_chess_adaptive_benchmark(nnue, jev, transformer, num_series=8)

    torch.save(eval_res, "data/chess_triprocess_results.pt")
    print("Saved raw benchmark data to data/chess_triprocess_results.pt")

    plot_chess_adaptive_results(eval_res, num_series=8)
