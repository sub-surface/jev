"""
=============================================================================
⚡ JEV AUTORESEARCH: AUTONOMOUS EXPERIMENT & LEADERBOARD DRIVER
=============================================================================
Karpathy-style autonomous research loop for Jevformer:
1. Benchmarks current candidate against deterministic ground truth (20 tactical, 10 quiet).
2. Calculates composite North Star Score:
   Score = 50 * (Solved/20) + 30 * CrisisRecall + 20 * QuietPrecision
3. Promotes new champions if Score > Champion.
4. Updates Markdown Leaderboard: jev-vault/analysis/AUTORESEARCH_LEADERBOARD.md.
5. Saves animated GIF of best self-play game.
=============================================================================
"""

import os
import sys
import time
import json
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Rule 1: Windows Unicode encoding hygiene
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from benchmark_harness import evaluate_engine_checkpoint, TACTICAL_CRISIS_SUITE, QUIET_CONTROL_SUITE
from eret_engine import ERETChessEngine, encode_board_13
from fast_parallel_selfplay import run_tier2_selfplay_batch
from batched_bitter_mcts import render_game_gif

LEADERBOARD_PATH = "jev-vault/analysis/AUTORESEARCH_LEADERBOARD.md"
CHAMPION_META_PATH = "data/champion_metadata.json"


def update_leaderboard(
    run_id: str,
    arch_desc: str,
    results: dict,
    status: str,
):
    """Appends an entry to the markdown leaderboard."""
    row = (
        f"| {'👑 ' if status == 'PROMOTED' else ''}{run_id} | {arch_desc} | "
        f"{results['tactical_solve_pct']}% | {results['crisis_recall_pct']}% | "
        f"{results['quiet_precision_pct']}% | {results['median_latency_ms']}ms | "
        f"**{results['composite_score']}** | {status} |\n"
    )

    if not os.path.exists(LEADERBOARD_PATH):
        header = (
            "# 🏆 JEVFORMER AUTORESEARCH LEADERBOARD\n\n"
            "| Run ID | Architecture & Hypothesis | Tactical Solve % | Crisis Recall % | Quiet Precision % | Median Latency | North Star Score | Status |\n"
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        )
        content = header + row
    else:
        with open(LEADERBOARD_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Insert row into the table
        table_end_idx = len(lines)
        for i, line in enumerate(lines):
            if line.startswith("| :---"):
                table_end_idx = i + 1
                break
        lines.insert(table_end_idx, row)
        content = "".join(lines)

    with open(LEADERBOARD_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated Leaderboard at: {LEADERBOARD_PATH}", flush=True)


def get_current_champion_score() -> float:
    if os.path.exists(CHAMPION_META_PATH):
        try:
            with open(CHAMPION_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
                return float(meta.get("composite_score", 52.50))
        except Exception:
            pass
    return 52.50


def save_champion_metadata(run_id: str, results: dict, checkpoint_path: str):
    meta = {
        "run_id": run_id,
        "composite_score": results["composite_score"],
        "tactical_solve_pct": results["tactical_solve_pct"],
        "crisis_recall_pct": results["crisis_recall_pct"],
        "quiet_precision_pct": results["quiet_precision_pct"],
        "median_latency_ms": results["median_latency_ms"],
        "checkpoint_path": checkpoint_path,
        "timestamp": time.time(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(CHAMPION_META_PATH)), exist_ok=True)
    with open(CHAMPION_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Autoresearch Experiment 1: Calibrated Epistemic Arbitration + Mate Gate
# ---------------------------------------------------------------------------
def run_experiment_1():
    print("\n=======================================================", flush=True)
    print("🔬 AUTORESEARCH EXP-01: Calibrated Epistemic Arbitration + Mate Gate", flush=True)
    print("Hypothesis: Resolving Aporia Inversion and 1-ply Mate Gates lifts score >60.0", flush=True)
    print("=======================================================\n", flush=True)

    from chess_8x8_engine import JevChess8x8Evaluator, score_move_candidates, negamax_alpha_beta

    device = torch.device("cpu")
    model = JevChess8x8Evaluator(in_channels=13, num_filters=64).to(device)
    model.load_state_dict(torch.load("data/jev_chess_8x8_gui.pt", map_location=device))
    model.eval()

    def exp1_select_fn(board: chess.Board):
        # 1. Direct Checkmate Gate (<0.1ms)
        for m in board.legal_moves:
            board.push(m)
            if board.is_checkmate():
                board.pop()
                return m, 0.01, 1
            board.pop()

        candidates = score_move_candidates(board, model)
        if not candidates:
            return None, 0.5, 1

        top_val = candidates[0]["val"]
        second_val = candidates[1]["val"] if len(candidates) > 1 else top_val
        margin = top_val - second_val
        has_tactical = any(c.get("is_tactical", False) for c in candidates[:4])
        is_check = board.is_check()

        # Epistemic calibration: peaceful indifference != crisis
        if is_check:
            noul = 0.10
        elif has_tactical and (abs(top_val) > 0.50 or margin > 0.35):
            noul = 0.20
        elif has_tactical:
            noul = 0.45
        else:
            noul = 0.88

        if noul >= 0.70:
            return chess.Move.from_uci(candidates[0]["uci"]), noul, 1

        # Search top candidates with dedicated timer
        search_cands = [c for c in candidates if c.get("is_tactical", False) or c["val"] >= top_val - 0.30][:8]
        if not search_cands:
            search_cands = candidates[:5]

        current_turn = 1 if board.turn == chess.WHITE else -1
        best_move = chess.Move.from_uci(candidates[0]["uci"])
        best_score = -999999.0
        alpha = -999999.0
        beta = 999999.0
        tt = {}
        t0_search = time.time()
        max_duration = 0.25

        for cand in search_cands:
            m = chess.Move.from_uci(cand["uci"])
            board.push(m)
            score = -negamax_alpha_beta(board, 2, -beta, -alpha, -current_turn, model, t0_search, max_duration, tt)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = m
            alpha = max(alpha, score)
            if best_score > 90000.0:
                break
            if time.time() - t0_search > max_duration:
                break

        return best_move, noul, 3

    results = evaluate_engine_checkpoint(exp1_select_fn)
    champ_score = get_current_champion_score()

    if results["composite_score"] > champ_score:
        status = "PROMOTED"
        print(f"🎉 NEW CHAMPION! Score {results['composite_score']} > {champ_score}", flush=True)
        save_champion_metadata("EXP-01-CALIBRATED-GATE", results, "data/jev_chess_8x8_gui.pt")
    else:
        status = "REVERTED"

    update_leaderboard("EXP-01-CALIBRATED-GATE", "Calibrated Noul + 1-ply Mate Gate", results, status)
    return results


# ---------------------------------------------------------------------------
# Autoresearch Experiment 2: ERET Equilibrium Self-Play Fast Training
# ---------------------------------------------------------------------------
def run_experiment_2(epochs: int = 4, games_per_epoch: int = 16):
    print("\n=======================================================", flush=True)
    print("🔬 AUTORESEARCH EXP-02: ERET Krasnoselskii-Mann Equilibrium Training", flush=True)
    print("Hypothesis: Looped equilibrium reasoning resolves latent piece tension", flush=True)
    print("=======================================================\n", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training ERET on: {device}", flush=True)

    model = ERETChessEngine(max_unrolls=4, tau_halt=0.70).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    replay_states = []
    replay_values = []
    replay_policies = []
    last_game_moves = []
    last_game_res = ""

    t0_train = time.time()

    for epoch in range(1, epochs + 1):
        # 1. Tier-2 Actor-Batcher self-play
        states, vals, pols, moves, res = run_tier2_selfplay_batch(
            model=model,
            device=device,
            num_games=games_per_epoch,
            num_threads=4,
        )
        replay_states.extend(states)
        replay_values.extend(vals)
        replay_policies.extend(pols)
        last_game_moves = moves
        last_game_res = res

        # 2. Train on replay buffer
        model.train()
        X_t = torch.tensor(np.array(replay_states), dtype=torch.float32, device=device)
        y_val_t = torch.tensor(np.array(replay_values), dtype=torch.float32, device=device)
        y_pol_t = torch.tensor(np.array(replay_policies), dtype=torch.float32, device=device)

        indices = np.arange(len(X_t))
        batch_size = 64
        for _ in range(2):
            np.random.shuffle(indices)
            for s_idx in range(0, len(X_t), batch_size):
                b_idx = indices[s_idx:s_idx + batch_size]
                optimizer.zero_grad()
                v_pred, p_pred, noul_pred, crelu, _ = model(X_t[b_idx])
                loss_v = F.mse_loss(v_pred, y_val_t[b_idx])
                loss_p = -torch.sum(y_pol_t[b_idx] * F.log_softmax(p_pred, dim=-1), dim=-1).mean()
                loss_sparse = 0.02 * torch.mean(torch.abs(crelu))
                loss = loss_v + loss_p + loss_sparse
                loss.backward()
                optimizer.step()

        print(f"Epoch {epoch}/{epochs} | Replay Positions: {len(replay_states)} | Loss V: {loss_v.item():.4f} | Loss P: {loss_p.item():.4f}", flush=True)

    dt = time.time() - t0_train
    print(f"ERET Self-Play & Training completed in {dt:.1f}s!", flush=True)

    # Save model and GIF
    ckpt_path = "data/eret_v1.pt"
    torch.save(model.state_dict(), ckpt_path)
    gif_path = "jev-vault/figures/eret_v1_selfplay.gif"
    if last_game_moves:
        render_game_gif(last_game_moves, gif_path, result_str=last_game_res)
        print(f"🎬 Rendered self-play animation to: {gif_path}", flush=True)

    # 3. Benchmark ERET Engine
    model.eval()
    def eret_select_fn(board: chess.Board):
        # 1-ply mate
        for m in board.legal_moves:
            board.push(m)
            if board.is_checkmate():
                board.pop()
                return m, 0.01, 1
            board.pop()

        legal = list(board.legal_moves)
        if not legal:
            return None, 0.5, 1

        enc = encode_board_13(board)
        tx = torch.tensor(enc, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            v, p, noul, _, unrolls = model(tx)

        noul_val = float(noul.item())
        pol_np = p[0].cpu().numpy()
        legal_indices = [m.from_square * 64 + m.to_square for m in legal]
        sub_logits = pol_np[legal_indices]
        best_legal_idx = int(np.argmax(sub_logits))
        chosen_m = legal[best_legal_idx]

        depth = unrolls
        return chosen_m, noul_val, depth

    results = evaluate_engine_checkpoint(eret_select_fn)
    champ_score = get_current_champion_score()

    if results["composite_score"] > champ_score:
        status = "PROMOTED"
        print(f"🎉 NEW CHAMPION! Score {results['composite_score']} > {champ_score}", flush=True)
        save_champion_metadata("EXP-02-ERET-EQUILIBRIUM", results, ckpt_path)
    else:
        status = "REVERTED"

    update_leaderboard("EXP-02-ERET-EQUILIBRIUM", "ERET Looped Krasnoselskii-Mann Equilibrium", results, status)
    return results


if __name__ == "__main__":
    # Run Experiment 1 (Calibrated Gate)
    r1 = run_experiment_1()
