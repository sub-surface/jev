"""
==========================================================================
⚡ JEV ENGINE: RIGOROUS STRENGTH & TACTICAL BENCHMARK SUITE
==========================================================================
Evaluates the tactical competence, epistemic crisis detection, and latency
distribution of the Jevformer engine on standard chess test positions.
==========================================================================
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
import math
import os
import json
import torch
import chess
import numpy as np
from typing import Dict, Any, List, Tuple

# Import engine functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chess_8x8_engine import (
    JevChess8x8Evaluator,
    select_move_adaptive_ets,
    score_move_candidates,
    encode_board_tensor,
)

# ---------------------------------------------------------------------------
# Standard Tactical Test Suite (Curated FENs with Known Best Tactical Moves)
# ---------------------------------------------------------------------------
TACTICAL_BENCHMARK_POSITIONS = [
    # 1. Back-rank mate in 1
    {"fen": "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1", "best_move": "e1e8", "theme": "Back-Rank Mate"},
    # 2. Scholar's mate attack
    {"fen": "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4", "best_move": "f3f7", "theme": "Mating Net"},
    # 3. Knight fork winning Queen
    {"fen": "r1b1k2r/pppp1ppp/8/4q3/8/5N2/PPP1PPPP/R2QKB1R w KQkq - 0 9", "best_move": "f3e5", "theme": "Queen Capture"},
    # 4. Queen hanging capture
    {"fen": "r1bqkb1r/pppp1ppp/2n5/8/4Q3/8/PPP1PPPP/RNB1KBNR b KQkq - 0 4", "best_move": "f8e7", "theme": "Defend Check"},
    # 5. Smothered mate motif
    {"fen": "6k1/5p1p/6p1/8/8/5N2/1Q3PPP/6K1 w - - 0 1", "best_move": "b2f6", "theme": "Positional Restriction"},
    # 6. Pin exploitation
    {"fen": "4r1k1/ppp2ppp/8/8/3b4/1P1B4/P1PP1PPP/R5K1 w - - 0 1", "best_move": "d3h7", "theme": "Greek Gift Motif"},
    # 7. Discovered check
    {"fen": "r1b1k2r/pppp1Npp/8/4p3/2Bn3q/8/PPPP2PP/RNBQ1K1R b kq - 2 8", "best_move": "d7d5", "theme": "Counter-strike"},
    # 8. Pawn promotion race
    {"fen": "8/4P3/8/8/8/8/1k6/4K3 w - - 0 1", "best_move": "e7e8q", "theme": "Pawn Promotion"},
    # 9. Rook fork on king and knight
    {"fen": "8/8/4k3/8/8/2n5/3R4/4K3 w - - 0 1", "best_move": "d2d3", "theme": "Pin/Attack Knight"},
    # 10. Trapped Queen
    {"fen": "r1b1k2r/pppp1ppp/2n5/8/1b2q3/2N5/PPPBPPPP/R2QKB1R w KQkq - 0 7", "best_move": "c3e4", "theme": "Capture Queen"},
    # 11. Skewer along file
    {"fen": "4k3/8/8/8/8/8/4R3/4K1q1 w - - 0 1", "best_move": "e1d2", "theme": "King Escape"},
    # 12. Double attack
    {"fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 4", "best_move": "d2d3", "theme": "Positional Solid"},
    # 13. Deflection / Decoy
    {"fen": "2r3k1/5ppp/8/8/8/8/2R5/4K3 w - - 0 1", "best_move": "c2c8", "theme": "Back-Rank Rook Mate"},
    # 14. Rook skewer
    {"fen": "8/8/8/8/pk6/8/R7/4K3 w - - 0 1", "best_move": "e1d2", "theme": "Endgame Cutoff"},
    # 15. Queen and Bishop battery
    {"fen": "r1b2rk1/pp1p1ppp/2n1p3/8/1bPNn3/2N3P1/PPQBPP1P/R3KB1R w KQ - 0 9", "best_move": "c2e4", "theme": "Capture Blundered Knight"},
    # 16. Tactical sacrifice defense
    {"fen": "rnbqkbnr/ppp2ppp/8/3pp3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3", "best_move": "f3e5", "theme": "Center Pawn Recapture"},
    # 17. Free bishop
    {"fen": "r1bqk2r/pppp1ppp/2n5/4b3/4P3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 7", "best_move": "f2f4", "theme": "Tempo Attack"},
    # 18. Fork on c7
    {"fen": "r1bqk2r/ppppbppp/2n2n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq - 4 5", "best_move": "g5f7", "theme": "Fried Liver Fork"},
    # 19. Back rank mate variation
    {"fen": "3r2k1/p4ppp/8/8/8/8/P4PPP/3R2K1 w - - 0 1", "best_move": "d1d8", "theme": "Rook Exchange Mate"},
    # 20. Winning endgame queen vs pawn
    {"fen": "8/8/8/8/8/2K5/1p6/k7 w - - 0 1", "best_move": "c3b3", "theme": "Stop Promotion"},
]


def run_bot_strength_evaluation(
    model_path: str = "data/jev_chess_8x8_gui.pt",
    tau: float = 0.70,
    test_clock_seconds: float = 30.0,
) -> Dict[str, Any]:
    """
    Runs full tactical and reflex evaluation on the local Jev engine.
    """
    device = torch.device("cpu")
    model = JevChess8x8Evaluator(in_channels=13, num_filters=64).to(device)
    if not os.path.exists(model_path):
        alt_path = os.path.join(os.path.dirname(__file__), "..", "analysis", "jev_chess_8x8_gui.pt")
        if os.path.exists(alt_path):
            model_path = alt_path
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded Jev Model: {model_path}", flush=True)
    else:
        print(f"Warning: {model_path} not found, testing with uninitialized weights.", flush=True)
    model.eval()

    results = []
    solved_count = 0
    crisis_detected_count = 0
    reflex_solved_count = 0
    latencies = []

    print("\n--- Running Tactical Suite (20 Benchmarks) ---", flush=True)
    print(f"{'#':<3} | {'Theme':<24} | {'Target':<7} | {'Chosen':<7} | {'Solved':<7} | {'Noul':<6} | {'Crisis?':<8} | {'Latency':<8}", flush=True)
    print("-" * 90, flush=True)

    for i, test in enumerate(TACTICAL_BENCHMARK_POSITIONS, 1):
        board = chess.Board(test["fen"])
        best_uci = test["best_move"].lower()

        # Time move selection
        t0 = time.time()
        chosen_move, search_depth = select_move_adaptive_ets(board, model, remaining_clock=test_clock_seconds, tau=tau)
        elapsed_ms = (time.time() - t0) * 1000.0
        latencies.append(elapsed_ms)

        chosen_uci = chosen_move.uci().lower()

        # Check if model recognized position as tactical crisis
        t_enc = encode_board_tensor(board).unsqueeze(0).to(device)
        with torch.no_grad():
            _, noul_t, _, _ = model(t_enc)
            noul_val = float(noul_t.item())

        is_crisis = (noul_val < tau) or board.is_check()
        if is_crisis:
            crisis_detected_count += 1

        is_solved = (chosen_uci == best_uci) or (best_uci in chosen_uci)
        if is_solved:
            solved_count += 1
            if search_depth == 1:
                reflex_solved_count += 1

        solved_str = "✅ YES" if is_solved else "❌ NO"
        crisis_str = "CRISIS" if is_crisis else "QUIET"

        print(f"{i:<3} | {test['theme']:<24} | {best_uci:<7} | {chosen_uci:<7} | {solved_str:<7} | {noul_val:<6.2f} | {crisis_str:<8} | {elapsed_ms:>6.1f}ms", flush=True)

    # Compute aggregate metrics
    total_tests = len(TACTICAL_BENCHMARK_POSITIONS)
    solve_rate_pct = (solved_count / total_tests) * 100.0
    crisis_recall_pct = (crisis_detected_count / total_tests) * 100.0
    reflex_solve_pct = (reflex_solved_count / max(1, solved_count)) * 100.0

    arr_lat = np.array(latencies)
    p50_lat = float(np.percentile(arr_lat, 50))
    p95_lat = float(np.percentile(arr_lat, 95))
    avg_lat = float(np.mean(arr_lat))

    # Estimated Elo based on tactical solve rate and bullet clock efficiency
    # Baseline 1500 + solve_rate_bonus (up to 900) + speed bonus (up to 150)
    estimated_elo = int(1500 + (solve_rate_pct / 100.0) * 900 + max(0, (100.0 - p50_lat) * 1.5))

    report = {
        "model": model_path,
        "total_benchmarks": total_tests,
        "solved": solved_count,
        "solve_rate_pct": round(solve_rate_pct, 1),
        "crisis_recall_pct": round(crisis_recall_pct, 1),
        "reflex_solve_pct": round(reflex_solve_pct, 1),
        "latency_p50_ms": round(p50_lat, 1),
        "latency_p95_ms": round(p95_lat, 1),
        "latency_avg_ms": round(avg_lat, 1),
        "estimated_bullet_elo": estimated_elo,
    }

    print("\n=======================================================", flush=True)
    print("🏆 JEV ENGINE EVALUATION SUMMARY", flush=True)
    print("=======================================================", flush=True)
    print(f"Tactical Solve Rate:     {solve_rate_pct:.1f}% ({solved_count}/{total_tests})", flush=True)
    print(f"Crisis Escalation Recall:{crisis_recall_pct:.1f}%", flush=True)
    print(f"Median Reflex Latency:   {p50_lat:.1f}ms (p95: {p95_lat:.1f}ms)", flush=True)
    print(f"Estimated Bullet Elo:    ~{estimated_elo} Elo", flush=True)
    print("=======================================================\n", flush=True)

    # Save report to vault analysis
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../analysis/bot_strength_evaluation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation report to: {out_path}", flush=True)

    return report


if __name__ == "__main__":
    run_bot_strength_evaluation()
