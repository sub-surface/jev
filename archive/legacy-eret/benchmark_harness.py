"""
=============================================================================
⚡ JEV AUTORESEARCH: DETERMINISTIC BENCHMARK HARNESS
=============================================================================
The frozen ground truth for automated hypothesis evaluation:
1. 20 Tactical Crisis Positions (Sacrifices, Mates, Pins, Forks, Counter-attacks)
2. 10 Quiet Control Positions (Openings, Symmetrical Pawns, Theoretical Endgames)
3. Outputs single composite score:
   Score = 50 * (Solved/20) + 30 * CrisisRecall + 20 * QuietPrecision
=============================================================================
"""

import os
import sys
import time
import math
import json
from typing import Dict, Any, List, Tuple, Optional
import chess
import numpy as np
import torch

# Rule 1: Windows Unicode encoding hygiene
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Benchmark Suite: 20 Tactical Crises + 10 Quiet Controls
# ---------------------------------------------------------------------------
TACTICAL_CRISIS_SUITE = [
    {"id": "T01", "theme": "Back-Rank Mate", "fen": "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1", "best_move": "e1e8"},
    {"id": "T02", "theme": "Mating Net", "fen": "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4", "best_move": "f3f7"},
    {"id": "T03", "theme": "Queen Capture", "fen": "r1b1k2r/pppp1ppp/8/4q3/8/5N2/PPP1PPPP/R2QKB1R w KQkq - 0 9", "best_move": "f3e5"},
    {"id": "T04", "theme": "Defend Check", "fen": "r1bqkb1r/pppp1ppp/2n5/8/4Q3/8/PPP1PPPP/RNB1KBNR b KQkq - 0 4", "best_move": "f8e7"},
    {"id": "T05", "theme": "Positional Restriction", "fen": "6k1/5p1p/6p1/8/8/5N2/1Q3PPP/6K1 w - - 0 1", "best_move": "b2f6"},
    {"id": "T06", "theme": "Greek Gift Sacrifice", "fen": "4r1k1/ppp2ppp/8/8/3b4/1P1B4/P1PP1PPP/R5K1 w - - 0 1", "best_move": "d3h7"},
    {"id": "T07", "theme": "Center Counter-strike", "fen": "r1b1k2r/pppp1Npp/8/4p3/2Bn3q/8/PPPP2PP/RNBQ1K1R b kq - 2 8", "best_move": "d7d5"},
    {"id": "T08", "theme": "Pawn Promotion Race", "fen": "8/4P3/8/8/8/8/1k6/4K3 w - - 0 1", "best_move": "e7e8q"},
    {"id": "T09", "theme": "Pin Knight", "fen": "8/8/4k3/8/8/2n5/3R4/4K3 w - - 0 1", "best_move": "d2d3"},
    {"id": "T10", "theme": "Capture Hanging Queen", "fen": "r1b1k2r/pppp1ppp/2n5/8/1b2q3/2N5/PPPBPPPP/R2QKB1R w KQkq - 0 7", "best_move": "c3e4"},
    {"id": "T11", "theme": "King Escape Skewer", "fen": "3k4/8/8/8/8/8/4R3/4K1q1 w - - 0 1", "best_move": "e1d2"},
    {"id": "T12", "theme": "Solid Central Guard", "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 4", "best_move": "d2d3"},
    {"id": "T13", "theme": "Back-Rank Deflection", "fen": "2r3k1/5ppp/8/8/8/8/2R5/4K3 w - - 0 1", "best_move": "c2c8"},
    {"id": "T14", "theme": "Endgame Cutoff", "fen": "8/8/8/8/pk6/8/R7/4K3 w - - 0 1", "best_move": "e1d2"},
    {"id": "T15", "theme": "Capture Blundered Piece", "fen": "r1b2rk1/pp1p1ppp/2n1p3/8/1bPNn3/2N3P1/PPQBPP1P/R3KB1R w KQ - 0 9", "best_move": "c2e4"},
    {"id": "T16", "theme": "Center Recapture", "fen": "rnbqkbnr/ppp2ppp/8/3pp3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3", "best_move": "f3e5"},
    {"id": "T17", "theme": "Tempo Attack", "fen": "r1bqk2r/pppp1ppp/2n5/4b3/4P3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 7", "best_move": "f2f4"},
    {"id": "T18", "theme": "Fried Liver Attack", "fen": "r1bqk2r/ppppbppp/2n2n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq - 4 5", "best_move": "g5f7"},
    {"id": "T19", "theme": "Rook Exchange Mate", "fen": "3r2k1/p4ppp/8/8/8/8/P4PPP/3R2K1 w - - 0 1", "best_move": "d1d8"},
    {"id": "T20", "theme": "King Stop Promotion", "fen": "8/8/8/8/8/2K5/1p6/k7 w - - 0 1", "best_move": "c3b3"},
]

QUIET_CONTROL_SUITE = [
    {"id": "Q01", "theme": "Starting Position", "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},
    {"id": "Q02", "theme": "Queen's Gambit Declined", "fen": "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4"},
    {"id": "Q03", "theme": "Ruy Lopez Quiet", "fen": "r1bqk2r/pppp1ppp/2n2n2/4p3/1bB1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5"},
    {"id": "Q04", "theme": "English Opening Quiet", "fen": "rnbqkb1r/pppppppp/5n2/8/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 1 2"},
    {"id": "Q05", "theme": "French Defense Closed", "fen": "rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3"},
    {"id": "Q06", "theme": "Caro-Kann Advance", "fen": "rn1qkbnr/pp2pppp/2p5/3pP3/3P4/8/PPP2PPP/RNBQKBNR w KQkq - 1 4"},
    {"id": "Q07", "theme": "King's Indian Setup", "fen": "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R b KQ - 1 6"},
    {"id": "Q08", "theme": "Quiet Rook Endgame", "fen": "8/5pk1/7p/7P/8/4r1P1/5RK1/8 w - - 1 45"},
    {"id": "Q09", "theme": "Symmetrical Pawn Endgame", "fen": "8/4k3/4p3/4P3/8/4K3/8/8 w - - 0 1"},
    {"id": "Q10", "theme": "Lucena Bridge Theoretical", "fen": "1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1"},
]


def evaluate_engine_checkpoint(
    select_fn,
    tau: float = 0.70,
) -> Dict[str, Any]:
    """
    Evaluates select_fn across the entire benchmark suite.
    select_fn(board) must return (move: chess.Move, noul: float, depth: int)
    """
    tactical_solved = 0
    crisis_detected = 0
    latencies = []

    print("\n" + "=" * 90)
    print("🎯 AUTORESEARCH BENCHMARK EVALUATION")
    print("=" * 90)
    print(f"{'ID':<4} | {'Theme':<24} | {'Target':<7} | {'Chosen':<7} | {'Solved':<7} | {'Noul':<6} | {'Status':<10} | {'Latency':<8}")
    print("-" * 90)

    for item in TACTICAL_CRISIS_SUITE:
        board = chess.Board(item["fen"])
        best_uci = item["best_move"].lower()

        t0 = time.time()
        move, noul, depth = select_fn(board)
        dt_ms = (time.time() - t0) * 1000.0
        latencies.append(dt_ms)

        chosen_uci = move.uci().lower() if move else "none"
        is_solved = (chosen_uci == best_uci) or (best_uci in chosen_uci)
        if is_solved:
            tactical_solved += 1

        is_crisis = (depth > 1) or (noul < tau) or board.is_check()
        if is_crisis:
            crisis_detected += 1

        solved_str = "✅ YES" if is_solved else "❌ NO"
        stat_str = f"CRISIS(d{depth})" if is_crisis else "QUIET(d1)"
        print(f"{item['id']:<4} | {item['theme']:<24} | {best_uci:<7} | {chosen_uci:<7} | {solved_str:<7} | {noul:<6.2f} | {stat_str:<10} | {dt_ms:>6.1f}ms")

    # Quiet Controls
    quiet_passed = 0
    print("-" * 90)
    print("🛡️ QUIET POSITIONAL CONTROLS (Expect Noul >= 0.70 & Fast Reflex)")
    print("-" * 90)
    for item in QUIET_CONTROL_SUITE:
        board = chess.Board(item["fen"])
        t0 = time.time()
        move, noul, depth = select_fn(board)
        dt_ms = (time.time() - t0) * 1000.0
        latencies.append(dt_ms)

        is_quiet = (noul >= tau) and (depth <= 1)
        if is_quiet:
            quiet_passed += 1

        pass_str = "✅ PASS" if is_quiet else "⚠️ ESCALATED"
        print(f"{item['id']:<4} | {item['theme']:<24} | {'-':<7} | {move.uci() if move else 'none':<7} | {pass_str:<7} | {noul:<6.2f} | {'QUIET' if is_quiet else 'CRISIS':<10} | {dt_ms:>6.1f}ms")

    n_tactical = len(TACTICAL_CRISIS_SUITE)
    n_quiet = len(QUIET_CONTROL_SUITE)

    solve_pct = (tactical_solved / n_tactical) * 100.0
    crisis_recall_pct = (crisis_detected / n_tactical) * 100.0
    quiet_precision_pct = (quiet_passed / n_quiet) * 100.0
    median_lat = float(np.median(latencies))

    # Composite Score: 0 to 100
    composite_score = 0.50 * solve_pct + 0.30 * crisis_recall_pct + 0.20 * quiet_precision_pct

    print("\n" + "=" * 90)
    print(f"📊 BENCHMARK RESULTS SUMMARY:")
    print(f"   • Tactical Solve Rate:       {solve_pct:.1f}% ({tactical_solved}/{n_tactical})")
    print(f"   • Crisis Escalation Recall:  {crisis_recall_pct:.1f}% ({crisis_detected}/{n_tactical})")
    print(f"   • Quiet Control Precision:   {quiet_precision_pct:.1f}% ({quiet_passed}/{n_quiet})")
    print(f"   • Median Latency:            {median_lat:.1f}ms")
    print(f"   ⭐ COMPOSITE NORTH STAR SCORE: {composite_score:.2f} / 100.00")
    print("=" * 90 + "\n")

    return {
        "tactical_solve_pct": round(solve_pct, 1),
        "crisis_recall_pct": round(crisis_recall_pct, 1),
        "quiet_precision_pct": round(quiet_precision_pct, 1),
        "median_latency_ms": round(median_lat, 1),
        "composite_score": round(composite_score, 2),
    }


if __name__ == "__main__":
    # Test benchmark harness with current champion
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from chess_8x8_engine import JevChess8x8Evaluator, select_move_adaptive_ets, score_move_candidates

    device = torch.device("cpu")
    model = JevChess8x8Evaluator(in_channels=13, num_filters=64).to(device)
    model_path = "data/jev_chess_8x8_gui.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded: {model_path}")
    model.eval()

    def baseline_select_fn(board: chess.Board):
        chosen_move, depth = select_move_adaptive_ets(board, model, remaining_clock=30.0, tau=0.70)
        candidates = score_move_candidates(board, model)
        if candidates:
            top_val = candidates[0]["val"]
            second_val = candidates[1]["val"] if len(candidates) > 1 else top_val
            margin = top_val - second_val
            has_tactical = any(c.get("is_tactical", False) for c in candidates[:4])
            is_check = board.is_check()
            noul = float(math.tanh(margin * 3.0))
            if has_tactical or is_check:
                noul *= 0.40
        else:
            noul = 0.50
        return chosen_move, noul, depth

    evaluate_engine_checkpoint(baseline_select_fn)
