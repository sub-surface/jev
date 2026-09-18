"""
=============================================================================
⚡ JEV / ERET EVALUATION ENTRYPOINT
=============================================================================
Evaluates the champion ERET (Epistemic Recurrent Equilibrium Transformer)
model against the frozen 30-position benchmark suite and official Stockfish 19.

Usage:
    python evaluate.py                       # Run 30-position tactical & quiet benchmark
    python evaluate.py --stockfish           # Run benchmark + Stockfish 19 agreement & match
    python evaluate.py --model path/to.pt    # Evaluate specific checkpoint
=============================================================================
"""

import os
import sys
import argparse

# Rule 1: Windows Unicode encoding hygiene
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add jev-vault/src to path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jev-vault", "src")
sys.path.insert(0, SRC_DIR)

import torch
from eret_engine import ERETChessEngine
from benchmark_harness import evaluate_engine_checkpoint
from stockfish_validator import select_eret_move, run_benchmark_stockfish_agreement, play_match_vs_stockfish


def main():
    parser = argparse.ArgumentParser(description="JEV / ERET Neural Chess Evaluation Harness")
    parser.add_argument("--model", type=str, default="data/jev_champion.pt", help="Path to model weights (.pt)")
    parser.add_argument("--stockfish", action="store_true", help="Run full validation against Stockfish 19 UCI engine")
    parser.add_argument("--sf-depth", type=int, default=10, help="Stockfish search depth (default: 10)")
    parser.add_argument("--sf-match", action="store_true", help="Play 2-game head-to-head match vs Stockfish Level 1")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, cpu)")
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"⚡ JEV Evaluation Harness | Device: {device} | Model: {args.model}")

    # Load model
    model = ERETChessEngine().to(device)
    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f" Loaded weights from: {args.model}")
    else:
        print(f"⚠️ Model checkpoint not found at {args.model}. Evaluating untrained initialization.")
    model.eval()

    # 1. Run Frozen 30-Position Ground Truth Benchmark
    print("\n[Phase 1] Evaluating Frozen 30-Position Ground Truth Suite...")
    bm_results = evaluate_engine_checkpoint(
        select_fn=lambda b: select_eret_move(b, model, device),
        tau=0.35,
    )

    # 2. Run Stockfish Agreement & Matches (if requested)
    if args.stockfish or args.sf_match:
        print("\n[Phase 2] Evaluating Agreement Against Stockfish 19...")
        sf_res = run_benchmark_stockfish_agreement(model=model, device=device, depth=args.sf_depth)

        if args.sf_match:
            print("\n[Phase 3] Playing Head-to-Head Matches vs Stockfish Level 1...")
            play_match_vs_stockfish(model=model, device=device, sf_skill_level=1, num_games=2, max_plies=60)


if __name__ == "__main__":
    main()
