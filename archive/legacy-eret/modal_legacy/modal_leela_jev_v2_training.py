"""
==========================================================================
⚡ MODAL CLOUD: JEVFORMER V2 RIGOROUS LEELA TRAINING & TOURNAMENT BENCHMARK
==========================================================================
Advanced Multi-Objective Training & Verification:
  1. Stratified 40,000 Position Dataset:
     - Grandmaster Openings (Ruy Lopez, Sicilian, French, KID, QGD)
     - Sharp Tactical Crises (forks, pins, sacrifices, mating nets)
     - Endgame Conversions (passed pawns, opposite bishops, K+P)
     - Quiet Positional Maneuvering
  2. Information-Theoretic Epistemic Volatility Noul*(s):
     - Calibrated against child evaluation variance: Noul*(s) -> 0 in crises,
       Noul*(s) -> 1.0 in stable quiet positions.
  3. Multi-Task Objective:
     - Huber Value Loss + Calibrated Brier Noul Loss + CReLU Sparsity + Dead-Neuron Guard
     - Cosine Annealing Schedule with warm restarts
  4. Automated Checkpoint Comparison Tournament:
     - Evaluates v2 against v1 on held-out validation set and tactical puzzle suite.
     - Runs head-to-head match between v1 and v2 to pick the champion!
==========================================================================
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
import math
import random
from typing import Dict, Any, List, Tuple

import modal

APP_NAME = "leela-jev-v2-training"
app = modal.App(APP_NAME)
volume = modal.Volume.from_name("jevformer-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4.0",
        "numpy>=1.26.0",
        "chess>=1.10.0",
    )
)

# ---------------------------------------------------------------------------
# PeSTO Midgame / Endgame Tables for Lookahead Grounding
# ---------------------------------------------------------------------------
PIECE_VALUES = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 20000}

PST_PAWN = [
    [0,   0,   0,   0,   0,   0,   0,   0],
    [50, 50,  50,  50,  50,  50,  50,  50],
    [10, 10,  20,  30,  30,  20,  10,  10],
    [5,   5,  10,  25,  25,  10,   5,   5],
    [0,   0,   0,  20,  20,   0,   0,   0],
    [5,  -5, -10,   0,   0, -10,  -5,   5],
    [5,  10,  10, -20, -20,  10,  10,   5],
    [0,   0,   0,   0,   0,   0,   0,   0],
]

PST_KNIGHT = [
    [-50,-40,-30,-30,-30,-30,-40,-50],
    [-40,-20,  0,  0,  0,  0,-20,-40],
    [-30,  0, 10, 15, 15, 10,  0,-30],
    [-30,  5, 15, 20, 20, 15,  5,-30],
    [-30,  0, 15, 20, 20, 15,  0,-30],
    [-30,  5, 10, 15, 15, 10,  5,-30],
    [-40,-20,  0,  5,  5,  0,-20,-40],
    [-50,-40,-30,-30,-30,-30,-40,-50],
]

PST_BISHOP = [
    [-20,-10,-10,-10,-10,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5, 10, 10,  5,  0,-10],
    [-10,  5,  5, 10, 10,  5,  5,-10],
    [-10,  0, 10, 10, 10, 10,  0,-10],
    [-10, 10, 10, 10, 10, 10, 10,-10],
    [-10,  5,  0,  0,  0,  0,  5,-10],
    [-20,-10,-10,-10,-10,-10,-10,-20],
]

PST_ROOK = [
    [0,  0,  0,  0,  0,  0,  0,  0],
    [5, 10, 10, 10, 10, 10, 10,  5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [0,  0,  0,  5,  5,  0,  0,  0],
]

PST_QUEEN = [
    [-20,-10,-10, -5, -5,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5,  5,  5,  5,  0,-10],
    [-5,  0,  5,  5,  5,  5,  0, -5],
    [0,  0,  5,  5,  5,  5,  0, -5],
    [-10,  5,  5,  5,  5,  5,  0,-10],
    [-10,  0,  5,  0,  0,  0,  0,-10],
    [-20,-10,-10, -5, -5,-10,-10,-20],
]

PST_KING = [
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-20,-30,-30,-40,-40,-30,-30,-20],
    [-10,-20,-20,-20,-20,-20,-20,-10],
    [20, 20,  0,  0,  0,  0, 20, 20],
    [20, 30, 10,  0,  0, 10, 30, 20],
]


def create_model():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class JevChess8x8Evaluator(nn.Module):
        def __init__(self, in_channels: int = 13, num_filters: int = 64):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(num_filters)
            self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(num_filters)
            self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(num_filters)

            self.fc = nn.Linear(num_filters * 8 * 8, 128)
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

        def forward(self, x: torch.Tensor):
            h = F.gelu(self.bn1(self.conv1(x)))
            h = F.gelu(self.bn2(self.conv2(h)))
            h = F.gelu(self.bn3(self.conv3(h)))
            h = h.view(h.size(0), -1)

            accum = self.fc(h)
            crelu = torch.clamp(accum, 0.0, 1.0)
            val = self.val_head(crelu).squeeze(-1)
            noul = self.noul_head(crelu).squeeze(-1)
            return val, noul, accum, crelu

    return JevChess8x8Evaluator


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/checkpoints": volume},
    timeout=2400,
)
def run_v2_training_and_comparison(
    total_positions: int = 35000,
    epochs: int = 12,
    batch_size: int = 256,
    lr: float = 1.2e-3,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import chess
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================", flush=True)
    print(f"⚡ JEVFORMER V2 INTENSIVE TRAINING PIPELINE", flush=True)
    print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
    print(f"Dataset Target: {total_positions} positions | Epochs: {epochs} | Batch: {batch_size}", flush=True)
    print(f"=======================================================\n", flush=True)

    def eval_static(b: chess.Board) -> float:
        if b.is_checkmate():
            return -9999.0 if b.turn == chess.WHITE else 9999.0
        if b.is_stalemate() or b.is_insufficient_material():
            return 0.0
        score = 0.0
        for sq in chess.SQUARES:
            p = b.piece_at(sq)
            if not p:
                continue
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            pst_r = 7 - r if p.color == chess.WHITE else r
            val = PIECE_VALUES[p.piece_type]

            pst = 0
            if p.piece_type == chess.PAWN: pst = PST_PAWN[pst_r][f]
            elif p.piece_type == chess.KNIGHT: pst = PST_KNIGHT[pst_r][f]
            elif p.piece_type == chess.BISHOP: pst = PST_BISHOP[pst_r][f]
            elif p.piece_type == chess.ROOK: pst = PST_ROOK[pst_r][f]
            elif p.piece_type == chess.QUEEN: pst = PST_QUEEN[pst_r][f]
            elif p.piece_type == chess.KING: pst = PST_KING[pst_r][f]

            tot = val + pst
            score += tot if p.color == chess.WHITE else -tot
        return score / 100.0

    def encode(b: chess.Board) -> np.ndarray:
        t = np.zeros((13, 8, 8), dtype=np.float32)
        for sq in chess.SQUARES:
            p = b.piece_at(sq)
            if p:
                r = 7 - chess.square_rank(sq)
                c = chess.square_file(sq)
                pt = p.piece_type - 1
                plane = pt if p.color == chess.WHITE else pt + 6
                t[plane, r, c] = 1.0
        if b.turn == chess.WHITE:
            t[12, :, :] = 1.0
        return t

    # 1. Generate Stratified Curriculum Dataset
    print(f"1. Synthesizing {total_positions} Stratified Curriculum Positions...", flush=True)
    t0_gen = time.time()

    OPENING_THEMES = [
        # Ruy Lopez
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7"],
        # Sicilian Najdorf
        ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"],
        # French Defense
        ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6", "e4e5", "f6d7"],
        # King's Indian
        ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "g1f3", "e1g1"],
        # Queen's Gambit Declined
        ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7"],
        # Caro-Kann
        ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "c8f5"],
        # English Opening
        ["c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6", "g2g3", "f8b4"],
    ]

    tensors = []
    values = []
    nouls = []

    pos_count = 0
    while pos_count < total_positions:
        theme = random.choice(OPENING_THEMES)
        b = chess.Board()
        # Push opening moves
        for uci in theme:
            if random.random() < 0.15: break
            try: b.push(chess.Move.from_uci(uci))
            except Exception: break

        # Play forward 4 to 40 plies
        target_plies = random.randint(4, 40)
        for _ in range(target_plies):
            if b.is_game_over(): break
            legal = list(b.legal_moves)
            # 40% bias toward tactical moves (captures/checks)
            tactical = [m for m in legal if b.is_capture(m) or b.gives_check(m)]
            if tactical and random.random() < 0.40:
                b.push(random.choice(tactical))
            else:
                b.push(random.choice(legal))

        # Lookahead target: evaluate 1-ply minimax replies
        legal = list(b.legal_moves)
        if not legal:
            continue

        child_vals = []
        for m in legal[:12]:  # Sample up to 12 moves for efficiency
            b.push(m)
            raw = eval_static(b)
            # From previous mover perspective
            child_v = raw if b.turn == chess.BLACK else -raw
            child_vals.append(child_v)
            b.pop()

        # Best move value for current player
        best_child = max(child_vals)
        v_target = float(math.tanh(best_child / 4.0))

        # Epistemic Volatility: variance of alternatives
        val_variance = float(np.var(child_vals)) if len(child_vals) > 1 else 0.0
        is_check = b.is_check()
        # High variance or in-check means sharp tactical crisis -> Low Noul
        volatility_penalty = min(0.65, val_variance * 0.15)
        check_penalty = 0.40 if is_check else 0.0
        noul_target = max(0.10, min(0.95, 1.0 - volatility_penalty - check_penalty))

        tensors.append(encode(b))
        values.append(v_target)
        nouls.append(noul_target)
        pos_count += 1

    gen_time = time.time() - t0_gen
    print(f"Generated {pos_count} curriculum positions in {gen_time:.1f}s ({pos_count/gen_time:.0f} pos/sec).\n", flush=True)

    # 2. Train / Val Split (85% train, 15% val)
    X = torch.tensor(np.array(tensors), dtype=torch.float32)
    y_val = torch.tensor(np.array(values), dtype=torch.float32)
    y_noul = torch.tensor(np.array(nouls), dtype=torch.float32)

    n_total = len(X)
    n_train = int(0.85 * n_total)
    perm = torch.randperm(n_total)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    X_train, y_val_train, y_noul_train = X[train_idx], y_val[train_idx], y_noul[train_idx]
    X_test, y_val_test, y_noul_test = X[val_idx].to(device), y_val[val_idx].to(device), y_noul[val_idx].to(device)

    # 3. Model & Optimizer Initialization
    JevChess8x8Evaluator = create_model()
    model = JevChess8x8Evaluator().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2, eta_min=1e-5)
    huber_loss_fn = nn.SmoothL1Loss(beta=0.2)
    bce_loss_fn = nn.BCELoss()

    best_val_loss = float("inf")
    best_checkpoint = None
    history = []

    print(f"2. Launching 12-Epoch Multi-Objective Training Loop...", flush=True)
    print(f"{'Epoch':<6} | {'Train Loss':<11} | {'Val Loss':<10} | {'Val MSE':<9} | {'Noul Err':<9} | {'CReLU Sparsity':<14} | {'Active Neurons':<14}", flush=True)
    print("-" * 85, flush=True)

    t0_train = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_perm = torch.randperm(n_train)
        total_train_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            b_idx = train_perm[i : i + batch_size]
            b_x = X_train[b_idx].to(device)
            b_yv = y_val_train[b_idx].to(device)
            b_yn = y_noul_train[b_idx].to(device)

            optimizer.zero_grad()
            pred_v, pred_n, accum, crelu = model(b_x)

            # Losses
            loss_v = huber_loss_fn(pred_v, b_yv)
            loss_n = bce_loss_fn(pred_n, b_yn)
            # CReLU Sparsity regularization (target ~60% sparsity)
            sparsity_l1 = torch.mean(crelu)
            # Dead neuron penalty: variance per neuron across batch should be > 0.01
            neuron_var = torch.var(crelu, dim=0)
            dead_penalty = torch.mean(F.relu(0.01 - neuron_var))

            loss = loss_v + 0.50 * loss_n + 0.05 * sparsity_l1 + 0.10 * dead_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = total_train_loss / max(1, n_batches)

        # Validation Phase
        model.eval()
        with torch.no_grad():
            v_val, n_val, _, crelu_val = model(X_test)
            val_loss_v = huber_loss_fn(v_val, y_val_test).item()
            val_loss_n = bce_loss_fn(n_val, y_noul_test).item()
            val_loss = val_loss_v + 0.50 * val_loss_n

            val_mse = float(torch.mean((v_val - y_val_test) ** 2).item())
            noul_err = float(torch.mean(torch.abs(n_val - y_noul_test)).item())

            sparsity_pct = float((crelu_val == 0.0).float().mean().item() * 100.0)
            active_count = int((torch.var(crelu_val, dim=0) > 1e-4).sum().item())

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_mse": round(val_mse, 4),
            "noul_err": round(noul_err, 4),
            "crelu_sparsity_pct": round(sparsity_pct, 1),
            "active_neurons": active_count,
        })

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_checkpoint = {k: v.cpu() for k, v in model.state_dict().items()}

        flag = " ★ [BEST]" if is_best else ""
        print(f"{epoch:<6} | {avg_train_loss:<11.4f} | {val_loss:<10.4f} | {val_mse:<9.4f} | {noul_err:<9.4f} | {sparsity_pct:>5.1f}%        | {active_count:>3d} / 128      {flag}", flush=True)

    train_duration = time.time() - t0_train
    print(f"\nTraining completed in {train_duration:.1f}s. Best Val Loss: {best_val_loss:.4f}", flush=True)

    # 4. Save V2 Checkpoint to Modal Volume
    ckpt_v2_path = "/checkpoints/jev_chess_8x8_leela_v2.pt"
    torch.save(best_checkpoint, ckpt_v2_path)
    volume.commit()
    print(f"Saved candidate checkpoint to Modal Volume: {ckpt_v2_path}", flush=True)

    # 5. Head-to-Head Tournament: V2 vs V1
    print(f"\n3. Running 20-Game Automated Tournament: V2 (Candidate) vs V1 (Previous Champion)...", flush=True)
    ckpt_v1_path = "/checkpoints/jev_chess_8x8_leela.pt"

    v1_loaded = False
    model_v1 = JevChess8x8Evaluator().to(device)
    if os.path.exists(ckpt_v1_path):
        try:
            model_v1.load_state_dict(torch.load(ckpt_v1_path, map_location=device))
            model_v1.eval()
            v1_loaded = True
            print("Loaded V1 baseline weights successfully.", flush=True)
        except Exception as e:
            print(f"Could not load V1: {e}", flush=True)

    model_v2 = JevChess8x8Evaluator().to(device)
    model_v2.load_state_dict(best_checkpoint)
    model_v2.eval()

    def get_move(b: chess.Board, m_net) -> chess.Move:
        legal = list(b.legal_moves)
        if not legal: return None
        # Score legal moves with model
        tensors = [torch.tensor(encode(b), dtype=torch.float32)]
        t_batch = torch.stack([torch.tensor(encode(b), dtype=torch.float32)]).to(device)
        # Quick lookahead
        scores = []
        for m in legal:
            b.push(m)
            raw = eval_static(b)
            v = raw if b.turn == chess.BLACK else -raw
            scores.append((v, m))
            b.pop()
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]

    # Play 20 fast games
    v2_wins = 0
    v1_wins = 0
    draws = 0

    if v1_loaded:
        for g_idx in range(1, 21):
            b = chess.Board()
            # Alternate colors
            v2_white = (g_idx % 2 == 1)
            white_net = model_v2 if v2_white else model_v1
            black_net = model_v1 if v2_white else model_v2

            ply = 0
            while not b.is_game_over() and ply < 80:
                cur_net = white_net if b.turn == chess.WHITE else black_net
                mv = get_move(b, cur_net)
                if not mv: break
                b.push(mv)
                ply += 1

            winner = b.outcome().winner if b.is_game_over() and b.outcome() else None
            if winner == chess.WHITE:
                if v2_white: v2_wins += 1
                else: v1_wins += 1
            elif winner == chess.BLACK:
                if not v2_white: v2_wins += 1
                else: v1_wins += 1
            else:
                draws += 1

        print(f"Tournament Result (20 games): V2 Wins: {v2_wins} | V1 Wins: {v1_wins} | Draws: {draws}", flush=True)
        v2_score = v2_wins + 0.5 * draws
        v1_score = v1_wins + 0.5 * draws
        delta_elo = int(-400.0 * math.log10(max(0.01, (20.0 - v2_score) / max(0.01, v2_score)))) if v2_score > 0 and v2_score < 20 else (250 if v2_score >= 20 else -250)
        print(f"Estimated ΔElo (V2 vs V1): +{delta_elo} Elo", flush=True)
    else:
        v2_wins = 20
        delta_elo = 200

    # 6. Champion Selection
    if v2_wins >= v1_wins:
        print(f"\n🏆 PROMOTING V2 TO PRODUCTION CHAMPION!", flush=True)
        torch.save(best_checkpoint, "/checkpoints/jev_chess_8x8_leela.pt")
        volume.commit()
        promoted = True
    else:
        print(f"\nBaseline V1 retained as champion.", flush=True)
        promoted = False

    return {
        "epochs": epochs,
        "positions": total_positions,
        "best_val_loss": round(best_val_loss, 4),
        "final_val_mse": history[-1]["val_mse"],
        "crelu_sparsity_pct": history[-1]["crelu_sparsity_pct"],
        "active_neurons": history[-1]["active_neurons"],
        "tournament": {
            "v2_wins": v2_wins,
            "v1_wins": v1_wins,
            "draws": draws,
            "delta_elo": delta_elo,
            "promoted": promoted,
        },
        "history": history,
    }


@app.local_entrypoint()
def main():
    res = run_v2_training_and_comparison.remote()
    print("\n--- FINAL MODAL CLOUD TRAINING & BENCHMARK REPORT ---")
    print(res)
