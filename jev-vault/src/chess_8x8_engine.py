"""
Standard 8x8 Bullet Chess Engine with Jev Epistemic Gating & Negamax Search
===========================================================================
Authors: Leon & The Research Collective

Provides:
  - Full standard 8x8 chess rules via `python-chess` (castling, en passant, promotion,
    checks, checkmate, stalemate, 50-move rule, threefold repetition).
  - Neural Jev Epistemic Evaluator (JevChess8x8Evaluator):
      * Outputs Value V(s) in [-1, 1] and Epistemic Certainty Noul(s) in [0, 1].
      * Raw 128-dim accumulator representation for Mechanistic Interpretability.
  - Adaptive Epistemic Tree Search (ETS):
      * 1-ply reflex (~0.5ms) during quiet positional play to preserve bullet clock.
      * Gated Alpha-Beta / Negamax (Depth 2-4) during sharp tactical crises (Noul < tau).
  - Move Credence & Ranking Engine:
      * Live scoring, credences P(m), Noul, and tactical classification for all legal moves.
"""

from __future__ import annotations

import os
import sys
import time
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BOARD_SIZE = 8

# Standard piece values in centipawns
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# Piece-Square Tables (from White's perspective: rank 0 = rank 1, rank 7 = rank 8)
# We flatten or index as [rank][file]
PST_PAWN = np.array([
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [ 5,  5, 10, 25, 25, 10,  5,  5],
    [ 0,  0,  0, 20, 20,  0,  0,  0],
    [ 5, -5,-10,  0,  0,-10, -5,  5],
    [ 5, 10, 10,-20,-20, 10, 10,  5],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
], dtype=np.float32)

PST_KNIGHT = np.array([
    [-50,-40,-30,-30,-30,-30,-40,-50],
    [-40,-20,  0,  0,  0,  0,-20,-40],
    [-30,  0, 10, 15, 15, 10,  0,-30],
    [-30,  5, 15, 20, 20, 15,  5,-30],
    [-30,  0, 15, 20, 20, 15,  0,-30],
    [-30,  5, 10, 15, 15, 10,  5,-30],
    [-40,-20,  0,  5,  5,  0,-20,-40],
    [-50,-40,-30,-30,-30,-30,-40,-50],
], dtype=np.float32)

PST_BISHOP = np.array([
    [-20,-10,-10,-10,-10,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5, 10, 10,  5,  0,-10],
    [-10,  5,  5, 10, 10,  5,  5,-10],
    [-10,  0, 10, 10, 10, 10,  0,-10],
    [-10, 10, 10, 10, 10, 10, 10,-10],
    [-10,  5,  0,  0,  0,  0,  5,-10],
    [-20,-10,-10,-10,-10,-10,-10,-20],
], dtype=np.float32)

PST_ROOK = np.array([
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 5, 10, 10, 10, 10, 10, 10,  5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [ 0,  0,  0,  5,  5,  0,  0,  0],
], dtype=np.float32)

PST_QUEEN = np.array([
    [-20,-10,-10, -5, -5,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5,  5,  5,  5,  0,-10],
    [ -5,  0,  5,  5,  5,  5,  0, -5],
    [  0,  0,  5,  5,  5,  5,  0, -5],
    [-10,  5,  5,  5,  5,  5,  0,-10],
    [-10,  0,  5,  0,  0,  0,  0,-10],
    [-20,-10,-10, -5, -5,-10,-10,-20],
], dtype=np.float32)

PST_KING_MIDGAME = np.array([
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-20,-30,-30,-40,-40,-30,-30,-20],
    [-10,-20,-20,-20,-20,-20,-20,-10],
    [ 20, 20,  0,  0,  0,  0, 20, 20],
    [ 20, 30, 10,  0,  0, 10, 30, 20],
], dtype=np.float32)


def evaluate_static_board(board: chess.Board) -> float:
    """
    Computes fast static positional evaluation in centipawns (+ for White, - for Black).
    """
    if board.is_checkmate():
        # Checkmated side has lost
        return -99999.0 if board.turn == chess.WHITE else 99999.0
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition():
        return 0.0

    score = 0.0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        pt = piece.piece_type
        color = piece.color  # True for White, False for Black
        val = PIECE_VALUES[pt]

        # PST coordinates: White uses rank, Black mirrors rank
        r = chess.square_rank(sq)
        f = chess.square_file(sq)
        pst_r = 7 - r if color == chess.WHITE else r

        pst_val = 0.0
        if pt == chess.PAWN:
            pst_val = PST_PAWN[pst_r, f]
        elif pt == chess.KNIGHT:
            pst_val = PST_KNIGHT[pst_r, f]
        elif pt == chess.BISHOP:
            pst_val = PST_BISHOP[pst_r, f]
        elif pt == chess.ROOK:
            pst_val = PST_ROOK[pst_r, f]
        elif pt == chess.QUEEN:
            pst_val = PST_QUEEN[pst_r, f]
        elif pt == chess.KING:
            pst_val = PST_KING_MIDGAME[pst_r, f]

        total_piece_score = val + pst_val
        if color == chess.WHITE:
            score += total_piece_score
        else:
            score -= total_piece_score

    return score / 100.0  # Normalized to pawn units (e.g. +1.5 = +1.5 pawns)


def encode_board_tensor(board: chess.Board) -> torch.Tensor:
    """
    Encodes chess.Board into a 13 x 8 x 8 float tensor.
      Planes 0..5: White P, N, B, R, Q, K
      Planes 6..11: Black P, N, B, R, Q, K
      Plane 12: Turn indicator (1.0 for White, 0.0 for Black)
    """
    t = np.zeros((13, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        # row index in grid (0 at top / rank 8, 7 at bottom / rank 1)
        r = 7 - chess.square_rank(sq)
        c = chess.square_file(sq)
        pt = piece.piece_type - 1  # 0 to 5
        plane = pt if piece.color == chess.WHITE else pt + 6
        t[plane, r, c] = 1.0

    if board.turn == chess.WHITE:
        t[12, :, :] = 1.0

    return torch.tensor(t, dtype=torch.float32)


# ----------------------------------------------------------------------
# 2. Neural Architecture: Jev 8x8 Chess Evaluator
# ----------------------------------------------------------------------

class JevChess8x8Evaluator(nn.Module):
    """
    TypeSafe Jev Evaluator for Standard 8x8 Chess:
      Outputs:
        - Value V(s) in [-1, 1]: Advantage in tanh scale (+ favors current player).
        - Noul(s) in [0, 1]: Epistemic certainty / tactical quietness.
          (Low Noul = tactical crisis, piece hangs, checks, search required).
        - Raw 128-dim accumulator representation & CReLU for MechInterp.
    """
    def __init__(self, in_channels: int = 13, num_filters: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filters)

        self.fc = nn.Linear(num_filters * BOARD_SIZE * BOARD_SIZE, 128)
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Feature extraction
        h = F.gelu(self.bn1(self.conv1(x)))
        h = F.gelu(self.bn2(self.conv2(h)))
        h = F.gelu(self.bn3(self.conv3(h)))
        h = h.view(h.size(0), -1)

        # 128-dim accumulator representation
        accum = self.fc(h)
        # CReLU activation for mechanistic sparsity
        crelu = torch.clamp(accum, 0.0, 1.0)

        val = self.val_head(crelu).squeeze(-1)
        noul = self.noul_head(crelu).squeeze(-1)
        return val, noul, accum, crelu


# ----------------------------------------------------------------------
# 3. Model Training & Checkpoint Management
# ----------------------------------------------------------------------

def train_jev_chess8x8_model(
    model: JevChess8x8Evaluator,
    num_samples: int = 4000,
    epochs: int = 4,
    batch_size: int = 64
) -> None:
    """
    Generates diverse standard chess positions and trains the Jev evaluator.
    Positions are labeled with normalized static evaluation and tactical crisis flags.
    """
    print(f"Generating {num_samples} diverse 8x8 chess positions...", flush=True)
    tensors = []
    values = []
    nouls = []

    # Common openings to seed realistic structures
    openings = [
        [],
        ["e2e4", "e7e5"],
        ["d2d4", "d7d5"],
        ["e2e4", "c7c5"],
        ["d2d4", "g8f6"],
        ["c2c4", "e7e5"],
        ["g1f3", "d7d5"],
        ["e2e4", "e7e6"],
    ]

    for _ in range(num_samples):
        b = chess.Board()
        # Seed opening
        op = random.choice(openings)
        for u in op:
            try:
                b.push_san(u)
            except Exception:
                pass

        # Random playout of 0-25 plies
        plies = random.randint(0, 25)
        for _ in range(plies):
            if b.is_game_over():
                break
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(random.choice(moves))

        score = evaluate_static_board(b)
        # Score from perspective of side to move
        turn_score = score if b.turn == chess.WHITE else -score
        val_target = float(math.tanh(turn_score / 4.0))

        # Check tactical volatility
        in_check = b.is_check()
        has_captures = any(b.is_capture(m) for m in b.legal_moves)
        tactical_crisis = in_check or (has_captures and random.random() < 0.6)

        if tactical_crisis:
            noul_target = random.uniform(0.20, 0.60)
        else:
            noul_target = random.uniform(0.75, 0.98)

        t_enc = encode_board_tensor(b)
        tensors.append(t_enc)
        values.append(val_target)
        nouls.append(noul_target)

    x_train = torch.stack(tensors).to(device)
    y_val = torch.tensor(values, dtype=torch.float32, device=device)
    y_noul = torch.tensor(nouls, dtype=torch.float32, device=device)

    dataset = torch.utils.data.TensorDataset(x_train, y_val, y_noul)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn_val = nn.MSELoss()
    loss_fn_noul = nn.BCELoss()

    model.train()
    print(f"Training Jev 8x8 Evaluator on {device} ({epochs} epochs)...", flush=True)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for bx, bv, bn in loader:
            optimizer.zero_grad()
            pred_val, pred_noul, _, _ = model(bx)
            l_val = loss_fn_val(pred_val, bv)
            l_noul = loss_fn_noul(pred_noul, bn)
            loss = l_val + 0.8 * l_noul
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.4f}", flush=True)
    print(f"Training completed in {time.time() - t0:.2f}s.", flush=True)


# ----------------------------------------------------------------------
# 4. Search Engines: Reflex, Negamax & Adaptive ETS
# ----------------------------------------------------------------------

def score_move_candidates(
    board: chess.Board,
    model: JevChess8x8Evaluator,
    temperature: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Evaluates every legal move from the current position.
    Returns structured move diagnostics sorted descending by score:
      - move_uci: string e.g. "e2e4"
      - move_san: string e.g. "e4"
      - from_coords: [r1, c1] in 0..7 grid
      - to_coords: [r2, c2] in 0..7 grid
      - val: expected position value after move from current player's perspective
      - noul: epistemic certainty after move
      - credence: softmax probability P(m)
      - tactical_type: "CHECK", "CAPTURE", "PROMOTION", "QUIET"
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return []

    results = []
    tensors = []
    moves_meta = []

    for m in legal_moves:
        san = board.san(m)
        is_check = board.gives_check(m)
        is_capture = board.is_capture(m)
        is_promo = m.promotion is not None

        if is_check:
            tactical = "CHECK"
        elif is_capture:
            tactical = "CAPTURE"
        elif is_promo:
            tactical = "PROMOTION"
        else:
            tactical = "QUIET"

        # Coordinates for frontend (0..7 where row 0 is rank 8)
        r1 = 7 - chess.square_rank(m.from_square)
        c1 = chess.square_file(m.from_square)
        r2 = 7 - chess.square_rank(m.to_square)
        c2 = chess.square_file(m.to_square)

        # Clone and push
        board.push(m)
        # Child score from current player's perspective:
        # After move, board.turn has flipped, so static eval for previous player is:
        # -child_turn * evaluate_static(child)
        child_score = evaluate_static_board(board)
        # If board.turn == chess.BLACK now, White just moved, so score is child_score
        # If board.turn == chess.WHITE now, Black just moved, so score is -child_score
        mover_val = child_score if board.turn == chess.BLACK else -child_score

        # Prepare tensor for neural batch eval
        t_enc = encode_board_tensor(board)
        tensors.append(t_enc)

        moves_meta.append({
            "uci": m.uci(),
            "san": san,
            "from_coords": [r1, c1],
            "to_coords": [r2, c2],
            "static_val": mover_val,
            "tactical_type": tactical,
        })
        board.pop()

    # Neural batch inference
    batch_t = torch.stack(tensors).to(device)
    model.eval()
    with torch.no_grad():
        vals, nouls, _, _ = model(batch_t)
        vals = vals.cpu().numpy()
        nouls = nouls.cpu().numpy()

    # Combine static heuristics with neural evaluation
    combined_scores = []
    for i, meta in enumerate(moves_meta):
        # Val in [-1, 1], convert static_val to tanh scale
        v_neural = float(vals[i])
        v_static = float(math.tanh(meta["static_val"] / 4.0))
        # Blended value: 60% neural + 40% static tactical
        blended_val = 0.60 * v_neural + 0.40 * v_static
        # Tactical bonus for captures/checks if static shows advantage
        if meta["tactical_type"] in ("CHECK", "CAPTURE") and meta["static_val"] > 0:
            blended_val += 0.10

        combined_scores.append(blended_val)

    # Softmax credences
    arr_scores = np.array(combined_scores)
    exp_scores = np.exp((arr_scores - np.max(arr_scores)) / max(0.1, temperature))
    credences = exp_scores / np.sum(exp_scores)

    for i, meta in enumerate(moves_meta):
        results.append({
            "uci": meta["uci"],
            "san": meta["san"],
            "from_coords": meta["from_coords"],
            "to_coords": meta["to_coords"],
            "val": round(float(combined_scores[i]), 3),
            "noul": round(float(nouls[i]), 3),
            "credence": round(float(credences[i]) * 100.0, 1),
            "tactical_type": meta["tactical_type"],
        })

    # Sort descending by score
    results.sort(key=lambda x: x["val"], reverse=True)
    return results


# ----------------------------------------------------------------------
# 4. Opening Book & Fast Epistemic Move Selection
# ----------------------------------------------------------------------

OPENING_BOOK: Dict[str, List[str]] = {
    # Start position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": ["e2e4", "d2d4", "c2c4", "g1f3"],
    # 1. e4
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": ["c7c5", "e7e5", "c7c6", "e7e6"],
    # 1. d4
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1": ["d7d5", "g8f6", "e7e6"],
    # 1. c4
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1": ["e7e5", "c7c5", "g8f6"],
    # 1. Nf3
    "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1": ["d7d5", "g8f6", "c7c5"],
    # 1. e4 e5
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": ["g1f3", "f1c4", "b1c3"],
    # 1. e4 c5 (Sicilian)
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": ["g1f3", "b1c3", "c2c3"],
    # 1. e4 e6 (French)
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": ["d2d4", "g1f3"],
    # 1. e4 c6 (Caro-Kann)
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": ["d2d4", "b1c3"],
    # 1. d4 d5
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2": ["c2c4", "g1f3", "b1c3"],
    # 1. d4 Nf6
    "rnbqkbnr/pppppppp/8/8/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq - 1 2": ["e7e6", "g7g6", "d7d5"],
    # 1. e4 e5 2. Nf3 Nc6
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 1 3": ["f1c4", "f1b5", "d2d4"],
}

def get_opening_book_move(board: chess.Board) -> Optional[chess.Move]:
    """Returns a high-quality opening move in 0.0ms if board matches known book FEN."""
    fen = board.fen()
    candidates = OPENING_BOOK.get(fen)
    if candidates:
        legal_uci = {m.uci(): m for m in board.legal_moves}
        valid = [legal_uci[uci] for uci in candidates if uci in legal_uci]
        if valid:
            return random.choice(valid)
    return None


def select_move_reflexive_jev(board: chess.Board, model: JevChess8x8Evaluator) -> chess.Move:
    """1-ply reflex move selection via scored move candidates (sub-1ms execution)."""
    candidates = score_move_candidates(board, model)
    if not candidates:
        return random.choice(list(board.legal_moves))
    best_uci = candidates[0]["uci"]
    return chess.Move.from_uci(best_uci)


def negamax_alpha_beta(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    current_turn: int,  # 1 for White, -1 for Black
    model: JevChess8x8Evaluator,
    start_time: float,
    max_duration: float = 0.25,
    tt: Optional[Dict[str, Tuple[int, float, int]]] = None,
) -> float:
    """
    Standard Negamax with Alpha-Beta Pruning and Transposition Table (TT).
    Strictly follows: score = -negamax(child, depth - 1, -beta, -alpha, -current_turn).
    Static evaluation returns positive values for current_turn: current_turn * evaluate_static().
    """
    # Time cutoff check
    if time.time() - start_time > max_duration:
        return current_turn * evaluate_static_board(board)

    if depth <= 0 or board.is_game_over():
        return current_turn * evaluate_static_board(board)

    key = board._transposition_key() if hasattr(board, "_transposition_key") else board.fen()
    orig_alpha = alpha

    # TT Lookup
    if tt is not None and key in tt:
        tt_depth, tt_score, tt_flag = tt[key]
        if tt_depth >= depth:
            if tt_flag == 0:  # EXACT
                return tt_score
            elif tt_flag == 1:  # LOWERBOUND
                alpha = max(alpha, tt_score)
            elif tt_flag == 2:  # UPPERBOUND
                beta = min(beta, tt_score)
            if alpha >= beta:
                return tt_score

    max_eval = -999999.0
    legal_moves = list(board.legal_moves)
    # Tactical move ordering: captures and checks first
    legal_moves.sort(key=lambda m: (board.is_capture(m), board.gives_check(m)), reverse=True)

    for m in legal_moves:
        board.push(m)
        evaluation = -negamax_alpha_beta(
            board, depth - 1, -beta, -alpha, -current_turn, model, start_time, max_duration, tt
        )
        board.pop()

        max_eval = max(max_eval, evaluation)
        alpha = max(alpha, evaluation)
        if alpha >= beta:
            break  # Beta cutoff

        if time.time() - start_time > max_duration:
            break

    # TT Store
    if tt is not None:
        if max_eval <= orig_alpha:
            flag = 2  # UPPERBOUND
        elif max_eval >= beta:
            flag = 1  # LOWERBOUND
        else:
            flag = 0  # EXACT
        tt[key] = (depth, max_eval, flag)

    return max_eval


def select_move_adaptive_ets(
    board: chess.Board,
    model: JevChess8x8Evaluator,
    remaining_clock: float,
    tau: float = 0.70
) -> Tuple[chess.Move, int]:
    """
    Lightning Adaptive Epistemic Tree Search (ETS):
      1. Opening Book (<0.2ms instant response).
      2. 1-ply batch neural scoring + classical PeSTO heuristics.
      3. Epistemic Certainty Noul(s):
         - Quiet position (Noul >= tau, no check, no direct capture threats):
           Plays reflex candidate immediately (<15ms). Banks clock!
         - Tactical crisis (Noul < tau, in check, or active threats):
           Deploys strict, time-capped Negamax search with Transposition Table.
    """
    t0 = time.time()

    # 1. Opening Book check
    book_move = get_opening_book_move(board)
    if book_move:
        return book_move, 1

    # 2. Score 1-ply candidates
    candidates = score_move_candidates(board, model)
    if not candidates:
        return random.choice(list(board.legal_moves)), 1

    # 3. Position certainty Noul
    t_curr = encode_board_tensor(board).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        val_cur, noul_cur, _, _ = model(t_curr)
        noul_val = float(noul_cur.item())

    is_check = board.is_check()
    has_capture_threat = any(cand["tactical_type"] == "CAPTURE" for cand in candidates[:2])
    is_tactical_crisis = (noul_val < tau) or is_check or has_capture_threat

    # Epistemic Rule: In quiet positions, play the 1-ply reflex move immediately!
    # Latency: ~10ms. Clock preserved!
    if not is_tactical_crisis or remaining_clock < 2.0:
        return chess.Move.from_uci(candidates[0]["uci"]), 1

    # 4. Tactical Crisis: Gated Search with TT and Dynamic Time Allocation
    if remaining_clock < 5.0:
        search_depth = 2
        max_duration = 0.05
    elif remaining_clock < 20.0:
        search_depth = 2
        max_duration = 0.12
    elif remaining_clock < 60.0:
        search_depth = 3
        max_duration = 0.20
    else:
        search_depth = 3
        max_duration = 0.28  # Strict sub-300ms ceiling even with abundant clock

    current_turn = 1 if board.turn == chess.WHITE else -1
    best_move = chess.Move.from_uci(candidates[0]["uci"])
    best_score = -999999.0
    alpha = -999999.0
    beta = 999999.0
    tt: Dict[str, Tuple[int, float, int]] = {}

    # Search top 5 candidates (pre-sorted by neural + PeSTO)
    for cand in candidates[:5]:
        m = chess.Move.from_uci(cand["uci"])
        board.push(m)
        score = -negamax_alpha_beta(
            board, search_depth - 1, -beta, -alpha, -current_turn, model, t0, max_duration, tt
        )
        board.pop()

        if score > best_score:
            best_score = score
            best_move = m
        alpha = max(alpha, score)

        if time.time() - t0 > max_duration:
            break

    return best_move, search_depth
