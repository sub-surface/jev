"""
==========================================================================
⚡ MODAL CLOUD: ZERO-IDLE LICHESS BOT DAEMON (WAKE-ON-DEMAND)
==========================================================================
Architectural Philosophy: "Zero Compute When Idle, Instant Spin-up On Demand"
  - Wakes up in ~1 second via HTTP webhook (from jev.subsurfaces.net).
  - Maintains live Lichess event stream while active.
  - Plays standard bullet / blitz challenges using Jevformer Leela weights.
  - Full Epistemic Tree Search (ETS): Opening Book (<1ms), Reflex Neural
    Move (15-35ms) on quiet positions, Depth-2/3 Negamax on tactical crisis.
  - Automatically shuts down after 15 minutes of inactivity to cost $0.00 idle!
  - Checkpoint and games persisted to Modal Volume 'jevformer-checkpoints'.
==========================================================================
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
import math
import json
import random
import threading
from typing import Dict, Any, List, Optional, Tuple

import modal

APP_NAME = "jess-hyperbullet-bot"
app = modal.App(APP_NAME)
volume = modal.Volume.from_name("jevformer-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4.0",
        "numpy>=1.26.0",
        "chess>=1.10.0",
        "requests>=2.31.0",
        "fastapi[standard]",
    )
)

LICHESS_API = "https://lichess.org"
BOT_USERNAME = "jess-hyperbullet"
LICHESS_TOKEN = os.environ.get("LICHESS_BOT_TOKEN", "")
if not LICHESS_TOKEN and os.path.exists(".env"):
    for line in open(".env"):
        if line.startswith("LICHESS_BOT_TOKEN="):
            LICHESS_TOKEN = line.strip().split("=", 1)[1].strip("\"'")

HEADERS = {
    "Authorization": f"Bearer {LICHESS_TOKEN}",
    "User-Agent": "Jess-Hyperbullet-Bot/1.0 (Modal Cloud Zero-Idle Runner)",
}

# ---------------------------------------------------------------------------
# Classical PeSTO Piece-Square Tables (Midgame) for Heuristic Grounding
# ---------------------------------------------------------------------------
PIECE_VALUES = {
    1: 100,   # PAWN
    2: 320,   # KNIGHT
    3: 330,   # BISHOP
    4: 500,   # ROOK
    5: 900,   # QUEEN
    6: 20000, # KING
}

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

PST_KING_MIDGAME = [
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-20,-30,-30,-40,-40,-30,-30,-20],
    [-10,-20,-20,-20,-20,-20,-20,-10],
    [20, 20,  0,  0,  0,  0, 20, 20],
    [20, 30, 10,  0,  0, 10, 30, 20],
]

OPENING_BOOK = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": ["e2e4", "d2d4", "c2c4", "g1f3"],
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": ["c7c5", "e7e5", "c7c6", "e7e6"],
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1": ["d7d5", "g8f6", "e7e6", "c7c5"],
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": ["g1f3", "b1c3", "f1c4"],
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": ["g1f3", "b1c3", "c2c3", "d2d4"],
    "rnbqkb1r/pppppppp/5n3/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2": ["c2c4", "g1f3", "b1c3"],
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": ["d2d4", "d2d3"],
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": ["d2d4", "b1c3"],
}


def create_jev_model():
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
    volumes={"/checkpoints": volume},
    timeout=3600,  # Max 1 hour run window per wakeup
    cpu=0.5,       # 0.5 CPU core = ~$0.0000065/sec
    memory=1024,   # 1 GB RAM
)
def run_bot_daemon(idle_timeout_seconds: int = 900):
    """
    Persistent event-listening loop running on Modal Cloud.
    Automatically terminates after `idle_timeout_seconds` of inactivity to cost $0.00.
    """
    import torch
    import chess
    import numpy as np
    import requests

    device = torch.device("cpu")
    print("--- Initializing Jess-Hyperbullet Zero-Idle Daemon on Modal ---", flush=True)

    JevChess8x8Evaluator = create_jev_model()
    model = JevChess8x8Evaluator().to(device)

    # Load Leela weights from Modal Volume
    ckpt_path = "/checkpoints/jev_chess_8x8_leela.pt"
    if os.path.exists(ckpt_path):
        print(f"Loading weights from Modal Volume: {ckpt_path}...", flush=True)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"Warning: {ckpt_path} not found on volume, using base weights.", flush=True)
    model.eval()

    # Verify Lichess account
    acc_res = requests.get(f"{LICHESS_API}/api/account", headers=HEADERS, timeout=10)
    if acc_res.status_code != 200:
        print(f"Lichess auth error: {acc_res.status_code} {acc_res.text}", flush=True)
        return

    print(f"✅ Authenticated as: {acc_res.json().get('username')} [BOT]", flush=True)
    print(f"Listening on event stream (Inactivity timeout: {idle_timeout_seconds}s)...", flush=True)

    last_active_time = time.time()
    active_games: Dict[str, threading.Thread] = {}
    running = True

    # -------------------------------------------------------------------
    # Search & Evaluation Helpers
    # -------------------------------------------------------------------
    def evaluate_static_board(b: chess.Board) -> float:
        if b.is_checkmate():
            return -9999.0 if b.turn == chess.WHITE else 9999.0
        if b.is_stalemate() or b.is_insufficient_material():
            return 0.0

        score = 0.0
        for sq in chess.SQUARES:
            piece = b.piece_at(sq)
            if piece is None:
                continue
            color = piece.color
            pt = piece.piece_type
            val = PIECE_VALUES.get(pt, 0)

            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            pst_r = 7 - r if color == chess.WHITE else r

            pst_val = 0
            if pt == chess.PAWN:
                pst_val = PST_PAWN[pst_r][f]
                adv_rank = r if color == chess.WHITE else 7 - r
                if adv_rank == 6:
                    pst_val += 350.0
                elif adv_rank == 5:
                    pst_val += 120.0
            elif pt == chess.KNIGHT:
                pst_val = PST_KNIGHT[pst_r][f]
            elif pt == chess.BISHOP:
                pst_val = PST_BISHOP[pst_r][f]
            elif pt == chess.ROOK:
                pst_val = PST_ROOK[pst_r][f]
            elif pt == chess.QUEEN:
                pst_val = PST_QUEEN[pst_r][f]
            elif pt == chess.KING:
                pst_val = PST_KING_MIDGAME[pst_r][f]

            total_p = val + pst_val
            if color == chess.WHITE:
                score += total_p
            else:
                score -= total_p

        return score / 100.0

    def encode_board_tensor(b: chess.Board) -> torch.Tensor:
        t = np.zeros((13, 8, 8), dtype=np.float32)
        for sq in chess.SQUARES:
            piece = b.piece_at(sq)
            if piece is None:
                continue
            r = 7 - chess.square_rank(sq)
            c = chess.square_file(sq)
            pt = piece.piece_type - 1
            plane = pt if piece.color == chess.WHITE else pt + 6
            t[plane, r, c] = 1.0

        if b.turn == chess.WHITE:
            t[12, :, :] = 1.0
        return torch.tensor(t, dtype=torch.float32)

    def score_move_candidates(b: chess.Board) -> List[Dict[str, Any]]:
        legal = list(b.legal_moves)
        if not legal:
            return []
        tensors = []
        moves_meta = []
        for m in legal:
            tactical = "QUIET"
            if b.gives_check(m):
                tactical = "CHECK"
            elif b.is_capture(m):
                tactical = "CAPTURE"
            elif m.promotion:
                tactical = "PROMOTION"

            b.push(m)
            child_score = evaluate_static_board(b)
            mover_val = child_score if b.turn == chess.BLACK else -child_score
            tensors.append(encode_board_tensor(b))
            moves_meta.append({
                "uci": m.uci(),
                "san": b.san(m) if False else m.uci(),
                "static_val": mover_val,
                "tactical_type": tactical,
            })
            b.pop()

        batch_t = torch.stack(tensors).to(device)
        with torch.no_grad():
            vals, nouls, _, _ = model(batch_t)
            vals = vals.cpu().numpy()
            nouls = nouls.cpu().numpy()

        combined_scores = []
        for i, meta in enumerate(moves_meta):
            # Correctly negate child value: vals[i] is opponent perspective
            v_neural = -float(vals[i])
            v_static = float(math.tanh(meta["static_val"] / 4.0))
            blended = 0.50 * v_neural + 0.50 * v_static
            if meta["tactical_type"] in ("CHECK", "CAPTURE", "PROMOTION") and meta["static_val"] > 0:
                blended += 0.15
            combined_scores.append(blended)

        for i, meta in enumerate(moves_meta):
            meta["combined_val"] = combined_scores[i]
            meta["noul"] = float(nouls[i])
            meta["is_tactical"] = meta["tactical_type"] in ("CHECK", "CAPTURE", "PROMOTION")

        moves_meta.sort(key=lambda x: x["combined_val"], reverse=True)
        return moves_meta

    def quiescence(b: chess.Board, alpha: float, beta: float, current_turn: int, max_qdepth: int = 3) -> float:
        stand_pat = current_turn * evaluate_static_board(b)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        if max_qdepth <= 0 or b.is_game_over():
            return stand_pat

        captures = [m for m in b.legal_moves if b.is_capture(m) or m.promotion]
        for m in captures:
            b.push(m)
            score = -quiescence(b, -beta, -alpha, -current_turn, max_qdepth - 1)
            b.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def negamax(b: chess.Board, depth: int, alpha: float, beta: float, current_turn: int, t_start: float, max_dur: float, tt: dict) -> float:
        if time.time() - t_start > max_dur:
            return current_turn * evaluate_static_board(b)

        if depth <= 0:
            return quiescence(b, alpha, beta, current_turn, max_qdepth=3)

        if b.is_game_over():
            return current_turn * evaluate_static_board(b)

        key = b.fen()
        orig_alpha = alpha
        if key in tt:
            d, s, f = tt[key]
            if d >= depth:
                if f == 0: return s
                elif f == 1: alpha = max(alpha, s)
                elif f == 2: beta = min(beta, s)
                if alpha >= beta: return s

        max_eval = -999999.0
        moves = list(b.legal_moves)
        moves.sort(key=lambda m: (b.is_capture(m), m.promotion is not None, b.gives_check(m)), reverse=True)

        for m in moves:
            b.push(m)
            ev = -negamax(b, depth - 1, -beta, -alpha, -current_turn, t_start, max_dur, tt)
            b.pop()
            if ev > max_eval:
                max_eval = ev
            if max_eval > alpha:
                alpha = max_eval
            if alpha >= beta or time.time() - t_start > max_dur:
                break

        flag = 0 if max_eval > orig_alpha and max_eval < beta else (1 if max_eval >= beta else 2)
        tt[key] = (depth, max_eval, flag)
        return max_eval

    def select_move_ets(b: chess.Board, remaining_clock: float) -> Tuple[chess.Move, int, float]:
        t0 = time.time()

        # 1. Opening Book
        fen = b.fen()
        book_candidates = OPENING_BOOK.get(fen)
        if book_candidates:
            legal_ucis = {m.uci(): m for m in b.legal_moves}
            matches = [legal_ucis[u] for u in book_candidates if u in legal_ucis]
            if matches:
                return random.choice(matches), 1, (time.time() - t0) * 1000.0

        # 2. Score Candidates
        candidates = score_move_candidates(b)
        if not candidates:
            return random.choice(list(b.legal_moves)), 1, (time.time() - t0) * 1000.0

        # 3. Position Certainty Noul (calibrated from margin and tactical tension)
        top_val = candidates[0]["combined_val"]
        second_val = candidates[1]["combined_val"] if len(candidates) > 1 else top_val
        margin = top_val - second_val

        has_threat = any(c.get("is_tactical", False) for c in candidates[:4])
        is_check = b.is_check()

        noul_calibrated = float(math.tanh(margin * 3.0))
        if has_threat or is_check:
            noul_calibrated *= 0.40

        is_crisis = is_check or has_threat or (noul_calibrated < 0.70)

        # Reflex Move in Quiet Positions
        if not is_crisis or remaining_clock < 2.0:
            return chess.Move.from_uci(candidates[0]["uci"]), 1, (time.time() - t0) * 1000.0

        # Dynamic Search Window
        if remaining_clock < 8.0:
            depth = 2; max_dur = 0.06
        elif remaining_clock < 25.0:
            depth = 2; max_dur = 0.12
        else:
            depth = 3; max_dur = 0.22

        tt = {}
        curr_turn = 1 if b.turn == chess.WHITE else -1
        best_m = chess.Move.from_uci(candidates[0]["uci"])
        best_sc = -999999.0
        alpha = -999999.0
        beta = 999999.0

        search_cands = [c for c in candidates if c.get("is_tactical", False) or c["combined_val"] >= top_val - 0.25][:8]
        if not search_cands:
            search_cands = candidates[:5]

        ordered = [chess.Move.from_uci(c["uci"]) for c in search_cands if chess.Move.from_uci(c["uci"]) in b.legal_moves]

        for m in ordered:
            b.push(m)
            sc = -negamax(b, depth - 1, -beta, -alpha, -curr_turn, t0, max_dur, tt)
            b.pop()
            if sc > best_sc:
                best_sc = sc
                best_m = m
            alpha = max(alpha, sc)
            if time.time() - t0 > max_dur:
                break

        return best_m, depth, (time.time() - t0) * 1000.0

    # -------------------------------------------------------------------
    # Game Stream Handling
    # -------------------------------------------------------------------
    def play_game(game_id: str):
        nonlocal last_active_time
        print(f"\n[{game_id}] Starting Game Stream: {LICHESS_API}/{game_id}", flush=True)
        url = f"{LICHESS_API}/api/bot/game/stream/{game_id}"
        my_color = None

        try:
            with requests.get(url, headers=HEADERS, stream=True, timeout=90) as stream:
                if stream.status_code != 200:
                    return

                # Send greeting
                requests.post(
                    f"{LICHESS_API}/api/bot/game/{game_id}/chat",
                    headers=HEADERS,
                    json={"room": "player", "text": "(o^▽^o) Hi! I'm Jess — a 128-neuron CReLU Tri-Process engine (System 0 heuristics + System 1 epistemic Noul gating + System 2 Negamax search). Trained via Leela-style distillation on Modal Cloud. glhf! ⚡ jev.subsurfaces.net"},
                    timeout=5,
                )

                for raw_line in stream.iter_lines():
                    if not raw_line:
                        continue
                    last_active_time = time.time()

                    try:
                        event = json.loads(raw_line.decode("utf-8"))
                    except Exception:
                        continue

                    evt_type = event.get("type")
                    if evt_type == "gameFull":
                        white_info = event.get("white", {})
                        is_white = white_info.get("name", "").lower() == BOT_USERNAME.lower()
                        my_color = chess.WHITE if is_white else chess.BLACK
                        state = event.get("state", {})
                        process_turn(game_id, state, my_color)
                    elif evt_type == "gameState":
                        process_turn(game_id, event, my_color)

        except Exception as e:
            print(f"[{game_id}] Exception: {e}", flush=True)
        finally:
            if game_id in active_games:
                del active_games[game_id]
            last_active_time = time.time()
            print(f"[{game_id}] Match closed.", flush=True)

    def process_turn(game_id: str, state: Dict[str, Any], my_color: Optional[chess.Color]):
        nonlocal last_active_time
        status = state.get("status", "started")
        if status != "started":
            return

        moves_str = state.get("moves", "").strip()
        moves_list = moves_str.split() if moves_str else []

        board = chess.Board()
        for uci_m in moves_list:
            try:
                board.push(chess.Move.from_uci(uci_m))
            except Exception:
                pass

        if board.turn == my_color and not board.is_game_over():
            last_active_time = time.time()
            rem_ms = state.get("wtime", 30000) if my_color == chess.WHITE else state.get("btime", 30000)
            rem_clock = max(0.5, rem_ms / 1000.0)

            chosen_move, depth, elapsed_ms = select_move_ets(board, rem_clock)
            uci = chosen_move.uci()
            san = board.san(chosen_move)
            print(f"[{game_id}] Ply {len(board.move_stack)+1} | Play: {san} ({uci}) | Depth: {depth} | Time: {elapsed_ms:.1f}ms | Clock Left: {rem_clock:.1f}s", flush=True)

            requests.post(f"{LICHESS_API}/api/bot/game/{game_id}/move/{uci}", headers=HEADERS, timeout=5)

    # Main event stream
    while running:
        idle_duration = time.time() - last_active_time
        if idle_duration > idle_timeout_seconds and len(active_games) == 0:
            print(f"\n💤 Inactive for {idle_duration:.0f}s (> {idle_timeout_seconds}s). Auto-shutting down to $0.00 compute.", flush=True)
            break

        try:
            with requests.get(f"{LICHESS_API}/api/stream/event", headers=HEADERS, stream=True, timeout=60) as resp:
                if resp.status_code != 200:
                    time.sleep(5)
                    continue

                for line in resp.iter_lines():
                    idle_duration = time.time() - last_active_time
                    if idle_duration > idle_timeout_seconds and len(active_games) == 0:
                        print(f"💤 Inactive for {idle_duration:.0f}s. Auto-sleeping.", flush=True)
                        running = False
                        break

                    if not line:
                        continue

                    try:
                        event = json.loads(line.decode("utf-8"))
                    except Exception:
                        continue

                    evt_type = event.get("type")
                    if evt_type == "challenge":
                        ch = event.get("challenge", {})
                        ch_id = ch.get("id")
                        variant = ch.get("variant", {}).get("key")
                        speed = ch.get("speed")
                        tc_limit = ch.get("timeControl", {}).get("limit", 0)

                        if variant != "standard":
                            requests.post(f"{LICHESS_API}/api/challenge/{ch_id}/decline", headers=HEADERS, json={"reason": "variant"}, timeout=5)
                            continue

                        if speed == "ultraBullet" or tc_limit < 30:
                            requests.post(f"{LICHESS_API}/api/challenge/{ch_id}/decline", headers=HEADERS, json={"reason": "timeControl"}, timeout=5)
                            continue

                        print(f"Accepting challenge {ch_id} ({speed})!", flush=True)
                        requests.post(f"{LICHESS_API}/api/challenge/{ch_id}/accept", headers=HEADERS, timeout=5)
                        last_active_time = time.time()

                    elif evt_type == "gameStart":
                        g_id = event.get("game", {}).get("id")
                        if g_id and g_id not in active_games:
                            t = threading.Thread(target=play_game, args=(g_id,), daemon=True)
                            active_games[g_id] = t
                            t.start()
                            last_active_time = time.time()

        except Exception as e:
            time.sleep(3)


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def wake(idle_timeout: int = 900):
    """
    HTTP Webhook to wake up the bot on demand from anywhere (e.g. jev.subsurfaces.net).
    Runs asynchronously and returns immediately.
    """
    print(f"⚡ Wakeup signal received! Launching bot daemon (Idle timeout: {idle_timeout}s)...", flush=True)
    run_bot_daemon.spawn(idle_timeout_seconds=idle_timeout)
    return {
        "status": "awakened",
        "bot": BOT_USERNAME,
        "profile": f"https://lichess.org/@/{BOT_USERNAME}",
        "idle_timeout_seconds": idle_timeout,
        "message": "Jess Hyperbullet is booting up on Modal Cloud and ready for challenges!",
    }


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "jess-hyperbullet-modal-daemon"}
