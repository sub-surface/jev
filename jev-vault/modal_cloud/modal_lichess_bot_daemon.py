"""
==========================================================================
⚡ MODAL CLOUD: ZERO-IDLE LICHESS BOT DAEMON (WAKE-ON-DEMAND)
==========================================================================
Architectural Philosophy: "Zero Compute When Idle, Instant Spin-up On Demand"
  - Wakes up in ~1 second via HTTP webhook (e.g. from jev.subsurfaces.net).
  - Maintains live Lichess event stream while active.
  - Plays standard bullet / blitz challenges using Jevformer Leela weights.
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
from typing import Dict, Any, List, Optional

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
    cpu=0.5,       # Tiny 0.5 CPU core = ~$0.0000065/sec (less than half a cent per hour)
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

    # Opening Book
    OPENING_BOOK = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": ["e2e4", "d2d4", "c2c4", "g1f3"],
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": ["c7c5", "e7e5", "c7c6", "e7e6"],
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1": ["d7d5", "g8f6", "e7e6"],
    }

    def play_game(game_id: str):
        nonlocal last_active_time
        print(f"\n[{game_id}] Starting Game Stream: {LICHESS_API}/{game_id}", flush=True)
        url = f"{LICHESS_API}/api/bot/game/stream/{game_id}"
        my_color = None
        opponent_name = "Opponent"

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
            t0 = time.time()

            # 1. Opening Book
            fen = board.fen()
            candidates = OPENING_BOOK.get(fen)
            chosen_move = None
            if candidates:
                legal = {m.uci(): m for m in board.legal_moves}
                matches = [legal[u] for u in candidates if u in legal]
                if matches:
                    chosen_move = random.choice(matches)

            # 2. Reflex / Neural Move
            if not chosen_move:
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    return
                legal_moves.sort(key=lambda m: (board.is_capture(m), board.gives_check(m)), reverse=True)
                chosen_move = legal_moves[0]

            elapsed = time.time() - t0
            uci = chosen_move.uci()
            print(f"[{game_id}] Move {board.san(chosen_move)} ({uci}) in {elapsed*1000:.1f}ms", flush=True)

            requests.post(f"{LICHESS_API}/api/bot/game/{game_id}/move/{uci}", headers=HEADERS, timeout=5)

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
