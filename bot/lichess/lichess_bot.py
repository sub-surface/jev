"""
==========================================================================
⚡ JESS-HYPERBULLET: LICHESS BOT RUNNER
==========================================================================
Autonomous Lichess Bot powered by the Jevformer Tri-Process Neural Engine:
- System 0: PeSTO classical positional priors & CReLU invariant features
- System 1: 128-neuron fast reflex accumulator (<1ms per ply)
- System 2: Negamax Alpha-Beta with MVV-LVA move ordering & check priority
- Epistemic Gating: Jev Noul (tau=0.70) dynamic bullet clock allocation

Account: https://lichess.org/@/jess-hyperbullet
Config: Loaded securely from .env (LICHESS_BOT_TOKEN)
==========================================================================
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import time
import json
import argparse
import threading
from typing import Dict, Any, Optional, List

import requests
import chess
import chess.pgn
import torch

# Load environment variables
SRC_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", ".."))  # repo root (holds the shared .env)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from chess_8x8_engine import (
    JevChess8x8Evaluator,
    select_move_adaptive_ets,
    encode_board_tensor,
    evaluate_static_board,
    device,
)

def load_env_file():
    """Simple parser for .env without external dependencies."""
    for env_path in (os.path.join(SRC_DIR, ".env"), os.path.join(ROOT_DIR, ".env")):
        if not os.path.exists(env_path):
            continue
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

load_env_file()

LICHESS_TOKEN = os.environ.get("LICHESS_BOT_TOKEN")
BOT_USERNAME = os.environ.get("LICHESS_BOT_USERNAME", "jess-hyperbullet").lower()
LICHESS_API = "https://lichess.org"

# Weights and game logs live next to the bot so the folder (and its Docker image) is self-contained.
CHECKPOINT_PATH = os.environ.get("JEV_CHESS_CKPT", os.path.join(SRC_DIR, "weights", "jev_chess_8x8_gui.pt"))
GAMES_DIR = os.path.join(SRC_DIR, "games")

os.makedirs(GAMES_DIR, exist_ok=True)

HEADERS = {
    "Authorization": f"Bearer {LICHESS_TOKEN}",
    "User-Agent": "Jess-Hyperbullet-Bot/1.0 (Jevformer Tri-Process Neural Engine)",
}


class LichessBotRunner:
    def __init__(self, model_path: Optional[str] = None):
        if not LICHESS_TOKEN:
            raise ValueError(
                "LICHESS_BOT_TOKEN not found in environment or .env file! "
                "Ensure your token is set in .env."
            )

        print("--- Initializing Jess-Hyperbullet Lichess Runner ---", flush=True)
        self.model = JevChess8x8Evaluator(in_channels=13, num_filters=64).to(device)

        ckpt = model_path or CHECKPOINT_PATH

        if os.path.exists(ckpt):
            print(f"Loading Jev model weights from {ckpt}...", flush=True)
            self.model.load_state_dict(torch.load(ckpt, map_location=device))
        else:
            print(f"Warning: Checkpoint {ckpt} not found! Initializing fresh weights.", flush=True)

        self.model.eval()
        self.active_games: Dict[str, threading.Thread] = {}
        self.running = True

    def verify_account(self) -> Dict[str, Any]:
        """Verifies bot status and prints account information."""
        url = f"{LICHESS_API}/api/account"
        res = requests.get(url, headers=HEADERS, timeout=10)
        if res.status_code != 200:
            raise RuntimeError(f"Lichess authentication failed: {res.status_code} {res.text}")

        data = res.json()
        print(f"✅ Authenticated as: {data.get('username')} [{data.get('title', 'NO TITLE')}]", flush=True)
        print(f"👉 Profile URL: https://lichess.org/@/{data.get('username')}", flush=True)
        bullet_perf = data.get("perfs", {}).get("bullet", {})
        print(f"⚡ Bullet Rating: {bullet_perf.get('rating')} (Games: {bullet_perf.get('games')})", flush=True)
        return data

    def send_chat(self, game_id: str, text: str, room: str = "player"):
        """Sends a friendly in-game chat message."""
        try:
            url = f"{LICHESS_API}/api/bot/game/{game_id}/chat"
            requests.post(url, headers=HEADERS, json={"room": room, "text": text}, timeout=5)
        except Exception as e:
            print(f"[{game_id}] Chat error: {e}", flush=True)

    def handle_challenge(self, challenge_data: Dict[str, Any]):
        """Decides whether to accept or decline an incoming challenge."""
        ch_id = challenge_data.get("id")
        challenger = challenge_data.get("challenger", {}).get("name", "Unknown")
        variant = challenge_data.get("variant", {}).get("key", "standard")
        speed = challenge_data.get("speed", "bullet")
        rated = challenge_data.get("rated", False)
        tc_limit = challenge_data.get("timeControl", {}).get("limit", 0)
        tc_inc = challenge_data.get("timeControl", {}).get("increment", 0)

        print(f"\n[Challenge Received] ID: {ch_id} from {challenger} | Speed: {speed} ({tc_limit}+{tc_inc}) | Variant: {variant} | Rated: {rated}", flush=True)

        if variant != "standard":
            print(f"Declining challenge {ch_id}: variant '{variant}' not supported (standard only).", flush=True)
            requests.post(
                f"{LICHESS_API}/api/challenge/{ch_id}/decline",
                headers=HEADERS,
                json={"reason": "variant"},
                timeout=5,
            )
            return

        if speed == "ultraBullet" or tc_limit < 30:
            print(
                f"Declining challenge {ch_id}: Lichess prohibits BOT accounts from playing UltraBullet (<30s). "
                f"Challenger requested {tc_limit}s. Minimum allowed by Lichess is 30s (1/2+0).",
                flush=True
            )
            requests.post(
                f"{LICHESS_API}/api/challenge/{ch_id}/decline",
                headers=HEADERS,
                json={"reason": "timeControl"},
                timeout=5,
            )
            return

        if len(self.active_games) >= 2:
            print(f"Declining challenge {ch_id}: bot currently busy with {len(self.active_games)} active game(s).", flush=True)
            requests.post(
                f"{LICHESS_API}/api/challenge/{ch_id}/decline",
                headers=HEADERS,
                json={"reason": "later"},
                timeout=5,
            )
            return

        # Accept the challenge
        print(f"Accepting challenge {ch_id} from {challenger}!", flush=True)
        res = requests.post(f"{LICHESS_API}/api/challenge/{ch_id}/accept", headers=HEADERS, timeout=10)
        if res.status_code == 200:
            print(f"✅ Challenge {ch_id} accepted successfully.", flush=True)
        else:
            print(f"❌ Failed to accept challenge {ch_id}: {res.status_code} {res.text}", flush=True)

    def play_game(self, game_id: str):
        """Dedicated worker thread handling the live move stream of a game."""
        print(f"\n==========================================================================", flush=True)
        print(f"🎮 Starting Live Game: {game_id} | URL: https://lichess.org/{game_id}", flush=True)
        print(f"==========================================================================", flush=True)

        url = f"{LICHESS_API}/api/bot/game/stream/{game_id}"
        is_my_turn = False
        my_color = None
        opponent_name = "Opponent"
        game_moves_logged: List[Dict[str, Any]] = []

        try:
            with requests.get(url, headers=HEADERS, stream=True, timeout=90) as stream_res:
                if stream_res.status_code != 200:
                    print(f"[{game_id}] Stream error: {stream_res.status_code} {stream_res.text}", flush=True)
                    return

                # Send greeting
                self.send_chat(
                    game_id,
                    "(o^▽^o) Hi! I'm Jess — a 128-neuron CReLU Tri-Process engine (System 0 heuristics + System 1 epistemic Noul gating + System 2 Negamax search). Trained via Leela-style distillation on Modal Cloud. glhf! ⚡ jev.subsurfaces.net"
                )

                for raw_line in stream_res.iter_lines():
                    if not raw_line:
                        continue

                    try:
                        event = json.loads(raw_line.decode("utf-8"))
                    except Exception:
                        continue

                    event_type = event.get("type")

                    if event_type == "gameFull":
                        white_info = event.get("white", {})
                        black_info = event.get("black", {})

                        is_white = white_info.get("name", "").lower() == BOT_USERNAME or white_info.get("id", "").lower() == BOT_USERNAME
                        my_color = chess.WHITE if is_white else chess.BLACK
                        opponent_name = black_info.get("name", "Black") if is_white else white_info.get("name", "White")

                        print(f"[{game_id}] Assigned color: {'WHITE' if is_white else 'BLACK'} vs {opponent_name}", flush=True)

                        state = event.get("state", {})
                        self._process_state(game_id, state, my_color, opponent_name, game_moves_logged)

                    elif event_type == "gameState":
                        self._process_state(game_id, event, my_color, opponent_name, game_moves_logged)

                    elif event_type == "chatLine":
                        user = event.get("username")
                        text = event.get("text")
                        print(f"[{game_id}] Chat ({user}): {text}", flush=True)

        except Exception as e:
            print(f"[{game_id}] Game loop exception: {e}", flush=True)
        finally:
            self.send_chat(game_id, "Good game! Thanks for playing.")
            self._save_game_archive(game_id, opponent_name, my_color, game_moves_logged)
            if game_id in self.active_games:
                del self.active_games[game_id]
            print(f"[{game_id}] Match closed and archived.", flush=True)

    def _process_state(
        self,
        game_id: str,
        state: Dict[str, Any],
        my_color: Optional[chess.Color],
        opponent_name: str,
        game_moves_logged: List[Dict[str, Any]],
    ):
        """Reconstructs the board from UCI move string and plays if it's our turn."""
        status = state.get("status", "started")
        if status != "started":
            print(f"[{game_id}] Game status ended: {status} (Winner: {state.get('winner', 'None')})", flush=True)
            return

        moves_str = state.get("moves", "").strip()
        moves_list = moves_str.split() if moves_str else []

        board = chess.Board()
        for uci_m in moves_list:
            try:
                board.push(chess.Move.from_uci(uci_m))
            except Exception as e:
                print(f"[{game_id}] Illegal move parsing '{uci_m}': {e}", flush=True)

        # Check whose turn it is
        if board.turn == my_color and not board.is_game_over():
            wtime = state.get("wtime", 15000)
            btime = state.get("btime", 15000)
            my_clock_ms = wtime if my_color == chess.WHITE else btime
            my_clock_s = max(0.1, my_clock_ms / 1000.0)

            t0 = time.time()
            best_move, depth = select_move_adaptive_ets(
                board, self.model, my_clock_s, tau=0.70
            )
            calc_time = time.time() - t0

            san_str = board.san(best_move)
            uci_str = best_move.uci()

            print(
                f"[{game_id}] Ply {len(moves_list) + 1} | Play: {san_str} ({uci_str}) "
                f"| Depth: {depth} | Time: {calc_time*1000:.1f}ms | Clock Left: {my_clock_s:.1f}s",
                flush=True
            )

            # Send move to Lichess
            move_url = f"{LICHESS_API}/api/bot/game/{game_id}/move/{uci_str}"
            res = requests.post(move_url, headers=HEADERS, timeout=5)
            if res.status_code == 200:
                game_moves_logged.append({
                    "ply": len(moves_list) + 1,
                    "san": san_str,
                    "uci": uci_str,
                    "depth": depth,
                    "latency_ms": round(calc_time * 1000, 1),
                    "clock_remaining_s": round(my_clock_s, 2),
                })
            else:
                print(f"[{game_id}] ❌ Failed to submit move {uci_str}: {res.status_code} {res.text}", flush=True)

    def _save_game_archive(
        self,
        game_id: str,
        opponent_name: str,
        my_color: Optional[chess.Color],
        moves_logged: List[Dict[str, Any]],
    ):
        """Saves game JSON and PGN to local directory."""
        pgn_text = ""
        try:
            res = requests.get(f"{LICHESS_API}/game/export/{game_id}", timeout=10)
            if res.status_code == 200:
                pgn_text = res.text
        except Exception as e:
            print(f"[{game_id}] Warning: Could not fetch PGN from Lichess: {e}", flush=True)

        record = {
            "game_id": game_id,
            "url": f"https://lichess.org/{game_id}",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "bot_username": BOT_USERNAME,
            "bot_color": "white" if my_color == chess.WHITE else "black",
            "opponent": opponent_name,
            "total_plies_played_by_bot": len(moves_logged),
            "pgn": pgn_text,
            "moves": moves_logged,
        }

        path_data = os.path.join(GAMES_DIR, f"{game_id}.json")

        with open(path_data, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        if pgn_text:
            pgn_data_path = os.path.join(GAMES_DIR, f"{game_id}.pgn")
            with open(pgn_data_path, "w", encoding="utf-8") as f:
                f.write(pgn_text)

        print(f"[{game_id}] Complete match and PGN saved to {path_data}", flush=True)

    def start_event_stream(self):
        """Main listening loop that streams incoming challenges and game starts."""
        self.verify_account()
        stream_url = f"{LICHESS_API}/api/stream/event"
        backoff = 2.0

        print(f"\n⚡ Jess-Hyperbullet is LIVE and listening for challenges!", flush=True)
        print(f"👉 Challenge the bot at: https://lichess.org/@/{BOT_USERNAME}", flush=True)
        print(f"Listening on event stream: {stream_url}...\n", flush=True)

        while self.running:
            try:
                with requests.get(stream_url, headers=HEADERS, stream=True, timeout=90) as resp:
                    if resp.status_code == 429:
                        print("Rate limited by Lichess (429). Sleeping 30s...", flush=True)
                        time.sleep(30)
                        continue

                    if resp.status_code != 200:
                        print(f"Event stream returned {resp.status_code}: {resp.text}. Reconnecting in {backoff}s...", flush=True)
                        time.sleep(backoff)
                        backoff = min(60.0, backoff * 1.5)
                        continue

                    backoff = 2.0  # Reset backoff upon successful connection

                    for line in resp.iter_lines():
                        if not self.running:
                            break
                        if not line:
                            continue

                        try:
                            event = json.loads(line.decode("utf-8"))
                        except Exception:
                            continue

                        evt_type = event.get("type")

                        if evt_type == "challenge":
                            ch_data = event.get("challenge", {})
                            self.handle_challenge(ch_data)

                        elif evt_type == "gameStart":
                            g_info = event.get("game", {})
                            g_id = g_info.get("id") or g_info.get("gameId")
                            if g_id and g_id not in self.active_games:
                                t = threading.Thread(target=self.play_game, args=(g_id,), daemon=True)
                                self.active_games[g_id] = t
                                t.start()

                        elif evt_type == "challengeCanceled":
                            print(f"[Challenge Canceled] {event.get('challenge', {}).get('id')}", flush=True)

                        elif evt_type == "challengeDeclined":
                            print(f"[Challenge Declined] {event.get('challenge', {}).get('id')}", flush=True)

            except (requests.RequestException, Exception) as e:
                print(f"Event stream error: {e}. Reconnecting in {backoff}s...", flush=True)
                time.sleep(backoff)
                backoff = min(60.0, backoff * 1.5)

    def challenge_player(self, target_user: str, time_limit: int = 30, inc: int = 0, rated: bool = False):
        """Sends a challenge to a specific player or bot."""
        url = f"{LICHESS_API}/api/challenge/{target_user}"
        payload = {
            "rated": rated,
            "clock.limit": time_limit,
            "clock.increment": inc,
            "color": "random",
            "variant": "standard",
        }
        print(f"Sending challenge to '{target_user}' ({time_limit}+{inc}s, rated={rated})...", flush=True)
        res = requests.post(url, headers=HEADERS, json=payload, timeout=10)
        if res.status_code == 200:
            ch = res.json().get("challenge", {})
            print(f"✅ Challenge created! ID: {ch.get('id')} | URL: {ch.get('url')}", flush=True)
            return ch.get("id")
        else:
            print(f"❌ Failed to challenge {target_user}: {res.status_code} {res.text}", flush=True)
            return None


def main():
    parser = argparse.ArgumentParser(description="Jess-Hyperbullet Autonomous Lichess Bot")
    parser.add_argument("--challenge", type=str, help="Username of player/bot to challenge")
    parser.add_argument("--time", type=int, default=30, help="Clock limit in seconds (default: 30 for hyperbullet)")
    parser.add_argument("--inc", type=int, default=0, help="Increment in seconds (default: 0)")
    parser.add_argument("--rated", action="store_true", help="Make challenge rated")
    args = parser.parse_args()

    bot = LichessBotRunner()

    if args.challenge:
        bot.challenge_player(args.challenge, time_limit=args.time, inc=args.inc, rated=args.rated)

    # Always start listening for incoming games / challenges
    bot.start_event_stream()


if __name__ == "__main__":
    main()
