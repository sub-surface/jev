"""
=============================================================================
⚡ MODAL CLOUD: LARGE-SCALE ERET TRAINING (A100-80GB / H100)
=============================================================================
Scales the Champion ERET (Epistemic Recurrent Equilibrium Transformer)
across large numbers of games (1,000 - 5,000+ games) on NVIDIA A100:
1. Tier-2 Actor-Batcher: 16 concurrent CPU workers feeding an async GPU GEMM queue.
2. Looped Krasnoselskii-Mann Equilibrium: Inner micro-time relaxation (k in [1, 4]).
3. Brier Proper Scoring Noul: Intrinsic epistemic calibration without Aporia Inversion.
4. Pure Bitter Lesson: Terminal grounded rewards, Dirichlet exploration, Temperature schedule.
5. Automated Checkpointing & Epoch GIF rendering to Modal Volume.
=============================================================================
"""

import modal
import os
import sys

# Rule 1: Windows Unicode encoding hygiene
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

app = modal.App("eret-large-scale")
volume = modal.Volume.from_name("jevformer-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("fonts-dejavu-core", "fonts-freefont-ttf")
    .pip_install(
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "chess>=1.10.0",
        "pillow>=10.0.0",
    )
)

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/checkpoints": volume},
    timeout=3600,
)
def run_large_scale_eret(
    total_games: int = 1000,
    epochs: int = 5,
    num_workers: int = 16,
    batch_size: int = 128,
    lr: float = 1e-3,
):
    import time
    import math
    import queue
    import threading
    from typing import List, Tuple, Dict, Optional
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import chess
    from PIL import Image, ImageDraw, ImageFont

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=======================================================", flush=True)
    print("⚡ LARGE-SCALE ERET TRAINING ON NVIDIA A100-80GB", flush=True)
    print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
    print(f"Target: {total_games} Games across {epochs} Epochs ({total_games // epochs} games/epoch)", flush=True)
    print(f"Parallel Architecture: Tier-2 Actor-Batcher ({num_workers} Async CPU Workers)", flush=True)
    print("Inner Micro-Time: Krasnoselskii-Mann Equilibrium Relaxation (k in [1, 4])", flush=True)
    print("=======================================================\n", flush=True)

    # -------------------------------------------------------------------------
    # 1. ERET Model Definition (1.70M Parameters)
    # -------------------------------------------------------------------------
    class ResidualBlock(nn.Module):
        def __init__(self, channels: int = 64):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, channels // 4),
                nn.ReLU(),
                nn.Linear(channels // 4, channels),
                nn.Sigmoid()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            res = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            b, c, _, _ = out.shape
            w = self.se(out).view(b, c, 1, 1)
            out = out * w
            return F.relu(out + res)

    class ERETChessEngine(nn.Module):
        def __init__(self, max_unrolls: int = 4, tau_halt: float = 0.70):
            super().__init__()
            self.max_unrolls = max_unrolls
            self.tau_halt = tau_halt

            self.stem = nn.Sequential(
                nn.Conv2d(13, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            self.eq_block = ResidualBlock(channels=64)
            self.noul_sensor = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            self.fc_accum = nn.Linear(64 * 8 * 8, 128)
            self.ln_accum = nn.LayerNorm(128)
            self.val_head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Tanh()
            )
            self.policy_head = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 4096)
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
            B = x.shape[0]
            h = self.stem(x)
            unrolls_taken = 0
            final_noul = torch.zeros((B,), device=x.device)

            for k in range(self.max_unrolls):
                unrolls_taken += 1
                noul_k = self.noul_sensor(h).squeeze(-1)
                final_noul = noul_k

                if torch.all(noul_k >= self.tau_halt) and k >= 1:
                    break

                gamma_k = (1.0 - noul_k).view(B, 1, 1, 1)
                h = (1.0 - gamma_k) * h + gamma_k * self.eq_block(h)

            flat_h = h.view(B, -1)
            accum = self.ln_accum(self.fc_accum(flat_h))
            crelu = torch.clamp(accum, 0.0, 1.0)
            val = self.val_head(crelu).squeeze(-1)
            pol_logits = self.policy_head(crelu)
            return val, pol_logits, final_noul, crelu, unrolls_taken

    def encode_board_13(board: chess.Board) -> np.ndarray:
        t = np.zeros((13, 8, 8), dtype=np.float32)
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p:
                r = 7 - chess.square_rank(sq)
                c = chess.square_file(sq)
                pt = p.piece_type - 1
                plane = pt if p.color == chess.WHITE else pt + 6
                t[plane, r, c] = 1.0
        if board.turn == chess.WHITE:
            t[12, :, :] = 1.0
        return t

    # -------------------------------------------------------------------------
    # 2. Asynchronous Central GPU Dispatcher Queue
    # -------------------------------------------------------------------------
    class GPUDispatcher:
        def __init__(self, model: nn.Module, device: torch.device, max_batch: int = 128):
            self.model = model
            self.device = device
            self.max_batch = max_batch
            self.req_queue = queue.Queue()
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

        def request_eval(self, tensor_np: np.ndarray, holder: List, event: threading.Event):
            self.req_queue.put((tensor_np, holder, event))

        def stop(self):
            self.running = False
            self.thread.join(timeout=1.0)

        def _loop(self):
            while self.running:
                requests = []
                try:
                    req = self.req_queue.get(timeout=0.005)
                    requests.append(req)
                except queue.Empty:
                    continue

                t0 = time.time()
                while len(requests) < self.max_batch and (time.time() - t0 < 0.001):
                    try:
                        requests.append(self.req_queue.get_nowait())
                    except queue.Empty:
                        break

                tensors = [r[0] for r in requests]
                batch_x = torch.tensor(np.array(tensors), dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    vals, pols, nouls, _, _ = self.model(batch_x)

                vals_np = vals.cpu().numpy()
                pols_np = pols.cpu().numpy()
                nouls_np = nouls.cpu().numpy()

                for i, (_, holder, evt) in enumerate(requests):
                    v = float(vals_np[i]) if len(requests) > 1 else float(vals_np.item())
                    p = pols_np[i] if len(requests) > 1 else pols_np[0]
                    n = float(nouls_np[i]) if len(requests) > 1 else float(nouls_np.item())
                    holder.extend([v, p, n])
                    evt.set()

    # -------------------------------------------------------------------------
    # 3. Vector GIF Renderer
    # -------------------------------------------------------------------------
    UNICODE_PIECES = {
        chess.Piece(chess.PAWN, chess.WHITE): "♙", chess.Piece(chess.KNIGHT, chess.WHITE): "♘",
        chess.Piece(chess.BISHOP, chess.WHITE): "♗", chess.Piece(chess.ROOK, chess.WHITE): "♖",
        chess.Piece(chess.QUEEN, chess.WHITE): "♕", chess.Piece(chess.KING, chess.WHITE): "♔",
        chess.Piece(chess.PAWN, chess.BLACK): "♟", chess.Piece(chess.KNIGHT, chess.BLACK): "♞",
        chess.Piece(chess.BISHOP, chess.BLACK): "♝", chess.Piece(chess.ROOK, chess.BLACK): "♜",
        chess.Piece(chess.QUEEN, chess.BLACK): "♛", chess.Piece(chess.KING, chess.BLACK): "♚",
    }

    def render_game_gif(move_history: List[chess.Move], out_path: str, result_str: str = "") -> str:
        sq_size = 44
        header_h = 32
        footer_h = 28
        w = sq_size * 8
        h = w + header_h + footer_h

        font_piece = None
        for font_path in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
        ]:
            if os.path.exists(font_path):
                try:
                    font_piece = ImageFont.truetype(font_path, int(sq_size * 0.68))
                    break
                except Exception:
                    pass

        board = chess.Board()
        frames = []

        for ply in range(len(move_history) + 1):
            last_m = move_history[ply - 1] if ply > 0 else None
            im = Image.new("RGB", (w, h), color="#0D1117")
            draw = ImageDraw.Draw(im)

            t_str = "White" if board.turn == chess.WHITE else "Black"
            stat = f"Ply {ply} • {t_str} to move" if ply < len(move_history) else f"Result: {result_str}"
            draw.rectangle([(0, 0), (w, header_h)], fill="#161B22")
            draw.text((10, 8), stat, fill="#38BDF8" if ply < len(move_history) else "#10B981")

            for r in range(8):
                for c in range(8):
                    x1 = c * sq_size
                    y1 = header_h + r * sq_size
                    x2 = x1 + sq_size
                    y2 = y1 + sq_size
                    sq_col = "#E2E8F0" if (r + c) % 2 == 0 else "#475569"
                    sq = chess.square(c, 7 - r)

                    if last_m and sq == last_m.from_square: sq_col = "#38BDF8"
                    elif last_m and sq == last_m.to_square: sq_col = "#10B981"

                    draw.rectangle([(x1, y1), (x2, y2)], fill=sq_col)
                    p = board.piece_at(sq)
                    if p:
                        glyph = UNICODE_PIECES.get(p, p.symbol().upper())
                        col = "#FFFFFF" if p.color == chess.WHITE else "#0F172A"
                        if font_piece:
                            draw.text((x1 + 6, y1 + 2), glyph, font=font_piece, fill=col)
                        else:
                            draw.text((x1 + 14, y1 + 10), p.symbol().upper(), fill=col)

            draw.rectangle([(0, h - footer_h), (w, h)], fill="#161B22")
            draw.text((10, h - footer_h + 6), f"Move: {last_m.uci() if last_m else 'Start'} | ERET A100 Large-Scale", fill="#94A3B8")

            frames.append(im)
            if ply < len(move_history):
                board.push(move_history[ply])

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=350, loop=0)
        return out_path

    # -------------------------------------------------------------------------
    # 4. Multi-Worker Self-Play Simulation
    # -------------------------------------------------------------------------
    def worker_sim_game(dispatcher: GPUDispatcher) -> Tuple[List, float, str]:
        board = chess.Board()
        history = []
        ply = 0
        max_plies = 70

        while ply < max_plies and not board.is_game_over():
            legal = list(board.legal_moves)
            if not legal:
                break

            # 1-ply immediate checkmate gate
            mate_m = None
            for m in legal:
                board.push(m)
                if board.is_checkmate():
                    mate_m = m
                    board.pop()
                    break
                board.pop()

            if mate_m:
                pi_vec = np.zeros(4096, dtype=np.float32)
                pi_vec[mate_m.from_square * 64 + mate_m.to_square] = 1.0
                history.append((encode_board_13(board), board.turn == chess.WHITE, pi_vec, mate_m))
                board.push(mate_m)
                break

            holder = []
            evt = threading.Event()
            dispatcher.request_eval(encode_board_13(board), holder, evt)
            evt.wait()
            _, pol_logits, _ = holder

            legal_indices = [m.from_square * 64 + m.to_square for m in legal]
            sub_logits = pol_logits[legal_indices]
            sub_logits = sub_logits - np.max(sub_logits)
            probs = np.exp(sub_logits) / (np.sum(np.exp(sub_logits)) + 1e-9)

            if ply < 15 and len(legal) > 1:
                noise = np.random.dirichlet([0.3] * len(legal))
                probs = 0.75 * probs + 0.25 * noise

            temp = 1.0 if ply < 15 else 0.1
            if temp <= 0.05:
                chosen_idx = int(np.argmax(probs))
            else:
                scaled = probs ** (1.0 / temp)
                p_sample = scaled / scaled.sum()
                chosen_idx = np.random.choice(len(legal), p=p_sample)

            chosen_m = legal[chosen_idx]
            pi_vec = np.zeros(4096, dtype=np.float32)
            for idx, m in enumerate(legal):
                pi_vec[m.from_square * 64 + m.to_square] = probs[idx]

            history.append((encode_board_13(board), board.turn == chess.WHITE, pi_vec, chosen_m))
            board.push(chosen_m)
            ply += 1

        if board.is_checkmate():
            winner = chess.BLACK if board.turn == chess.WHITE else chess.WHITE
            z = 1.0 if winner == chess.WHITE else -1.0
            res_str = "1-0 (White Win)" if winner == chess.WHITE else "0-1 (Black Win)"
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
            z = 0.0
            res_str = "1/2-1/2 (Draw)"
        else:
            z = 0.0
            res_str = "Adjudicated Draw (Plies limit)"

        return history, z, res_str

    # -------------------------------------------------------------------------
    # 5. Training Loop across Large Number of Games
    # -------------------------------------------------------------------------
    model = ERETChessEngine().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    games_per_epoch = total_games // epochs
    replay_states = []
    replay_values = []
    replay_policies = []
    replay_nouls = []

    t0_all = time.time()

    for epoch in range(1, epochs + 1):
        print(f"\n--- EPOCH {epoch}/{epochs}: Simulating {games_per_epoch} Games via Tier-2 Batcher ---", flush=True)
        model.eval()
        dispatcher = GPUDispatcher(model, device, max_batch=128)

        epoch_results = [None] * games_per_epoch
        threads = []

        def worker_batch(indices):
            for idx in indices:
                epoch_results[idx] = worker_sim_game(dispatcher)

        chunk = math.ceil(games_per_epoch / num_workers)
        t_ep_start = time.time()
        for w_idx in range(num_workers):
            s = w_idx * chunk
            e = min(games_per_epoch, s + chunk)
            if s < e:
                t = threading.Thread(target=worker_batch, args=(range(s, e),))
                threads.append(t)
                t.start()

        for t in threads:
            t.join()

        dispatcher.stop()
        dt_selfplay = time.time() - t_ep_start

        # Process games
        ep_plies = 0
        last_moves = []
        last_res = ""
        outcomes = {"white": 0, "black": 0, "draw": 0}

        for hist, z, res_str in epoch_results:
            ep_plies += len(hist)
            if z == 1.0: outcomes["white"] += 1
            elif z == -1.0: outcomes["black"] += 1
            else: outcomes["draw"] += 1

            for enc, is_white_turn, pi, m in hist:
                s_z = z if is_white_turn else -z
                replay_states.append(enc)
                replay_values.append(s_z)
                replay_policies.append(pi)
                # Calibrated noul label: 0.1 for high tension / checks, 0.85 for quiet
                replay_nouls.append(0.15 if abs(s_z) > 0.5 else 0.85)

            last_moves = [item[3] for item in hist]
            last_res = res_str

        plies_per_sec = ep_plies / dt_selfplay if dt_selfplay > 0 else 0
        print(f"Epoch {epoch} Self-Play: {games_per_epoch} games ({ep_plies} plies) in {dt_selfplay:.1f}s ({plies_per_sec:.1f} plies/s)", flush=True)
        print(f"W/B/D: {outcomes['white']}/{outcomes['black']}/{outcomes['draw']} | Cumulative Positions: {len(replay_states)}", flush=True)

        # Render GIF
        gif_out = f"/checkpoints/eret_epoch_{epoch}.gif"
        try:
            render_game_gif(last_moves, gif_out, result_str=last_res)
            print(f"🎬 Saved game animation: {gif_out} ({len(last_moves)} plies | {last_res})", flush=True)
        except Exception as e:
            print(f"Warning: GIF failed: {e}", flush=True)

        # Training Step
        model.train()
        train_len = min(len(replay_states), 30000)
        X_t = torch.tensor(np.array(replay_states[-train_len:]), dtype=torch.float32, device=device)
        y_v_t = torch.tensor(np.array(replay_values[-train_len:]), dtype=torch.float32, device=device)
        y_p_t = torch.tensor(np.array(replay_policies[-train_len:]), dtype=torch.float32, device=device)
        y_n_t = torch.tensor(np.array(replay_nouls[-train_len:]), dtype=torch.float32, device=device)

        indices = np.arange(train_len)
        for _ in range(3):
            np.random.shuffle(indices)
            for s_idx in range(0, train_len, batch_size):
                b_idx = indices[s_idx:s_idx + batch_size]
                optimizer.zero_grad()
                v_pred, p_pred, noul_pred, crelu, _ = model(X_t[b_idx])
                loss_v = F.mse_loss(v_pred, y_v_t[b_idx])
                loss_p = -torch.sum(y_p_t[b_idx] * F.log_softmax(p_pred, dim=-1), dim=-1).mean()
                loss_noul = F.mse_loss(noul_pred, y_n_t[b_idx])
                loss_sparse = 0.02 * torch.mean(torch.abs(crelu))
                loss = loss_v + loss_p + 0.3 * loss_noul + loss_sparse
                loss.backward()
                optimizer.step()

        print(f"Epoch {epoch} Training Complete | Val MSE: {loss_v.item():.4f} | Pol CE: {loss_p.item():.4f} | Noul: {loss_noul.item():.4f}", flush=True)

        # Volume commit per epoch (Rule 7)
        torch.save(model.state_dict(), f"/checkpoints/eret_epoch_{epoch}.pt")
        torch.save(model.state_dict(), "/checkpoints/eret_large_scale_latest.pt")
        volume.commit()
        print(f"💾 Checkpoint and GIF committed to Modal Volume.", flush=True)

    total_time = time.time() - t0_all
    print("\n=======================================================", flush=True)
    print(f"🏆 LARGE-SCALE ERET TRAINING COMPLETE IN {total_time:.1f}s!", flush=True)
    print(f"Total Games: {total_games} | Total Positions: {len(replay_states)}")
    print("=======================================================\n", flush=True)

    return {
        "status": "success",
        "total_games": total_games,
        "positions": len(replay_states),
        "total_time_seconds": total_time,
        "final_val_mse": loss_v.item(),
        "final_policy_ce": loss_p.item(),
    }

@app.local_entrypoint()
def main(games: int = 1000, epochs: int = 5, workers: int = 16):
    print("Launching Large-Scale ERET Training on Modal A100-80GB...")
    res = run_large_scale_eret.remote(
        total_games=games,
        epochs=epochs,
        num_workers=workers,
    )
    print("Execution Result:", res)
