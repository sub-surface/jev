# jess-hyperbullet — Lichess bot

A small chess bot: a 623k-parameter CNN (`chess_8x8_engine.JevChess8x8Evaluator`) ranks candidate
moves, and a confidence gate decides whether to play the top move immediately or run a short
alpha-beta search (depth 2–3, material evaluation at the leaves). Lichess: <https://lichess.org/@/jess-hyperbullet>.

It is kept as-is because it is a ready-made **System-1 / System-2 testbed** (see `../../RESEARCH.md`,
E002). Be honest about what it is today: the network's "noul" confidence head was trained on
randomly generated targets, and the gate uses `tanh(3 * margin)` — so the confidence is **not
calibrated against anything yet**. Strength comes mostly from the handcrafted search.

## Files

| file | what |
|---|---|
| `lichess_bot.py` | local runner: streams Lichess events, plays games. CLI: `python lichess_bot.py [--challenge USER]` |
| `app.py` | the same runner plus a tiny health-check HTTP server (for container hosts) |
| `chess_8x8_engine.py` | model, board encoding, search |
| `weights/` | `jev_chess_8x8_gui.pt` (default; == the old `v3`), `_leela.pt`, `_v2.pt` — all load into the same model |
| `games/` | logs of games played by the local runner |
| `modal_lichess_bot_daemon.py` | **the currently deployed production bot** (Modal app `jess-hyperbullet-bot`) |

## Running

Token: `LICHESS_BOT_TOKEN` in `bot/lichess/.env` or the repo-root `.env`. Override weights with
`JEV_CHESS_CKPT=path`. Docker: `docker build -t jess bot/lichess && docker run -e LICHESS_BOT_TOKEN=... jess`
(the image now ships its weights; previously it silently fell back to random init).

## ⚠ Modal — read before touching

- The deployed Modal app is **self-contained**: it defines its own copy of the model and loads
  weights from the Modal volume `jevformer-checkpoints` (`jev_chess_8x8_leela_v3.pt`), not from this folder.
  Editing files here does **not** change production until someone runs `modal deploy`.
- It exposes a **public, unauthenticated** `wake` endpoint that spawns a daemon (0.5 CPU, 1 GB,
  ≤1 h, exits after 15 min idle). It is cheap, but anyone who finds the URL can trigger it, and
  repeated calls spawn parallel daemons. The old website called it; the new one does not.
- Never `modal run` / `modal deploy` anything in this repo without deciding to spend money.
