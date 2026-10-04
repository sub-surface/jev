# archive/ — pre-2026-10-03 material (untrusted)

Everything here dates from before the teardown. Most of it was written with an AI assistant, and many of its claims do not survive an audit. Read `../REVIEW.md` before relying on any of it, including code, figures, docs and "theorems".

| path | what |
|---|---|
| `contracts/02-rlcd-decision/` | The old option-marker decision model ("RLCD"), its Modal training scripts and its result JSONs. Contains Modal entrypoints: don't run them by accident. |
| `contracts/01-discrete-coupling/`, `contracts/03-typed-selfplay/` | "Tri-process" / VQ-coupling toy benchmarks; an epiplexity toy |
| `legacy-eret/` | ERET chess engine, the old training and evaluation scripts, Modal training scripts, and a Stockfish Windows binary. The binary is useful for E002. |
| `early_explorations/`, `plots/` | Older toys. Several plot scripts synthesise data with `np.random`. |
| `jev-vault/` | The Obsidian vault: theory, experiments and docs |
| `docs/` | The old root docs: README, WHITEPAPER, DEVLOG, ROADMAP, HANDOFF, CHANGELOG, GEMINI |
| `data/`, `figures/` | Old checkpoints and result `.pt` files, and generated figures |
| `old-site/edge-cockpit/` | The previous jev.subsurfaces.net Worker. It calls the bot's public Modal `wake` endpoint. |

Safe to delete once a snapshot commit or tag exists, because git history keeps everything.
