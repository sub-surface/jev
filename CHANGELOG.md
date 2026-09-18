# 📜 Jevformer & Jev Architecture: Changelog

All notable changes, architectural experiments, and deployment milestones for the **Jev** decision engine and **Jevformer** neural models.

---

## [v2.2.0] — 2026-09-18
### 🚀 Permanent Cloudflare Edge & Zero-Idle Modal Deployment
- **Edge Cockpit:** Deployed interactive chess cockpit and architecture explorer to [`https://jev.subsurfaces.net`](https://jev.subsurfaces.net) on Cloudflare Anycast Edge.
- **Serverless Bot Daemon:** Launched `modal_cloud/modal_lichess_bot_daemon.py` on Modal Cloud with instant HTTP wake webhook (`sub-surface--jess-hyperbullet-bot-wake.modal.run`).
- **Zero-Idle Guarantee:** Automatic container shutdown after 15 minutes of inactivity ($0.00 idle cost).
- **Lichess BOT Verification:** Verified official Lichess bot account [`@jess-hyperbullet`](https://lichess.org/@/jess-hyperbullet).

---

## [v2.1.0] — 2026-09-18
### ⚡ Latency Overhaul & Adaptive Epistemic Tree Search (ETS)
- **Opening Book Integration:** Instant sub-1ms response for classical opening lines (Ruy Lopez, Sicilian, French, Caro-Kann, Queen's Gambit).
- **Epistemic Noul Gating:** Calibrated threshold ($\tau = 0.70$). Quiet positions trigger 1-ply reflex moves in **15ms–35ms**; tactical crises dynamically route to depth-2/3 Negamax search.
- **Transposition Table (TT):** Added Zobrist/FEN transposition caching to alpha-beta unrolling, eliminating redundant branch re-evaluations.
- **Live Match Verification:** Latency decreased from ~750ms to 31ms–66ms in live 30s bullet on Lichess.

---

## [v2.0.0] — 2026-09-18
### 🧠 Leela-Style Distillation on NVIDIA A10G Cloud GPU
- **Massive Training Run:** Trained on 20,000 diverse master positions across 10 epochs using Modal Cloud GPU compute.
- **Loss Convergence:** Value MSE dropped from **0.4519 to 0.0207** (-95.4%).
- **Mechanistic Sparsity:** 128-neuron CReLU accumulator settled at **60.0% sparsity**, demonstrating crisp discrete representations with zero dead-neuron leakage.
- **Volume Persistence:** Persisted Leela weights to Modal Volume `jevformer-checkpoints`.

---

## [v1.5.0] — 2026-09-18
### ♟️ Scaling to Standard 8x8 Chess & Local Cockpit
- **Standard 8x8 Ruleset:** Full support for castling, en passant, promotions, checks, threefold repetition, fifty-move rule, and insufficient material.
- **Minimal Monochromatic UI:** High-contrast, borderless aesthetic with SVG vector pieces, click-to-move and drag-and-drop mechanics.
- **Mechanistic Probing Panel:** Real-time visualization of 128 CReLU activations, Noul certainty curve, and candidate move credences table.
- **Stockfish Integration:** Installed local Stockfish 17 universal engine for real-time game reviews and evaluation accuracy benchmarking.

---

## [v1.0.0] — 2026-09-18
### 🔬 Foundational Research & Tri-Process Model
- **Tri-Process Coupling:** Formalized System 0 (discrete heuristic accumulator), System 1 (epistemic confidence gating $\text{Noul}$), and System 2 (symbolic constraint search).
- **Cycle Entrapment Avoidance (Domain C):** Proven and mitigated 94%+ cycle trap rates in reversible discrete state manifolds (modular register machines, grid operations).
- **Continuous Cellular Automata (Domain B):** Simulated neural CA emergent pattern formation with stable localized solitons.
- **Theorem 2 Proof:** Proved the mathematical impossibility of invalid output symbols via type-safe projective manifold representations.
