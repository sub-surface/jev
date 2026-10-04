# 📜 Jevformer & Jev Architecture: Changelog

All notable changes, architectural experiments, and deployment milestones for the **Jev** decision engine and **Jevformer** neural models.

---

## [v2.3.0] — 2026-09-19
### 🎯 Multi-Task RLCD Scaled Cloud Run & Epistemic Calibration Breakthrough
- **Cloud Infrastructure:** Scaled multi-task training on NVIDIA L40S (48GB Ada Lovelace) via serverless Modal cloud. Total training completed in **101.6 seconds** ($0.0550 compute spend, total dispatch $0.0681).
- **Data Foundation:** Streaming ingestion of 15 diverse tasks from `tasksource/tasksource-instruct-v0` (3,073 examples) across NLI, QA, Paraphrase, and Other reasoning families.
- **Zero-Shot Epistemic Preservation:** Eliminated cross-entropy zero-shot overconfidence collapse ($\text{ECE} = 87.81\% \to \mathbf{5.12\%}$, a **17.1x reduction in calibration error**).
- **Zero Generalization Drop:** Out-of-domain evaluation on unseen `paraphrase` family reached **44.11% accuracy** (vs 43.94% in-task, $\Delta = -0.17\%$) and **5.12% ECE** (vs 6.86% in-task, $\Delta = -1.75\%$).
- **Selective Risk-Coverage Monotonicity:** Confirmed calibrated quality scaling under Noul deferral: accuracy climbs to **57.69%** (in-task) and **53.85%** (zero-shot) at 10% coverage budget.
- **Cardinality Temperature Scaling:** Fitted optimal softmax scaling per candidate cardinality: $T^*(2)=1.0000, T^*(3\text{--}5)=0.9439, T^*(6\text{--}10)=0.8625, T^*(11+)=0.8454$.
- **Volume Checkpointing & Ledger:** Champion checkpoint persisted to Modal Volume `jev-model-artifacts/jev_qwen05b_rlcd_champion.pt`. Total cumulative Modal spend is **$0.1896 USD**, preserving **$27.7004 USD (99.32%)** of project budget.
- **Publication Figures:** Generated and verified [Fig 37: Modal Scaled RLCD Multi-Task Generalization & Epistemic Calibration](file:///C:/Users/Leon/Desktop/Psychograph/jev/jev-vault/figures/fig37_modal_scaled_rlcd_generalization.png) and documented in `jev-vault/Experiments/EXP-005-Modal-L40S-Scaled-RLCD-MultiTask.md`.

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
