# Jevformer & Tri-Process Research Vault (Map of Content)

Welcome to the internal research vault for **Jevformer & The Tri-Process Neural Architecture (NNUE + Jev + In-Context Transformer)**.

This vault organizes the theoretical foundations, architectural innovations, empirical benchmarks, and mechanistic interpretability for unifying **System 0 (Microsecond NNUE Sparse Accumulators)**, **System 1 (Calibrated TypeSafe Jev Decisions)**, and **System 2 (In-Context Transformer Deliberation)**.

---

## 🗺️ Navigation Map

### 1. The Tri-Process Neural Hierarchy (The New Paradigm)
* [[src/triprocess_frontier.py]]: Frontier architecture with 4 coupling modes (Dense, Multiplicative, Discrete Mask, VQ Bottleneck).
* [[src/mechinterp_engine.py]]: Mechanistic Interpretability engine for CReLU activation physics, dead neuron leakage, and SVD spectral geometry.
* [[Theory/The-Law-of-Discrete-Invariant-Coupling]]: Proof and empirical demonstration of why continuous dense additive modulation causes feature interference, and why discrete symbolic invariants yield optimal generalization.
* [[Theory/Discrete-Bottlenecks-and-VQ-Coupling]]: Vector-Quantized (VQ) Shannon channel bottlenecks with Straight-Through Estimators.
* [[Literature/Key-Literature-Index]]: Citations and technical synthesis of RLCD (arXiv:2307.12950), Epiplexity (arXiv:2601.03220), and Stockfish NNUE.
* [[analysis/Analysis_Dashboard]]: Live Dataview dashboard synthesizing all empirical frontiers.
* [[analysis/empirical_results_master.csv]]: Master tabular spreadsheet tracking win rates, expansions, latencies, and failure modes across all experiments.

### 2. Empirical Benchmark Suite (Frontiers 1–8)
* **P1: Scaled 2D Labyrinths (12x12 & 16x16):** [[modal_cloud/modal_epistemic_search_scaled.py]] — Collapse of ERET (1.0%) vs. Triumph of Adaptive ETS (73.0%).
* **P2: Real-Time Tetris (Heuristic & Pure RL):** [[src/tetris_arena.py]] & [[src/tetris_pure_rl.py]] — 2x faster line clears via reflexive drops on flat surfaces.
* **P3: Gardner Bullet Micro-Chess (2.5s Clock):** [[src/gardner_chess_arena.py]] — Flawless 32-0 tournament sweep via dynamic clock allocation.
* **P4: 6x6 Los Alamos Mini-Chess (1.5s Clock):** [[src/bullet_chess_6x6.py]] — 36 squares, beating Fixed Depth 3 (5-1) by exploiting opponent clock forfeits (0 flags).
* **P5: Deep 20-Hop Symbolic Program Deduction:** [[src/symbolic_proof_deep.py]] — ERET trapped in cycles (94.3%) vs. Adaptive ToT (88.6% accuracy, 0% cycles).
* **P6: Countdown Arithmetic Target Synthesis:** [[benchmarks/benchmark_countdown_triprocess.py]] & [[benchmarks/benchmark_coupling_frontiers.py]] — Discrete factor sub-goals boost accuracy to 64.0%.
* **P7: Mini-ARC Symbolic Grid Induction:** [[benchmarks/benchmark_mini_arc_triprocess.py]] & [[benchmarks/test_coupling_physics.py]] — 100.0% exact-match inductive generalization via discrete invariant coupling.
* **P8: Multi-Scale Coupling Physics & VQ-Bottleneck:** [[modal_cloud/modal_frontier_scaled.py]] & [[benchmarks/benchmark_coupling_frontiers.py]] — CReLU dead neuron leakage (29.8% dense vs. 0.0% VQ/multiplicative).

### 3. Scaled Modal Cloud Infrastructure
* [[modal_cloud/modal_frontier_scaled.py]]: Scaled Frontier runner executing Countdown, Mini-ARC, 6x6 Chess, and MechInterp extraction on NVIDIA A10G.
* [[modal_cloud/modal_triprocess_scaled.py]]: Initial scaled Tri-Process runner.
* [[modal_cloud/modal_epistemic_search_scaled.py]]: 16x16 maze scaling on Modal Volume `/checkpoints`.

### 4. Theoretical Foundations & History
* [[Theory/The-Shannon-Rate-Distortion-Frontier]]: Channel capacity, rate-distortion bounds, and why discrete bottlenecks outperform continuous latents.
* [[Theory/The-Law-of-Discrete-Invariant-Coupling]]: Why continuous dense additive vectors corrupt Clipped ReLU zero-thresholds.
* [[Theory/Discrete-Bottlenecks-and-VQ-Coupling]]: Vector-quantized epistemic bottlenecks and rate-distortion theory.
* [[Theory/Domain-B-Cellular-Automata-ARC]]: Spatial invariant induction in ARC-AGI & 2D Cellular Automata without CReLU threshold leakage.
* [[Theory/Domain-C-Formal-Proof-Synthesis]]: Topological contraction dynamics, Lyapunov epistemic potential, and cycle elimination in Lean 4 formal proofs.
* [[Theory/The-TypeSafe-Jev-Theorem]]: Dual Impossibility Theorem proving mathematically impossible ill-typed symbols and cycle non-occurrence.
* [[Theory/Beyond-Next-Token-Prediction-and-Looped-Architectures]]: The limits of autoregression and why latent recurrence fails without discrete branching.
* [[Theory/RLCD-Proper-Scoring-Calibration]]: Derivation of Reinforcement Learning from Contrastive Distillation and Brier proper scoring.
* [[Theory/Epiplexity-and-Bounded-Information]]: Computation-bounded observer entropy ($S_{\mathcal{F}, T}$).
* [[src/chess_8x8_engine.py]]: Standard 8x8 FIDE Chess engine with Jev Epistemic Gating and Negamax Alpha-Beta search.
* [[src/chess_gui_server.py]]: Real-time 8x8 Bullet Chess GUI, drag-and-drop arena, dual-sparkline graph, and live move credences table.
* [[src/lichess_bot.py]]: Autonomous Lichess Bot daemon for `@jess-hyperbullet`.
* [[Lichess-Bot-Guide]]: Setup, API mechanics, and matchmaking guide for `@jess-hyperbullet`.
* [[Permanent-Bot-Hosting-Guide]]: Free 24/7 hosting architecture on Hugging Face Spaces & Cloudflare (`jev.subsurfaces.net`).
* [[Theory/Leela-Style-TriProcess-Training]]: Dual-head policy + value + Noul formulation on public games.
* [[modal_cloud/modal_leela_jev_training.py]]: Modal cloud Leela-style training on NVIDIA A10G GPU.

---

## 📊 Core Empirical Figures Gallery

### The Frontier Coupling & MechInterp Discovery
* `[[figures/ca_triprocess_evolution.gif]]`: Animated Cellular Automata evolution comparing TypeSafe discrete invariant (0.0% leakage) vs. continuous dense additive perturbation.
* `[[figures/fig32_typesafe_soundness_theorem.png]]`: Empirical Verification of Theorem 3 (0.0% ill-typed symbols, 0.0% cycle entrapment, 100% solvency).
* `[[figures/fig31_cellular_automata_physics.png]]`: Cellular Automata 2D CReLU Physics (Dead-cell leakage and Hamming divergence from invariant).
* `[[figures/fig30_shannon_channel_rate_distortion.png]]`: The Shannon Rate-Distortion Frontier (Distortion D(B), accuracy phase transition, CReLU dead neuron leakage across bits).
* `[[figures/fig29_frontier_coupling_physics.png]]`: The Physics of Multi-Scale Coupling (Mini-ARC exact match, Countdown Pareto frontier, CReLU dead neuron leakage).
* `[[figures/fig28_coupling_physics_synthesis.png]]`: The Law of Discrete Invariant Coupling (Dense vs. Discrete across Mini-ARC and Countdown).
* `[[figures/fig25_countdown_triprocess_results.png]]`: Countdown Arithmetic training convergence, accuracy, and node expansions.
* `[[figures/fig26_mini_arc_triprocess_results.png]]`: Mini-ARC inductive generalization rate and Noul confidence distribution.
* `[[figures/fig27_chess_adaptive_triprocess.png]]`: 6x6 Chess in-context adaptation curves across multi-game matches.

### Trajectory Visualizations & Harder Arenas
* `[[figures/fig24_deep_deduction_path_render.png]]`: 20-hop register state evolution (Adaptive ToT vs. ERET cyclic drift).
* `[[figures/fig23_bullet_chess_6x6_filmstrip.png]]`: 6-ply 6x6 Bullet Chess filmstrip with millisecond clock countdowns.
* `[[figures/fig21_deep_deduction_scaling.png]]`: 8-hop, 14-hop, and 20-hop deduction horizon scaling.
* `[[figures/fig22_bullet_chess_6x6_tournament.png]]`: 6x6 Bullet Chess tournament outcomes and flag distributions.
* `[[figures/fig19_tetris_decision_filmstrip.png]]`: Real-time Tetris filmstrip (Reflexive drops vs. Lookahead search).
* `[[figures/fig20_bullet_chess_gameplay_render.png]]`: 5x5 Gardner Chess checkmate sequence.
* `[[figures/fig15_epistemic_search_scaled_pareto.png]]`: Modal cloud A10G Pareto frontier on 12x12 and 16x16 mazes.
