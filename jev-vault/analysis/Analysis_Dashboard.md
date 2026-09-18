---
type: dashboard
project: Jevformer & Tri-Process Architecture
created: 2026-09-18
tags: [analysis, dataview, benchmarks, triprocess, bitter-lesson]
---

# 🔬 Jevformer & Tri-Process Empirical Analysis Dashboard

This dashboard organizes the empirical outcomes across all seven benchmark frontiers, contrasting continuous latent recurrence, discrete tree search, and our unified **Tri-Process Architecture (NNUE + Jev + In-Context Transformer)**.

The underlying raw data is tracked in [[empirical_results_master.csv]]

---

## 📊 1. Master Empirical Summary Table

| Frontier | Domain               | Architecture                   | Accuracy / Win Rate | Expansions / Ply | Latency (ms) | Primary Failure Mode           |
| :------- | :------------------- | :----------------------------- | :------------------ | :--------------- | :----------- | :----------------------------- |
| **P1**   | 2D Mazes (16x16)     | ERET (Latent Loop)             | 1.0%                | 0.0              | 18.7         | Wall Collision (93%)           |
| **P1**   | 2D Mazes (16x16)     | Adaptive ETS (Jev-Search)      | **73.0%**           | 53.0             | 68.2         | None (0 Collisions)            |
| **P2**   | Tetris (Pure RL)     | ERET (Latent Loop)             | 0.0%                | 0.0              | 4.2          | Board Topped Out               |
| **P2**   | Tetris (Pure RL)     | Adaptive ETS (tau=0.80)        | **100.0%**          | 2.0              | 9.1          | None (2x Faster than Uniform)  |
| **P3**   | Gardner Chess (2.5s) | Reflexive Jev                  | 0.0%                | 0.0              | 0.5          | Tactical Blunder (0-16)        |
| **P3**   | Gardner Chess (2.5s) | Fixed Depth 2                  | 0.0%                | 15.2             | 168.0        | Clock Forfeit (15 flags)       |
| **P3**   | Gardner Chess (2.5s) | Adaptive ETS                   | **100.0%**          | 3.8              | 14.2         | None (32-0 Tournament Sweep)   |
| **P4**   | 6x6 Chess (1.5s)     | Fixed Depth 3                  | 16.7%               | 28.4             | 210.0        | Clock Forfeit (5 flags)        |
| **P4**   | 6x6 Chess (1.5s)     | Adaptive ETS                   | **83.3%**           | 6.2              | 22.4         | None (0 Clock Flags)           |
| **P5**   | 20-Hop Deduction     | ERET (Latent Loop)             | 14.3%               | 0.0              | 319.2        | Cycle Entrapment (94.3%)       |
| **P5**   | 20-Hop Deduction     | S1 Greedy                      | 48.6%               | 0.0              | 117.4        | Attractor Basins (77.1%)       |
| **P5**   | 20-Hop Deduction     | Adaptive Epistemic ToT         | **88.6%**           | 131.3            | 712.6        | None (0.0% Cycles)             |
| **P6**   | Countdown Math       | Dense Additive Tri-Process     | 36.0%               | 147.7            | 2112.0       | Representation Interference    |
| **P6**   | Countdown Math       | Discrete Invariant Tri-Process | **58.0%**           | 90.9             | 410.2        | None (-29.5% Search Compute)   |
| **P6**   | Countdown Math       | VQ-TriProcess (4-bit Latent)   | **64.0%**           | 64.4             | 534.5        | None (Learned STE Codebook)    |
| **P7**   | Mini-ARC Grids       | Dense Additive Tri-Process     | 6.0% - 46.0%        | 5.0              | 19.5         | Feature Saturation & Leakage   |
| **P7**   | Mini-ARC Grids       | Multiplicative Gated           | 32.0%               | 5.0              | 18.2         | Zero-Preserving but Drifted    |
| **P7**   | Mini-ARC Grids       | VQ-TriProcess (4-bit Latent)   | **94.0%**           | 1.0              | 8.1          | None (Discrete Bottleneck)     |
| **P7**   | Mini-ARC Grids       | Discrete Invariant Tri-Process | **100.0%**          | 1.0              | 8.4          | None (Flawless Generalization) |
| **P8**   | Rate-Distortion      | Shannon VQ Sweep (B=1..8)      | **D=0.000 (B>=1)**  | 1.0              | 4.2          | CReLU Threshold Leakage at B=inf |
| **P9**   | Standard 8x8 Chess   | Adaptive ETS + Jev 8x8 S1      | **Active Arena**    | 1 - 3            | 0.8 - 35.0   | Drag-and-Drop Bullet Arena     |
| **Dom B**| ARC-AGI & 2D CA      | 2D CReLU CA + VQ Invariants    | **100% Induction**  | 1.0              | <5.0         | Zero-Leakage Spatial Gating    |
| **Dom C**| Lean 4 & Registers   | Lyapunov Epistemic Halting     | **0.0% Cycles**     | Contractive      | <2.0         | Universal Cycle Elimination    |
| **EXP-10**| Theorem 3 Soundness | TypeSafe Jev (Fiber + Lyap)    | **100.0%**          | 8.8              | 1.2          | 0.0% Ill-Typed, 0.0% Cycles    |

---

## 📈 2. Core Scientific Figures

### Theorem 3: TypeSafe Jev Invariant & Soundness
![[figures/fig32_typesafe_soundness_theorem.png]]
> *Figure 32: Empirical Verification of Theorem 3 across 80 formal rewrite tasks. Left: Ill-Typed symbol emissions (Unconstrained 49.7% vs. TypeSafe Jev 0.0%). Middle: Limit cycle entrapment frequency (Syntactic Mask alone 67.5% vs. TypeSafe Jev 0.0%). Right: Formal Q.E.D. solvency pass rate (Unconstrained 32.5% vs. TypeSafe Jev 100.0%).*

### Cellular Automata Physics & Zero-Leakage Invariant Evolution
![[figures/fig31_cellular_automata_physics.png]]
> *Figure 31: 2D Cellular Automata CReLU Physics under Continuous vs. TypeSafe Discrete Invariant Steering. Dense additive latent perturbation triggers 5.58% parasitic dead-cell leakage and 12.6% structural Hamming divergence, while TypeSafe discrete invariant steering guarantees 0.0% leakage and flawless structural conservation.*

### Animated Cellular Automata Evolution
![[figures/ca_triprocess_evolution.gif]]
> *Figure 31-GIF: 36-step real-time side-by-side evolution comparing TypeSafe Discrete Invariant steering (left, emerald) vs. Continuous Dense Additive modulation (right, crimson).*

### The Physics of Multi-Scale Coupling & MechInterp
![[figures/fig29_frontier_coupling_physics.png]]
> *Figure 29: Mechanistic Interpretability of Multi-Scale Coupling. Left: Mini-ARC inductive generalization (Dense Additive collapses to 6.0% vs. VQ-TriProcess 94.0% and Discrete Invariant 88.0%-100%). Middle: Countdown Pareto efficiency. Right: CReLU dead neuron leakage rate (Dense additive induces 29.8% spurious reactivation of dead features, while Multiplicative and VQ Bottlenecks maintain exactly 0.0% leakage).*

### The Law of Discrete Invariant Coupling
![[figures/fig28_coupling_physics_synthesis.png]]
> *Figure 28: Comparison of Dense Additive Modulation vs. Discrete Sub-Goal Invariant Coupling across Mini-ARC and Countdown. Continuous vector injection causes feature interference at the Clipped ReLU threshold, whereas discrete invariant coupling yields 100% exact-match generalization.*

### 20-Hop Deep Deduction Trajectory
![[figures/fig24_deep_deduction_path_render.png]]
> *Figure 24: Register state evolution comparing Adaptive Epistemic ToT (top) vs. ERET Latent Recurrence (bottom). ERET stalls in cyclic attractor basins, while counterfactual search reaches the target assertion.*

### 6x6 Bullet Chess Decision Filmstrip
![[figures/fig23_bullet_chess_6x6_filmstrip.png]]
> *Figure 23: Six-ply board progression showing Adaptive ETS dynamically switching between 1ms reflexive moves on quiet plies and deep minimax calculations during tactical crises under a 1.5s clock.*

---

## 🔌 3. Recommended Obsidian Plugins for this Vault

To maximize the analytical and visual power of `jev-vault`, the following community plugins are recommended:

1. **Dataview (Essential):**
   * Enables live SQL-like querying over file frontmatter, tasks, and empirical CSV tables.
   * Query example:
     ```sql
     TABLE file.name, Verification_Accuracy_Pct, Failure_Mode
     FROM "analysis"
     WHERE Verification_Accuracy_Pct > 50
     ```
2. **Obsidian Charts:**
   * Directly plots interactive line charts, bar charts, and radar plots inside Markdown notes using Chart.js syntax.
3. **Advanced Tables:**
   * Auto-formatting, Excel-like formulas, and keyboard shortcuts for managing tabular experimental data.
4. **Excalidraw / Mermaid Diagrams:**
   * Visualizing multi-agent workflows, System 0/1/2 hierarchies, and neural compute graphs.
5. **Omnisearch:**
   * High-speed indexed search with text-in-image OCR to search within publication figures and generated heatmaps.
6. **Local REST API / Obsidian Git:**
   * Automatic commit history and programmatic synchronization with background training scripts.
