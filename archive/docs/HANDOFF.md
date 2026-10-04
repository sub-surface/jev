# 🧭 Jevformer & Tri-Process Architecture: Research Handoff Document

> **Status:** All 7 Benchmark Frontiers Complete | Tri-Process Architecture Validated | Repository Reorganized into `jev-vault/`  
> **Date:** September 18, 2026  
> **Authors:** Leon 

---

## 📌 Executive Summary & Key Theoretical Discoveries

Over the course of this research trajectory, we systematically investigated the boundary between **continuous representation** and **discrete search**, moving from our original Jevformer / ERET architecture to the unified **Tri-Process Neural Hierarchy (NNUE + Jev + In-Context Transformer)**.

### 1. The Fall of Latent Recurrence (ERET)
* We tested unrolling weight-tied Krasnoselskii-Mann continuous equilibrium loops across 2D labyrinths, real-time Tetris, bullet micro-chess, and deep 20-hop symbolic deductions.
* In every single domain, **continuous latent loops collapsed ($0.0\% - 14.3\%$ accuracy)**. 
* *The Mathematical Reason:* Continuous state relaxations cannot represent discrete counterfactual branches. In complex state spaces with loops or obstacles, continuous vectors drift into orbital limit cycles (94.3%–100% cycle trap rate).

### 2. The Bitter Lesson Ascendant (Discrete Search)
* Explicit counterfactual tree search (System 2) scaled monotonically with compute across every domain, providing zero-shot out-of-distribution transfer without retraining (85%–100% accuracy).

### 3. The Law of Discrete Invariant Coupling
* When synthesizing an **In-Context Transformer (System 2)** with an **NNUE Sparse Accumulator (System 0)**, we initially tried **continuous additive modulation** ($\vec{a}_{\text{new}} = \vec{a} + \vec{m}$).
* *The Anomaly:* Adding continuous latent vectors into NNUE's Clipped ReLU activation shifted neuron zero-thresholds, creating **representation interference** and degrading performance (Countdown dropped from 48% to 36%; Mini-ARC dropped from 98% to 46%).
* *The Solution:* Replacing continuous modulation with **Discrete Sub-Goal Invariant Coupling** (System 2 emits discrete symbolic factor masks or candidate operator invariants).
* *The Empirical Result:* 
  * Mini-ARC: **100.0% exact-match inductive generalization** (+54.0% gain).
  * Countdown: **58.0% accuracy** with **90.9 expansions** (+10% accuracy, -29.5% search compute).

---

## 🏗️ Repository Architecture & `jev-vault` Organization

The repository has been restructured so that the **`jev-vault/`** directory functions as a self-contained, elegant **Obsidian Vault** for reviewing code, reading docs, querying experimental data, and inspecting figures:

```
jev/
├── jev-vault/                           <-- OPEN THIS AS AN OBSIDIAN VAULT
│   ├── Index.md                         <-- Main Map of Content (wikilinks to everything)
│   ├── analysis/
│   │   ├── empirical_results_master.csv <-- Complete tabular spreadsheet of all runs
│   │   └── Analysis_Dashboard.md        <-- Dataview dashboard & summary tables
│   ├── src/                             <-- Core Neural Engines & Game Simulators
│   │   ├── triprocess_engine.py         <-- Unified NNUE + Jev + Transformer architecture
│   │   ├── bullet_chess_6x6.py          <-- 6x6 Los Alamos Chess (1.5s clock)
│   │   ├── symbolic_proof_deep.py       <-- 20-hop guarded register machine
│   │   ├── tetris_pure_rl.py            <-- Real-time Tetris TD(0) Bellman engine
│   │   ├── gardner_chess_arena.py       <-- 5x5 Gardner Chess engine
│   │   └── render_harder_frontiers.py   <-- Trajectory filmstrip renderer
│   ├── benchmarks/                      <-- Benchmark Suites
│   │   ├── benchmark_countdown_triprocess.py
│   │   ├── benchmark_mini_arc_triprocess.py
│   │   ├── benchmark_chess_adaptive_triprocess.py
│   │   └── test_coupling_physics.py     <-- Dense vs Discrete coupling validation
│   ├── modal_cloud/                     <-- Scaled Cloud Infrastructure
│   │   ├── modal_triprocess_scaled.py   <-- Scaled multi-benchmark runner on A10G
│   │   └── modal_epistemic_search_scaled.py
│   ├── figures/                         <-- High-resolution figures (fig1 to fig28)
│   └── docs/                            <-- Core Publications & Logs
│       ├── DEVLOG.md                    <-- Comprehensive chronological lab notebook
│       ├── ROADMAP.md                   <-- Empirical roadmap
│       └── WHITEPAPER.md                <-- Formal mathematical monograph
├── data/                                <-- Raw PyTorch .pt evaluation tensors
└── .venv/                               <-- Python 3.12 virtual environment (CUDA active)
```

---

## 📊 Summary of Benchmark Frontiers (1 to 7)

| Paradigm | Domain | Best Model | Accuracy / Win Rate | Compute Metric | Key Publication Figure |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P1** | 2D Mazes ($16\times 16$) | Adaptive ETS | **73.0%** (vs 1% ERET) | 53 expansions | `fig15_epistemic_search_scaled_pareto.png` |
| **P2** | Real-Time Tetris | Adaptive ETS | **100.0%** (vs 0% ERET) | 2.0 expansions/drop | `fig16_tetris_epistemic_arena.png`, `fig19` |
| **P3** | 5x5 Bullet Chess (2.5s) | Adaptive ETS | **100.0%** (32-0 sweep) | 0 clock flags | `fig17_bullet_chess_epistemic_arena.png`, `fig20` |
| **P4** | 6x6 Chess (1.5s) | Adaptive ETS | **83.3%** (5-1 vs D3) | 0 clock flags | `fig22_bullet_chess_6x6_tournament.png`, `fig23` |
| **P5** | 20-Hop Deduction | Adaptive ToT | **88.6%** (vs 14% ERET) | 0% cycle traps | `fig21_deep_deduction_scaling.png`, `fig24` |
| **P6** | Countdown Math | Discrete Tri-Process | **58.0%** (vs 36% Dense) | 90.9 expansions | `fig25_countdown_triprocess_results.png`, `fig28` |
| **P7** | Mini-ARC Grids | Discrete Tri-Process | **100.0%** (Exact Match) | 1 step induction | `fig26_mini_arc_triprocess_results.png`, `fig28` |

---

## 🔌 Obsidian Setup & Recommended Plugins

To explore `jev-vault` in Obsidian with full interactivity:
1. Open Obsidian -> "Open folder as vault" -> select `C:\Users\Leon\Desktop\Psychograph\jev\jev-vault`.
2. Install the following Community Plugins:
   * **Dataview:** Renders live tables and queries over `analysis/empirical_results_master.csv` and Markdown notes.
   * **Obsidian Charts:** Directly plots bar, line, and radar charts inside notes.
   * **Advanced Tables:** Format, sort, and navigate experimental tables with Excel shortcuts.
   * **Omnisearch:** Fast fuzzy search with OCR indexing across all generated figures and heatmaps.
   * **Excalidraw:** Sketching and annotating multi-scale architecture diagrams.

---

## ☁️ Running the Scaled Modal Experiment

To launch the scaled Tri-Process benchmarks on Modal cloud compute (NVIDIA A10G):

```powershell
# From workspace root:
.\.venv\Scripts\modal run modal_triprocess_scaled.py
```

### What this executes on Modal:
1. **Scaled Countdown:** 100 trials on 8-number pools with targets $T \in [1000, 9999]$.
2. **Scaled Mini-ARC:** 100 multi-color visual grid induction tasks.
3. **Scaled 6x6 Los Alamos Chess:** 16 match series (96 games) evaluating in-context opponent adaptation.
4. Persists checkpoint payload to `/checkpoints/triprocess_scaled_results.pt` on the persistent volume `jevformer-checkpoints`.

---

## 🎯 Next Steps for Fresh Context: Hybrid Loss Training

### 1. The Core Focus: Hybrid Model Distillation
* **Discovery:**
  - `data/jev_champion.pt` is a **tactical sniper**: 100/100 benchmark score, detects mates and sacrifices instantly, but loses in long-horizon games vs Stockfish (mated at ply 26 and 35).
  - `data/h100_eret_latest.pt` is a **positional grinder**: 5,000 games on H100 SXM5, held Stockfish Level 1 to a **60-ply draw as White (1/2 - 1/2)** and survived 47 plies as Black, but its Noul stays too high during tactical crises (~0.78), scoring 45.0% on tactical puzzles.
* **The Mission:** Unify both models through multi-task hybrid training:
  $$\mathcal{L}_{\text{hybrid}} = \mathcal{L}_{\text{selfplay\_policy\_value}} + 0.5 \mathcal{L}_{\text{tactical\_curriculum}} + 0.2 \mathcal{L}_{\text{epistemic\_noul\_calibration}}$$
* **Implementation Plan:**
  - In `modal_cloud/modal_h100_hybrid_scale.py` (or a dedicated local script `train_hybrid_loss.py`), interleave self-play replay buffer batches with tactical crisis curriculum batches in every optimizer step.
  - Evaluate against Stockfish 19 (`python evaluate.py --model ... --stockfish --sf-match`).

### 2. Operational Status:
* **Background Tasks:** All background tasks and the GUI server have been terminated.
* **Git Repository:** Fully consolidated, committed, and pushed to `master`.
* **Live Deployment:** Edge worker at `https://jev.subsurfaces.net` is live with the 6-stage forward pass simulator.
* **Vector GIF Renderer:** Fully upgraded with PySide6 QtSvg vector pieces and clean typography.

