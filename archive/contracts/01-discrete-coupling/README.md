# Contract 01: Discrete Invariant Coupling & Tri-Process Neural Hierarchy

> **Status:** Frontiers P1–P8 Complete | Theoretical Proofs Verified | Master CSV Indexed  
> **Key Finding:** Dense continuous perturbations into Clipped ReLU cause fatal threshold drift; System 2 must couple to System 0 via discrete invariants and symbolic factor masks.

---

## 🏛️ Executive Overview

Contract 01 establishes the foundational coupling physics between fast neural accumulators and discrete deliberation. Over seven empirical benchmark frontiers, continuous latent recurrence (ERET) systematically failed due to orbital limit cycles. In response, we developed the **Tri-Process Neural Hierarchy**:

```mermaid
flowchart TD
    State([Input State / Board / Grid]) --> S0[System 0: Sparse NNUE Accumulator<br/>Latency < 1ms | CReLU Clamp]
    State --> S1[System 1: Calibrated Jev Router<br/>Latency < 50ms | Choice, Score, Noul]
    
    S1 --> Gate{"Epistemic Gate<br/>Noul >= tau & Solved?"}
    Gate -->|YES: Reflex| Action([Fast Executed Move])
    
    Gate -->|NO: Crisis / Low Noul| S2[System 2: In-Context Deliberation / Search<br/>Tree of Thoughts / MCTS]
    
    S2 -.->|Discrete Invariant Coupling<br/>Factor Masks / Symbolic Pruning| S0
    S0 --> S2
    S2 --> Action
```

---

## 🔬 The Law of Discrete Invariant Coupling

When synthesizing an **In-Context Deliberator (System 2)** with an **NNUE Sparse Accumulator (System 0)**:
1. **Continuous Additive Modulation ($\vec{a}_{\text{new}} = \vec{a} + \vec{m}$):**
   - Shifts the zero-crossing threshold across all neurons simultaneously.
   - Degrades accuracy: Countdown dropped from 48% to 36%; Mini-ARC dropped from 98% to 46%; dead-neuron leakage rose to 29.8%.
2. **Discrete Sub-Goal Invariant Coupling:**
   - System 2 emits discrete symbolic factor masks or operator constraints:
     $$\mathcal{M} = \text{TopK}(\text{softmax}(W_{\text{gate}} \cdot z_{S2})) \in \{0, 1\}^D$$
   - Preserves $\ge 50\%$ guaranteed CReLU sparsity and eliminates dead-neuron threshold drift.
   - **Empirical Results:** Mini-ARC reached **100.0%** exact match; Countdown reached **58.0%** with **29.5% fewer expansions**.

---

## 📊 Summary of Benchmark Frontiers (P1 to P8)

| Frontier | Domain | Architecture | Performance | Key Discovery |
| :--- | :--- | :--- | :--- | :--- |
| **P1** | 2D Mazes ($16\times 16$) | Adaptive ETS | **73.0%** (vs 1% ERET) | Eliminates orbital cycles |
| **P2** | Real-Time Tetris | Adaptive ETS | **100.0%** (vs 0% ERET) | Dynamic depth saves clock |
| **P3** | 5x5 Gardner Chess | Adaptive ETS | **100.0%** (32-0 sweep) | Zero clock forfeits |
| **P4** | 6x6 Los Alamos Chess | Adaptive ETS | **83.3%** (5-1 vs D3) | Exploits opponent clock flags |
| **P5** | 20-Hop Deduction | Adaptive ToT | **88.6%** (vs 14% ERET) | Cycle non-occurrence theorem |
| **P6** | Countdown Arithmetic | Discrete Tri-Process | **58.0%** (vs 36% Dense) | Discrete factor sub-goals |
| **P7** | Mini-ARC Grids | Discrete Tri-Process | **100.0%** Exact Match | Zero threshold leakage |
| **P8** | Cellular Automata | VQ / Invariant | **0.0%** Dead Leakage | Straight-Through VQ channel |

---

## 📂 Directory Layout

```
contracts/01-discrete-coupling/
├── README.md                          # This contract document
├── src/                               # Neural engines & game simulators
│   ├── triprocess_engine.py           # Unified NNUE + Jev + Transformer architecture
│   ├── triprocess_frontier.py         # 4-mode coupling physics implementation
│   ├── mechinterp_engine.py           # SVD geometry & CReLU activation physics
│   ├── fused_crelu_kernel.py          # Fast CReLU kernel
│   ├── bitter_lesson_triprocess.py    # Compute scaling searcher
│   ├── bitter_lesson_mcts_diverse.py  # Monte Carlo tree search
│   ├── batched_bitter_mcts.py         # Batched rollouts
│   ├── bullet_chess_6x6.py            # 6x6 Los Alamos engine
│   ├── gardner_chess_arena.py         # 5x5 Gardner engine
│   ├── tetris_pure_rl.py              # Tetris TD(0) Bellman engine
│   ├── symbolic_proof_deep.py         # 20-hop guarded register machine
│   └── render_harder_frontiers.py     # Trajectory visualizer
│
├── benchmarks/                        # Frontier benchmark harnesses
│   ├── benchmark_countdown_triprocess.py
│   ├── benchmark_mini_arc_triprocess.py
│   ├── benchmark_chess_adaptive_triprocess.py
│   ├── benchmark_cellular_automata_triprocess.py
│   ├── benchmark_coupling_frontiers.py
│   ├── benchmark_shannon_rate_distortion.py
│   ├── benchmark_typesafe_cycle_elimination.py
│   ├── arena_intertwined_architectures.py
│   └── test_coupling_physics.py
│
└── modal/                             # Scaled Cloud GPU runners (Modal)
    ├── modal_frontier_scaled.py
    └── modal_epistemic_search_scaled.py
```

---

## 🚀 Quickstart

Run coupling physics validation:
```bash
python contracts/01-discrete-coupling/benchmarks/test_coupling_physics.py
```

Run Countdown Tri-Process benchmark:
```bash
python contracts/01-discrete-coupling/benchmarks/benchmark_countdown_triprocess.py
```

Run Mini-ARC grid induction benchmark:
```bash
python contracts/01-discrete-coupling/benchmarks/benchmark_mini_arc_triprocess.py
```
