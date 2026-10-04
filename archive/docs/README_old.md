# ⚡ JEV: Calibrated Decision Intelligence & Tri-Process Architecture

[![Edge Cockpit](https://img.shields.io/badge/Edge%20Cockpit-jev.subsurfaces.net-blue)](https://jev.subsurfaces.net)
[![Lichess Bot](https://img.shields.io/badge/Lichess-@jess--hyperbullet-brightgreen)](https://lichess.org/@/jess-hyperbullet)
[![Ecosystem Radar](https://img.shields.io/badge/Awesome%20Jev-691%2B%20Projects-2563eb)](https://github.com/logicrw/awesome-jev-projects)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **"Scale compute and search through principled epistemic invariants, not unbounded parameter sprawl."**

**JEV** is a first-principles research laboratory and codebase engineered to develop **fast, calibrated, type-safe decision intelligence**. It bridges microsecond neural accumulators (**System 0**), calibrated typed discrete decision reflex (**System 1**), and in-context deliberative search (**System 2**).

---

## 🏛️ The Tri-Process Neural Hierarchy

```mermaid
flowchart TD
    State([Unstructured State / Context]) --> S0[System 0: Sparse Accumulator<br/>Latency < 1ms | Clipped ReLU >= 50% Sparsity]
    State --> S1[System 1: TypeSafe Jev Decision Engine<br/>Latency < 100ms | Choice, Score, Noul]
    
    S1 --> Gate{"Epistemic Gate<br/>Noul >= tau & Calibrated?"}
    Gate -->|YES: Reflex| S0Action([Instant Reflex Action])
    
    Gate -->|NO: Crisis / Low Noul| S2[System 2: In-Context Deliberation / Search<br/>Tree of Thoughts / MCTS]
    
    S2 -.->|Discrete Invariant Coupling<br/>Symbolic Factor Masks & Sub-Goals| S0
    S0 --> S2
    S2 --> S2Action([Deliberated Action])
```

---

## 📂 The Three Distinct Research Contracts

The repository is strictly structured around three distinct, rigorous research contracts:

```
jev/
├── contracts/
│   ├── 01-discrete-coupling/   # Contract 01: Tri-Process & Law of Discrete Invariant Coupling
│   ├── 02-rlcd-decision/        # Contract 02: TypeSafe Jev Decision Foundation Model (RLCD)
│   └── 03-typed-selfplay/       # Contract 03: Typed Self-Play Pretraining & Epiplexity (Zero Data)
│
├── deployments/                 # Working live applications & edge services
│   ├── edge-cockpit/            # Cloudflare Worker live visualizer (jev.subsurfaces.net)
│   └── lichess-bot/             # 24/7 autonomous Lichess bot daemon (@jess-hyperbullet)
│
├── jev-vault/                   # Dedicated Obsidian Knowledge Vault (Theory, Lit, Plans, Figures)
└── archive/                     # Archived explorations & falsified continuous recurrence
    ├── legacy-eret/             # Archived Krasnoselskii-Mann continuous recurrence & chess scripts
    └── early_explorations/      # Historical early prototypes
```

---

### 1. [Contract 01: Discrete Invariant Coupling](contracts/01-discrete-coupling/README.md)
* **Question:** How do continuous deliberative representations couple into fast sparse accumulators without representation interference?
* **Core Law:** *The Law of Discrete Invariant Coupling* — Continuous dense additive vectors into Clipped ReLU corrupt zero-crossing thresholds (Countdown dropped 48% $\to$ 36%; Mini-ARC dropped 98% $\to$ 46%). System 2 must couple to System 0 via **discrete symbolic factor masks and sub-goal invariants** (Mini-ARC: 100.0% exact match; Countdown: 58.0% with 29.5% fewer expansions).
* **Validation:** Verified across 8 benchmark frontiers (2D Mazes, Tetris, Gardner Chess, 6x6 Los Alamos, 20-Hop Deduction, Countdown, Mini-ARC, Cellular Automata).

### 2. [Contract 02: TypeSafe Jev Decision Model (RLCD)](contracts/02-rlcd-decision/README.md)
* **Question:** How do we train sub-100ms language/decision models that emit calibrated confidence without human preference sycophancy?
* **Formulation:** **Reinforcement Learning from Contrastive Distillation (RLCD)** (Yang et al., 2023) combined with strictly convex **Proper Scoring Rules** (Brier loss, Ranked Probability Score, Spherical score).
* **High-Performance Stack:** Fused analytic Brier GPU kernel (machine-verified to $6.94 \times 10^{-18}$ error), batched option tensor gather (6.35x speedup), temperature scaling.
* **Scale-up:** Ingests `tasksource-instruct-v0` (5.3M examples, 485 tasks) with strictly held-out zero-shot validation (Commonsense, Paraphrase, Fact Verification). Compatible with 691+ open-source projects on [Awesome Jev Projects](https://github.com/logicrw/awesome-jev-projects).

### 3. [Contract 03: Typed Self-Play Pretraining & Epiplexity (Zero Data)](contracts/03-typed-selfplay/README.md)
* **Question:** Can a System 1 decision model acquire structured predictive priors from synthetic computable structure alone—zero natural data—before task training?
* **Theoretical Anchors:** *Self-Play Pretraining with Zero Data* (Cowsik et al., arXiv:2609.30063) and *Epiplexity* (Finzi et al., arXiv:2601.03220).
* **Core Bet:** A *typed* program space (e.g. typed stack VM) avoids the 99.3% crash rate of untyped Turing machines (Brainf\*ck), and *epiplexity* curates learnable structure rather than arbitrary noise.
* **Toy 2 Discoveries:** Tested 7 token-stream families. Uncovered severe **negative transfer** ($R = -0.151$, `modmul -> modadd`) due to mismatched underlying computational primitives, and demonstrated **observer-relative invisibility** (`modadd` invisible to weak MLP).

---

## 🚀 Live Deployments

* **[Edge Cockpit (Cloudflare Worker)](deployments/edge-cockpit/):** Live interactive forward-pass visualizer, 13-bitplane slicer, channel excitation heatmap, and CReLU accumulator inspector running at **[jev.subsurfaces.net](https://jev.subsurfaces.net)**.
* **[Lichess Bot Daemon](deployments/lichess-bot/):** 24/7 autonomous hyperbullet engine [@jess-hyperbullet](https://lichess.org/@/jess-hyperbullet).

---

## 📚 Obsidian Knowledge Vault (`jev-vault/`)

Open `jev-vault/` as a dedicated vault in [Obsidian](https://obsidian.md) for full interactive research navigation:
* **`Literature/`:** Reading notes and citations (RLCD, Epiplexity, Zero-Data Self-Play, Rewarding Doubt).
* **`Theory/`:** Formal proofs (Discrete Invariant Coupling, Rate-Distortion Bounds, TypeSafe Jev Theorem).
* **`Plans/`:** Research roadmaps (`PLAN-001`, `PLAN-002`, `PLAN-003`).
* **`figures/`:** High-resolution trajectory renders, pareto curves, and gifs (fig1 through fig38).
* **`analysis/`:** Master tabular CSV (`empirical_results_master.csv`) and Dataview dashboards.

---

## ⚰️ Archived Explorations (`archive/`)

* **[Legacy ERET](archive/legacy-eret/README.md):** Scientific autopsy documenting why continuous Krasnoselskii-Mann latent recurrence entered orbital limit cycles (94.3%–100% trap rate) in reversible state spaces, and housing historical training scripts (`train.py`, `evaluate.py`, `gui.py`).
