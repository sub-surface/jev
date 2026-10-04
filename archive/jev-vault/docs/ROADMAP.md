# The Epistemic Search Trilogy: Research Roadmap
**Authors:** Leon & Ilya Sutskever persona  
**Date:** September 18, 2026  
**Objective:** Scaling Calibrated Dual-Process Cognition (Jev + Epistemic Search) across Three Graded Frontiers  

---

## The Core Thesis
Current foundation models treat all tokens and states with uniform computational depth. Our empirical discoveries on 2D Labyrinths and Tetris have proven two foundational principles:
1. **Continuous latent equilibrium recurrence (ERET) collapses** when deployed in closed-loop interactive environments because continuous relaxation cannot perform counterfactual branching.
2. **Adaptive Epistemic Tree Search (ETS / Jev-Search)** is strictly Pareto-superior: using a calibrated System 1 evaluator (Jev) to dynamically gate explicit System 2 search expansions achieves higher task success with up to 78% fewer FLOPs.

We now formalize a three-stage experimental progression, ordered by structural complexity:

```mermaid
flowchart LR
    F1["Frontier 1: Pure RL Tetris<br>(Single-Agent Stochastic)"] --> F2["Frontier 2: Bullet Micro-Chess<br>(Adversarial 2-Player Clock)"]
    F2 --> F3["Frontier 3: Symbolic Proof Tracing<br>(Hierarchical Deductive Verification)"]
```

---

## Frontier 1: Pure Reinforcement Learning Tetris (The Bitter Lesson Grounding)
* **Status:** In Progress (First Implementation)
* **Complexity Level:** Medium (Single-agent, stochastic piece queue, immediate physics transitions)
* **The Core Departure:** In our previous run, Jev was bootstrapped using Dellacherie heuristic feature weights. Sutton’s Bitter Lesson demands that we **strip all human heuristics**.
* **Method:**
  * Define scalar environment reward: $R = \Delta \text{Lines} \times 10.0 - 0.5 \times \Delta \text{Holes} - 0.1 \times \text{Height} - 10.0 \times \mathbf{1}_{\text{GameOver}}$.
  * Train Jev's Policy-Value-Noul network purely through autonomous trial-and-error (Temporal Difference / Policy Improvement).
  * Evaluate dynamic lookahead search under a simulated time budget (ms per piece).
* **Key Metric:** Total pieces survived, line clearing rate, and compute Pareto frontier under purely learned values.

---

## Frontier 2: Bullet Micro-Chess (Adversarial Epistemic Time Allocation)
* **Status:** Designed (To follow Frontier 1)
* **Complexity Level:** High (Two-player zero-sum, minimax tree search, strict physical clock)
* **The Core Departure:** Applying Jev to physical time allocation. In bullet chess (e.g. 5-second total chess clock on a 5x5 Gardner board), players cannot search every move.
* **Method:**
  * 5x5 Gardner Chess engine (Pawns, Knights, Bishops, Rooks, Kings).
  * System 1 Jev outputs position evaluation $V(s)$ and tactical volatility $\text{Noul}(s) \in [0, 1]$ (peaceful quiet vs tactical capture/check).
  * When position is quiet ($\text{Noul} \ge \tau$): Move in $1\text{ms}$ (Reflexive).
  * When position is tactical ($\text{Noul} < \tau$): Allocate chess clock time to unroll Alpha-Beta / MCTS search.
* **Key Metric:** Win rate against fixed-depth engines under strict time-forfeit clocks.

---

## Frontier 3: Symbolic Code & Proof Deduction (Epistemic Tree-of-Thought)
* **Status:** Designed (To follow Frontier 2)
* **Complexity Level:** Frontier / AGI-Hard (Discrete symbolic state, multi-step dependency, zero surface leakage)
* **The Core Departure:** Extending the dual-process engine from spatial games to discrete programmatic deduction and formal theorem verification.
* **Method:**
  * Symbolic register machine / lambda expression evaluation with branches and assertions.
  * System 1 Jev: Step-level correctness verifier ($\text{Noul} = P(\text{step valid})$).
  * System 2 ETS: Explores counterfactual deductive branches, backtracking when assertions fail.
* **Key Metric:** Out-of-distribution reasoning depth (solving 15-step proofs when trained on 5-step proofs).

---

*Execution commences immediately on Frontier 1: Pure RL Scaled Tetris.*
