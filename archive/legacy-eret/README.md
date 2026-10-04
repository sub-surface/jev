# Archive: Legacy ERET & Continuous Latent Recurrence

> **Status:** Archived & Falsified | Replaced by Tri-Process Hierarchy (Contract 01) & RLCD (Contract 02)  
> **Theoretical Verdict:** Continuous latent recurrence in zero-sum adversarial and reversible state spaces enters orbital limit cycles (94.3%–100% trap rate) and cannot represent discrete counterfactual branch points.

---

## ⚰️ Scientific Autopsy: The Fall of Latent Recurrence

The initial hypothesis of JEV / ERET was that iterating a weight-tied residual block through damped Krasnoselskii-Mann fixed-point updates:
$$h_{k+1} = (1 - \gamma_k) h_k + \gamma_k \mathcal{B}_\theta(h_k)$$
would converge to a contractive positional equilibrium that captures deep search without explicit tree unrolling.

### Why It Failed Across All Complex Domains:
1. **Orbital Limit Cycles:** In state spaces with reversible transitions (modular arithmetic, mazes, register machines, piece maneuvers), continuous state trajectories enter infinite circular orbits rather than fixed points (0.0%–14.3% accuracy, 94.3% cycle entrapment).
2. **Inability to Branch Counterfactually:** Continuous latents cannot superimpose mutually exclusive tactical branches without catastrophic interference.
3. **Threshold Corruption:** Injecting continuous dense latents into Clipped ReLU accumulators destroys neuron sparsity and desensitizes discrete heuristic ordering.

### The Successor:
Discrete search and symbolic invariant coupling (the Bitter Lesson) succeeded where continuous recurrence failed, leading directly to **Contract 01: Discrete Invariant Coupling** and **Contract 02: TypeSafe Jev Decision Models**.

---

## 📂 Archived Artifacts Inventory

- **Root Entrypoint Scripts:**
  - `train.py`: Legacy local ERET curriculum training script.
  - `evaluate.py`: Legacy 30-position tactical suite & Stockfish 19 UCI validator.
  - `gui.py`: Local web GUI cockpit launcher.
  - `demo.py`: Command-line interactive chess game demo.
- **Engine Modules:**
  - `eret_engine.py`: Original Krasnoselskii-Mann recurrent architecture.
  - `stockfish_validator.py`: Stockfish 19 protocol interface and GIF generator.
  - `train_eret_tactical_curriculum.py`: Tactical puzzle curriculum optimizer.
  - `fast_parallel_selfplay.py`: Multi-threaded chess rollout generator.
  - `game_reviewer.py`: PGN game annotator.
  - `evaluate_bot_strength.py`: Elo rating evaluator.
  - `benchmark_harness.py`: Frozen 30-position ground truth suite.
  - `chess_8x8_engine.py` / `chess_gui_server.py`: Standalone 8x8 engines.
- **Cloud & Tooling:**
  - `modal_h100_hybrid_scale.py`: Scaled H100 self-play pipeline.
  - `stockfish/`: Official Stockfish 19 binary.
