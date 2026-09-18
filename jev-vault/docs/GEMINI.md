# Engineering Principles & Project Rules

## Core Principles
* **First Principles:** Write code that is clean, mathematically sound, type-safe, and stripped of unnecessary abstraction.
* **Clarity over Cleverness:** Ensure data structures and flow of logic are transparent and robust.
* **No Sloppiness:** Maintain impeccable hygiene with environments, dependencies, schemas, and error boundaries.

---

## Critical Gotchas & Common Pitfalls

### 1. Windows Unicode & Standard Output Encoding (cp1252 Crash)
* **Problem:** Windows PowerShell / CMD defaults to `cp1252`, which immediately crashes with `UnicodeEncodeError: 'charmap' codec can't encode character` when printing Greek letters ($\tau, \gamma, \Delta$), mathematical symbols ($\pm, \le$), or Unicode chess glyphs.
* **Rule:** Always add the following at the very top of executable Python scripts:
  ```python
  import sys
  sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  sys.stderr.reconfigure(encoding="utf-8", errors="replace")
  ```

### 2. Output Buffering in Asynchronous / Background Tasks
* **Problem:** Standard Python buffers stdout when not connected to an interactive TTY. Background tasks appear stuck with empty logs until the process terminates.
* **Rule:** Always pass `flush=True` to `print()` statements in training, evaluation, and benchmark loops, or run scripts with `python -u`.

### 3. Negamax vs. Minimax Sign Inversions
* **Problem:** In adversarial zero-sum games, mixing Minimax (where White maximizes and Black minimizes) with Negamax (where the current player always maximizes their own score and negates child scores) causes sign inversion bugs where the searcher intentionally plays losing moves.
* **Rule:** Strictly adhere to the Negamax recurrence:
  ```python
  score = -negamax(child, depth - 1, -beta, -alpha, -current_turn)
  ```
  Ensure static evaluation returns positive values for the current player (`current_turn * evaluate_static()`).

### 4. Tri-Process Model Coupling (Discrete Invariants vs. Dense Perturbations)
* **Problem:** Adding continuous dense latent vectors ($\vec{a}_{\text{new}} = \vec{a} + \vec{m}$) into a discrete accumulator or sparse activation layer (such as Clipped ReLU in Stockfish-style NNUE) shifts the zero-point threshold for every neuron simultaneously. This corrupts inactive features and desensitizes discrete heuristic ordering.
* **Rule:** System 2 (Transformers/LLMs) must steer System 0 (NNUE/discrete searchers) through **discrete symbolic constraints, sub-goal factor masks, and epistemic routing**, never through raw continuous additive latent perturbations.

### 5. Cycle Entrapment in Deep Combinatorial Search
* **Problem:** In state spaces with reversible transitions (modular arithmetic, register machines, grid operations, game boards), continuous latent recurrence and naive greedy searchers enter infinite orbital limit cycles (e.g. 94%+ cycle trap rate).
* **Rule:** Always maintain path history / visited state sets and explicitly prune visited cycles during search unrolling.

### 6. Strict Clock Management in Bullet Game Search
* **Problem:** Fixed-depth searchers (e.g., Depth 3 on a 36-square 6x6 board) suffer from exponential branch explosions and flag out on time under tight bullet clocks (e.g., 1.5s total).
* **Rule:** Use calibrated epistemic confidence ($\text{Noul}$) or remaining time thresholds to dynamically gate search depth: spend $\sim 1\text{ms}$ on quiet positional plies, and only allocate deep search time during sharp tactical crises.

### 7. Modal Cloud Volume Commits
* **Problem:** Modal cloud functions writing artifacts or checkpoints to mounted volumes (e.g., `/checkpoints`) will lose data unless explicitly committed before the container shuts down.
* **Rule:** Always call `volume.commit()` after writing any file to a Modal Volume.

### 8. Repository Structure & Obsidian Vault Integrity
* **Problem:** Research documentation, Dataview queries, and figures can become fragmented across the root directory.
* **Rule:** All Obsidian notes, Dataview dashboards, analysis spreadsheets, figures, and organized source modules belong in `jev-vault/`. Root files should remain minimal entrypoints.
