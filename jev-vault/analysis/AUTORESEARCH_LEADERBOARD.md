# 🏆 JEVFORMER AUTORESEARCH LEADERBOARD
*Autonomous Hypothesis Exploration & Empirical Progression (2026)*

## Benchmark Protocol
- **Tactical Benchmark:** 20 Curated Crisis Positions (Sacrifices, Mates, Pins, Counter-strikes)
- **Quiet Control Suite:** 10 Peaceful Control Positions (Openings, Symmetrical Endgames)
- **North Star Metric:** $\text{Score} = 50 \cdot (\text{Solved}/20) + 30 \cdot \text{CrisisRecall} + 20 \cdot \text{QuietPrecision}$

---

## Leaderboard Standings

| Rank | Run ID | Architecture & Hypothesis | Tactical Solve % | Crisis Recall % | Quiet Precision % | Median Latency | North Star Score | Status |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 👑 EXP-04-CALIBRATED-ERET | ERET with Calibrated Tau (0.35) & Queen Promo Prior | 100.0% | 100.0% | 100.0% | 12.4ms | **100.0** | PROMOTED |
| 👑 EXP-03-ERET-CURRICULUM | ERET + Brier Noul Calibration + Policy Ordering | 95.0% | 100.0% | 20.0% | 10.9ms | **81.5** | PROMOTED |
| EXP-02-ERET-EQUILIBRIUM | ERET Looped Krasnoselskii-Mann Equilibrium | 25.0% | 100.0% | 0.0% | 11.8ms | **42.5** | REVERTED |
| 👑 EXP-01-CALIBRATED-GATE | Calibrated Noul + 1-ply Mate Gate | 55.0% | 90.0% | 50.0% | 124.0ms | **64.5** | PROMOTED |
| 👑 1 | `BASELINE-V3` | Conv4 + PeSTO Distillation + Heuristic Margin Noul | 45.0% (9/20) | 100.0% (20/20) | 0.0% (0/10) | 228.6ms | **52.50** | 👑 Current Champion |

---

## Empirical Hypotheses Queue & Log

### Run 1: `EXP-01-ERET-EQUILIBRIUM`
* **Hypothesis:** Replacing static feedforward with Epistemic Recurrent Equilibrium (ERET Krasnoselskii-Mann inner iterations $\gamma_k = 1 - \text{Noul}_k$) will allow latent feature cross-talk to resolve tactical forks and pins, improving tactical solve rate without increasing parameters.
* **Status:** In progress.

### Run 2: `EXP-02-INTRINSIC-NOUL-CALIBRATION`
* **Hypothesis:** Training an intrinsic Noul head directly on tactical crisis labels ($y=1$ quiet, $y=0$ crisis) via Brier proper scoring will fix the Aporia Inversion bug where peaceful openings with margin $\approx 0$ triggered false-alarm depth 3 search, jumping Quiet Precision from $0\% \to 90\%+$.
* **Status:** Queued.

### Run 3: `EXP-03-TIER2-ACTOR-BATCHER`
* **Hypothesis:** Decoupling board rollouts across 16 CPU workers with an asynchronous batched queue to A100-80GB ($B=256$) will accelerate self-play throughput by $>15\times$, enabling rapid policy iteration.
* **Status:** Queued.
