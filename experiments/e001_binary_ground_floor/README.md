# E001: the ground floor, calibrated prediction of binary sequences

**Status:** done, 2026-10-04. `python run.py` takes about 30 s on CPU.
**Outputs:** `results_full.json`, `fig_ledger_full.png`, `fig_jumps_full.png`, `log_full.txt`.
**Concepts:** [[code-length-is-log-loss]], [[corp-decomposition]] in `../../vault/`.

**Why binary sequences.** Log loss on bits *is* code length, so every claim in this experiment is a claim in bits. Each result below is one line of an exact ledger:

**bits/symbol = MCB − DSC + UNC**

- **MCB:** the cost of being miscalibrated.
- **DSC:** the bits actually extracted.
- **UNC:** the base-rate entropy.

The decomposition is the CORP split computed by isotonic regression (Arnold et al. 2023), and it is exact with no binning.

**Setup.**
- **Sources.** 8 sources, n = 8192 bits each, 3 seeds.
  - noise
  - an order-3 Markov chain
  - a period-23 pattern with 5% noise
  - Thue–Morse
  - an LCG top bit (pseudorandom)
  - the logistic map
  - binary expansions of Minkowski's ?(u), i.e. Stern–Brocot paths
  - a regime shift
- **Observers.** A compute ladder: KT estimators at context order 0–12, plus CTW-12. CTW is the exact Bayesian mixture over all tree structures of depth ≤ 12.

## Results

1. **Inference over structure beats any fixed amount of memory.**
   - CTW matches or beats the best fixed order on every source *without being told the order*:

     | Source | CTW | Best fixed order |
     |---|---|---|
     | markov-3 | 0.471 | KT-4, 0.472 |
     | thue-morse | 0.177 | KT-8, 0.181 |
     | periodic | 0.436 | KT-12, 0.447 |

   - Larger fixed orders get *worse* on everything without deep structure.
2. **Overfitting shows up as miscalibration, and inference removes it.**
   - On noise, pseudorandom and chaotic sources, KT-12's excess bits are almost all MCB: 0.12 bits/symbol on bernoulli, 0.11 on LCG and 0.09 on logistic.
   - CTW's MCB stays ≤ 0.008 on every stationary source. The Bayesian mixture is calibrated by construction within its class.
   - Calibration error here is a *symptom of committing to too much structure*.
3. **Computation creates information for bounded observers.**
   - The LCG source has a description of under 100 bits, yet every observer, CTW included, extracts **DSC = 0.000** bits and pays 1.00 bit/symbol.
   - A predictor able to run the generator would pay ≈ 0.
   - Randomness is observer-relative. This is the empirical anchor for epiplexity and V-information ([[bounded-observer-information]]).
4. **Calibration breaks at a shift, then recovers at a speed set by inference.** MCB, in bits/symbol, around the regime change:

   | Observer | Before | Just after | Late after |
   |---|---|---|---|
   | KT-3 | 0.005 | 0.44 | 0.11 |
   | CTW-12 | 0.006 | 0.13 | 0.024 |

   The structure-averaging observer recovers calibration about 4× faster. Neither is told a shift happened; that is the next obvious improvement (switching or forgetting priors).
5. **Confidence is a weak proxy for the value of escalating.** S1 is KT-2 and S2 is CTW-12. The table gives the fraction of symbols escalated to recover 99% of S2's savings:

   | Source | Confidence gate | Random | Perfect gate |
   |---|---|---|---|
   | markov-3 | 0.69 | 0.99 | 0.21 |
   | thue-morse | 0.66 | 0.99 | 0.49 |
   | periodic | 0.97 | 0.99 | 0.32 |
   | minkowski | 0.98 | 0.99 | 0.03 |
   | logistic | 1.0 | 0.99 | 0.08 |

   On noise and pseudorandom sources S2 saves < 0.01 bits/symbol, so the right policy is *never* escalate, even though S1 is maximally unsure there. The gate should predict S2's *gain*, not S1's doubt ([[value-of-computation]]).
6. **Competition: every predictor has an adversary.** Each observer, including a log-score market over {KT-0, KT-4, CTW-12}, pays ≈ 1.00 bit/symbol on its own diagonal sequence (the next bit is always the one it finds less likely).
   - Weak observers' diagonals are trivial for stronger ones. KT-0's diagonal costs CTW only 0.004 bits/symbol.
   - The diagonals of CTW and of the market resist every observer tested.
   - There is no computable universal predictor, but there is an arms race ([[competition-and-exchange]]).

## Limitations

- **The learning-cost (epiplexity-style) proxy is unreliable.** It is the area of the loss curve above its last-quarter mean. It goes negative on noise and shift sources, because the "asymptote" window isn't an asymptote there. It needs a held-out prequential definition before it is used for claims.
- **All observers here are count-based.** The neural and amortised observers arrive in E002.
