# E007: amortised inference. Can one forward pass do structure inference?

**Status:** done, 2026-10-04. This was the first cloud run.

**How to run:**
- `python sweep.py --local 120` runs a smoke test on the RTX 2060.
- `modal run sweep.py` runs the full sweep. **This spends money.**
- `python analyze.py results_modal.json` produces the analysis.

**Artifacts:**
- `probs_modal.npz`: float16 predictions.
- `meta_modal.json`.
- `analysis_results_modal.json`.
- `references.json`: CTW-12 and Bayes baselines on the identical sequences.
- `modal_log.txt`.

**Vault:** [[amortized-in-context-inference]], [[jumps-are-posterior-concentration]].

## Setup

**Models.** Causal transformers with bit tokens and a BOS token, sequence length 512.

**Training.** Each model is meta-trained on a *prior over generators*. Every sequence comes from a fresh random instance, and the model never sees which family:
- iid Bernoulli(p ~ U)
- variable-order Markov (order k ≤ 5, Beta(½,½) transitions)
- noisy periodic (period 2–24)
- noisy Thue–Morse
- Golay codeword streams (BSC with ε ∈ {0, 0.03})

**Held-out families:** LCG (pseudorandom), Minkowski ?(u), the logistic map, and a regime switch.

**Evaluation:** 64 fresh sequences per family, ledgered against CTW-12 and exact Bayes on the identical sequences.

**Compute** (napkin math in the commit message and `../COMPUTE_LOG.md`):
- One H100 container ran four sizes, each with a *fixed 220 s wall-time budget*, so the comparison is compute-matched.
- Training data came from a 122,880-sequence pool resident on the GPU. Generating it on the CPU per step would have idled the GPU about 75% of the time.
- Wall time was 901 s, including the image build. Cost was **about $0.95** against a $1.43 hard cap.

## Results: bits/symbol

| Model | Params | Tokens | iid | markov | periodic | thue | golay | lcg* | minkowski* | logistic* | regime* |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d64-L4 | 0.2M | 1467M | 0.717 | 0.720 | 0.859 | 0.821 | 1.009 | 1.008 | 0.660 | 0.999 | 0.754 |
| d128-L4 | 0.9M | 931M | 0.717 | 0.745 | 0.869 | 0.705 | 1.009 | 1.009 | 0.699 | 1.004 | 0.811 |
| **d256-L6** | 4.9M | 301M | 0.720 | **0.600** | **0.591** | **0.490** | 1.014 | 1.013 | 0.636 | 0.990 | **0.702** |
| d384-L8 | 14.4M | 137M | 0.721 | 0.613 | 0.634 | 0.518 | 1.017 | 1.017 | 0.634 | 0.996 | 0.728 |
| CTW-12 | n/a | n/a | 0.711 | 0.538 | 0.486 | 0.495 | 1.010 | 1.010 | 0.624 | 0.972 | 0.621 |
| exact Bayes | | | 0.709 | | | | 0.604 | | | | |

\* marks a family held out from training.

## Findings

1. **One forward pass does real structure inference.**
   - The d256 model reaches 0.490 bits on Thue–Morse, *matching* CTW-12 (0.495).
   - It is close on held-out Minkowski: 0.636 vs 0.624.
   - It is within 0.01 bits of exact Bayes on iid sources.
   - It trails CTW on Markov (+0.06), periodic (+0.11) and the regime switch (+0.08).
2. **Amortisation buys speed and costs the asymptote.** On Thue–Morse the transformer reaches low loss within about 40 positions, versus about 100 for CTW (Fig. 9B). The learned prior identifies the family fast. The exact Bayesian mixture overtakes it later. Prior-fitted inference is front-loaded.
3. **Calibration came free, everywhere.**
   - MCB ≤ 0.02 bits/symbol on every family, including held-out ones.
   - The pseudorandom LCG is reported honestly at ≈1 bit/symbol.
   - Compare E000, where an MLP trained on one task was badly overconfident.
   - Training on a *prior* makes the Bayes-optimal target calibrated by construction. This is the strongest practical argument for the PFN route to calibrated decision models.
4. **The parity barrier holds at transformer scale.** Golay stays at about 1.01 bits for every model, against a Bayes ceiling of 0.604. With about 3e8 tokens, SGD does not find the 8-wise constraints (E005 shows they are invisible below 7 bits). Symmetry and structure search (E003) remain qualitatively better here.
5. **There is a compute-optimal size.** At a fixed 220 s, d256 beats d384, which saw only 137M tokens. This is the usual Chinchilla trade-off, here on bits.

## Limitations

- One seed per size.
- 64 eval sequences per family, so differences under about 0.01 are noise.
- Fixed wall time is not fixed FLOPs: small models are launch-bound and get lower utilisation.
- No hyperparameter tuning.
- Next: a seed sweep, longer budgets for d256/d384, and a Golay-specific probe (does any amount of data crack it?).
