# E003: a learner that can represent the invariant

**Status:** done, 2026-10-04. `python run.py` takes about 100 s on CPU and runs 5 seeds. It writes `results_full.json` and `log_full.txt`.
**Follows from:** E002 (symmetry earns bits; SGD could not use it). **Vault:** [[golay-symmetry-testbed]], [[jumps-are-posterior-concentration]].

## The learner

The hypotheses are *structures*, not weights. For each training word w, the learner forms span(G·w), the smallest code that is both invariant under G = PSL(2,23) and linear over GF(2) and contains w. It adds the trivial code F₂²⁴ to the candidate set. It then does **exact** Bayesian model selection over the candidates under BSC(ε).

The likelihoods are computed exactly for any linear code. For a small code, enumerate its codewords. For a large code, use Poisson summation over the dual:

P(x_O | C) = 2^−|O| · Σ_{u ∈ C⊥, supp u ⊆ O} (−1)^{u·x} (1−2ε)^{|u|}

The control is the same learner without symmetry: its candidates are span(training words) and F₂²⁴.

## Results

Bits per symbol, mean over 5 seeds. The Bayes ceiling (a predictor that knows the code) is 0.500 at ε = 0 and 0.69 at ε = 0.03.

| Training codewords | sym + lin, ε=0 | lin only, ε=0 | sym + lin, ε=.03 | lin only, ε=.03 |
|---|---|---|---|---|
| 1 | **0.500** | 1.005 | 0.94 ± 0.12 | 1.006 |
| 2 | **0.500** | 1.005 | 0.75 ± 0.12 | 1.012 |
| 4 | **0.500** | 1.004 | 0.82 ± 0.15 | 1.021 |
| 16 | **0.500** | 0.500 | **0.695** | 1.000 |

For comparison, E002's results at N = 16 and ε = 0:

| Observer | bits/symbol |
|---|---|
| MLP | 6.6 |
| MLP + orbit augmentation | 1.21 |
| CTW | 1.01 |

1. **One example suffices when the hypothesis space is the right shape.** With symmetry and linearity together, a single codeword pins down the whole Golay code. A single octad's orbit spans all 12 dimensions. The learner then predicts at exactly the information-theoretic optimum of 0.500 bits/symbol, which is 12 bits per 24-bit block.
2. **This is the "jump", made exact.** The posterior moves from "anything" to "Golay" in one step, without any gradient descent. Under noise the outcome is bimodal at small N:
   - If any training word is clean (≈ 48% chance each), the learner jumps to near-optimal.
   - Otherwise it stays at 1 bit.

   That is why the spread is large at N = 1–4 and collapses by N = 16, where the learner sits at the ceiling.
3. **Symmetry and linearity are complementary.** Linearity alone needs 16 clean words, about 12 independent ones. Under noise it never finds the code, because spans of noisy words are junk. The group compresses the search from "find 12 independent codewords" to "find one clean word".
4. **The general lesson.** The gap E002 found was purely about *representation*. When the learner's hypothesis class is the space of G-invariant structures, the data cost of learning this code drops from "SGD never finds it" to **one example**. That is the strongest version yet of the "intelligence = inference over the right structure space" claim.

## Limitations

- **The group was given, not discovered.** Discovering symmetries from data is the next level: search over permutations that preserve the empirical code.
- **ε is assumed known.**
- **The candidate set is the orbit-spans of the training words themselves.** A noisier world would need candidate denoising first.
