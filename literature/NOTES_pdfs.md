# Read-outs of the PDFs in this folder

Written 2026-10-03 from the PDFs themselves. The older `.md` notes in this folder were not used, because they are AI-generated and unverified. Anything marked **[inference]** is the reader's own conclusion, not the paper's.

---

## 1. Calibration-Aware RL for decision-making LLMs — `2026.findings-acl.610.pdf`

**Citation.** Yaldiz, Spiliopoulou, Qi, Varia, Doss, Pappas. *Balancing Classification and Calibration Performance in Decision-Making LLMs via Calibration Aware Reinforcement Learning.* Findings of ACL 2026.

**What it shows.**
- RLVR (GRPO) raises accuracy but leaves models extremely overconfident.
  - More than 94% of base-model rollouts put p > 0.99 on the decision token.
  - Swapping in reasoning that argues for the opposite label flips 92–100% of decisions, yet confidence stays near 1. The decision token behaves like an extraction step.

**Method.**
- Loss: `L = L_GRPO + λ·L_CE(y_d)`. The advantage is zeroed at the decision token.
- The CE target is the gold one-hot if the model's answer was correct, and uniform if it was wrong.
- λ = 0.001.
- Setup: Qwen3 1.7B, 4B and 8B with LoRA, trained on 500 examples.

**Results (Qwen3-1.7B).**

| Setting | Method | Acc / ECE (%) |
|---|---|---|
| CSQA | GRPO | 73.67 / 24.39 |
| CSQA | Ours | 73.73 / 15.97 |
| CSQA | SFT | 68.55 / 7.36 |
| OBQA (OOD) | Ours | 88.33 / 5.27 |

- Post-hoc isotonic or Platt scaling on top gives the best ECE.

**Caveats.**
- λ-sensitive.
- Checkpoints were selected on the test set (Appendix A).
- **[inference]** The CE target is not a proper target, so it biases confidence upward.

**Implication.** This is the closest analogue to a one-pass option head. A distribution-level proper loss on the head matters more than trajectory reward.

---

## 2. TruthRL — `2509.25760v2.pdf`

**Citation.** Wei et al. (UVA, Meta). ICML 2026.

**Method.**
- GRPO with a ternary reward: +1 correct, 0 abstain, −1 wrong.
- An LLM verifier (Llama3.3-70B) judges correctness.

**Results.** On CRAG with Llama3.1-8B and retrieval:
- Hallucination falls from 43.5% to 19.4%.
- A binary reward gives 0.2% abstention.

**Caveats.**
- It needs a strong verifier. A rule-based verifier collapsed to always abstaining.
- No calibration metric is reported.

**Implication.** Abstention / "defer to System 2" can be an explicit option. **[inference]** A +1/0/−1 reward implies an abstain threshold at p = 0.5.

---

## 3. UAMDP — `2510.08226v2.pdf`

**Citation.** Koren, Peretz, Dinh, Yu. arXiv preprint, December 2025.

**Method.**
- Bayes-adaptive MDP: a GP or TFT forecaster plus a belief update.
- Thompson sampling, then MCTS with CVaR.

**Results.**
- Domains: S&P 500 and H&M demand.
- Claims CRPS −29% / −34%, Sharpe 1.54 → 1.74, and a shallower drawdown.

**Reader's cautions.**
- "Consistently outperforms" is overstated.
- Its 95%-interval coverage on S&P is 90.2%, the worst in its own table.
- Two ablation rows are identical.
- The two reliability diagrams look identical.

**Implication.** Least relevant here. It is classical System-2 machinery, exactly the kind of compute a calibrated System 1 would amortise. Calibration degrades at long horizons.

---

## 4. RLCR: Beyond Binary Rewards — `ICLR-2026-beyond-binary-rewards-...pdf`

**Citation.** Damani, Puri, Slocum, Shenfeld, Choshen, Kim, Andreas (MIT). ICLR 2026. arXiv:2507.16806.

**Method.**
- The model reasons, then emits an answer *y* and a verbalised confidence *q*.
- Reward: `R = 1[y≡y*] − (q − 1[y≡y*])²`.

**Theory.**
- **Theorem 1.** If success is Bernoulli(p_y), E[R] is maximised at q = p_y. Among calibrated predictions, it is maximised by the answer with the highest p_y.
- **General form (App. A).** `R = λc − S(q,c)`.
  - Lemma 1: the calibration incentive holds iff S is proper.
  - Lemma 2: the correctness incentive holds iff `S(p,1) − S(p,0) ≤ λ` for all p.
  - Corollary 1: bounded proper rules work. The unbounded log score fails for any finite λ.
  - For Brier, `S(p,1) − S(p,0) = 1−2p`, so λ = 1.

**Results.**

| Setting | Method | Accuracy | ECE |
|---|---|---|---|
| HotpotQA | RLVR | 63.0% | 0.37 |
| HotpotQA | RLCR | 62.1% | 0.03 |
| OOD average | base | — | 0.40 |
| OOD average | RLVR | — | 0.46 |
| OOD average | RLCR | — | 0.21 |

- Confidence-weighted voting beats plain majority voting.

**Limitations.**
- OOD ECE is still 0.21.
- Models can give high confidence to contradictory answers.
- One answer and one scalar only.

**Note for us.** This theorem concerns a *sampled* answer plus a scalar confidence, where correctness is not differentiable. A one-pass head that outputs a full distribution over options does not need it. Plain proper-score training of that distribution already targets calibration and accuracy.

---

## 5. Verified Uncertainty Calibration — `NeurIPS-2019-verified-uncertainty-calibration-Paper.pdf`

**Citation.** Kumar, Liang, Ma (Stanford). NeurIPS 2019. arXiv:1909.10155.

**Main points.**
- **Binning underestimates calibration error.** Prop. 3.3: `CE(f_B) ≤ CE(f)`. Example 3.2 constructs a predictor with binned CE = 0 but true CE ≥ 0.49.
- **Scaling-binning calibrator.** Fit a scaling function, bin it with uniform-mass bins, and output bin means. This needs O(B + 1/ε²) samples, versus O(B/ε²) for histogram binning.
- **Plug-in vs debiased estimators.**
  - Plug-in: `Ê²_pl = Σ p̂_s (s − ŷ_s)²`, bias ≈ B/n.
  - Debiased: `Ê²_db = Σ p̂_s[(s − ŷ_s)² − ŷ_s(1−ŷ_s)/(p̂_s n − 1)]`. It needs about √B/E² samples instead of B/E².
  - **Implemented in `calib/metrics.py::ece_l2_debiased`.**

**Implication.** The 10–15-bin plug-in ECE on a few hundred examples, as used throughout the old repo, is noisy and biased. Use debiased or smooth estimators with confidence intervals.

---

## 6. Calibrated Model-Based Deep RL — `malik19a.pdf`

**Citation.** Malik, Kuleshov, Song, Nemer, Seymour, Ermon. ICML 2019.

**Method.** Recalibrate a learned dynamics model on held-out data: `T̂ ← R∘T̂`.

**Theory.** Theorem 1: in a discrete MDP, a policy's value under a *calibrated* model equals its value under the true dynamics.

**Results.**
- Gains in contextual bandits on UCI data, e.g. Census 207.6 → 603.7.
- Inventory control improves.
- HalfCheetah reaches the same result with 50% fewer samples.

**Caveat.** Calibrated-but-diffuse models can still rank actions well, and sharpness is still needed.

**Implication.** Calibration is necessary for expected-value decisions but not sufficient. A cheap recalibration layer on a frozen model is a strong, general baseline.

---

## Cross-cutting

- **Only one formal theorem.** RLCR's Theorem 1 and its lemmas are the only theorem in the set.
- **OOD calibration is unsolved.** No paper here shows reliable out-of-distribution calibration in absolute terms. Kumar et al. list calibration under shift as an open problem.
- **The ECE metric itself is weak.** Papers 1 and 4 report plug-in ECE, which paper 5 shows is biased.
