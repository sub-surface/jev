# ML bibliography — calibrated one-pass decisions

Compiled 2026-10-03 by a research agent that searched the web.

**How to read the tags**
- **Verified:** arXiv IDs were checked against the arXiv API (title, authors, date) unless tagged otherwise.
- **[S]:** seen in search results only.
- **[?]:** from memory, not checked.
- **(unvetted):** a low-visibility 2026 preprint.
- Findings are paraphrased from the abstract. Read the paper before you rely on any of them.

## ⚑ Closest prior art: read this first

**Li, Miao, Krishnan, Padman (2026). *JEV-as-a-Judge: Accept When Confident, Escalate When Unsure.* arXiv:2609.26550.**

- **What it does:** a decision-only judge (TypeSafe's proprietary JEV) gives label probabilities. Verdicts it is confident about are accepted; the rest go to a reasoning judge.
- **Results on RewardBench:**
  - The decision-only judge scores 92.2%, against 93.5% for the reasoning judge.
  - The cascade beats the reasoning judge by 0.9 points at 41% of the cost.
- **Where it fails:**
  - It collapses on the reasoning-heavy JudgeBench (reasoning 68.4%, coding 76.2%).
  - Temperature scaling fitted on one dataset transferred poorly to others.

This is the **external** hosted Jev that the old repo tried to reproduce. It is *not* this repo.

## 1. Distilling System 2 into System 1 / amortised search

**Core results**
- **Expert Iteration.** Anthony, Tian & Barber 2017, arXiv:1705.08439. Search is the expert and the network is the apprentice.
- **AlphaZero.** Silver et al. 2017, arXiv:1712.01815.
- **Amortized Planning with Large-Scale Transformers (chess).** Ruoss et al. 2024, arXiv:2402.04494 (NeurIPS 2024). ChessBench: 10M games and 15B positions annotated by Stockfish. Reaches 2895 Lichess blitz Elo with no search.
  - **This is the reference for our chess testbed.** Whether its value bins are calibrated has not been checked.
- **Other chess work.**
  - Monroe & Chalmers 2024, arXiv:2409.12272.
  - Jenner et al. 2024, arXiv:2406.00877. Shows learned look-ahead inside Lc0's policy net.
  - Miłosz et al. 2026, arXiv:2608.27757 (unvetted). Searchless imitation plateaus on tactics.
- **Stop Regressing.** Farebrother et al. 2024, arXiv:2403.03950. Cross-entropy over value bins beats regression. This supports ordinal-bin heads.
- **Distilling System 2 into System 1.** Yu, Xu, Weston, Kulikov 2024, arXiv:2407.06023. They evaluate accuracy only, not calibration. [?] Chain-of-thought-heavy math distils poorly.
- **Search traces as training data.**
  - Searchformer: Lehnert et al. 2024, arXiv:2402.14083.
  - Stream of Search: Gandhi et al. 2024, arXiv:2404.03683.
- **Implicit chain of thought.**
  - Deng et al. 2023, arXiv:2311.01460.
  - Deng et al. 2024, arXiv:2405.14838.
  - Dualformer: arXiv:2410.09918.
  - CODI: arXiv:2502.21074.
- **Self-generated training data.**
  - STaR: arXiv:2203.14465.
  - ReST-EM: arXiv:2312.06585.

**Skeptical results**
- **Transformers Struggle to Learn to Search.** Saparov et al. 2024, arXiv:2412.04703.
- **Latent chain of thought.** Zou et al. 2026, arXiv:2602.01148 (unvetted). 97% on ProsQA but 34% on GSM8K.
- **The Illusion of Certainty.** Zhang et al. 2026, arXiv:2604.16830. **On-policy distillation raises accuracy but makes models overconfident.**

**Why distillation helps**
- Menon et al. 2020, arXiv:2005.10419.
- Hu et al. 2026, arXiv:2609.17474 (unvetted).

## 2. Amortised Bayesian inference / prior-fitted networks

- **PFNs.** Müller et al. 2021, arXiv:2112.10510.
- **TabPFN.** arXiv:2207.01848. Also Hollmann et al. 2025 in *Nature* [S].
- **Position paper on prior-fitted prediction.** Müller et al. 2025, arXiv:2505.23947.
- **Later TabPFN versions.**
  - TabPFN-2.5: arXiv:2511.08667.
  - TabPFN-3: arXiv:2605.13986. It has a "Thinking" mode, i.e. PFNs are gaining their own System 2.
  - TabPFN-3.5: arXiv:2609.17895.
- **TabICL.** arXiv:2502.05564 and arXiv:2602.11139.
- **TabPFN on frozen embeddings.** Zhang et al. 2026, arXiv:2607.11007 (unvetted). NLL is 48–62% lower and ECE is 2–5× lower than the baselines. **Potentially the cheapest calibrated decision head.**
- **Uncertainty quantification when data is scarce.** Johnson et al. 2026, arXiv:2606.01427 (unvetted). Gaussian processes beat TabPFN here.
- **Causal PFNs.**
  - Do-PFN: arXiv:2506.06039.
  - CausalPFN: arXiv:2506.07918.
  - LLM-written priors: arXiv:2609.06941 (single author, unvetted).
- **Time series.**
  - ForecastPFN: arXiv:2311.01933.
  - Chronos: arXiv:2403.07815.
- **Other amortised inference.**
  - Neural Processes: arXiv:1807.01622.
  - Simulation-based inference: arXiv:1911.01429.
  - BayesFlow: arXiv:2306.16015.
  - Bayesian Teaching: arXiv:2503.17523.

## 3. Calibration training and LLM confidence

**Foundations**
- Temperature scaling: Guo et al. 2017, arXiv:1706.04599.
- Kadavath et al. 2022, arXiv:2207.05221. Introduces P(IK) ("probability I know").
- Lin, Hilton & Evans 2022, arXiv:2205.14334.
- Tian et al. 2023, arXiv:2305.14975.

**Post-training hurts calibration**
- GPT-4 report, arXiv:2303.08774. RLHF degrades calibration [S].
- Luo et al. 2025, arXiv:2505.16690. Recalibrate a post-trained model using the pretrained model.

**Calibration as a training target**
- **Kapoor et al. 2024**, arXiv:2406.08391. Fine-tuning on about 1,000 graded examples beats prompt-based uncertainty.
- **RLCR.** Damani et al. 2025, arXiv:2507.16806.
- **Rewarding Doubt.** arXiv:2503.02623.
- RL-for-calibration follow-ups (unvetted): arXiv:2505.14489, 2512.19920, 2604.12632, 2609.34857.

**Uncertainty from sampling, and cheap substitutes**
- **Semantic entropy.**
  - arXiv:2302.09664.
  - *Nature* 630 [S].
- **Semantic entropy probes.** arXiv:2406.15927. **This compresses multi-sample uncertainty into one forward pass.**
- **Linear probes.** No training of the model itself is needed.
  - arXiv:2304.13734.
  - arXiv:2212.03827.
  - arXiv:2310.06824.
  - Judge probes: arXiv:2512.22245, about 10× cheaper.

**Known failure modes**
- **Option-position bias in multiple-choice selectors.** Zheng et al. 2023, arXiv:2309.03882. This directly affects option-marker heads.
- **Calibration versus hallucination.**
  - arXiv:2311.14648.
  - arXiv:2509.04664.

## 4. Calibration for decisions (theory)

- **Multicalibration.** arXiv:1711.08513.
- **Outcome indistinguishability.** arXiv:2011.13426.
- **Omnipredictors.**
  - arXiv:2109.05389.
  - arXiv:2210.08649.
  - arXiv:2302.06726.
- **Decision calibration.** Zhao et al. 2021, arXiv:2107.05719. With only a few options, it can be tested directly.
- **Measuring calibration error.**
  - Kumar et al. 2019, arXiv:1909.10155.
  - Błasiok et al. 2022, arXiv:2211.16886. Gives a unified theory of distance from calibration.
- **Calibration measures tied to decision loss.**
  - U-Calibration: arXiv:2307.00168.
  - **Hu & Wu 2024, Calibration Decision Loss.** arXiv:2404.13503. ECE does *not* bound decision loss; CDL does.
  - Smooth calibration and decision making: arXiv:2504.15582.
  - Later work: arXiv:2504.15615, 2505.16141, 2502.12564, 2501.17205, 2511.13699, 2510.23471, 2605.17749.

## 5. Selective prediction, deferral, cascades, adaptive compute

- **Abstention and learning to defer.**
  - Selective classification: arXiv:1705.08500.
  - Learning to defer: arXiv:2006.01862, 2202.03673, 2310.14772.
  - Human–AI triage: arXiv:1903.12220, 2005.00582.
- **Early exit and adaptive depth.**
  - ACT: arXiv:1603.08983.
  - PonderNet: arXiv:2107.05407.
  - BranchyNet: arXiv:1709.01686.
  - CALM: arXiv:2207.07061.
- **Cascades and routing.**
  - LM Cascades: arXiv:2207.10342.
  - FrugalGPT: arXiv:2305.05176.
  - RouteLLM: arXiv:2406.18665.
  - Hybrid LLM: arXiv:2404.14618.
  - AutoMix: arXiv:2310.12963.
  - Dekoninck et al.: arXiv:2410.10347.
- **Adaptive test-time compute.**
  - Snell et al. 2024, arXiv:2408.03314.
  - Setlur et al., arXiv:2502.12118.
  - Surveys: arXiv:2503.16419, 2507.02076.
- **Voting baselines a one-pass head must beat.**
  - Self-Consistency: arXiv:2203.11171.
  - arXiv:2502.06233.
  - arXiv:2502.18581.
  - DeepConf: arXiv:2508.15260.
- **Conformal guarantees.**
  - Tutorial: arXiv:2107.07511.
  - Conformal language modeling: arXiv:2306.10193.
  - arXiv:2402.10978.
  - KnowNo: arXiv:2307.01928.

## 6. Forecasting, judges, verifiers

- **Forecasting benchmarks and systems.**
  - Autocast: arXiv:2206.15474.
  - Halawi et al. 2024, arXiv:2402.18563.
  - ForecastBench: arXiv:2409.19839.
  - arXiv:2507.04562.
- **Training forecasters on outcomes.**
  - Turtel et al. 2025, arXiv:2505.17989. Outcome-based RL for forecasting.
  - arXiv:2512.25070.
  - Turtel et al. 2026, arXiv:2608.28482. How proper scoring rules shape LLM forecasting.
  - arXiv:2607.25554. Distilling temporal search.
  - arXiv:2610.01955.
  - Hindcast: arXiv:2607.14051. Leakage-free evaluation.
- **Judges.**
  - MT-Bench: arXiv:2306.05685.
  - G-Eval: arXiv:2303.16634. Uses an expected score over ordinal bins.
  - **Trust or Escalate.** arXiv:2407.18370. A judge cascade with guarantees.
- **Verifiers and reward models.**
  - arXiv:2110.14168.
  - arXiv:2305.20050.
  - Math-Shepherd: arXiv:2312.08935.
  - arXiv:2501.07301.
  - Generative Verifiers: arXiv:2408.15240.
  - Self-Taught Evaluators: arXiv:2408.02666.
  - arXiv:2605.11954 (unvetted). Soft-label distillation reduced ECE by about 43%.

## 7. Continual learning and escalate-then-distill

- **The loop, already in print.**
  - **Online Cascade Learning.** Nie et al. 2024, arXiv:2402.04513. Small models learn from LLM labels on the inputs they deferred.
  - **From Deferral to Learning.** Wu et al. 2025, arXiv:2509.22984. Reports 48% fewer expensive calls.
- **Test-time training.**
  - arXiv:1909.13231.
  - arXiv:2411.07279.
  - arXiv:2407.04620.
  - SEAL: arXiv:2506.10943.
  - TTRL: arXiv:2504.16084.
- **Self-distillation.**
  - Shenfeld et al. 2026, *Self-Distillation Enables Continual Learning*, arXiv:2601.19897.
  - *Denser ≠ Better*, arXiv:2607.01763. Self-distillation can amplify forgetting.
- **Learning from self-generated or verified outcomes.**
  - DeepSeek-R1: arXiv:2501.12948.
  - Absolute Zero: arXiv:2505.03335.

## 8. Applications where one pass could replace heavy compute

- **Theorem proving.**
  - arXiv:2009.03393.
  - HTPS: arXiv:2205.11491.
  - LeanDojo: arXiv:2306.15626.
  - Magnushammer: arXiv:2303.04488.
- **Code.**
  - CodeT: arXiv:2207.10397.
  - LEVER: arXiv:2302.08468.
  - SWE-Gym: arXiv:2412.21139.
  - R2E-Gym: arXiv:2504.07164.
- **Reranking.**
  - RankGPT: arXiv:2304.09542.
  - RankZephyr: arXiv:2312.02724.
- **Agents (unvetted).**
  - arXiv:2602.16699.
  - arXiv:2601.07264. Tools make agents overconfident.
  - arXiv:2609.07395.
- **Weather.**
  - GraphCast: arXiv:2212.12794.
  - GenCast: arXiv:2312.15796.
  - NeuralGCM: arXiv:2311.07222.
  - FGN: arXiv:2506.10772.
- **Systems.**
  - TVM cost model: arXiv:1805.08166.
  - TPU performance model: arXiv:2008.01040.
  - MLGO: arXiv:2101.04808.
  - Neo: arXiv:1904.03711.
  - Bao: arXiv:2004.03814.
  - Learned branching: arXiv:1906.01629.
  - Bengio et al.: arXiv:1811.06128.

## The agent's view of where the gaps are

1. **Calibration is the neglected half of System-2 distillation.** Distillation papers report accuracy or Elo. The only direct evidence on calibration after distillation is negative (the overconfidence result in §1).
2. **Calibration saves compute only when it comes with resolution.** On hard reasoning tasks, a one-pass model cannot tell right from wrong answers well enough. It stays calibrated, but it has to escalate almost everything.
3. **Labels arrive only for the inputs you escalate.** That is a selective-labels problem, and it was not found addressed for LLM cascades. A random audit slice plus importance weighting is a testable fix.
4. **Calibration under distribution shift is unsolved.**
5. **PFNs need priors.** No accepted prior exists for text or judging tasks.
6. **Evaluation hygiene.** Use smooth or debiased ECE plus decision calibration.
