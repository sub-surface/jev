---
status: established
area: inference
---
# amortized in context inference

A network meta-trained on samples from a prior over generators performs approximate Bayesian inference in its forward pass. Examples are prior-fitted networks (TabPFN) and transformers trained on UTM outputs. This is 'compress System 2 into System 1' applied to *inference itself*: the training target is the prior, and the forward pass is the posterior predictive.

**Result (E007):** Transformers meta-trained on a prior over 5 generator families do structure inference in one pass. Thue-Morse 0.490 bits vs CTW 0.495; iid within 0.01 of exact Bayes; held-out Minkowski 0.636 vs 0.624. They learn *faster* early in context than CTW (prior-front-loaded) and lose to it asymptotically on Markov and periodic. **Calibration came free:** MCB <= 0.02 on every family, held-out included. Golay stays at about 1 bit (the parity barrier).

**Source:** Muller et al. 2022 (PFNs); Grau-Moya et al. 2024 [verify]; Xie et al. 2022 (ICL as implicit Bayesian inference).

**Links:** [[intelligence-target]] [[universal-prediction]]
