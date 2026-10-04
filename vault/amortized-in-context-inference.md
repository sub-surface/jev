---
status: established
area: inference
---
# amortized in context inference

A network meta-trained on samples from a prior over generators performs approximate Bayesian inference in its forward pass. Examples are prior-fitted networks (TabPFN) and transformers trained on UTM outputs. This is 'compress System 2 into System 1' applied to *inference itself*: the training target is the prior, and the forward pass is the posterior predictive.

**Source:** Muller et al. 2022 (PFNs); Grau-Moya et al. 2024 [verify]; Xie et al. 2022 (ICL as implicit Bayesian inference).

**Links:** [[intelligence-target]] [[universal-prediction]]
