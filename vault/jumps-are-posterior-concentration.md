---
status: conjecture
area: inference
---
# jumps are posterior concentration

'Jumping through latent space to a new configuration' has a precise, non-mystical form: Bayesian model selection. When a mixture's posterior concentrates on a hypothesis, the predictive distribution switches abruptly, with no local gradient walk. CTW's loss drops in steps as weight moves to deeper trees, while a high-order counter descends slowly. Amortised in-context learners should reproduce these jumps.

**Source:** Standard Bayesian model averaging; E001 fig_jumps.

**Test / open:** Do meta-trained sequence models show abrupt in-context loss drops aligned with the switch points of the Bayes-optimal posterior (E002)?

**Links:** [[predictive-coding]] [[amortized-in-context-inference]] [[ctw-inference-over-structure]]
