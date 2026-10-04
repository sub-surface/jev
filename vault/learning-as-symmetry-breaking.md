---
status: conjecture
area: mathematics
---
# learning as symmetry breaking

There are two symmetries, with opposite roles. (1) **The prior's symmetry**, where hypotheses are exchangeable. Learning *breaks* it, and the information gained is KL(posterior || prior), at most log(orbit size) when the evidence selects one element. In this sense 'learning is symmetry breaking that increases information' is right. (2) **The world's symmetry** (the data's automorphism group). Learning should *respect* it, and knowing it saves data (E002). The Noether quantities of gradient flow are a third thing: memory of the initialisation, which weight decay and noise erase. Parameter-space symmetry breaking is forgetting the init; prior symmetry breaking is learning the world.

**Source:** Owner's intuition (2026-10-04); Kunin et al. 2021; standard Bayesian information gain.

**Test / open:** Measure KL(posterior || prior) in bits against the ledger's DSC on E001/E002 sources. They should match in the limit (prequential identity), and the gap measures inefficiency.

**Links:** [[symmetry-conservation-in-learning]] [[golay-symmetry-testbed]] [[resolution-is-extracted-information]]
