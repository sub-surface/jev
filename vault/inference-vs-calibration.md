---
status: conjecture
area: inference
---
# inference vs calibration

Calibration is about honesty; inference is about structure. A model has *inferred* structure when three things hold. (a) Its DSC approaches the best achievable for the source class. (b) Its calibration survives interventions or shifts that preserve the mechanism. (c) It gets there with a small learning cost. E000 showed (a) in part, and (b) failing: the model was calibrated in-distribution and broke under shift.

**Source:** Owner's critique, 2026-10-03; E000 Q4; Wald et al. 2021 on multi-domain calibration implying invariance [verify].

**Test / open:** Calibration preserved under mechanism-preserving shift should predict out-of-distribution accuracy better than in-distribution calibration does (E002).

**Links:** [[resolution-is-extracted-information]] [[intelligence-target]]
