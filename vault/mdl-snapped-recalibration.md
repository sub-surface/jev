---
status: speculative
area: program
---
# mdl snapped recalibration

A testable version of the `mod.py` intuition. Isotonic recalibration yields a few piecewise-constant levels, and each level must itself be described. Choose the levels from the Stern-Brocot tree to minimise MCB bits plus level-description bits (Stern-Brocot depth). Low-denominator probabilities are cheap to describe, so the recalibrator becomes an MDL object. It earns its place only if it beats plain PAV on held-out code length.

**Source:** Owner's `mod.py` sketch (illustrative, simulated data); [[mediants-are-bayesian-updating]].

**Test / open:** Compare held-out log loss of MDL-snapped recalibration, plain PAV and temperature scaling at small n.

**Links:** [[corp-decomposition]] [[mediants-are-bayesian-updating]]
