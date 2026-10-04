---
status: established
area: foundations
---
# ctw inference over structure

Context Tree Weighting computes the exact Bayesian mixture over all binary tree sources of depth <= D, at O(D) cost per symbol. It pays only the model's description cost plus its parameter cost over the best tree in hindsight. This is inference over *structure*, not a bigger lookup table, and it is the natural System-2 teacher for the ground floor.

**Source:** Willems, Shtarkov & Tjalkens 1995; E001 ledger (CTW matches or beats the best fixed order on every learnable source without being told the order).

**Links:** [[jumps-are-posterior-concentration]] [[universal-prediction]]
