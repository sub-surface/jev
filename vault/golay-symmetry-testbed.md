---
status: established (E002, this setting)
area: mathematics
---
# golay symmetry testbed

The extended binary Golay code [24, 12, 8] has 4096 codewords. Its automorphism group is the sporadic Mathieu group M24, the first step of the codes -> Leech -> Conway -> Monster chain, and it lives on the ground floor: it is made of bits. On a stream of noisy codewords, a code-aware Bayes predictor extracts far more DSC than CTW, because the last 12 bits of each codeword are determined by the first 12.

This is the first place where a *sporadic symmetry can earn bits*. If it does not pay here, the Monster is unlikely to pay anywhere in sequence prediction.

**Result (E002):** From only 4 codewords, the PSL(2,23) orbit (order 6072, transitive on the 759 octads) extracts about 0.2 bits/symbol of DSC. That beats linearity (0.00) and every generic observer (0.00). An SGD-trained MLP, *even with orbit augmentation*, extracts almost nothing, because the check bits are parities. Symmetry earns bits only through a learner that can represent the invariant.

**Result (E003):** Bayesian model selection over G-invariant linear codes (orbit-spans) reaches the exact optimum of 0.500 bits/symbol from **one** codeword. Linearity alone needs about 16, and SGD never gets there. Under noise (ε = 0.03) it reaches the ceiling of 0.69 by N = 16. The gap was entirely about representation.

**Source:** Conway & Sloane, SPLAG, ch. 10; the E001 ledger.

**Test / open:** Ledger CTW, a generic MLP, an M24-equivariant learner, and the Bayes decoder against the number of training codewords. Prediction: the equivariant learner approaches the Bayes predictor's DSC with orders of magnitude less data. Then compare Leech vs E8 vs scalar quantisation of a sequence model's latents, in bits per parameter.

**Links:** [[codes-lattices-moonshine]] [[nn-field-theory]] [[symmetry-conservation-in-learning]] [[intelligence-target]]
