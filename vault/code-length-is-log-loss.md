---
status: established
area: foundations
---
# code length is log loss

Cumulative log loss of a sequential predictor on bits *is* the length of the arithmetic code it induces: L(x_1..n) = sum_t -log2 p(x_t | x_<t). Prediction and compression are one act, so every quantity in this program can be stated in bits.

**Source:** Shannon 1948; Rissanen & Langdon 1979 (arithmetic coding); Deletang et al. 2024, 'Language Modeling Is Compression'.

**Links:** [[corp-decomposition]] [[universal-prediction]] [[library-of-babel-as-coordinates]]
