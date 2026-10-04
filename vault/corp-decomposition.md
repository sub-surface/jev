---
status: established
area: foundations
---
# corp decomposition

For any proper score, mean S(p) = MCB - DSC + UNC exactly, with MCB and DSC both >= 0. Here q is the isotonic (PAV) recalibration of p and r is the base rate. MCB = S(p) - S(q) is the calibration cost, DSC = S(r) - S(q) is discrimination, and UNC = S(r). Under the log score all three are in bits. Implemented and tested in `calib/metrics.py::corp_decomposition`.

**Source:** Dimitriadis, Gneiting & Jordan 2021 (PNAS, CORP); Arnold, Walz, Ziegel & Gneiting 2023 (arXiv:2311.14122, in literature/); Brocker 2009.

**Links:** [[calibration-is-cheap]] [[resolution-is-extracted-information]] [[mdl-snapped-recalibration]]
