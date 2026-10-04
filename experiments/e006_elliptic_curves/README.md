# E006: elliptic curves. CM curves are GL(1) in disguise; non-CM curves carry GL(2) information

**Status:** done, 2026-10-04. `python run.py` takes about 80 s; outputs `results_full.json` and `log_full.txt`.
**Vault:** [[langlands-ladder]], [[elliptic-curves-cm-vs-generic]].

## Setup

**Coefficients.** a_p(E) = −Σ_x ((f(x)/p)), computed by exact Legendre sums for every good prime 5 ≤ p < 40000, about 4200 primes per curve.
- Checked against curve 11a: a₅…a₁₉ = 1, −2, 4, −2, 0.
- The Hasse bound |a_p| ≤ 2√p is asserted for every prime.

**Bit streams over the primes.** For each curve, two streams:
- z_p = 1[a_p = 0], whether p is supersingular;
- s_p = 1[a_p > 0], the sign.

**Scoring.** Observers are scored on the second half of the primes.

**Rule search.** A residue-class modulus m is chosen by MDL on the first half.

| Curve | Stream | KT-0 | CTW-16 | Residue rule (m chosen by MDL) | GL(1) / Hecke-character observer |
|---|---|---|---|---|---|
| CM ℤ[i]: y² = x³ − x | z (a_p = 0) | 1.000 | 0.970 | **0.000** (m = 4) | — |
| CM ℤ[i] | s (sign) | 1.001 | 1.001 | 1.001 (m = 1: no residue helps) | **0.002** (features a, b mod 4 of p = a² + b²) |
| CM ℤ[ω]: y² = x³ + 1 | z | 1.000 | 0.977 | **0.000** (m = 3) | not built (Eisenstein analogue) |
| CM ℤ[ω] | s | 1.000 | 1.000 | 1.000 | — |
| non-CM y² = x³ − x + 1 | z | 0.042 | 0.042 | 0.042 | — |
| non-CM y² = x³ − x + 1 | s | 1.001 | 1.001 | 1.001 | — |
| non-CM 11a3 | z | 0.042 | 0.042 | **0.034** (m = 5) | — |
| non-CM 11a3 | s | 1.000 | 1.000 | 1.000 | — |

## What it shows

1. **CM curves reduce to GL(1).**
   - For y² = x³ − x, the supersingular primes are exactly p ≡ 3 (mod 4), which MDL found from data.
   - The *sign* of a_p is invisible to every residue class: 1 bit per symbol.
   - Yet it is fully determined by the Gaussian-integer data of p = a² + b², at 0.002 bits per symbol. CM curves come from Hecke characters of an imaginary quadratic field (automorphic induction), and the ledger shows it directly.
2. **Non-CM curves carry genuinely GL(2) information.**
   - Their signs cost 1.000 bit per symbol for every observer here (Sato–Tate).
   - No GL(1)-type feature helps, which is the defining difference between the two rungs.
3. **The learner found structure we didn't plant.**
   - For 11a3 the MDL search picked *m = 5*.
   - 11a has a rational 5-torsion point, so a_p ≡ p + 1 (mod 5), and a_p = 0 forces p ≡ 4 (mod 5).
   - That is a reducible mod-5 Galois representation, surfaced by a residue search from bits alone.

## Limitations and next

- **Observers are handed features.** They get p's residues, or p = a² + b². Discovering *which* field to look in is the next rung.
- **No ℤ[ω] Hecke-character observer was built.**
- **No murmurations yet.** They need curve families with known ranks, from LMFDB, plus averaging over thousands of curves.
