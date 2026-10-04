# E004: the GL(2) rung, rediscovering Hecke structure from Ramanujan's τ

**Status:** done, 2026-10-04. `python run.py` takes about 2 s. Outputs are `results_full.json` and `log_full.txt`.
**Vault:** [[hecke-multiplicativity]], [[langlands-ladder]], [[primes-are-the-information]].

## Setup

**Data.** τ(n) for n ≤ 4000 is computed independently from Δ = q ∏(1−qⁿ)²⁴, using exact integers and the pentagonal-number series. The script asserts τ(1..5) = 1, −24, 252, −1472, 4830.

**Discovery.** A program search over a small grammar of arithmetic relations, using **only n ≤ 2000**:

| Candidate relation | Result on n ≤ 2000 |
|---|---|
| complete multiplicativity τ(mn) = τ(m)τ(n) | holds for **59%** of pairs. *Rejected.* |
| coprime multiplicativity (gcd(m,n) = 1) | holds for **100%** of 3406 pairs. *Accepted.* |
| prime-power recursion τ(p^(k+1)) = τ(p)τ(p^k) − p^(w−1)τ(p^(k−1)), w ∈ 1..40 | **exactly one** consistent weight, **w = 12** (30 checks) |

The search rediscovers the Hecke relations and the weight of Δ = η²⁴. Complete multiplicativity, the GL(1) rule from E002, *fails* on this rung. Only coprime multiplicativity survives, plus a recursion at prime powers whose weight is learnable from data.

**Ledger.** The task is to predict the sign of τ(n) for n ∈ (2000, 4000]. Rule observers see past *values*; generic observers see only past sign bits. Each rule observer learns its confidence in its rule online, so a wrong rule is charged in bits.

| Observer | bits/symbol | DSC | MCB |
|---|---|---|---|
| KT-0 / CTW-16 (generic) | 1.000 | 0.000 | 0.000 |
| complete-multiplicative (wrong rule) | 0.925 | 0.076 | 0.001 |
| coprime-multiplicative | **0.128** | 0.872 | 0.000 |
| Hecke (coprime + recursion, w = 12) | **0.124** | 0.876 | 0.001 |

The two reference densities on the same range are 0.128 for prime powers and 0.1235 for primes:

- The coprime observer pays exactly the density of prime powers, since it cannot see across prime powers.
- The Hecke observer pays exactly the density of primes.
- On the GL(2) rung, as on GL(1), the primes carry the information. Hecke operators are the structure that makes this true.

**Side facts from the run:**
- **Sato–Tate.** 50.4% of τ(p) are positive. Prime-sign bits are fresh coins, so even a perfect observer pays about 1 bit per prime.
- **Lehmer.** τ(n) ≠ 0 for all n ≤ 4000.

## Limitations

- **The grammar is tiny and hand-written.** This is rediscovery within a 3-template program space, not open-ended discovery. The next step is a richer relation grammar searched by MDL.
- **Rule observers are handed factorisations,** i.e. arithmetic knowledge is assumed.
- **The next rung up is a_p(E) for elliptic curves.** Wiles' modularity means these are coefficients of weight-2 newforms. A further step is murmurations: rank as DSC.
