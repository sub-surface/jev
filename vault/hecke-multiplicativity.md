---
status: established (+ rediscovered in E004)
area: mathematics
---
# hecke multiplicativity

The coefficients a(n) of a normalised Hecke eigenform are multiplicative: a(mn) = a(m) a(n) for coprime m, n. At prime powers they follow a(p^(k+1)) = a(p) a(p^k) - p^(w-1) a(p^(k-1)). Hecke operators are therefore what turns 'a sequence' into 'data determined by its values at primes'. This is the structure E002's multiplicative observer exploits, in its GL(1) form (completely multiplicative).

**Result (E004):** a program search over a 3-template grammar, run on n <= 2000, does two things. It rejects complete multiplicativity (holds for 59% of pairs) and accepts coprime multiplicativity (100% of 3406 pairs). It also finds a unique prime-power weight, w = 12. On the ledger, the Hecke observer pays 0.124 bits/symbol, which equals the density of primes; generic observers pay 1.000.

**Source:** Hecke 1937; Diamond & Shurman, ch. 5.

**Test / open:** Next rung: Ramanujan tau(n) (from Delta = eta^24). Does a Hecke-aware observer pay only for the primes? And, harder: can a learner given factorisation features *discover* the Hecke recursion from data?

**Links:** [[primes-are-the-information]] [[langlands-ladder]] [[murmurations]]
