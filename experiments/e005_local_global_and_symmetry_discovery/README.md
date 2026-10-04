# E005: the local-to-global obstruction, and discovering the symmetry group from data

**Status:** done, 2026-10-04. `python run.py` takes about 40 s and writes `results_full.json` and `log_full.txt`.
**Vault:** [[local-global-obstruction]], [[golay-symmetry-testbed]], [[category-theory-fit]].

## A. The local-to-global profile

This is the measurable shadow of "the cohomology of the target". Enumerating all 4096 Golay codewords, the experiment computes exactly how much a *k-local* observer knows about one bit, I(X_j ; X_S) with |S| = k:

| k (bits seen) | 0–6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|
| mean I (bits) | **0.0000** | 0.003 | 0.003 | 0.023 | 0.100 | 0.283 |
| max I (bits) | **0.0000** | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 |

The prediction from the dual distance (8) is exact.
- **Up to 6 bits, nothing.** Any observer looking at 6 or fewer other bits extracts *nothing*: every set of 7 coordinates is exactly uniform.
- **7 bits on an octad, everything.** At k = 7 one bit becomes fully determined, but only when the 8 positions form an octad, i.e. the support of a dual codeword. The information exists *only globally*.

This is why CTW and MLPs failed in E002: their natural features are local. Pseudorandom sources are the extreme case. For bounded observers the obstruction has unbounded order, since no feasible window reveals anything (E001: DSC = 0 for the LCG).

**Interpretation.** It is not literally cohomology. It is the information-theoretic form of a local-to-global obstruction: locally everything is consistent with noise, and a global constraint carries the structure. Abramsky's sheaf-theoretic treatment of contextuality formalises the same pattern. The profile is a cheap, general diagnostic that can be computed for any source.

## B. Symmetry discovery: the learner finds M24 without being told

**Pipeline.** The learner gets codewords only, with no group supplied:
1. Take the GF(2) span of the codewords.
2. Extract the weight-8 blocks.
3. Find random permutations of the 24 coordinates that preserve the block system. This uses depth-first search with Steiner-system propagation: once 5 points of a block are mapped, the image block is determined.
4. Compute the order of the group they generate, by Schreier–Sims (sympy).

| Training codewords | Span dim | Weight-8 blocks | Discovered group order |
|---|---|---|---|
| 4 | 4 | 1 | n/a (no block system yet) |
| 8 | 8 | 42 | n/a |
| 12 | 11 | 407 | n/a |
| **16** | 12 | 759, a Steiner system S(5,8,24) | **244,823,040 = \|M24\|**, 5-transitive |
| 24 | 12 | 759 | 244,823,040 |

**Results.**
- From 16 codewords the learner recovers the **full Mathieu group M24**, about 40,000 times larger than the PSL(2,23) we handed it in E002/E003. The discovered group contains PSL(2,23), checked by membership.
- It takes 4 random automorphisms, found in about 20 s, to generate the group.

**Honest caveat.** Discovering the group needs essentially the whole code first: span dimension 12 at about 16 words. So *within one task*, symmetry discovery does not beat linearity on data efficiency. Its value is **reuse**. A group discovered once can then be applied to new data at one-example cost (E003), and that is where amortisation comes in (E007).
