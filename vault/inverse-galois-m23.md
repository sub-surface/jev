---
status: established (news; primary source read at abstract level)
area: mathematics
---
# inverse galois m23

M23, the last sporadic group whose realisation as a Galois group over Q was open, is now realised. The construction is an explicit degree-23 polynomial over Q, obtained from a regular Galois extension of Q(t).
- The usual **rigidity** method (Belyi, Fried, Matzat, Thompson) needs a triple of conjugacy classes whose Nielsen class contains essentially one generating triple up to conjugacy. M23 has no rigid triple.
- The authors used a **non-rigid** triple and computed the Belyi maps numerically.

**Why it matters to us.** Rigidity is a uniqueness-from-local-data principle: local data (ramification classes) pins down a global object (the Galois cover). It mirrors two of our findings:
- E003: in the right hypothesis space, one example determines the code.
- E005: a local-to-global obstruction.

When rigidity fails, the paper falls back to *computation*. That is "computation creates information" in pure mathematics.

**Source:** Huang, Jackson, Lee, Poonen, Pries & Zhang, 'The Mathieu group M23 is a Galois group over Q', arXiv:2608.08538 (2026-08-09). Builds on numerical Belyi-map algorithms by Klug, Musty, Schiavone, Sijsling & Voight.

**Test / open:** Is there an analogue of the rigidity count (generating tuples up to conjugacy) that predicts when a learner can identify a structure from one example? A candidate is the number of hypothesis-space orbits consistent with one observation.

**Links:** [[golay-symmetry-testbed]] [[local-global-obstruction]] [[langlands-ladder]] [[ai-verified-mathematics]]
