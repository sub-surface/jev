# Research program: calibrated decisions as compressed deliberation

*v1, 2026-10-04: the ground floor (binary sequences), the bits ledger and the intelligence target were added. v0 is kept below.*

## 0. The ground floor: everything in bits

**Prediction is compression.** On binary sequences, a predictor's cumulative log loss *is* the length of the arithmetic code it induces. The whole program can therefore be stated in one currency. Binary sequences are the universal testbed: any data, any decision, any program output can be written as bits. That makes them the right ground floor for a theory meant to be general ("it from bit").

**The ledger.** For any predictor, exactly:

> **bits/symbol = MCB − DSC + UNC** (the CORP decomposition, Arnold et al. 2023)

| Term | Meaning | What it shows |
|---|---|---|
| **UNC** | the base-rate entropy | what you pay knowing nothing |
| **DSC** | the bits *extracted*: I(Y; recalibrated forecast) | knowledge |
| **MCB** | the bits wasted on dishonest probabilities | removable by a monotone map |

**Calibration, inference and intelligence, separated.**
- **Calibration** (MCB → 0) is cheap. It is honesty, not knowledge.
- **Inference** is about which structure produced the DSC. It is shown by DSC near the source's achievable limit, by calibration that survives mechanism-preserving shift, and by a small learning cost.
- **Intelligence** is inference made efficient: bits extracted per unit of data, compute and prior description, across a family of sources, including ones never seen. See [[intelligence-target]].

**Information is observer-relative.** Shannon information can only be lost by processing (the data-processing inequality). But to a *bounded* observer, computation creates usable information. A pseudorandom stream of under 100 bits of description costs every bounded predictor 1 bit/symbol. Epiplexity, V-information and pseudo-entropy formalise this. Calibration relative to a class of observers (outcome indistinguishability) is the matching notion of honesty. **E001 measured each of these on 8 sources.** Its headline: inference over structure (CTW) beats any fixed memory, and is calibrated by construction. Overfitting appears as miscalibration in bits, and the confidence gate is a weak proxy for the value of escalating.

**The staged plan.** Each stage gets one experiment that can fail.

| Stage | Question | Experiment |
|---|---|---|
| **1. Ground floor** ✅ | Can every concept be measured in bits on binary sources: calibration, resolution, bounded information, shift, escalation, adversaries? | E001 (done) |
| **2a. Symmetry and arithmetic** ✅ | Does known structure (a group, multiplicativity) earn bits? Can learners use it? Can they rediscover it? | E002–E004 (done). The structure earns bits. Gradient descent doesn't find it. Inference over structure spaces gets the optimum from one example and rediscovers Hecke's weight 12. **Next on the ladder:** discover the symmetry itself (Q34), a_p(E) and murmurations, and amortise the structure search into a network (stage 2). |
| **2. Amortised inference** | Meta-train a small sequence model on a *prior over generating programs* (finite-state machines, tree sources, arithmetic and number-theoretic generators, compositions). In one forward pass, does it reach CTW-level code length on held-out programs? Does it show the posterior "jumps"? Does it stay calibrated on held-out *families* and under shift? Does it report ≈1 bit, honestly, on pseudorandom sources? This is the general training target: **the prior is the curriculum, and the forward pass is the posterior.** | E002 |
| **3. Competition and exchange** | Generators and predictors in self-play, where each side makes sequences the other cannot yet compress, plus log-score markets between predictors. Does the curriculum beat a fixed prior on held-out code length? | E003 |
| **4. Deliberation and continual learning** | A gate trained to predict *S2's gain* (not S1's doubt), with an ε-audit against selective labels, distilling back. Does the loop now beat random labelling, unlike E000? | E004 |
| **5. Leave the floor** | Port the stage-2/4 machinery to chess (search as S2) and to language decisions, using the same ledger. | E005+ |

**Kept with care, not discarded.** These are mathematical bridges, each in its own atomic note with a status:
- mediants as Bayesian updating;
- Minkowski ? between the Stern–Brocot and binary trees;
- surreal sign expansions;
- the chain codes → lattices → moonshine;
- Noether in gradient flow;
- loss-landscape orbifolds;
- field theories of networks;
- category-theoretic fits.

They enter experiments only when they make a prediction a baseline doesn't. The concept graph is in `vault/MAP.md`, and every open question raised so far is tracked in `vault/QUESTIONS.md`.

---

*v0 — 2026-10-03. This is a living document. Change it when evidence changes it.*

## 1. The thesis, stated so it can be wrong

A great deal of expensive computation exists to produce one small decision:
- chain-of-thought;
- search;
- best-of-N and voting;
- LLM-as-judge;
- simulation;
- running the test suite;
- asking a human.

The decision might be which option, yes or no, or a grade on a scale. We claim three things.

1. **Amortisation.** For many such decisions, one forward pass of a model trained on the slow procedure's outputs gets most of the way there. "Many" and "most" are to be measured, not assumed.
2. **Calibration buys the remainder.** If the one-pass model is *calibrated*, its confidence tells us where the slow procedure is worth running. Calibrated means that, of the answers it gives 80% to, 80% are right. On those uncertain cases the slow procedure runs, and the rest are answered instantly. The deliverable metric is **slow-path compute saved at matched decision quality**, not accuracy and not ECE.
3. **The loop compounds.** Cases that were escalated come back with labels and are distilled into the fast model. The fast model then escalates less over time. This is a narrow, testable form of continual learning, and it has well-known ways to fail (§5).

**The headline frame.** What "calibrated decision models" usually means is a calibrated classifier on a labelled dataset. That is solved well enough, and not interesting. The object studied here is different: **a calibrated predictor of what deliberation would conclude, indexed by how much deliberation.**

Hacking (1967) and Gaifman (2004) argued that probabilities can be relative to the reasoner's computational resources. Logical Induction (Garrabrant et al., 2016) is the idealised version: calibrated credences about deductive facts before the proofs are found. We want the practical version, which makes three things possible.
- **Labels for free.** The slow procedure is a label generator, so the bottleneck shifts from "find a clean dataset" to "find a slow procedure you trust".
- **A direct escalation signal.** "How likely is deliberation to *change my answer*?" is not the same question as "how likely am I to be right?"
- **Honest accounting of two calibrations.** Being calibrated to the *teacher* is not the same as being calibrated to the *world*. Both are measurable when ground truth exists at some fidelity.

## 2. What is already known (so we do not rediscover it)

Full bibliographies are in `literature/`. The load-bearing facts:

- **Distillation works for accuracy, and calibration is the neglected half.**
  - Accuracy-side results: Expert Iteration (Anthony 2017), AlphaZero, searchless grandmaster chess (Ruoss 2024), "Distilling System 2 into System 1" (Yu 2024), and implicit chain-of-thought (Deng 2024). All report accuracy or Elo.
  - The only direct evidence on calibration *after* distillation is negative: on-policy distillation makes models overconfident (arXiv:2604.16830).
- **Proper scoring rules are the right loss.** They are not magic: every strictly proper rule has the same population optimum. Differences between log, Brier and spherical come only from finite-sample and optimisation effects. RLCR's theorem (Damani 2026) matters only when a *sampled* answer comes with a separately reported confidence. A head that outputs a full distribution does not need RL at all.
- **Calibration is cheap; resolution is the product.**
  - A base-rate predictor is perfectly calibrated (Murphy decomposition).
  - Foster and Vohra show calibration can be achieved on any sequence.
  - ECE does not bound decision loss. Calibration decision loss (CDL) does (Hu & Wu 2024).
  - So every claim must report resolution or sharpness, decision loss, and the base-rate null.
- **Kahneman & Klein (2009).** Intuition is valid only in high-validity environments with rapid, unequivocal feedback, and subjective confidence does *not* signal when it is invalid. That is the map for where amortisation will work and where it will fail quietly (under distribution shift).
- **Rational metareasoning.** Russell & Wefald, and Lieder & Griffiths: the right gate is *value of computation*, not confidence. Deliberation cannot reduce irreducible uncertainty, so a coin-flip question should not be escalated however unsure the model is.
- **Escalate-then-distill already exists in pieces.**
  - Online cascade learning (Nie 2024).
  - "From Deferral to Learning" (Wu 2025, 48% fewer expensive calls).
  - "JEV-as-a-Judge" (2026). TypeSafe's hosted Jev as a confident-accept / escalate judge. It works on RewardBench and collapses on reasoning-heavy JudgeBench.

  Nobody we found treats the **selective-labels bias** of this loop, or measures calibration-to-world against calibration-to-teacher across compute levels.

## 3. Principles (falsifiable, adopted from the philosophy review)

| | Principle | Prediction | Falsified if |
|---|---|---|---|
| P1 | **Validity ceiling** | The student's resolution against ground truth is bounded by the teacher-ensemble's resolution. On zero-validity families, the student collapses to the base rate. | The student beats the ensemble on clean ground truth. |
| P2 | **Confidence can't see its own invalidity** | Under shift, ECE rises while mean confidence falls much less than accuracy. A separate familiarity / disagreement signal predicts errors beyond confidence. | Confidence alone matches the extra signal out of distribution. |
| P3 | **Gate on value of computation** | A learned gain predictor E[loss_S1 − loss_S2 \| x] beats a confidence threshold at matched escalation rate. | The two are within a pre-registered margin. |
| P4 | **The loop is a selective-labels system** | Without a random audit slice, calibration measured on audit data drifts across distillation rounds. A small ε of random escalation plus inverse-propensity weighting fixes it. | Audit-slice ECE stays flat without audits. |
| P5 | **Compression shows in resolution and decision loss, not ECE** | Equal-ECE models differ in decision regret. Worst-subgroup calibration is worse than global calibration. | ECE ranks the models the same way regret does. |

## 4. Where this applies (and what the "slow path" and labels are)

The test for a good domain: **a slow procedure you trust, cheap enough to run thousands of times to create labels, and expensive enough that skipping it matters.**

| Domain | Slow path today | Label source | One-pass decision | Laptop-feasible? |
|---|---|---|---|---|
| Games (chess) | alpha-beta / MCTS search | Stockfish at depth d, game results | move choice; win/draw/loss bins | **yes** (Stockfish binary in `archive/legacy-eret/stockfish`) |
| Self-consistency | sample N chains, vote | the votes themselves (free); answer keys | predict the *vote histogram* in one pass, calibrated to the answer key | small models yes |
| LLM-as-judge / reward models | long-CoT judge, pairwise sampling | strong judge, human preferences | choice / ordinal score | partly |
| Code agents | run tests, retry | test outcomes | "will this patch pass?" to rerank before running | needs SWE-Gym-scale data |
| Theorem proving | tactic search | Lean says yes or no | tactic or premise success probability | partly |
| Agents / tools | ReAct loops | task success | which tool, when to stop | later |
| Retrieval | LLM reranking | LLM reranker / clicks | relevance bins | yes |
| Forecasting | research and deliberation | **the world resolves it** | binary / ordinal credence | yes (continual, leakage is the enemy) |
| Systems | autotuning, query optimisers, MIP branching | run it and time it | which config / branch | yes |
| Scientific surrogates | simulators | simulator outputs | regime / outcome bins | domain-dependent |
| Triage / moderation | human review | human decisions | escalate-or-not | needs data |

Where it should *fail* (P1/P2, and worth showing):
- **Long-chain arithmetic and novel multi-step reasoning.** The answer is not readable from the surface, and Saparov 2024 shows transformers struggle to learn search.
- **Rare facts.** Kalai & Vempala: a calibrated model must hallucinate them.
- **Wicked feedback.** The model's own decisions shape its labels.

## 5. The loop, and how it breaks

```
query → S1 (answer, p, p_change) ── p_change low ──→ act
                     │
                     └─ p_change high (or ε random audit) → S2 → act + label → replay buffer → periodic distill
```

Failure modes to measure, not hand-wave:
- **Selective labels.** Labels exist only where the model escalated. Fix: ε-audit plus inverse-propensity weighting (P4).
- **Confirmation drift.** Confident-and-wrong regions are never escalated, so they are never corrected. The ε-audit is the only cure.
- **Forgetting.** Distilling hard cases erodes easy ones. Fix: a replay buffer, and a fixed audit set.
- **Teacher ≠ world.** Distilling the teacher's errors with confidence. Fix: keep a small ground-truth stream and report both calibrations.
- **Aggregation under-extremity.** Vote fractions are under-confident relative to truth (Baron 2014). Fix: recalibrate on ground truth, never on the teacher.
- **Collapse.** Training on your own outputs (Shumailov 2024). The loop only ever trains on S2 outputs, never on S1's, and that rule is enforced in code.

## 6. Experiment queue

Each experiment lives in `experiments/eNNN_*`. Each has:
- a README stating its pre-registered question and prediction;
- `run.py`, which saves every number it reports;
- `results_*.json` and figures.

| ID | Question | Status |
|---|---|---|
| **E000** | Toy with an exact oracle (shortest paths). How much is amortisable, what does the gate save, does least-confident labelling beat random, does calibration survive shift? | **done.** Results: 77% one-pass accuracy, with an MLP ceiling. Raw S1 is overconfident; one temperature fixes it in-distribution. The gate needs 65% of S2 calls, against 96% random and 22% oracle, so the bottleneck is resolution, not calibration. Calibration error triples under shift while accuracy holds. Least-confident labelling gives **no gain** over random. |
| **E003** | **Invariant-structure learner** (Golay). | **done.** One codeword gives the exact optimum (0.500 bits/symbol); without symmetry it takes 16; SGD never. |
| **E004** | **GL(2) rung: Ramanujan τ.** | **done.** Rediscovered coprime multiplicativity and weight 12; Hecke observer pays exactly the prime density. |
| **E002** | **Sporadic symmetry and arithmetic.** Golay/PSL(2,23); Legendre and Liouville. | **done.** Symmetry is the most data-efficient prior, but SGD can't use it (parity hardness). Multiplicativity-aware observers pay exactly the prime density. |
| **E001** | **Binary ground floor**: 8 sources × compute ladder, bits ledger, shift, escalation, diagonals. | **done.** See §0 and its README. |
| E000b | **Anytime credence and selective labels** (renumbered: folded into stage 4 / E004). Same toy, but S2 has a budget: k rounds of Bellman-Ford relaxation, so S2 is only right if the path has ≤ k hops. S1 predicts p(S2_k answer \| x) for all k jointly, plus ground truth. **Use a learner without E000's ceiling:** a GNN or edge-transformer, with an architecture sweep first. Tests: P3 (gate on predicted *change*, not confidence); P4 (least-confident vs random vs ε-audit + IPW); calibration-to-teacher vs calibration-to-world; and a familiarity signal for the shift failure E000 found (P2). CPU / RTX 2060, hours. | next |
| E005 | **Chess, System 1 vs search.** A small net predicts Stockfish depth-d best move and a WDL distribution, for several d. Calibrate. Gate the bot's search on predicted VOC. Plot the Elo-vs-compute Pareto against fixed local opponents (Stockfish skill levels). Compare with Ruoss 2024 value bins. Local, Stockfish labels free. | planned |
| E006 | **Honest redo of the language decision head.** On multiple-choice / yes-no / ordinal tasks with held-out *families*, matched compute, 3 seeds, compare: zero-shot option log-probs + temperature; linear probe on frozen embeddings; TabPFN head on frozen embeddings; option-marker LoRA head (log vs Brier loss); and the base-rate null. Metrics: proper scores, smECE with CIs, CDL, worst-family calibration. Qwen2.5-0.5B/1.5B on the local RTX 2060. | planned |
| E007 | **Vote-histogram compression.** Predict a sampled model's N-vote distribution in one pass (labels free). Then recalibrate to answer keys. Measure the compute saved against self-consistency at matched accuracy. | planned |
| E008 | **Resolving forecasts as a continual stream.** Strict temporal splits, prequential evaluation, leakage audit. | later |

**Cloud policy.**
- About $22.9 of the original $27.89 Modal budget remains.
- No `modal run` / `modal deploy` without a written note in the experiment README, covering: why a laptop can't do it, the expected cost, and the kill criteria.
- Nothing above E003 needs cloud.

## 7. Standards (lessons from the old repo, written as rules)

1. **Every reported number is produced and saved by a script in the repo.** No hand-typed figures, ever.
2. **Baselines first:** the base-rate null, a heuristic, temperature-scaled zero-shot, and a linear probe. A method that doesn't beat a probe isn't a method.
3. **At least 3 seeds and bootstrap CIs.** No calibration claim from fewer than about 2k evaluation examples.
4. **Recalibrators are fitted on a split disjoint from both training and evaluation.**
5. **Report:**
   - NLL;
   - Brier, with its resolution term;
   - smooth ECE or debiased ECE (not just 15-bin plug-in);
   - AURC;
   - the escalation curve against random and oracle gates.
6. **Held-out evaluation means held-out families or distributions, stated explicitly.** Report the size of the held-out set.
7. **Name things for what they are.** Supervised proper-score training is not RL. A formula is not an Elo. A lemma is not a law.
8. **Negative results go in the README** with the same prominence as positive ones.

## 8. Open questions for the owner

See the end-of-session summary. They are mostly about direction and outward-facing actions: deploying the site, the bot's wake endpoint, how to treat Modal, and whether to delete `archive/`.
