# E000: amortising an exact System 2, on a toy with a perfect oracle

**Status:** done, 2026-10-03.
**Run:** `python run.py` (about 10 min on CPU; `--quick` takes about 30 s).
**Artifacts:** `results_full.json` (summaries plus every raw row), `fig_full.png`, `fig_full_loop.png`, `log_full.txt`.

## Setup

- **Task.** Each problem is a random weighted digraph on 10 nodes with edge density 0.4. From node 0 there are 4 candidate first hops. The answer is the hop that lies on the shortest path from 0 to 9.
- **System 2.** Exact Floyd–Warshall, so it is always right.
- **System 1.** A 2×512 MLP that sees the weight matrix and an edge mask once and outputs a distribution over the 4 hops.
- **Training.** Cross-entropy, which is the log score. 30 epochs. A temperature is fitted on a 10% split of the training data that was held out from fitting the weights.
- **Evaluation.** 5,000 fresh problems, for 3 seeds; table values are mean ± sd over seeds.

## Pre-registered questions → what happened

### Q1 Amortisation: how far does one pass get?

Accuracy rises from 28% with 250 labels to 77% with 64k labels (chance is 25%). It is clearly flattening, and was still climbing slowly at 64k.

There is a dip at 16k labels, reproduced across all 3 seeds. It coincides with the worst raw overconfidence, so it is most likely over-training from the fixed 30-epoch schedule, not a real non-monotonicity.

A flat MLP cannot do the multi-hop relaxation that S2 does, so there is an **architectural amortisation ceiling**. The next step is to vary the architecture (a GNN, or a transformer over edges) to separate "data-limited" from "architecture-limited".

### Q1b Calibration: does proper-score training give calibration?

**No, not at finite data.** The raw S1 is strongly overconfident: smooth ECE is 0.11–0.19 for n ≥ 4k, and the fitted temperatures are 1.6–6.5.

One scalar temperature fixes this in-distribution: smooth ECE drops to 0.016–0.018, and the debiased L2 ECE to ≤ 0.011.

Lesson: the proper loss sets the *target*, but optimisation and overfitting decide where the model lands. Post-hoc recalibration on held-out data is not optional.

### Q2 Escalation: what does confidence buy?

The question is what fraction of queries must go to S2 to reach 99% of S2's accuracy. At 64k labels:

| Gate | Fraction escalated |
|---|---|
| Confidence gate | **65% ± 1%** |
| Random routing | 96% |
| Perfect oracle gate | 22% |

So calibrated confidence saves about 31 points of S2 calls relative to random routing. But it captures only about 40% of the achievable saving.

The gap to the oracle is a **resolution** problem, not a calibration problem: the model is calibrated but cannot tell its right answers from its wrong ones sharply enough. This is principle P5 in action.

### Q3 Distill-back loop: does labelling the least-confident queries beat random labelling?

**No.** The setup gives both strategies equal oracle budgets: 1k initial labels, plus 1k labels per round over 6 rounds, chosen from 10k fresh queries per round.

- Both strategies reach about 72% by round 2, then plateau.
- After that, least-confident selection is slightly *worse* on accuracy: 71.8% vs 72.5% at round 6, a gap of about 1–2 sd.
- It is slightly *better* calibrated: smooth ECE 0.017 vs 0.020, but within noise.

The plateau is the same architectural ceiling as in Q1, so selection cannot help. Uncertainty sampling also concentrates labels on intrinsically ambiguous problems. That is the selective-labels concern (P4) showing up as a mild harm rather than a gain.

**The loop's benefit is not established here.** E001 must use a learner without a hard ceiling, and must compare least-confident, random, and ε-audit mixed with inverse-propensity weighting.

### Q4 Distribution shift: train at density 0.4, test at 0.25

Accuracy *transfers*: the sparser graphs are, if anything, slightly easier (79% vs 77% at 64k). **Calibration does not transfer.**

| Labels | iid smooth ECE (after temp.) | Shift smooth ECE (after temp.) |
|---|---|---|
| 4k | 0.017 | 0.023 |
| 16k | 0.018 | 0.034 |
| 64k | 0.016 | 0.051 |

The shifted calibration error grows with more labels while iid stays flat. The sharper the amortised intuition, the worse its calibration off-distribution, even where its accuracy holds up. This is a small but clean instance of Kahneman–Klein / P2.

The confidence gate's *ranking* survives the shift: 60% escalated vs 65% iid. So the gate stays useful while the probabilities themselves become untrustworthy.

## Caveats

- **Toy problem, MLP, one task family.** Treat these results as a sanity-checked pipeline and a set of hypotheses, not findings about language models.
- **"99% of S2 accuracy" is one operating point.** The full curves are in `results_full.json`.
- **Hyperparameters were not tuned per dataset size,** which is the likely source of the 16k dip.
