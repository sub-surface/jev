# Philosophy & rationality grounding

Compiled 2026-10-03 by a research agent working from the web.

**How much of this was actually read.** Full text was read for only four sources:
- Kahneman–Klein 2009
- Logical Induction (sections 4.3–4.4)
- Cotra's post on Iterated Distillation and Amplification (IDA)
- Beren's LessWrong post on amortised optimisation

Everything else is at the level of the abstract. Items marked **[unverified]** come from memory and were not checked.

## 1. Accuracy, calibration, scoring

- **Joyce 1998**, "A Nonpragmatic Vindication of Probabilism", *Phil. Sci.* 65(4) — <https://philpapers.org/rec/JOYANV>
  **Pettigrew 2016**, *Accuracy and the Laws of Credence* — <https://philpapers.org/rec/PETAAT-7>
  **SEP entry on epistemic utility** — <https://plato.stanford.edu/entries/epistemic-utility/>
  - **Claim:** credences that are not probabilities are always beaten on accuracy, where accuracy is measured by a strictly proper score.
  - **For us:** proper-score training is the principled target. These theorems justify *coherence*, though, not calibration and not usefulness.

- **Gneiting & Raftery 2007**, *JASA* 102 — <https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf>
  - **Claim:** this is the reference text on proper scoring rules.

- **Who defends calibration, and who attacks it:**
  - **van Fraassen 1983** gives a frequency-based justification of calibration.
  - **Dawid 1982**, "The Well-Calibrated Bayesian", argues that a coherent Bayesian expects to be calibrated.
  - **Oakes 1985** shows there is no self-calibrating prior.
  - **Seidenfeld 1985** shows that scoring calibration over the short run is improper and invites hedging — <https://philpapers.org/rec/SEICCA>
  - **Foster & Vohra 1998** show that a forecaster who randomises can be calibrated on *any* sequence of outcomes, so calibration alone can be faked.
  - **Belot 2013** shows that calibration failure is topologically typical — <https://arxiv.org/abs/1306.4943>
  - **Levinstein 2017**, "Accuracy Uncomposed", argues that calibration should be explained by accuracy rather than added alongside it.

  **Implication: calibration alone is never evidence of skill.**

- **Dawid 1984**, the prequential principle: judge a forecaster only on its forecast–outcome pairs, evaluated test-then-train.

- **Brier score = reliability − resolution + uncertainty:**
  - **Murphy 1973** introduced this decomposition.
  - **DeGroot & Fienberg 1983**.
  - **Gneiting, Balabdaoui & Raftery 2007**: "maximise sharpness subject to calibration".
  - **Tetlock 2005**: perfect calibration can coexist with fence-sitting.

  **Implication: a predictor that always outputs the base rate is perfectly calibrated. It is the null baseline, and every report must show resolution.**

## 2. Bounded rationality and metareasoning

- **Simon 1955** (satisficing) and **Gigerenzer & Gaissmaier 2011** (*Annu. Rev. Psychol.* 62) on heuristics and "less is more".
  - **For us:** a distilled model must beat cheap heuristic baselines before we claim "compression".

- **Russell & Wefald 1991**, "Principles of metareasoning", *AIJ* 49.
  - **Claim:** computations are actions. Each has a cost and a value, the expected improvement in the decision.
  - **For us:** confidence is only a proxy for the value of computation. Deliberating cannot help when the uncertainty is irreducible.

- **Resource rationality:**
  - **Gershman, Horvitz & Tenenbaum 2015**, *Science* 349, "Computational rationality".
  - **Lieder & Griffiths 2020**, *BBS* 43, "Resource-rational analysis".
  - **Lieder & Griffiths 2017**, *Psych. Rev.* 124. People learn a predictive model of how well each strategy will perform.
  - **Callaway et al. 2022**, *Nat. Hum. Behav.* 6.
  - **Icard 2018**, *Phil. Sci.* 85.

  **For us:** the escalation controller should itself be a calibrated predictor of how much deliberation will gain.

- **LLM-era precedents:**
  - **De Sabbata et al. 2024** (arXiv:2410.05563) put a value-of-computation reward into Expert Iteration and used 38% fewer tokens.
  - **Snell et al. 2024** show test-time compute should adapt to difficulty.
  - **FrugalGPT** routes queries through a cascade of models.
  - **Fan et al. 2026** (arXiv:2608.07968) find that models ration a shared compute budget poorly.

## 3. Dual process, intuition, amortisation

- **Kahneman & Klein 2009**, "Conditions for Intuitive Expertise: A Failure to Disagree", *Am. Psychol.* 64(6) — full text: <https://www.hansfagt.dk/Kahneman_and_Klein(2009).pdf>
  - **Claim:** intuition is valid only where three things hold:
    - the environment has "sufficiently high validity",
    - there is adequate practice,
    - feedback is "both rapid and unequivocal".
  - Validity is not the same as certainty: poker is high-validity.
  - "Subjective confidence is ... an unreliable indication of the validity of intuitive judgments."
  - **This is the central map for the program.**

- **Simon 1992**: "Intuition is nothing more and nothing less than recognition."
  **Klein et al. 1986**, recognition-primed decision.
  - **Claim:** recognition proposes a single option, and mental simulation checks it.
  - **For us:** this is the escalate architecture.
  - **Hogarth et al. 2015** separate *kind* learning environments from *wicked* ones.

- **Evans & Stanovich 2013** (default-interventionist dual process) and **Dreyfus & Dreyfus 1980** (the five-stage skill model).
  - **For us:** treat System 1 / System 2 as an engineering decomposition, not as psychological evidence.

- **Gershman & Goodman 2014**, "Amortized inference in probabilistic reasoning", CogSci.
  **Beren 2022** on LessWrong, "Deconfusing direct vs amortised optimisation" — <https://www.lesswrong.com/posts/S54HKhxQyttNLATKu/>
  - **Claim:** amortisation pays off only when queries recur, and its main risk is misgeneralisation.

- **Anthony et al. 2017** (Expert Iteration) and **Yu et al. 2024** ("Distilling System 2 into System 1"). Neither evaluates calibration.

## 4. Logical uncertainty

- **Garrabrant et al. 2016**, "Logical Induction" — <https://arxiv.org/abs/1609.03543>
  - **Claim:** it learns calibrated credences about deductive facts long before it can prove them (section 4.3).
  - On sequences that are pseudorandom to a reasoner with the same runtime, it falls back to the base rate (section 4.4).
  - The authors call it "theoretically interesting but ultimately impractical".
  - **This is the formal template for compressing deduction into credence.**

- **Hacking 1967**, "Slightly more realistic personal probability".
  **Gaifman 2004**, *Synthese* 140.
  - **Claim:** probability can be relative to the resources available for computation.
  - **For us:** this legitimises "credence about what deliberation would conclude" as the student's target.

- **Christiano, Neyman & Xu 2022** (arXiv:2211.06738) and **Christiano et al. 2024** (arXiv:2410.01290), heuristic estimators.
  - **Claim:** an estimator "ought not to be able to predict its own errors".
  - **For us:** this gives a test. The student's residuals should be unpredictable from cheap features, which is a multicalibration-like condition.

## 5. Rationality community

- **Calibration training:**
  - Credence Calibration Game FAQ on LessWrong.
  - Open Philanthropy's write-up on calibration training.
  - PredictionBook.
  - ACX 2020 calibration results: the 90% bin went 16 for 19, so bins of 10–20 items are noise.

- **Tetlock and the Good Judgment Project:**
  - **Mellers et al. 2014/2015**: training, teaming and tracking improve both calibration and resolution.
  - **Baron et al. 2014**: aggregated forecasts are under-extreme, so distilling an average of teacher samples needs extremising.

- **Yudkowsky 2007**, "Making Beliefs Pay Rent", and **Xu 2021**, "Strong Evidence is Common".
  - **For us:** being calibrated to a teacher is not the same as being calibrated to the world.

- **Iterated Distillation and Amplification:**
  - **Christiano, Shlegeris & Amodei 2018**, arXiv:1810.08575.
  - **Cotra 2018** on the Alignment Forum. It names three unproven assumptions, including that distillation is robust.
  - **Yudkowsky 2018**, critique: imperfect imitation compounds over iterations.

- **Goodhart effects:**
  - **Manheim & Garrabrant 2019**, arXiv:1803.04585.
  - **Witkowski et al.**, *Mgmt. Sci.* 69(3): winner-take-all scoring rewards extreme reports.
  - **Turner 2022**, "Reward is not the optimization target".
  - **For us:** never select models on ECE alone.

- **Limits of calibration:**
  - **Hájek 2007**, the reference class problem.
  - **Multicalibration**: global calibration can hide miscalibration in subgroups.
  - **Kalai & Vempala 2024**: a calibrated language model *must* hallucinate rare facts.

- **LLM calibration and forecasting:**
  - **Kadavath et al. 2022**.
  - **RLCR**.
  - **Halawi et al. 2024**.
  - **Schoenegger et al. 2024**.
  - Metaculus AI benchmark: in Q2 2025, human professionals stay clearly ahead.
  - FutureEval, Spring 2026: the professionals-vs-bots gap is not significant (n = 99).
  - Wilson 2026 flags leakage in AI forecasting evaluations.
  - **For us:** the bottleneck is research (retrieval), not reflex.

## 6. Decision theory: strategic and performative limits

- **Good 1967**, "On the Principle of Total Evidence", *BJPS* 17 (verified via secondary sources only).
  - **Claim:** free evidence never hurts in expectation.
  - **This is the normative core of "deliberate when the expected gain exceeds the cost".**

- **Omnipredictors** (arXiv:2109.05389) and **Hu & Wu 2024** (calibration decision loss).
  - **For us:** "calibrated beliefs plus expected utility equals good decisions" holds for decision- or multi-calibration, not for scalar ECE.

- **Strategic and performative effects:**
  - **Haghtalab et al. 2023**: calibrated forecasts can be exploited.
  - **Perdomo et al. 2020**: performative prediction.
  - **Demski 2019**: the Predict-O-Matic parable.
  - **Lakkaraju et al. 2017**: selective labels.
  - **Ovadia et al. 2019**: uncertainty under dataset shift.
  - **Shumailov et al. 2024**: model collapse.
  - **For us:** a model that gates its own labels becomes part of its own data-generating process.

## Five falsifiable principles

These have been adopted into `../RESEARCH.md`.

- **P1. Validity ceiling.**
  - **Claim:** the student's held-out resolution is bounded by the resolution of a teacher ensemble against ground truth.
  - **Prediction:** on zero-validity families, the student collapses to the base rate.
  - **Falsified if:** the student beats the ensemble on clean ground truth.

- **P2. Confidence doesn't flag its own invalidity.**
  - **Claim:** under shift, ECE grows while mean confidence falls far less than accuracy does.
  - **Prediction:** a separate familiarity head adds information about errors.
  - **Falsified if:** the student's confidence alone matches the familiarity head out of distribution.

- **P3. Escalate on value of computation, not confidence.**
  - **Claim:** a learned gain predictor g(x) = E[loss_S1 − loss_S2 | x] beats a confidence threshold at a matched escalation rate.
  - **Falsified if:** the two are within the confidence interval, with the margin pre-registered.

- **P4. The loop is a selective-labels, wicked-feedback system.**
  - **Prediction:** without a random audit slice, ECE measured on that audit slice drifts upward across distillation rounds.
  - **Falsified if:** it stays flat.

- **P5. Calibration is cheap; compression shows up in resolution and decision loss.**
  - **Prediction:** among models with equal ECE, decision regret differs. The rank correlation between ECE and regret is weak, and calibration on the worst subgroup is worse than global calibration.
  - **Falsified if:** ECE ranks the models the same way regret does.
