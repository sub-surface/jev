# Review of the pre-teardown repo (2026-10-03)

This is a frank review of the repo before the teardown.

**Method.**
- The code was read directly.
- The documents were audited against the result artifacts by an independent agent, which did not take their claims on trust.
- The literature PDFs were read from source.

All old material is preserved under `archive/` and in git history. Nothing has been deleted.

## Verdict

There is one useful idea in here, buried under a lot of noise. The idea is a one-pass decision head:
- Put option markers after the option texts.
- Pool the hidden state at each marker.
- Score the markers into a calibrated distribution.
- Train with a proper scoring rule.

That is a legitimate, cheap architecture.

Almost everything built around it overstates what was shown:
- The "RL" is supervised learning.
- The headline comparisons are invalid.
- Several figures are hardcoded or synthetic.
- The "theorems" are either trivial or just named claims.

As a research record, the repo can't be trusted. As a codebase, it has a few reusable parts.

## What the core actually is (`archive/contracts/02-rlcd-decision`)

| Claimed | Actual |
|---|---|
| "RLCD — Reinforcement Learning for Calibrated Decisions", and "= Yang et al. 2023 RLCD" | Gradient descent on `-(Brier + 0.5·spherical)` of softmax probabilities, with Gaussian noise added to the logits. There is no policy gradient and no sampling of actions. Cross-entropy *is* the log score, so "RLCD vs CE" compares proper rule A against proper rule B. Yang et al.'s RLCD is an unrelated alignment method based on contrastive preference pairs. |
| Uses RLCR's (Damani et al.) Brier theorem | RLCR scores a *sampled* answer plus a *verbalised* confidence, where correctness is not differentiable. Here, `1[argmax = y]` has zero gradient, so the term is decorative. The theorem is irrelevant to a head that outputs a full distribution. |
| Logit noise is "exploration" | Noise inside a proper score makes the optimum slightly *un*-calibrated (biased toward sharper logits). It is harmless at σ = 0.01–0.1, but it is not a feature. |
| "Rewarding Doubt" implemented | It is implemented wrongly: it rewards `log(1 − p_true)` when wrong, which pushes probability *away* from the true label. It was dead code (never called). |
| Choice / Noul / Score primitives | The Score / ordinal path never runs. The loader never sets `is_ordinal`, so the RPS code is unreachable. |
| 5.3M examples, 485 tasks, 18 families | The real runs used the first 15–75 tasks in stream order: about 3–4k examples, with options extracted by fragile regexes (quoted strings, comma splits). |
| RLCD vs CE comparison | The two arms used different learning rates (5e-5 vs 1e-4 to 2e-4), and only the RLCD arm shuffled option order. Option-position bias is therefore a confound. |
| Per-cardinality temperature scaling | Temperatures were clamped to [0.5, 3] with an L2 prior. They were fitted on the same in-task validation set that in-task ECE was then reported on. The ≥11-option bucket was fitted on n = 1. |
| Triton fused Brier kernel, "6.35x" gather speedup | Never imported by the trainer or the Modal pipeline. The speedup comes from a single unsynchronised microbenchmark. |

## What the results actually show

Full table in the audit; the essentials:

- **"RLCD zero-shot ECE 5.12% vs CE 87.81% — 17.1× better": invalid.**
  - The CE number comes from a *synthetic toy* run: 4 unique prompts repeated, with n = 24.
  - The RLCD number comes from a different, real run.
  - No CE baseline was ever run on real data.
- **The 0.5B "zero generalisation drop" model is worse than chance.**
  - It scores 44% on *binary* held-out tasks (MRPC and QQP).
  - Its Brier is 0.543 against 0.5 for a uniform guess, and its NLL is 0.811 against ln 2 = 0.693.
  - Its low ECE comes from outputs that carry almost no information.
- **Qwen3-4B (87% in-task, 70% held-out paraphrase) is the only real result.**
  - It is one seed with n ≈ 400, and no base-model or CE baseline.
  - The JSON was hand-assembled from console output.
- **Hardcoded or fabricated figures.** Several figures and values were typed in or simulated rather than measured:
  - fig37
  - fig38 (typed-in loss curves)
  - the 0.5B temperatures
  - fig2 and fig4 (built with `np.random`, labelled empirical)
  - the "ECE 1.85% vs 25.70%" labels
  - a "94%" Mini-ARC CSV row (the artifact says 8%)
- **The chess "Elo 1927" is a formula, not a rating:** `1500 + 9·solve% + 1.5·(100 − latency)`. Three Lichess games are on disk.
  - The "100/100" score was tuned on 20 hand-labelled positions.
  - The deployed weights solve 9 of those 20.
- **Named theorems ("Law of Discrete Invariant Coupling", "TypeSafe Jev Theorem").**
  - Some are trivial: a masked softmax assigns zero probability to masked actions, and a strictly decreasing potential has no cycles.
  - The rest are assertions backed by n = 50 toy runs that don't replicate in the repo's own raw `.pt` files.
- **The hosted-Jev head-to-head and the DiffusionGemma evaluation were never run.**
- **Modal spend.**
  - The log totals **$5.02**, not the $2.08 the docs report.
  - 80% of that spend went to runs that have no reported results.

## The chess bot (kept, fixed, documented in `bot/lichess/README.md`)

**What it is.** A 623k-parameter CNN ranks moves, and a gate decides between replying instantly and running a short alpha-beta search. That search evaluates leaves by *material count*, not by the network. The "noul" confidence head was trained on `random.uniform(0.2, 0.6)` / `uniform(0.75, 0.98)` targets, so it encodes nothing.

**Fixes made in this pass.**
- The Docker build could not find any weights and silently played with random initialisation. The folder is now self-contained: weights and games are local, and `.dockerignore` keeps `.env` out of the image.
- The bot was verified offline: it loads the right weights and plays sensible moves.

**Production (the Modal daemon).**
- Untouched, and unaffected by this pass.
- It has a public, unauthenticated `wake` endpoint. The old site called it.

## Other findings

- **Dead or incomplete code.**
  - `openjev_engine_v2.py` writes stub source files into `~/openjev_research`.
  - `eval_diffusion_gemma.py` would put a 26B model on Modal.
- **Repo hygiene.** About 60 MB of `.pt` files are tracked in git.
- **Experiment bookkeeping.**
  - The budget log has no commit or config hashes.
  - `train_a10.py` silently falls back to synthetic data if the HF stream fails.
- **The literature folder's `.md` notes** are AI-written. They are quarantined in `literature/_old_ai_notes_unverified/`.

## What carried forward

1. **The option-marker head idea**, to be re-implemented cleanly when we reach language tasks (E003).
2. **Proper-scoring losses**, re-implemented and tested in `calib/scoring.py`.
3. **The bot**, as a System-1/System-2 testbed.
4. **The PDFs**, now with faithful read-outs (`literature/NOTES_pdfs.md`).
5. **Lessons, written into `RESEARCH.md` as rules:**
   - baselines first
   - multiple seeds with confidence intervals
   - no fitting on the evaluation set
   - debiased or smooth calibration metrics
   - every number produced by a script that saves it
   - no cloud compute without a written reason
