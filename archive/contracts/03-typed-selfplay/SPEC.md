# Typed Self-Play + Epiplexity: Working Spec

## Mission

Pretrain a Jev-style System 1 typed-decision model on synthetic, type-valid
computable structure alone — zero natural data, zero curated task data — such
that it acquires a reusable prior over structured decision processes before
it ever sees chess, Tetris, maze, or Mini-ARC data. The bet is that a
*typed* program space, curated by structural relevance rather than raw
difficulty, is a more sample- and compute-efficient substrate for this than
an untyped Solomonoff-style search (Cowsik et al., 2026) would be. This has
not been shown to work past toy scale. Two toy experiments have now run;
this spec states what they actually established, and what the theory
currently claims given that evidence.

## Working theory, in its current (revised) form

The research line rests on four separable claims, stacked in order of how
well-supported each one now is:

1. **Typedness is a validity precondition, not a selection signal.** It
   guarantees the generator isn't wasting search/RL budget on programs that
   crash. This is the weakest, cheapest claim, and it's the one with the
   most direct evidence behind it.
2. **Raw difficulty (frontier-matching) is necessary but not sufficient.**
   A program the learner already predicts well teaches nothing; a program
   far past the learner's capability also teaches nothing. This is well
   established in the literature this line builds on and isn't itself in
   question.
3. **Epiplexity is claimed to be a distinct axis from raw difficulty** —
   the amount of *learnable, reusable* structure, as opposed to just how
   much loss a family currently produces. This is the load-bearing claim of
   the whole programme, and it is currently the weakest-supported one: the
   only test run so far found epiplexity and final-loss rankings nearly
   identical.
4. **Structural relatedness between families predicts transfer**, and this
   relatedness should be defined by shared computational primitives, not by
   surface category (e.g. "both modular arithmetic"). This is now actively
   contradicted at toy scale by one clean result (see below), not merely
   untested.

## Findings so far

**Toy 1 (typed stack VM + curriculum controller).** Type-directed program
generation crashed 0/5,000 times; random untyped generation crashed 99.3%
of the time. A tiny learner's loss fell below a uniform baseline and the
curriculum controller correctly advanced difficulty once, then plateaued —
not because the data ran out, but because the learner's own capacity became
the bottleneck. Claim 1 is now well supported. Claim 2's mechanism
(difficulty should track learner capability) behaved as expected.

**Toy 2 (epiplexity zoo + transfer matrix).** Three results, in order of
how much they should move belief:

- **Negative transfer, clearly above noise.** Pretraining on `modmul`
  actively hurt performance on `modadd` (R = −0.151 ± 0.009, roughly 15x
  the seed-to-seed noise on every other pair), despite both being "modular
  arithmetic" on paper. The likely cause: mod-prime multiplication mostly
  produces short cycles, so `modmul` behaves more like disguised periodic
  memorization than like general modular computation — the two families
  don't actually share the primitive their labels suggest. This directly
  contradicts claim 4 as stated (surface relatedness) and supports the
  stronger version (shared primitive, independently verified, is what
  should predict transfer — and even that needs to be checked in each
  case, not assumed from a shared name).
- **A structure can be invisible to a weak observer.** `modadd` was never
  learned at all by the toy MLP — final loss sat at the uniform baseline
  across every seed. This isn't evidence `modadd` lacks structure; it's a
  direct, if accidental, demonstration that epiplexity is observer-relative
  (Finzi et al., 2026), and it means this particular transfer pair can't
  cleanly support or refute anything about shared mechanism, since one side
  was never learned to begin with.
- **Epiplexity ≈ difficulty at this scale, with this observer.** The
  epiplexity-ranked and final-loss-ranked family orderings were nearly
  identical. Claim 3 — the one the entire programme depends on — has not
  yet been shown to add information beyond raw difficulty. This may well be
  an artefact of using one weak MLP instead of the paper's own M0–M3
  observer spread, or of the crude area-under-the-curve proxy conflating
  "slow to optimize" with "genuinely high-information." It has not been
  ruled out as a real limitation of the concept, either.

## Open cruxes (unresolved, stated as testable propositions)

- Does epiplexity separate from raw final-loss difficulty once measured
  across a real spread of observer capacities (M0 n-gram through M3
  Jev-scale), rather than one small MLP? Untested.
- Can "shared computational primitive" be operationalized and checked
  *before* pairing two families for a transfer test, rather than inferred
  from a shared surface label after the fact? Currently the failure mode
  this project just hit.
- Is negative transfer (interference) common enough across the zoo that it
  needs to be an explicit term in the generator's reward, rather than an
  assumed non-issue? One data point so far says yes; needs more pairs.
- At what learner capacity does `modadd`-type structure stop being
  invisible, and does the epiplexity/difficulty gap open up once it does?

## Immediate next step implied by this spec

Before any RL-trained generator or real Jev checkpoint: rerun the zoo with
at least two observer capacities (the existing MLP plus one materially
larger model, even a slightly bigger numpy net) to test whether the
epiplexity/difficulty gap opens up with a stronger observer, and redefine
family pairs by an explicitly checked shared primitive rather than a shared
name before testing transfer again.
