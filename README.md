# jev — calibrated decisions as compressed deliberation

This is a small research lab. It asks when an expensive deliberative procedure can be replaced by one forward pass of a calibrated model. Examples of such procedures: search, chain-of-thought, voting, a judge model, a simulator, a test suite. The calibrated model's confidence also says when the expensive path is still worth taking.

**Start here:**
- [`RESEARCH.md`](RESEARCH.md): the program. Thesis, principles, where it applies, how the loop breaks, and the experiment queue.
- [`REVIEW.md`](REVIEW.md): a frank review of the repo's previous incarnation, and why it was torn down.

## Layout

```
calib/         tested core: proper-score losses, calibration metrics (debiased / smooth ECE, CIs),
               recalibration, escalation curves (compute saved at matched quality)
experiments/   one folder per pre-registered experiment; every reported number is saved by its run.py
tests/         pytest for calib/
vault/         atomic notes: one claim each, with status; MAP.md (concept graph), QUESTIONS.md (registry)
bot/lichess/   the jess-hyperbullet Lichess bot (kept; a System-1/System-2 testbed). Read its README before touching Modal.
site/          public page for jev.subsurfaces.net (static, Cloudflare Workers assets). Not yet deployed.
literature/    PDFs + read-outs + bibliographies (ML, philosophy/rationality). Old AI-written notes quarantined.
archive/       everything from before 2026-10-03, untrusted. See REVIEW.md.
```

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # (bin/ on Linux/macOS)
.venv/Scripts/python -m pytest tests -q
.venv/Scripts/python experiments/e000_shortest_path_amortization/run.py --quick   # ~30 s, CPU
```

Nothing in this repo spends cloud money unless you explicitly run a `modal` command, and the only Modal file outside `archive/` is the bot daemon. Cloud policy is in `RESEARCH.md` §6.
