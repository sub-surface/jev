---
status: conjecture
area: program
---
# intelligence target

Operational target, on a source family F: **bits extracted per unit of (data + compute + prior description), subject to calibration that survives shift.** Score each predictor by its code-length surface L(n, c), where n is data and c is compute. Split the surface into MCB / DSC / UNC, and add learning cost, transfer to held-out generators, and the escalation frontier. This joins Chollet (skill-acquisition efficiency), Solomonoff (code length) and resource rationality (cost).

**Source:** Chollet 2019; Legg & Hutter 2007; Lieder & Griffiths 2020; this program.

**Test / open:** See RESEARCH.md stages: E002 amortised inference over a program prior; E003 self-play curriculum; E004 escalation learned from gain.

**Links:** [[inference-vs-calibration]] [[bounded-observer-information]] [[library-of-babel-as-coordinates]] [[competition-and-exchange]]
