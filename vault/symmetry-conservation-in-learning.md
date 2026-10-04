---
status: established
area: mathematics
---
# symmetry conservation in learning

Noether's theorem holds literally in gradient flow. Every continuous symmetry of the loss (rescaling, translation, or rotation of parameters) yields a conserved quantity. Weight decay, finite step size and SGD noise break these in predictable ways. 'Learning and forgetting as symmetry breaking' is a *conjecture* built on this established base.

**Source:** Kunin et al. 2021, 'Neural Mechanics: Symmetry and Broken Conservation Laws in Deep Learning Dynamics' (ICLR); Tanaka & Kunin 2021.

**Test / open:** In continual sequence prediction with regime shifts, does the drift of these conserved quantities predict forgetting better than the loss curve does?

**Links:** [[loss-landscape-orbifold]] [[codes-lattices-moonshine]]
