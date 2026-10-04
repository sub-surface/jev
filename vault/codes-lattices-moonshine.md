---
status: established
area: mathematics
---
# codes lattices moonshine

There is a real chain from Shannon to moonshine. Binary error-correcting codes lead to the Golay code, then to the Leech lattice (the densest packing in 24 dimensions; E8 is the densest in 8). From there come the Conway groups and the Monster, whose smallest faithful representation has dimension 196883. Since 196884 = 196883 + 1 is a coefficient of j, this links to monstrous moonshine. Along the way, sphere packing is optimal coding, and E8 codebooks are already used to quantise LLM weights.

**Quantisation numbers (Conway & Sloane):** normalised second moment G: Z^n 0.0833, E8 0.0717, Leech 0.0658, sphere limit 0.0585. So Leech gives about 8% less distortion than E8 at the same rate. The cost is decoding: Leech at 2 bits/dim has 2^48 points, so it needs an algebraic decoder (Conway & Sloane; Vardy & Be'ery), not a lookup table.

**Source:** Conway & Sloane, SPLAG; Viazovska 2017; Borcherds 1992; Tseng et al. 2024, QuIP# (E8 lattice quantisation) [verify].

**Test / open:** Do lattice-quantised latents or forecasts give better bits per parameter for sequence predictors than scalar quantisation? This is the practical end of the chain.

**Links:** [[golay-symmetry-testbed]] [[symmetry-conservation-in-learning]] [[nn-field-theory]]
