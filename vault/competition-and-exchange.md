---
status: established
area: program
---
# competition and exchange

Competition: every predictor has an adversarial sequence (its diagonal), and calibrated forecasts can be exploited strategically. Exchange: a log-score market (Hanson's LMSR) is exactly Bayesian model averaging over its traders, so it aggregates calibrated beliefs with at most log(#traders) bits of regret. Self-play between generators and predictors is a natural curriculum: each side produces the sequences the other cannot yet compress.

**Source:** Hanson 2003 and 2007; Haghtalab et al. 2023; E001 section E.

**Test / open:** Does predictor-generator self-play on binary sequences produce a curriculum that improves held-out code length over a fixed prior (E003)?

**Links:** [[no-universal-computable-predictor]] [[intelligence-target]]
