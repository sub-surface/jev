---
status: established
area: foundations
---
# no universal computable predictor

Take any computable predictor A and build the diagonal sequence: at each step, emit the bit A finds less likely. That sequence is computable, yet it costs A at least 1 bit per symbol. So no computable predictor is universal, and every one has an adversary. E001 section E confirms it: each observer, including a log-score market of observers, loses about 1 bit per symbol on its own diagonal, while weak observers' diagonals are easy for stronger ones.

**Source:** Classical diagonalisation; Dawid 1985; Oakes 1985; E001 competition matrix.

**Links:** [[competition-and-exchange]] [[pseudorandomness]]
