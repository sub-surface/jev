"""E003 — A learner that can represent the invariant: Bayesian model selection over G-invariant linear codes.

E002 showed the Golay structure earns bits only through a learner that can represent it; SGD on an MLP could
not, even with the symmetry handed over. Here hypotheses ARE structures: for every training word w the learner
forms span(G·w), the smallest G-invariant linear code containing w (G = PSL(2,23), order 6072), adds the
trivial code F2^24, and does exact Bayesian model selection under a BSC(eps) with known eps.
Exact likelihoods for any linear code C via Poisson summation over the dual:
    P(x_O | C) = 2^-|O| * sum_{u in C_perp, supp(u) ⊆ O} (-1)^{u·x} (1-2eps)^{|u|}
(enumerating C instead when it is the smaller of the two). Control: the same learner WITHOUT symmetry
(candidates = span of the training words). Ledger on the same stream protocol as E002.
python run.py   (CPU, ~minutes)
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_s = importlib.util.spec_from_file_location("e002", ROOT / "experiments/e002_golay_symmetry/run.py")
e002 = importlib.util.module_from_spec(_s); _s.loader.exec_module(e002)
OUT = Path(__file__).parent
FULL = (1 << 24) - 1


def to_int(w):
    return int(sum(int(b) << i for i, b in enumerate(w)))


def rref(vecs):
    """Reduced basis over GF(2) as a dict pivot->vector (vectors are ints)."""
    basis = {}
    for v in vecs:
        for p in sorted(basis, reverse=True):
            if v >> p & 1:
                v ^= basis[p]
        if v:
            p = v.bit_length() - 1
            for q in list(basis):
                if basis[q] >> p & 1:
                    basis[q] ^= v
            basis[p] = v
    return basis


def enumerate_space(basis_vecs):
    out = [0]
    for b in basis_vecs:
        out += [x ^ b for x in out]
    return np.array(out, dtype=np.int64)


def dual_basis(basis):
    piv = sorted(basis)
    free = [i for i in range(24) if i not in basis]
    duals = []
    for f in free:  # u = e_f + sum over pivots p of (bit f of basis[p]) e_p  is orthogonal to every basis vector
        u = 1 << f
        for p in piv:
            if basis[p] >> f & 1:
                u |= 1 << p
        duals.append(u)
    return duals


class LinearCode:
    def __init__(self, basis):
        self.basis = basis
        self.k = len(basis)
        self.key = tuple(sorted(basis.items()))
        if self.k <= 12:
            self.mode, self.S = "code", enumerate_space(list(basis.values()))
        else:
            self.mode, self.S = "dual", enumerate_space(dual_basis(basis))
        self.wt = np.array([bin(int(s)).count("1") for s in self.S])

    def log_prob(self, x, mask, eps):
        """log P(observed bits | C) for observed positions `mask` (int bitmask), x as int."""
        if self.mode == "code":
            d = np.bitwise_count((self.S ^ x) & mask)
            n_obs = bin(mask).count("1")
            le, l1 = math.log(max(eps, 1e-300)), math.log(1 - eps)
            terms = d * le + (n_obs - d) * l1
            m = terms.max()
            return m + math.log(np.exp(terms - m).sum()) - self.k * math.log(2)
        sel = (self.S & ~mask & FULL) == 0
        U, W = self.S[sel], self.wt[sel]
        par = np.array([bin(int(v)).count("1") & 1 for v in (U & x)])
        tot = float(np.sum(np.where(par == 1, -1.0, 1.0) * (1 - 2 * eps) ** W))
        return (math.log(tot) if tot > 1e-300 else -1e9) - bin(mask).count("1") * math.log(2)


def orbit_span(w_int, G):
    bits = np.array([(w_int >> i) & 1 for i in range(24)], np.int8)
    imgs = e002.act(G, np.broadcast_to(bits, (len(G), 24)).copy())
    vals = (imgs.astype(np.int64) << np.arange(24)).sum(1)
    return rref(np.unique(vals).tolist())


def learner_stream(cands, train, stream, eps):
    """Bayesian model averaging over candidate codes: posterior from training words, updated online by stream
    blocks; within a block the predictive mixes P(x_t | prefix, C) with weights P(C | data, prefix)."""
    lw = np.array([sum(c.log_prob(to_int(w), FULL, eps) for w in train) for c in cands])
    post_report = None
    p = np.empty(stream.size)
    for b, blk in enumerate(stream):
        x = to_int(blk)
        prev = np.zeros(len(cands))  # log P(prefix | C)
        for t in range(24):
            mask_prev, mask_next = (1 << t) - 1, (1 << (t + 1)) - 1
            x1 = (x & mask_prev) | (1 << t)
            nxt = np.array([c.log_prob(x1, mask_next, eps) for c in cands])
            lp = lw + prev
            wts = np.exp(lp - lp.max()); wts /= wts.sum()
            p[b * 24 + t] = float(np.clip(wts @ np.exp(np.minimum(nxt - prev, 0)), 1e-12, 1 - 1e-12))
            prev = np.array([c.log_prob(x, mask_next, eps) for c in cands])
        lw = lw + prev
        if b == 0:
            post_report = lw.copy()
    post = np.exp(post_report - post_report.max()); post /= post.sum()
    return p, {f"dim{c.k}": round(float(q), 4) for c, q in zip(cands, post)}


def main():
    t0 = time.time()
    C, gens = e002.golay_code()
    G = e002.closure(gens)
    rows = []
    for eps in [0.0, 0.03]:
        for seed in range(5):
            rng = np.random.default_rng(seed)
            noisy = lambda k: C[rng.integers(0, 4096, k)] ^ (rng.uniform(size=(k, 24)) < eps).astype(np.int8)
            stream = noisy(150)
            for N in [1, 2, 4, 16]:
                train = noisy(N)
                for variant in ["symmetry + linearity", "linearity only"]:
                    if variant.startswith("symmetry"):
                        spaces = [orbit_span(to_int(w), G) for w in train]
                    else:
                        spaces = [rref([to_int(w) for w in train])]
                    spaces.append({i: 1 << i for i in range(24)})
                    uniq = {}
                    for s in spaces:
                        if s:
                            uniq.setdefault(tuple(sorted(s.items())), s)
                    cands = [LinearCode(s) for s in uniq.values()]
                    p, post = learner_stream(cands, train, stream, eps)
                    rows.append({"eps": eps, "seed": seed, "N": N, "learner": variant,
                                 "candidate_dims": sorted(c.k for c in cands), "posterior_after_train_and_1_block": post,
                                 **e002.ledger(p, stream)})
                    print(f"eps={eps} seed={seed} N={N:>2} {variant:>21}: bits {rows[-1]['bits']:.3f}  "
                          f"DSC {rows[-1]['DSC']:.3f}  MCB {rows[-1]['MCB']:.3f}  cands {rows[-1]['candidate_dims']}", flush=True)
    summ = {}
    for r in rows:
        summ.setdefault(f"eps={r['eps']} N={r['N']} {r['learner']}", []).append(r["bits"])
    summary = {k: [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)] for k, v in summ.items()}
    (OUT / "results_full.json").write_text(json.dumps({"summary": summary, "rows": rows,
                                                        "seconds": round(time.time() - t0)}, indent=1, default=str))
    for k, v in summary.items():
        print(k, v)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
