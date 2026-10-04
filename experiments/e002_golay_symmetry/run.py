"""E002 — Does a sporadic symmetry earn bits?  (and: number-theoretic sources on the ground floor)

A. GOLAY. The extended binary Golay code [24,12,8] is built as the extended quadratic-residue code mod 23.
   Its automorphism group is M24; we verify and use the subgroup PSL(2,23) (order 6072) acting on the
   projective line {0..22, inf}. Source: a stream of uniformly random codewords through a BSC(eps).
   Observers (all predict bit-by-bit, phase known), trained on N codewords from the source:
     KT-phase · CTW-12-phase (generic structure inference) · MLP (generic) · MLP + group orbit augmentation
     · orbit-Bayes (Bayes over the group orbit of the training words: symmetry, no network)
     · span-Bayes (Bayes over the GF(2) span of the training words: linearity, no symmetry; eps=0 only)
     · Bayes (knows the code: the ceiling).
   Question: how many training codewords does each observer need to approach the Bayes ledger?

B. ARITHMETIC. Completely multiplicative ±1 sequences: Liouville λ(n) and the Legendre symbol (n/p).
   A bounded generic observer (CTW) vs an observer that knows multiplicativity (composites are determined
   by earlier values; only primes carry fresh bits). Sarnak's Möbius-disjointness conjecture says λ/μ are
   pseudorandom to every zero-entropy observer — a number-theory statement of observer-relative randomness.

CPU, a few minutes. python run.py [--quick]
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calib.metrics import corp_decomposition  # noqa: E402

_spec = importlib.util.spec_from_file_location("e001", ROOT / "experiments/e001_binary_ground_floor/run.py")
e001 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e001)
KT, CTW = e001.KT, e001.CTW
OUT = Path(__file__).parent
INF = 23


# ── the Golay code and its symmetry ──────────────────────────────────────────

def golay_code():
    for g in ([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1]):  # g(x) and reciprocal
        words = []
        for m in range(4096):
            c = np.zeros(23, np.int8)
            for i in range(12):
                if m >> i & 1:
                    c[i:i + 12] ^= np.array(g, np.int8)
            words.append(np.append(c, c.sum() % 2))
        C = np.array(words, np.int8)
        gens = group_generators()
        if all(preserves(C, p) for p in gens):
            return C, gens
    raise RuntimeError("no labelling found where PSL(2,23) preserves the code")


def group_generators():
    inv = {i: pow(i, -1, 23) for i in range(1, 23)}
    shift = [(i + 1) % 23 for i in range(23)] + [INF]
    double = [(2 * i) % 23 for i in range(23)] + [INF]
    neg_inv = [INF] + [(-inv[i]) % 23 for i in range(1, 23)] + [0]  # i -> -1/i, 0 <-> inf
    return [np.array(shift), np.array(double), np.array(neg_inv)]


def act(p, C):  # (p·c)[p[i]] = c[i]; p may be one permutation or a batch aligned with rows of C
    out = np.empty_like(C)
    if p.ndim == 1:
        out[..., p] = C
    else:
        out[np.arange(len(p))[:, None], p] = C
    return out


def preserves(C, p):
    S = set(map(bytes, C))
    return all(bytes(w) in S for w in act(p, C[[1 << i for i in range(12)]]))  # basis suffices: linear code


def closure(gens):
    seen, frontier = {tuple(range(24))}, [tuple(range(24))]
    while frontier:
        new = []
        for e in frontier:
            for g in gens:
                h = tuple(g[list(e)])
                if h not in seen:
                    seen.add(h); new.append(h)
        frontier = new
    return np.array(sorted(seen))


# ── observers for section A ──────────────────────────────────────────────────

def bayes_stream(words_set, stream, eps, escape=0.0):
    """Exact posterior predictive over a finite candidate set of 24-bit words, per block, BSC(eps).
    escape > 0 mixes in a 'word outside my set' model (uniform bits) with that prior weight, updated
    by Bayes within each block, so a learned candidate set is never infinitely confident."""
    W = words_set.astype(np.float64)
    le, l1e = math.log(max(eps, 1e-12)), math.log(1 - eps)
    p = np.empty(stream.size)
    for b, blk in enumerate(stream):
        lw = np.zeros(len(W))
        lc, lu = math.log(1 - escape) if escape else 0.0, math.log(escape) if escape else -np.inf
        for t in range(24):
            m = lw.max(); w = np.exp(lw - m); Z = w.sum(); w /= Z
            q1 = (w @ W[:, t]) * (1 - eps) + (1 - w @ W[:, t]) * eps
            lev = lc + m + math.log(Z) - math.log(len(W))  # log evidence of the candidate model so far
            pc = 1.0 if not escape else 1 / (1 + math.exp(min(50, lu - lev)))
            p[b * 24 + t] = pc * q1 + (1 - pc) * 0.5
            lw += np.where(W[:, t] == blk[t], l1e, le)
            lu += math.log(0.5)
    return p


def phase_online(make, train, stream):
    """Online count/CTW observers, one per phase; trained on the training words, scored on the stream."""
    obs = [make() for _ in range(24)]
    hist, p = [], []
    for scored, words in ((False, train), (True, stream)):
        for blk in words:
            for t, x in enumerate(blk):
                o = obs[t]
                if scored:
                    p.append(o.predict(hist))
                o.update(hist, int(x)); hist.append(int(x))
    return np.array(p)


def mlp_observer(train, stream, group, seed, steps):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = nn.Sequential(nn.Linear(72, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    eye = np.eye(24, dtype=np.float32)

    def feats(words, ts):
        mask = (np.arange(24)[None, :] < ts[:, None]).astype(np.float32)
        return torch.tensor(np.concatenate([words * mask, mask, eye[ts]], 1))

    for _ in range(steps):
        w = train[rng.integers(0, len(train), 256)]
        if group is not None:
            w = np.stack([act(group[k], wi) for k, wi in zip(rng.integers(0, len(group), 256), w)])
        ts = rng.integers(0, 24, 256)
        y = torch.tensor(w[np.arange(256), ts], dtype=torch.float32)
        loss = nn.functional.binary_cross_entropy_with_logits(net(feats(w.astype(np.float32), ts)).squeeze(1), y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        S = np.repeat(stream.astype(np.float32), 24, 0); ts = np.tile(np.arange(24), len(stream))
        return torch.sigmoid(net(feats(S, ts)).squeeze(1)).numpy().astype(np.float64)


def gf2_span(words):
    basis = []  # row-reduced over GF(2), words as ints
    for w in words:
        v = int("".join(map(str, w)), 2)
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v)
    span = [0]
    for b in basis:
        span += [s ^ b for s in span]
    return np.array([[s >> (23 - i) & 1 for i in range(24)] for s in span], np.int8)


def ledger(p, stream):
    x = stream.reshape(-1).astype(float)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    d = corp_decomposition(p, x)
    return {"bits": float(-(x * np.log2(p) + (1 - x) * np.log2(1 - p)).mean()), **d}


def section_golay(quick):
    C, gens = golay_code()
    wd = np.bincount(C.sum(1), minlength=25)
    G = closure(gens)
    assert all(preserves(C, g) for g in G[:: max(1, len(G) // 50)])
    octads = C[C.sum(1) == 8]
    oset = {bytes(o): i for i, o in enumerate(octads)}
    orbit_id, n_orb = -np.ones(len(octads), int), 0
    for i in range(len(octads)):
        if orbit_id[i] < 0:
            for w in act(G, np.broadcast_to(octads[i], (len(G), 24)).copy()):
                orbit_id[oset[bytes(w)]] = n_orb
            n_orb += 1
    facts = {"weight_distribution": {int(k): int(v) for k, v in enumerate(wd) if v}, "group_order": int(len(G)),
             "octad_orbits_under_group": sorted(np.bincount(orbit_id).tolist())}
    print("facts:", facts, flush=True)

    Ns = [4, 16, 64, 256, 1024] if not quick else [4, 64]
    eps_list = [0.0, 0.03]
    seeds = [0, 1] if not quick else [0]
    steps = 3000 if not quick else 600
    rows = []
    for eps in eps_list:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            noisy = lambda k: C[rng.integers(0, 4096, k)] ^ (rng.uniform(size=(k, 24)) < eps).astype(np.int8)
            stream = noisy(300 if not quick else 80)
            rows.append({"eps": eps, "seed": seed, "N": None, "observer": "Bayes (knows code)", **ledger(bayes_stream(C, stream, eps), stream)})
            for N in Ns:
                train = noisy(N)
                obs = {
                    "KT-phase": lambda: phase_online(lambda: KT(0), train, stream),
                    "CTW-12-phase": lambda: phase_online(lambda: CTW(12), train, stream),
                    "MLP": lambda: mlp_observer(train, stream, None, seed, steps),
                    "MLP + group augmentation": lambda: mlp_observer(train, stream, G, seed, steps),
                }
                if N <= 64:
                    orb = np.unique(act(G[:, None, :].repeat(N, 1).reshape(-1, 24), np.tile(train, (len(G), 1))), axis=0)
                    obs["orbit-Bayes (symmetry only)"] = lambda orb=orb: bayes_stream(orb, stream, eps, escape=0.5)
                if eps == 0:
                    obs["span-Bayes (linearity only)"] = lambda: bayes_stream(gf2_span(train), stream, eps, escape=0.5)
                for name, f in obs.items():
                    rows.append({"eps": eps, "seed": seed, "N": N, "observer": name, **ledger(f(), stream)})
                print(f"  eps={eps} seed={seed} N={N}: " + "  ".join(
                    f"{r['observer'].split(' ')[0]}{'+g' if 'augment' in r['observer'] else ''} {r['bits']:.3f}"
                    for r in rows if r["eps"] == eps and r["seed"] == seed and r["N"] == N), flush=True)
    return facts, rows


# ── section B: arithmetic sources ────────────────────────────────────────────

def spf_sieve(n):
    s = np.arange(n + 1)
    for i in range(2, int(n ** 0.5) + 1):
        if s[i] == i:
            s[i * i::i][s[i * i::i] == np.arange(i * i, n + 1, i)] = i
    return s


class Multiplicative:
    """Knows complete multiplicativity: f(n) = f(p) f(n/p). Composites are determined by the past;
    primes get a KT estimate over previously seen prime values."""
    name = "multiplicative"

    def __init__(self, spf, start):
        self.spf, self.start, self.val, self.kt = spf, start, {}, [0, 0]

    def predict(self, hist):
        n = self.start + len(hist)
        p = self.spf[n]
        if p != n:
            return 1 - 1e-9 if self.val[p] * self.val[n // p] == 1 else 1e-9
        return (self.kt[1] + 0.5) / (sum(self.kt) + 1)

    def update(self, hist, x):
        n = self.start + len(hist)
        self.val[n] = 1 if x else -1
        if self.spf[n] == n:
            self.kt[x] += 1


def section_arith(quick):
    n = 4096 if quick else 16384
    start = 2
    spf = spf_sieve(start + n + 1)
    lam = np.empty(n, np.int8)
    for i, m in enumerate(range(start, start + n)):
        k, v = 0, m
        while v > 1:
            v //= spf[v]; k += 1
        lam[i] = 1 if k % 2 == 0 else 0
    P = 1_000_003  # (n/P) for n < P via Euler's criterion
    leg = np.array([1 if pow(m, (P - 1) // 2, P) == 1 else 0 for m in range(start, start + n)], np.int8)
    out = {}
    for name, x in [("liouville lambda(n)", lam), ("legendre (n/1000003)", leg)]:
        res = {}
        for obs in [KT(0), KT(8), CTW(16), Multiplicative(spf, start)]:
            p = np.clip(e001.run_observer(obs, x), 1e-9, 1 - 1e-9)
            d = corp_decomposition(p, x.astype(float))
            res[obs.name] = {"bits": float(e001.bits(p, x).mean()), "DSC": d["DSC"], "MCB": d["MCB"]}
        primes = sum(1 for m in range(start, start + n) if spf[m] == m)
        res["prime_density"] = primes / n
        out[name] = res
        print(f"  {name}: " + "  ".join(f"{k} {v['bits']:.3f}" for k, v in res.items() if isinstance(v, dict))
              + f"  | prime density {primes / n:.3f}", flush=True)
    return out


def figure(rows, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, eps in zip(axs, [0.0, 0.03]):
        R = [r for r in rows if r["eps"] == eps]
        bayes = np.mean([r["bits"] for r in R if r["N"] is None])
        for name in sorted({r["observer"] for r in R if r["N"] is not None}):
            Ns = sorted({r["N"] for r in R if r["observer"] == name})
            ax.plot(Ns, [np.mean([r["bits"] for r in R if r["observer"] == name and r["N"] == N]) for N in Ns], "o-", label=name)
        ax.axhline(bayes, color="k", ls="--", label="Bayes (knows the code)")
        ax.set_xscale("log"); ax.set_xlabel("training codewords N"); ax.set_title(f"Golay stream, BSC eps={eps}")
    axs[0].set_ylabel("bits / symbol"); axs[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / f"fig_golay_{tag}.png", dpi=120)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    t0 = time.time(); torch.set_num_threads(4)
    tag = "quick" if a.quick else "full"
    print("A golay", flush=True); facts, rows = section_golay(a.quick)
    print("B arithmetic", flush=True); arith = section_arith(a.quick)
    (OUT / f"results_{tag}.json").write_text(json.dumps({"facts": facts, "golay": rows, "arithmetic": arith,
                                                          "seconds": round(time.time() - t0)}, indent=1))
    figure(rows, tag)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
