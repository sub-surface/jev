"""E001 — The ground floor: calibrated prediction of binary sequences.

Every number here is a code length (bits) because log loss on bits *is* arithmetic-coding length.
For each (source, observer) pair we report the prequential code length and its exact CORP split
    bits/symbol = MCB (calibration cost) - DSC (bits extracted) + UNC (base-rate entropy)
plus a learning-cost proxy (area of the loss curve above its own asymptote, cf. epiplexity).

Sources span the space a bounded observer cares about: pure noise, learnable finite-state structure,
deterministic-but-aperiodic, deterministic-but-pseudorandom (computation creates information for
bounded observers), chaotic, number-theoretic (Minkowski ?: Stern–Brocot paths = binary expansions),
and a regime shift. Observers form a compute ladder: KT order-k estimators and Context Tree Weighting
(an exact Bayesian mixture over all tree sources of depth <= D — inference over structure).

Sections:  A ledger · B jumps (posterior concentration vs. slow counting) · C shift
           D escalation S1->S2 · E competition: diagonal adversaries and a log-score market
CPU only, a few minutes.  Usage: python run.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from calib.metrics import corp_decomposition  # noqa: E402

OUT = Path(__file__).parent
LN2 = math.log(2)


# ── sources ──────────────────────────────────────────────────────────────────

def src_bernoulli(n, rng):
    return (rng.uniform(size=n) < 0.3).astype(np.int8)


def src_markov3(n, rng):
    P = rng.beta(0.5, 0.5, size=8)  # random order-3 chain
    x = list(rng.integers(0, 2, 3))
    for _ in range(n - 3):
        x.append(int(rng.uniform() < P[x[-3] * 4 + x[-2] * 2 + x[-1]]))
    return np.array(x, np.int8)


def src_periodic_noise(n, rng):
    pat = rng.integers(0, 2, 23)
    return (np.resize(pat, n) ^ (rng.uniform(size=n) < 0.05)).astype(np.int8)


def src_thue_morse(n, rng):
    off = int(rng.integers(0, 10_000))
    return np.array([bin(i + off).count("1") & 1 for i in range(n)], np.int8)


def src_pseudorandom(n, rng):
    """Top bit of a 32-bit LCG: deterministic, ~64 bits of description, ~1 bit/symbol to bounded observers."""
    s, a, c = int(rng.integers(1, 2**32)), 1664525, 1013904223
    out = np.empty(n, np.int8)
    for i in range(n):
        s = (a * s + c) & 0xFFFFFFFF
        out[i] = s >> 31
    return out


def src_logistic(n, rng):
    x, out = float(rng.uniform(0.1, 0.9)), np.empty(n, np.int8)
    for i in range(n):
        x = 3.99 * x * (1 - x)
        out[i] = x > 0.5
    return out


def src_minkowski(n, rng):
    """Binary expansions of ?(u), u ~ U(0,1): runs of alternating bits with lengths = continued-fraction
    digits of u (Gauss–Kuzmin distributed). This is u's Stern–Brocot path written in binary."""
    out, b = [], 0
    while len(out) < n:
        u = rng.uniform()
        for _ in range(12):  # first 12 partial quotients are numerically reliable
            u = 1.0 / u
            a = int(u); u -= a
            if a == 0 or u < 1e-9:
                break
            out.extend([b] * min(a, 64)); b ^= 1
    return np.array(out[:n], np.int8)


def src_regime_shift(n, rng):
    a, b = src_markov3(n // 2, rng), src_markov3(n - n // 2, np.random.default_rng(rng.integers(1 << 30)))
    return np.concatenate([a, b])


SOURCES = {"bernoulli(0.3)": src_bernoulli, "markov-3": src_markov3, "periodic-23+5%": src_periodic_noise,
           "thue-morse": src_thue_morse, "pseudorandom (LCG)": src_pseudorandom, "logistic map": src_logistic,
           "minkowski ?(u)": src_minkowski, "regime shift": src_regime_shift}


# ── observers (sequential predictors). each returns p_t = P(x_t = 1 | x_<t) ───

class KT:
    """Krichevsky–Trofimov estimator on the last k bits. Compute ~ O(1); memory ~ 2^k."""
    def __init__(self, k): self.k, self.c, self.name = k, {}, f"KT-{k}"

    def predict(self, hist):
        a, b = self.c.get(tuple(hist[-self.k:]) if self.k else (), (0, 0))
        return (b + 0.5) / (a + b + 1.0)

    def update(self, hist, x):
        key = tuple(hist[-self.k:]) if self.k else ()
        a, b = self.c.get(key, (0, 0))
        self.c[key] = (a + (x == 0), b + (x == 1))


class CTW:
    """Context Tree Weighting (Willems, Shtarkov, Tjalkens 1995), depth D. Exact Bayesian mixture over
    all binary tree sources of depth <= D; redundancy <= model cost + (|S|/2) log n + |S| bits.
    Conditional predictions use the 'stop here' posterior beta_s = Pe(s)/2 / Pw(s) at each node."""
    def __init__(self, D):
        self.D, self.name = D, f"CTW-{D}"
        self.cnt, self.le, self.lw = {}, {}, {}  # counts, log Pe, log Pw (natural log)

    def _path(self, hist):
        ctx = list(hist[-self.D:])[::-1] + [0] * max(0, self.D - len(hist))
        return [tuple(ctx[:d]) for d in range(self.D + 1)]

    def predict(self, hist):
        path, p = self._path(hist), None
        for d in range(self.D, -1, -1):
            s = path[d]
            a, b = self.cnt.get(s, (0, 0))
            pe = (b + 0.5) / (a + b + 1.0)
            if d == self.D or p is None:
                p = pe
            else:
                beta = math.exp(math.log(0.5) + self.le.get(s, 0.0) - self.lw.get(s, 0.0))
                p = beta * pe + (1 - beta) * p
        return p

    def update(self, hist, x):
        path = self._path(hist)
        for d in range(self.D, -1, -1):
            s = path[d]
            a, b = self.cnt.get(s, (0, 0))
            self.le[s] = self.le.get(s, 0.0) + math.log(((b if x else a) + 0.5) / (a + b + 1.0))
            self.cnt[s] = (a + (x == 0), b + (x == 1))
            if d == self.D:
                self.lw[s] = self.le[s]
            else:
                c0, c1 = s + (0,), s + (1,)
                lc = self.lw.get(c0, 0.0) + self.lw.get(c1, 0.0)
                m = max(self.le[s], lc)
                self.lw[s] = m + math.log(0.5 * math.exp(self.le[s] - m) + 0.5 * math.exp(lc - m))


def run_observer(obs, x):
    p, hist = np.empty(len(x)), []
    for t, xt in enumerate(x):
        p[t] = obs.predict(hist)
        obs.update(hist, int(xt)); hist.append(int(xt))
    return p


def bits(p, x):
    return -np.log2(np.where(x == 1, p, 1 - p))


def observers(D):
    return [KT(0), KT(1), KT(2), KT(4), KT(8), KT(12), CTW(D)]


# ── sections ─────────────────────────────────────────────────────────────────

def ledger(n, seeds, D):
    rows, curves = [], {}
    for name, f in SOURCES.items():
        for seed in seeds:
            x = f(n, np.random.default_rng(seed))
            for obs in observers(D):
                p = run_observer(obs, x)
                l = bits(p, x)
                tail = l[-n // 4:].mean()
                corp = corp_decomposition(p, x.astype(float), "log")
                rows.append({"source": name, "seed": seed, "observer": obs.name, "bits_per_symbol": float(l.mean()),
                             "asymptote": float(tail), "learning_cost_bits": float((l - tail).sum()), **corp})
                if seed == seeds[0]:
                    curves[(name, obs.name)] = l
    return rows, curves


def shift_calibration(n, seeds, D):
    """Calibration cost (MCB) and bits in windows before/after the regime change."""
    out = []
    for seed in seeds:
        x = src_regime_shift(n, np.random.default_rng(seed))
        for obs in [KT(3), CTW(D)]:
            p = run_observer(obs, x)
            for lo, hi, tag in [(n // 4, n // 2, "before"), (n // 2, n // 2 + n // 8, "just after"), (3 * n // 4, n, "late after")]:
                d = corp_decomposition(p[lo:hi], x[lo:hi].astype(float))
                out.append({"seed": seed, "observer": obs.name, "window": tag, **d})
    return out


def escalation(n, seeds, D):
    """S1 = KT-2 (cheap) answers; least-confident fraction q escalated to S2 = CTW-D. Quality = bits/symbol."""
    res = {}
    for name, f in SOURCES.items():
        need, saves = {"gated": [], "random": [], "oracle": []}, []
        for seed in seeds:
            x = f(n, np.random.default_rng(seed))
            p1 = run_observer(KT(2), x)
            l1, l2 = bits(p1, x), bits(run_observer(CTW(D), x), x)
            saves.append(l1.mean() - l2.mean())
            conf = np.abs(p1 - 0.5)
            gain = l1 - l2
            target = l2.mean() + 0.01 * (l1.mean() - l2.mean())  # recover 99% of the bits S2 saves
            qs = np.linspace(0, 1, 201)
            orders = {"gated": np.argsort(conf, kind="stable"), "oracle": np.argsort(-gain, kind="stable"),
                      "random": np.random.default_rng(seed + 99).permutation(n)}
            for k, o in orders.items():
                cum = l1.mean() - np.concatenate([[0], np.cumsum(gain[o])])[np.round(qs * n).astype(int)] / n
                hit = np.nonzero(cum <= target + 1e-12)[0]
                need[k].append(float(qs[hit[0]]) if len(hit) else 1.0)
        res[name] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in need.items()}
        res[name]["s2_saves_bits_per_symbol"] = float(np.mean(saves))
        if np.mean(saves) < 0.01:  # S2 has nothing to add: escalation is pointless, not "needed 100%"
            res[name]["note"] = "S2 saves <0.01 bits/symbol; never escalate"
    return res


def competition(n, D):
    """Diagonal adversaries: for predictor A, emit the bit A thinks less likely. Then score everyone on it.
    'market' = Bayesian mixture (log-score market) over {KT-0, KT-4, CTW-D} with weights ∝ past likelihood."""
    def make(name):
        return {"KT-0": lambda: KT(0), "KT-4": lambda: KT(4), f"CTW-{D}": lambda: CTW(D), "market": lambda: Market(D)}[name]

    names = ["KT-0", "KT-4", f"CTW-{D}", "market"]
    seqs = {}
    for a in names:
        obs, hist = make(a)(), []
        for _ in range(n):
            xt = int(obs.predict(hist) < 0.5)
            obs.update(hist, xt); hist.append(xt)
        seqs[a] = np.array(hist, np.int8)
    return {a: {b: float(bits(run_observer(make(b)(), seqs[a]), seqs[a]).mean()) for b in names} for a in names}


class Market:
    """Exchange between predictors: a log-score (LMSR-equivalent) market = Bayesian model averaging."""
    def __init__(self, D):
        self.m, self.lw, self.name = [KT(0), KT(4), CTW(D)], np.zeros(3), "market"

    def predict(self, hist):
        self._p = np.array([m.predict(hist) for m in self.m])
        w = np.exp(self.lw - self.lw.max()); w /= w.sum()
        return float(np.clip(w @ self._p, 1e-12, 1 - 1e-12))

    def update(self, hist, x):
        self.lw += np.log(np.where(x == 1, self._p, 1 - self._p))
        for m in self.m:
            m.update(hist, x)


# ── figures ──────────────────────────────────────────────────────────────────

def figures(rows, curves, shift, D, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    obs_names = [o.name for o in observers(D)]
    srcs = list(SOURCES)
    agg = lambda s, o, k: np.mean([r[k] for r in rows if r["source"] == s and r["observer"] == o])

    fig, axs = plt.subplots(2, 4, figsize=(17, 7.5), sharey=True)
    for ax, s in zip(axs.flat, srcs):
        unc = [agg(s, o, "UNC") for o in obs_names]; dsc = [agg(s, o, "DSC") for o in obs_names]
        mcb = [agg(s, o, "MCB") for o in obs_names]; tot = [agg(s, o, "score") for o in obs_names]
        xs = np.arange(len(obs_names))
        ax.bar(xs, np.array(unc) - np.array(dsc), color="#4c72b0", label="UNC − DSC (bits not extracted)")
        ax.bar(xs, mcb, bottom=np.array(unc) - np.array(dsc), color="#dd8452", label="MCB (calibration cost)")
        ax.plot(xs, unc, "k_", ms=18, label="UNC (base-rate entropy)")
        ax.plot(xs, tot, "w.", ms=6)
        ax.set_xticks(xs, obs_names, rotation=45, fontsize=8); ax.set_title(s, fontsize=10); ax.set_ylim(0, 1.15)
    axs[0, 0].set_ylabel("bits / symbol"); axs[1, 0].set_ylabel("bits / symbol")
    axs[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Code length = UNC − DSC + MCB, per source × observer (compute ladder →)")
    fig.tight_layout(); fig.savefig(OUT / f"fig_ledger_{tag}.png", dpi=120)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    for ax, s in zip(axs, ["markov-3", "thue-morse", "regime shift"]):
        for o in ["KT-2", "KT-4", "KT-12", f"CTW-{D}"]:
            l = curves[(s, o)]
            k = 64
            ax.plot(np.convolve(l, np.ones(k) / k, "valid"), label=o, lw=1)
        ax.set_title(f"{s}: running bits/symbol (window 64)"); ax.set_xscale("log"); ax.set_xlabel("t"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / f"fig_jumps_{tag}.png", dpi=120)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    t0 = time.time()
    n, seeds, D = (2048, [0], 8) if a.quick else (8192, [0, 1, 2], 12)
    tag = "quick" if a.quick else "full"
    print("A ledger", flush=True); rows, curves = ledger(n, seeds, D)
    print("C shift", flush=True); sh = shift_calibration(n, seeds, D)
    print("D escalation", flush=True); esc = escalation(n, seeds, D)
    print("E competition", flush=True); comp = competition(min(n, 4096), D)
    summary = {}
    for r in rows:
        summary.setdefault(r["source"], {}).setdefault(r["observer"], []).append(r)
    table = {s: {o: {k: round(float(np.mean([r[k] for r in rs])), 4) for k in
                     ["bits_per_symbol", "asymptote", "learning_cost_bits", "MCB", "DSC", "UNC"]}
                 for o, rs in d.items()} for s, d in summary.items()}
    (OUT / f"results_{tag}.json").write_text(json.dumps(
        {"config": {"n": n, "seeds": seeds, "ctw_depth": D}, "seconds": round(time.time() - t0),
         "ledger": table, "shift": sh, "escalation": esc, "competition": comp, "raw": rows}, indent=1))
    figures(rows, curves, sh, D, tag)
    for s, d in table.items():
        print(f"{s:>20}: " + "  ".join(f"{o} {v['bits_per_symbol']:.3f}" for o, v in d.items()))
    print("escalation (fraction to S2 for 99%):", {s: v["gated"][0] for s, v in esc.items()})
    print("competition (row = adversary of, col = observer):"); [print(f"  {k:>8}: {v}") for k, v in comp.items()]
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
