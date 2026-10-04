"""E004 — The GL(2) rung: can a learner rediscover Hecke structure from Ramanujan's tau?

Data: tau(n) computed independently from Delta = q * prod (1 - q^n)^24 (pentagonal-number series, exact ints).
Discovery (program search on the FIRST half, n <= N/2, values only):
  R_complete : tau(mn) = tau(m) tau(n) for all m, n            (GL(1)-style complete multiplicativity)
  R_coprime  : tau(mn) = tau(m) tau(n) when gcd(m, n) = 1      (Hecke multiplicativity)
  R_power(w) : tau(p^(k+1)) = tau(p) tau(p^k) - p^(w-1) tau(p^(k-1)), search w in 1..40
Ledger (SECOND half): predict the sign bit of tau(n). Generic observers see sign bits only; rule observers see
the past values and apply a rule, with their confidence in the rule learned online (KT on its hit rate), so a
wrong rule is charged in bits rather than crashing. Primes are 'fresh' for every rule (KT on prime signs).
python run.py   (CPU, ~1 min)
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calib.metrics import corp_decomposition  # noqa: E402

_s = importlib.util.spec_from_file_location("e001", ROOT / "experiments/e001_binary_ground_floor/run.py")
e001 = importlib.util.module_from_spec(_s); _s.loader.exec_module(e001)
OUT = Path(__file__).parent


def ramanujan_tau(N):
    E = {}  # Euler: prod (1 - q^n) = sum_k (-1)^k q^{k(3k-1)/2}, k in Z
    k = 0
    while True:
        done = True
        for kk in (k, -k) if k else (0,):
            e = kk * (3 * kk - 1) // 2
            if e < N:
                E[e] = (-1) ** (kk % 2); done = False
        if done and k:
            break
        k += 1
    a = [1] + [0] * (N - 1)
    sparse = sorted(E.items())
    for _ in range(24):
        b = [0] * N
        for i, ai in enumerate(a):
            if ai:
                for e, s in sparse:
                    if i + e >= N:
                        break
                    b[i + e] += s * ai
        a = b
    return [None] + a[: N - 1]  # tau(n) = coeff of q^(n-1) in prod^24; index from 1


def factor(n):
    f, p = {}, 2
    while p * p <= n:
        while n % p == 0:
            f[p] = f.get(p, 0) + 1; n //= p
        p += 1
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def discover(tau, M):
    rep = {}
    pairs = [(m, n) for m in range(2, M + 1) for n in range(m, M // m + 1)]
    rep["R_complete_holds_fraction"] = sum(tau[m] * tau[n] == tau[m * n] for m, n in pairs) / len(pairs)
    cop = [(m, n) for m, n in pairs if gcd(m, n) == 1]
    rep["R_coprime_holds_fraction"] = sum(tau[m] * tau[n] == tau[m * n] for m, n in cop) / len(cop)
    rep["R_coprime_pairs_tested"] = len(cop)
    primes = [p for p in range(2, M + 1) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    checks = [(p, k) for p in primes for k in range(1, 12) if p ** (k + 1) <= M]
    hits = [w for w in range(1, 41) if all(
        tau[p ** (k + 1)] == tau[p] * tau[p ** k] - p ** (w - 1) * (tau[p ** (k - 1)] if k >= 1 else 0) for p, k in checks)]
    rep["R_power_weights_consistent"] = hits
    rep["R_power_checks"] = len(checks)
    return rep, (hits[0] if len(hits) == 1 else None)


class RuleObserver:
    """Predicts sign(tau(n)) from past values via a rule; confidence in the rule learned online (KT)."""
    def __init__(self, name, tau, start, mode, w=None):
        self.name, self.tau, self.start, self.mode, self.w = name, tau, start, mode, w
        self.hit, self.prime_kt, self._r = [0, 0], [0, 0], None

    def rule(self, n):
        if n == 1:
            return 1  # tau(1) = 1 by normalisation
        f = factor(n)
        if len(f) == 1 and list(f.values())[0] == 1:
            return None  # prime: fresh information
        if self.mode == "complete":
            s = 1
            for p, k in f.items():
                s *= (1 if self.tau[p] > 0 else -1) ** k
            return int(s > 0)
        if len(f) >= 2:  # coprime split into prime powers, values known from the past
            s = 1
            for p, k in f.items():
                s *= 1 if self.tau[p ** k] > 0 else -1
            return int(s > 0)
        if self.mode == "hecke":  # n = p^k, k >= 2: recursion with the discovered weight
            (p, k), = f.items()
            v = self.tau[p] * self.tau[p ** (k - 1)] - p ** (self.w - 1) * self.tau[p ** (k - 2)]
            return int(v > 0)
        return None  # coprime-only observer: prime powers are fresh

    def predict(self, hist):
        n = self.start + len(hist)
        self._r = self.rule(n)
        if self._r is None:
            return (self.prime_kt[1] + 0.5) / (sum(self.prime_kt) + 1)
        c = (self.hit[1] + 0.5) / (sum(self.hit) + 1)  # P(rule correct)
        return c if self._r == 1 else 1 - c

    def update(self, hist, x):
        if self._r is None:
            self.prime_kt[x] += 1
        else:
            self.hit[int(self._r == x)] += 1


def main():
    t0 = time.time()
    N = 4000
    tau = ramanujan_tau(N + 1)
    assert tau[1:6] == [1, -24, 252, -1472, 4830], tau[1:6]
    M = N // 2
    rep, w = discover(tau, M)
    print("discovery on n <=", M, ":", rep, flush=True)
    sign = np.array([1 if tau[n] > 0 else 0 for n in range(1, N + 1)], np.int8)
    zero = [n for n in range(1, N + 1) if tau[n] == 0]
    primes2 = [n for n in range(M + 1, N + 1) if len(factor(n)) == 1 and list(factor(n).values())[0] == 1]
    ppow2 = [n for n in range(M + 1, N + 1) if len(factor(n)) == 1]
    st_pos = np.mean([tau[p] > 0 for p in range(2, N + 1) if len(factor(p)) == 1 and list(factor(p).values())[0] == 1])
    res = {}
    for obs in [e001.KT(0), e001.KT(8), e001.CTW(16),
                RuleObserver("complete-multiplicative", tau, 1, "complete"),
                RuleObserver("coprime-multiplicative", tau, 1, "coprime"),
                RuleObserver(f"hecke (w={w})", tau, 1, "hecke", w)]:
        p = np.clip(e001.run_observer(obs, sign), 1e-12, 1 - 1e-12)
        x2, p2 = sign[M:], p[M:]
        d = corp_decomposition(p2, x2.astype(float))
        res[obs.name] = {"bits": float(e001.bits(p2, x2).mean()), "DSC": d["DSC"], "MCB": d["MCB"]}
        print(f"  {obs.name:>26}: bits {res[obs.name]['bits']:.3f}  DSC {d['DSC']:.3f}  MCB {d['MCB']:.3f}", flush=True)
    out = {"N": N, "discovery": rep, "discovered_weight": w, "ledger_second_half": res,
           "prime_density_second_half": len(primes2) / (N - M), "prime_power_density_second_half": len(ppow2) / (N - M),
           "fraction_tau_p_positive (Sato-Tate predicts ~1/2)": float(st_pos), "tau_zero_n (Lehmer)": zero,
           "seconds": round(time.time() - t0)}
    (OUT / "results_full.json").write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k not in ("ledger_second_half", "discovery")})


if __name__ == "__main__":
    main()
