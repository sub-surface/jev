"""E006 — Elliptic curves: CM curves are GL(1) in disguise; non-CM curves carry genuinely GL(2) information.

a_p(E) = p + 1 - #E(F_p) = -sum_x ((f(x)/p)) for y^2 = f(x), computed by exact Legendre-symbol sums for all
good primes 5 <= p < P. Curves:
  CM by Z[i]   : y^2 = x^3 - x            (a_p = 0 for p = 3 mod 4)
  CM by Z[w]   : y^2 = x^3 + 1            (a_p = 0 for p = 2 mod 3)
  non-CM       : y^2 = x^3 - x + 1,  11a3: (2y+1)^2 = 4x^3 - 4x^2 + 1
Streams over primes: z_p = 1[a_p = 0] ("supersingular"), s_p = 1[a_p > 0] among a_p != 0.
Observers (online ledger, second half scored): KT-0, CTW-16, residue (KT per class p mod m, m chosen by MDL on
the first half among 1..24), and for Z[i]: a Hecke-character observer (KT per class of (a mod 4, b mod 4) with
p = a^2 + b^2, a odd) — i.e. an observer whose features are the GL(1) data over Q(i).
Also: Sato–Tate distributions of a_p / (2 sqrt p).   python run.py  (CPU, ~1 min)
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
from calib.metrics import corp_decomposition  # noqa: E402

_s = importlib.util.spec_from_file_location("e001", ROOT / "experiments/e001_binary_ground_floor/run.py")
e001 = importlib.util.module_from_spec(_s); _s.loader.exec_module(e001)
OUT = Path(__file__).parent

CURVES = {  # polynomial coefficients of f(x) = c3 x^3 + c2 x^2 + c1 x + c0
    "CM Z[i]: y^2=x^3-x": (1, 0, -1, 0),
    "CM Z[w]: y^2=x^3+1": (1, 0, 0, 1),
    "non-CM: y^2=x^3-x+1": (1, 0, -1, 1),
    "non-CM 11a3": (4, -4, 0, 1),
}


def primes_upto(n):
    s = np.ones(n, bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.nonzero(s)[0]


def a_p(coeffs, p):
    x = np.arange(p, dtype=np.int64)
    c3, c2, c1, c0 = coeffs
    f = (((c3 * x % p) * x % p + c2 * x) % p * x % p + c1 * x + c0) % p
    sq = np.zeros(p, np.int8); sq[(x * x) % p] = 1
    leg = np.where(f == 0, 0, np.where(sq[f] == 1, 1, -1))
    return int(-leg.sum())


def disc_bad(coeffs, p):
    """True if f has a repeated root mod p (bad reduction): gcd(f, f') nontrivial, checked by brute force."""
    x = np.arange(p, dtype=np.int64)
    c3, c2, c1, c0 = coeffs
    f = (((c3 * x % p) * x % p + c2 * x) % p * x % p + c1 * x + c0) % p
    df = ((3 * c3 * x % p) * x % p + 2 * c2 * x + c1) % p
    return bool(np.any((f == 0) & (df == 0)))


def two_squares(p):
    """p = a^2 + b^2 with a odd, for p = 1 mod 4 (Cornacchia via brute force over a)."""
    for a in range(1, int(math.isqrt(p)) + 1, 2):
        b2 = p - a * a
        b = math.isqrt(b2)
        if b * b == b2:
            return a, b
    return None


class ClassKT:
    """KT estimator per discrete feature class (feature function of the prime p)."""
    def __init__(self, name, feat, ps):
        self.name, self.feat, self.ps, self.c = name, feat, ps, {}

    def predict(self, hist):
        a, b = self.c.get(self.feat(self.ps[len(hist)]), (0, 0))
        return (b + 0.5) / (a + b + 1)

    def update(self, hist, x):
        k = self.feat(self.ps[len(hist)]); a, b = self.c.get(k, (0, 0))
        self.c[k] = (a + (x == 0), b + (x == 1))


def mdl_modulus(bits, ps):
    """Choose m in 1..24 minimising prequential code length of KT-per-(p mod m) on the given bits."""
    best = None
    for m in range(1, 25):
        o = ClassKT(f"mod{m}", lambda p, m=m: p % m, ps)
        L = float(e001.bits(e001.run_observer(o, bits), bits).sum())
        if best is None or L < best[1]:
            best = (m, L)
    return best[0]


def main():
    t0 = time.time()
    P = 40000
    ps_all = primes_upto(P); ps_all = ps_all[ps_all >= 5]
    res, st = {}, {}
    for name, co in CURVES.items():
        ps = np.array([p for p in ps_all if not disc_bad(co, int(p))])
        ap = np.array([a_p(co, int(p)) for p in ps])
        assert np.all(np.abs(ap) <= 2 * np.sqrt(ps) + 1e-9), "Hasse bound violated"
        st[name] = (ap / (2 * np.sqrt(ps))).tolist()
        half = len(ps) // 2
        z = (ap == 0).astype(np.int8)
        nz = ap != 0
        s, ps_s = (ap[nz] > 0).astype(np.int8), ps[nz]
        out = {"n_primes": int(len(ps)), "frac_supersingular": float(z.mean()), "frac_positive": float(s.mean())}
        for stream, x, pp in [("z (a_p = 0)", z, ps), ("s (sign a_p)", s, ps_s)]:
            h = len(x) // 2
            m = mdl_modulus(x[:h], pp[:h])
            obs = [e001.KT(0), e001.CTW(16), ClassKT(f"residue p mod {m}", lambda p, m=m: p % m, pp)]
            if name.startswith("CM Z[i]") and stream.startswith("s"):
                obs.append(ClassKT("Hecke character (a,b mod 4)", lambda p: (lambda ab: (ab[0] % 4, ab[1] % 4) if ab else "inert")(two_squares(int(p))) if p % 4 == 1 else "inert", pp))
            row = {}
            for o in obs:
                p_ = np.clip(e001.run_observer(o, x), 1e-12, 1 - 1e-12)
                d = corp_decomposition(p_[h:], x[h:].astype(float))
                row[o.name] = {"bits": float(e001.bits(p_[h:], x[h:]).mean()), "DSC": d["DSC"], "MCB": d["MCB"]}
            out[stream] = {"mdl_modulus": m, "observers": row}
            print(f"{name:>22} | {stream:>13} | m*={m:>2} | " + "  ".join(f"{k}: {v['bits']:.3f}" for k, v in row.items()), flush=True)
        res[name] = out
    (OUT / "results_full.json").write_text(json.dumps({"P": P, "results": res, "sato_tate": st,
                                                        "seconds": round(time.time() - t0)}))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
