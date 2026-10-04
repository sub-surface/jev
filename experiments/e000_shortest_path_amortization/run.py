"""E000 — Amortising an exact System-2 into a calibrated System-1, on a toy with a perfect oracle.

Task. Random weighted digraph on N nodes. From node 0 there are K candidate first hops
(nodes 1..K). Question: which first hop lies on the shortest 0 -> N-1 path?
System 2 = exact all-pairs shortest paths (Floyd–Warshall). System 1 = an MLP that sees the
weight matrix once and outputs a distribution over the K hops.

Questions (each a falsifiable claim, see README.md):
  Q1  amortisation: how do S1 accuracy and calibration scale with oracle labels?
  Q2  escalation: what fraction of S2 calls does a confidence gate save at 99% of S2 quality,
      versus random routing?
  Q3  distill-back loop: labelling S1's *least confident* queries (the ones you would have
      escalated anyway) vs random queries, at equal oracle budget — which learns faster, and
      what happens to calibration?
  Q4  shift: train on edge density 0.4, test on 0.25. Does calibration survive? Does the gate?

CPU only, ~minutes. No cloud. Usage:  python experiments/e000_shortest_path_amortization/run.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from calib import escalation, metrics  # noqa: E402
from calib.recalibrate import TemperatureScaler  # noqa: E402

N, K = 10, 4
OUT = Path(__file__).parent


# ── System 2: the oracle ─────────────────────────────────────────────────────

def make_problems(n: int, density: float, rng: np.random.Generator):
    """Return weight matrices W (n,N,N; inf = no edge), labels y (n,), and S2 cost-to-go."""
    W = np.where(rng.uniform(size=(n, N, N)) < density, rng.uniform(0.1, 1.0, (n, N, N)), np.inf)
    W[:, np.arange(N), np.arange(N)] = 0.0
    W[:, 0, :] = np.inf
    W[:, 0, 0] = 0.0
    W[:, 0, 1:K + 1] = rng.uniform(0.1, 1.0, (n, K))  # source reaches exactly the K candidates
    D = W.copy()
    for k in range(N):  # batched Floyd–Warshall: this is the "deliberation"
        D = np.minimum(D, D[:, :, k:k + 1] + D[:, k:k + 1, :])
    via = W[:, 0, 1:K + 1] + D[:, 1:K + 1, N - 1]
    ok = np.isfinite(via).any(1)
    return W[ok], via[ok].argmin(1), via[ok]


def featurise(W: np.ndarray) -> torch.Tensor:
    present = np.isfinite(W)
    return torch.tensor(np.concatenate([np.where(present, W, 0.0), present], 1).reshape(len(W), -1),
                        dtype=torch.float32)


# ── System 1 ─────────────────────────────────────────────────────────────────

def new_model(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(2 * N * N, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, K))


def train(model: nn.Module, X: torch.Tensor, y: np.ndarray, epochs: int, seed: int) -> nn.Module:
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    yt = torch.tensor(y)
    steps = max(1, epochs * len(X) // 128)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(X), (128,), generator=g)
        loss = nn.functional.cross_entropy(model(X[idx]), yt[idx])  # = log score
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model.eval()


@torch.no_grad()
def logits_of(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    return model(X).double().numpy()


def evaluate(model, ts, X, y) -> dict:
    p = ts(logits_of(model, X))
    r = metrics.report(p, y, ci=False)
    conf, ok = metrics.top_label(p, y)
    curve = escalation.escalation_curve(conf, ok, np.ones_like(ok))
    r["s2_calls_needed_gated"] = escalation.escalation_needed(curve, 0.99)
    r["s2_calls_needed_random"] = escalation.escalation_needed(curve, 0.99, key="random")
    r["s2_calls_needed_oracle"] = escalation.escalation_needed(curve, 0.99, key="oracle")
    return r


def fit_with_temperature(X, y, seed, epochs, val_frac=0.1):
    n_val = max(200, int(len(X) * val_frac))
    m = train(new_model(seed), X[n_val:], y[n_val:], epochs, seed)
    ts = TemperatureScaler().fit(logits_of(m, X[:n_val]), y[:n_val])
    return m, ts


def identity_ts():
    ts = TemperatureScaler(); ts.T = 1.0
    return ts


# ── experiments ──────────────────────────────────────────────────────────────

def q1_q2_q4(seeds, sizes, epochs, n_test):
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        Wte, yte, _ = make_problems(n_test, 0.40, rng)
        Wsh, ysh, _ = make_problems(n_test, 0.25, rng)
        Xte, Xsh = featurise(Wte), featurise(Wsh)
        Wtr, ytr, _ = make_problems(max(sizes) + 2000, 0.40, rng)
        Xtr = featurise(Wtr)
        for n in sizes:
            m, ts = fit_with_temperature(Xtr[:n], ytr[:n], seed, epochs)
            for split, X, y in [("iid", Xte, yte), ("shift", Xsh, ysh)]:
                for cal, t in [("raw", identity_ts()), ("temp", ts)]:
                    rows.append({"seed": seed, "n_labels": n, "split": split, "cal": cal, "T": ts.T,
                                 **evaluate(m, t, X, y)})
            print(f"  seed {seed} n={n:>6}: iid acc {rows[-3]['acc']:.3f} smECE {rows[-3]['smece']:.3f} "
                  f"S2 needed {rows[-3]['s2_calls_needed_gated']:.2f} | shift acc {rows[-1]['acc']:.3f} "
                  f"smECE {rows[-1]['smece']:.3f}", flush=True)
    return rows


def q3_loop(seeds, n0, rounds, per_round, pool, epochs, n_test):
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(2000 + seed)
        Wte, yte, _ = make_problems(n_test, 0.40, rng)
        Xte = featurise(Wte)
        W0, y0, _ = make_problems(n0, 0.40, rng)
        for strategy in ["least_confident", "random"]:
            srng = np.random.default_rng(3000 + seed)
            X, y = featurise(W0), y0.copy()
            m, ts = fit_with_temperature(X, y, seed, epochs)
            for r in range(rounds + 1):
                res = evaluate(m, ts, Xte, yte)
                rows.append({"seed": seed, "strategy": strategy, "round": r, "oracle_labels": len(y), **res})
                if r == rounds:
                    break
                Wp, yp, _ = make_problems(pool, 0.40, srng)  # a fresh stream of queries
                Xp = featurise(Wp)
                conf = ts(logits_of(m, Xp)).max(1)
                pick = np.argsort(conf)[:per_round] if strategy == "least_confident" \
                    else srng.choice(len(Xp), per_round, replace=False)
                X, y = torch.cat([X, Xp[pick]]), np.concatenate([y, yp[pick]])
                m, ts = fit_with_temperature(X, y, seed, epochs)
            print(f"  seed {seed} {strategy:>15}: final acc {rows[-1]['acc']:.3f} smECE {rows[-1]['smece']:.3f}",
                  flush=True)
    return rows


def summarise(rows, keys, metrics_=("acc", "smece", "ece_l2_debiased", "nll", "s2_calls_needed_gated",
                                     "s2_calls_needed_random", "s2_calls_needed_oracle")):
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for g, rs in sorted(groups.items()):
        d = dict(zip(keys, g)); d["n_seeds"] = len(rs)
        for m in metrics_:
            v = np.array([r[m] for r in rs])
            d[m] = [round(float(v.mean()), 4), round(float(v.std()), 4)]
        out.append(d)
    return out


def plot(summary_q1, summary_q3, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for split, style in [("iid", "-"), ("shift", "--")]:
        s = [d for d in summary_q1 if d["split"] == split and d["cal"] == "temp"]
        n = [d["n_labels"] for d in s]
        ax[0].errorbar(n, [d["acc"][0] for d in s], [d["acc"][1] for d in s], ls=style, marker="o", label=f"acc ({split})")
        ax[1].errorbar(n, [d["smece"][0] for d in s], [d["smece"][1] for d in s], ls=style, marker="o", label=f"smECE ({split})")
        ax[2].errorbar(n, [d["s2_calls_needed_gated"][0] for d in s], [d["s2_calls_needed_gated"][1] for d in s],
                       ls=style, marker="o", label=f"gated ({split})")
    s = [d for d in summary_q1 if d["split"] == "iid" and d["cal"] == "temp"]
    ax[2].plot([d["n_labels"] for d in s], [d["s2_calls_needed_random"][0] for d in s], "k:", label="random gate (iid)")
    ax[2].plot([d["n_labels"] for d in s], [d["s2_calls_needed_oracle"][0] for d in s], "g:", label="oracle gate (iid)")
    for a, t in zip(ax, ["S1 accuracy", "S1 smooth ECE (after temp. scaling)", "fraction of queries escalated to S2\nto reach 99% of S2 accuracy"]):
        a.set_xscale("log"); a.set_xlabel("oracle labels used for amortisation"); a.set_title(t); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for strat in ["least_confident", "random"]:
        s = [d for d in summary_q3 if d["strategy"] == strat]
        x = [d["round"] for d in s]
        ax[0].errorbar(x, [d["acc"][0] for d in s], [d["acc"][1] for d in s], marker="o", label=strat)
        ax[1].errorbar(x, [d["smece"][0] for d in s], [d["smece"][1] for d in s], marker="o", label=strat)
    ax[0].set_title("distill-back loop: S1 accuracy"); ax[1].set_title("distill-back loop: smooth ECE")
    for a in ax:
        a.set_xlabel("round (equal oracle labels per round)"); a.legend()
    fig.tight_layout(); fig.savefig(str(path).replace(".png", "_loop.png"), dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    t0 = time.time()
    if a.quick:
        cfg = dict(seeds=[0], sizes=[500, 4000], epochs=20, n_test=2000, n0=500, rounds=2, per_round=500, pool=4000)
    else:
        cfg = dict(seeds=[0, 1, 2], sizes=[250, 1000, 4000, 16000, 64000], epochs=30, n_test=5000,
                   n0=1000, rounds=6, per_round=1000, pool=10000)
    print("Q1/Q2/Q4: amortisation, escalation, shift", flush=True)
    r1 = q1_q2_q4(cfg["seeds"], cfg["sizes"], cfg["epochs"], cfg["n_test"])
    print("Q3: distill-back loop", flush=True)
    r3 = q3_loop(cfg["seeds"], cfg["n0"], cfg["rounds"], cfg["per_round"], cfg["pool"], cfg["epochs"], cfg["n_test"])
    s1, s3 = summarise(r1, ["n_labels", "split", "cal"]), summarise(r3, ["strategy", "round"])
    tag = "quick" if a.quick else "full"
    (OUT / f"results_{tag}.json").write_text(json.dumps({"config": cfg, "seconds": round(time.time() - t0),
                                                          "q1_q2_q4": s1, "q3": s3, "raw_q1": r1, "raw_q3": r3},
                                                         indent=1, default=float))
    plot(s1, s3, OUT / f"fig_{tag}.png")
    print(f"done in {time.time() - t0:.0f}s -> results_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
