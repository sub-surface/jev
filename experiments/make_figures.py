"""Publication figures for E000–E004. Reads ONLY saved results_*.json (plus one cheap recomputation of
E001 loss curves, deterministic), writes figures/figNN_*.png + .pdf. python experiments/make_figures.py"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP, FIG = ROOT / "experiments", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# validated categorical palette (dataviz reference instance, light mode; validator: all checks pass)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]  # one hue, light -> dark
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
MK = ["o", "s", "^", "D", "v", "P"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelcolor": INK2, "axes.edgecolor": "#c9c8c2", "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 6,
    "legend.frameon": False, "legend.fontsize": 8, "text.color": INK,
})


def load(path):
    return json.loads((EXP / path).read_text(encoding="utf-8"))


def save(fig, name, caption):
    fig.text(0.01, 0.005, caption, fontsize=7.5, color=INK2, ha="left", va="bottom", wrap=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ── Fig 1 · E001 bits ledger ─────────────────────────────────────────────────
def fig_ledger():
    r = load("e001_binary_ground_floor/results_full.json")["ledger"]
    obs = ["KT-0", "KT-2", "KT-4", "KT-8", "KT-12", "CTW-12"]
    src = list(r)
    fig, axs = plt.subplots(2, 4, figsize=(12, 5.6), sharey=True)
    for ax, s in zip(axs.flat, src):
        unc = np.array([r[s][o]["UNC"] for o in obs]); dsc = np.array([r[s][o]["DSC"] for o in obs])
        mcb = np.array([r[s][o]["MCB"] for o in obs]); x = np.arange(len(obs))
        ax.bar(x, unc - dsc, 0.72, color=C[0], label="not extracted  (UNC − DSC)")
        ax.bar(x, dsc, 0.72, bottom=unc - dsc, color="none", edgecolor=INK2, hatch="////", linewidth=0.0,
               label="extracted  (DSC), up to the entropy tick")
        ax.bar(x, mcb, 0.72, bottom=unc - dsc, color=C[1], label="miscalibration  (MCB): bar top = total bits",
               edgecolor=SURF, linewidth=1)
        ax.hlines(unc, x - 0.36, x + 0.36, color=INK, linewidth=1)
        best = int(np.argmin(unc - dsc + mcb))
        ax.annotate(f"{(unc - dsc + mcb)[best]:.2f}", (best, (unc - dsc + mcb)[best]), xytext=(0, -11),
                    textcoords="offset points", ha="center", fontsize=7, color=SURF, fontweight="bold")
        ax.set_title(s, loc="left"); ax.set_xticks(x, obs, rotation=50, fontsize=7.5); ax.set_ylim(0, 1.2)
        ax.grid(axis="x", visible=False)
    for a in axs[:, 0]:
        a.set_ylabel("bits / symbol")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=3, bbox_to_anchor=(0.995, 1.02))
    fig.suptitle("Fig. 1 · Every prediction is a code: bits/symbol = UNC − DSC + MCB  (exact CORP split, log score)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(fig, "fig01_ledger", "E001. 8 sources × compute ladder (KT order-k estimators → CTW-12, a Bayesian mixture over all context "
         "trees). Black tick = UNC (base-rate entropy); white number = best observer. Inference over structure (CTW) matches the "
         "best fixed order everywhere; over-large fixed orders pay in MCB; the pseudorandom LCG yields DSC = 0 to all bounded observers. "
         "n = 8192, 3 seeds.")


# ── Fig 2 · E001 jumps (recomputed, deterministic) ───────────────────────────
def fig_jumps():
    spec = importlib.util.spec_from_file_location("e001", EXP / "e001_binary_ground_floor/run.py")
    e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, s in zip(axs, ["markov-3", "thue-morse", "regime shift"]):
        x = e.SOURCES[s](8192, np.random.default_rng(0))
        ends = []
        for i, obs in enumerate([e.KT(2), e.KT(12), e.CTW(12)]):
            l = e.bits(e.run_observer(obs, x), x)
            cum = np.cumsum(l) / np.arange(1, len(l) + 1)
            ax.plot(np.arange(1, len(l) + 1), cum, color=C[[0, 1, 2][i]], label=obs.name, lw=1.8)
            ends.append((cum[-1], obs.name))
        ends.sort()
        for j, (yv, nm) in enumerate(ends):  # de-collide end labels: keep >= 0.06 apart
            y = yv if j == 0 else max(yv, last + 0.06)
            ax.annotate(nm, (len(x), yv), xytext=(len(x) * 1.15, y), fontsize=7.5, color=INK2, va="center")
            last = y
        if s == "regime shift":
            ax.axvline(4096, color=INK2, lw=1, ls=":"); ax.text(4200, 1.05, "shift", fontsize=7.5, color=INK2)
        ax.set_xscale("log"); ax.set_title(s, loc="left"); ax.set_xlabel("symbols seen  (log)"); ax.set_xlim(8, 2.2e4)
    axs[0].set_ylabel("cumulative bits / symbol"); axs[0].set_ylim(0, 1.15)
    fig.suptitle("Fig. 2 · Inference jumps: a Bayesian mixture over structures commits fast; big tables learn slowly",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save(fig, "fig02_jumps", "E001, seed 0. Cumulative code length per symbol. CTW-12 tracks the best fixed order without being "
         "told it; KT-12 pays a long learning cost (memory without inference).")


# ── Fig 3 · E001 shift, escalation, competition ──────────────────────────────
def fig_dynamics():
    r = load("e001_binary_ground_floor/results_full.json")
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1, 1.4, 1.1]})
    ax = axs[0]
    wins = ["before", "just after", "late after"]
    for i, o in enumerate(["KT-3", "CTW-12"]):
        m = [np.mean([s["MCB"] for s in r["shift"] if s["observer"] == o and s["window"] == w]) for w in wins]
        ax.bar(np.arange(3) + (i - 0.5) * 0.38, m, 0.36, color=C[i], label=o, edgecolor=SURF, linewidth=1)
        for j, v in enumerate(m):
            ax.text(j + (i - 0.5) * 0.38, v + 0.01, f"{v:.3f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(range(3), wins); ax.set_ylabel("MCB (bits / symbol)"); ax.legend(); ax.grid(axis="x", visible=False)
    ax.set_title("A · Calibration breaks at a shift", loc="left")

    ax = axs[1]
    esc = {k: v for k, v in r["escalation"].items() if "note" not in v}
    names = list(esc)
    for i, (k, lab) in enumerate([("oracle", "perfect gate"), ("gated", "confidence gate"), ("random", "random")]):
        ax.scatter([esc[n][k][0] for n in names], np.arange(len(names)), color=[C[2], C[0], INK2][i], marker=MK[i],
                   s=42, label=lab, zorder=3)
    for y, n in enumerate(names):
        ax.hlines(y, esc[n]["oracle"][0], esc[n]["random"][0], color=GRID, lw=3, zorder=1)
    ax.set_yticks(range(len(names)), names); ax.set_xlim(0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.set_xlabel("fraction of symbols escalated to S2 to recover 99% of its savings")
    ax.set_title("B · Confidence is a weak proxy for value of computation", loc="left")

    ax = axs[2]
    comp = r["competition"]; keys = list(comp)
    M = np.array([[comp[a][b] for b in keys] for a in keys])
    im = ax.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", SEQ), vmin=0, vmax=1.05)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color=SURF if M[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(keys)), keys, rotation=30); ax.set_yticks(range(len(keys)), [f"anti-{k}" for k in keys])
    ax.set_xlabel("observer"); ax.grid(False); ax.set_title("C · Every predictor has an adversary (bits/symbol)", loc="left")
    fig.suptitle("Fig. 3 · Dynamics on the ground floor: shift, escalation, competition", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    save(fig, "fig03_dynamics", "E001, 3 seeds. A: CORP miscalibration in windows around a regime change (structure-averaging "
         "CTW recovers ~4× faster). B: S1 = KT-2, S2 = CTW-12; sources where S2 saves < 0.01 bits omitted (never escalate). "
         "C: row = diagonal sequence of a predictor (always emit its less-likely bit); market = log-score market over {KT-0, KT-4, CTW-12}.")


# ── Fig 4 · Golay: symmetry earns bits, only through the right learner ───────
def fig_golay():
    g2 = load("e002_golay_symmetry/results_full.json")["golay"]
    g3 = load("e003_invariant_code_learner/results_full.json")["rows"]
    series = [("KT-phase", "generic counts (KT)", g2), ("CTW-12-phase", "generic structure (CTW-12)", g2),
              ("MLP + group augmentation", "MLP + symmetry augmentation", g2),
              ("linearity only", "Bayes over GF(2) spans", g3), ("symmetry + linearity", "Bayes over G-invariant codes", g3)]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, eps in zip(axs, [0.0, 0.03]):
        bayes = np.mean([r["bits"] for r in g2 if r["N"] is None and r["eps"] == eps])
        ax.axhline(bayes, color=INK, ls="--", lw=1)
        ax.text(1000, bayes + 0.015, f"Bayes, knows the code: {bayes:.2f}", fontsize=7.5, color=INK2, ha="right")
        for i, (key, lab, rows) in enumerate(series):
            field = "learner" if rows is g3 else "observer"
            R = [r for r in rows if r.get(field) == key and r["eps"] == eps]
            Ns = sorted({r["N"] for r in R})
            m = np.array([np.mean([r["bits"] for r in R if r["N"] == n]) for n in Ns])
            sd = np.array([np.std([r["bits"] for r in R if r["N"] == n]) for n in Ns])
            ax.errorbar(Ns, np.minimum(m, 1.48), yerr=sd, color=C[i], marker=MK[i], capsize=2, lw=1.8, label=lab)
        mlp = [r for r in g2 if r["observer"] == "MLP" and r["eps"] == eps and r["N"] <= 64]
        ax.annotate(f"plain MLP: {np.mean([r['bits'] for r in mlp]):.1f} bits (off scale,\nconfidently wrong)",
                    (16, 1.47), xytext=(30, 1.3), fontsize=7.5, color=INK2,
                    arrowprops={"arrowstyle": "->", "color": INK2, "lw": 0.8})
        ax.set_xscale("log"); ax.set_xlabel("training codewords N  (log)"); ax.set_ylim(0.4, 1.5)
        ax.set_title(f"BSC noise ε = {eps}", loc="left")
    axs[0].set_ylabel("bits / symbol on held-out stream")
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.085))
    fig.suptitle("Fig. 4 · A sporadic symmetry earns bits — but only through a learner that can represent it",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.16, 1, 0.94))
    save(fig, "fig04_golay", "E002 + E003. Extended Golay code [24,12,8], G = PSL(2,23) ⊂ M24 (order 6072, verified; transitive on "
         "the 759 octads). Bayes over G-invariant linear codes reaches the optimum 0.500 from ONE codeword; linearity alone needs 16; "
         "SGD (even with orbit augmentation) extracts ~nothing — the check bits are parities. Error bars: sd over seeds (2–5).")


# ── Fig 5 · The arithmetic ladder: GL(1) → GL(2) ─────────────────────────────
def fig_arith():
    a2 = load("e002_golay_symmetry/results_full.json")["arithmetic"]
    a4 = load("e004_hecke_rediscovery/results_full.json")
    groups = [
        ("Liouville λ(n)\nGL(1)", [("CTW-16", a2["liouville lambda(n)"]["CTW-16"]["bits"]),
                                     ("multiplicative", a2["liouville lambda(n)"]["multiplicative"]["bits"])],
         a2["liouville lambda(n)"]["prime_density"]),
        ("Legendre (n/1000003)\nGL(1)", [("CTW-16", a2["legendre (n/1000003)"]["CTW-16"]["bits"]),
                                          ("multiplicative", a2["legendre (n/1000003)"]["multiplicative"]["bits"])],
         a2["legendre (n/1000003)"]["prime_density"]),
        ("sign τ(n), Ramanujan Δ\nGL(2)", [(k.split(" (")[0], v["bits"]) for k, v in a4["ledger_second_half"].items()
                                           if k in ("CTW-16", "complete-multiplicative", "coprime-multiplicative") or k.startswith("hecke")],
         a4["prime_density_second_half"]),
    ]
    color_of = {"CTW-16": INK2, "complete-multiplicative": C[1], "multiplicative": C[0],
                "coprime-multiplicative": C[3], "hecke": C[2]}
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x0 = 0
    for gname, bars, dens in groups:
        xs = x0 + np.arange(len(bars))
        for x, (nm, v) in zip(xs, bars):
            ax.bar(x, v, 0.75, color=color_of[nm], edgecolor=SURF, linewidth=1)
            ax.text(x, v + 0.02, f"{v:.3f}", ha="center", fontsize=8, color=INK)
            ax.text(x, -0.05, nm.replace("-multiplicative", "-mult.").replace("hecke", "Hecke (w=12)"), ha="center",
                    va="top", fontsize=7.5, color=INK2, rotation=0)
        ax.hlines(dens, xs[0] - 0.45, xs[-1] + 0.45, color=INK, ls="--", lw=1)
        ax.text(xs[-1] + 0.48, dens, f"prime density\n{dens:.3f}", fontsize=7, color=INK2, va="center")
        ax.text(xs.mean(), 1.12, gname, ha="center", fontsize=9, fontweight="bold", color=INK)
        x0 = xs[-1] + 2.2
    ax.set_ylim(0, 1.25); ax.set_xlim(-0.8, x0 - 0.6); ax.set_xticks([]); ax.grid(axis="x", visible=False)
    ax.set_ylabel("bits / symbol")
    fig.suptitle("Fig. 5 · Climbing the Langlands ladder: the primes carry all the information", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    save(fig, "fig05_arithmetic_ladder", "E002B, E004. Generic observers see pure noise (≈1 bit/symbol; Sarnak-style pseudorandomness). "
         "Observers that know the multiplicative structure pay exactly the prime density. On GL(2) a program search on n ≤ 2000 rejects "
         "complete multiplicativity (59% of pairs), accepts coprime multiplicativity (100% of 3406) and finds the unique weight w = 12; "
         "the coprime-only observer pays the prime-POWER density (0.128). Ledger on n ∈ (2000, 4000].")


# ── Fig 6 · E000: amortisation, calibration, escalation ──────────────────────
def fig_e000():
    r = load("e000_shortest_path_amortization/results_full.json")
    s = r["q1_q2_q4"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for i, split in enumerate(["iid", "shift"]):
        d = [x for x in s if x["split"] == split and x["cal"] == "temp"]
        n = [x["n_labels"] for x in d]
        axs[0].errorbar(n, [x["acc"][0] for x in d], [x["acc"][1] for x in d], color=C[i], marker=MK[i], label=split, capsize=2)
        axs[1].errorbar(n, [x["smece"][0] for x in d], [x["smece"][1] for x in d], color=C[i], marker=MK[i], label=f"{split}, temp-scaled", capsize=2)
    raw = [x for x in s if x["split"] == "iid" and x["cal"] == "raw"]
    axs[1].plot([x["n_labels"] for x in raw], [x["smece"][0] for x in raw], color=C[1], marker=MK[2], ls=":", label="iid, raw")
    d = [x for x in s if x["split"] == "iid" and x["cal"] == "temp"]
    for k, lab, col, m in [("s2_calls_needed_random", "random gate", INK2, MK[2]), ("s2_calls_needed_gated", "confidence gate", C[0], MK[0]),
                           ("s2_calls_needed_oracle", "perfect gate", C[2], MK[1])]:
        axs[2].plot([x["n_labels"] for x in d], [x[k][0] for x in d], color=col, marker=m, label=lab)
    titles = ["A · One-pass accuracy (chance 0.25)", "B · Calibration (smooth ECE)",
              "C · S2 calls for 99% of S2 accuracy"]
    for a, t in zip(axs, titles):
        a.set_xscale("log"); a.set_xlabel("oracle labels  (log)"); a.set_title(t, loc="left"); a.legend()
    axs[0].set_ylabel("accuracy"); axs[1].set_ylabel("smooth ECE"); axs[2].set_ylabel("fraction escalated")
    fig.suptitle("Fig. 6 · Amortising an exact System 2 (shortest paths) into one pass", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    save(fig, "fig06_amortisation", "E000. 10-node graphs, 4 candidate first hops, 2×512 MLP, 3 seeds. Shift = edge density 0.4 → 0.25.")


# ── Fig 7 · E005: local-to-global obstruction ────────────────────────────────
def fig_local_global():
    r = load("e005_local_global_and_symmetry_discovery/results_full.json")
    k = [d["k"] for d in r["profile"]]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(k, [d["max_bits"] for d in r["profile"]], color=C[1], marker=MK[1], label="max over k-sets (an octad completes at k = 7)")
    ax.plot(k, [d["mean_bits"] for d in r["profile"]], color=C[0], marker=MK[0], label="mean over random k-sets")
    ax.axvspan(-0.3, 6.5, color=GRID, alpha=0.6, lw=0)
    ax.text(3.1, 0.55, "exactly 0 bits:\nevery 7 coordinates are uniform\n(dual distance 8)", ha="center", fontsize=8, color=INK2)
    d = [x for x in r["discovery"] if "group_order" in x][0]
    ax.text(11, 0.45, f"symmetry discovered from {d['N']} codewords:\n|G| = {d['group_order']:,} = |M24|\n"
            f"{d['transitivity_degree']}-transitive, contains PSL(2,23)", ha="right", fontsize=8, color=INK, bbox={"fc": SURF, "ec": GRID})
    ax.set_xlabel("k = number of other coordinates observed"); ax.set_ylabel("I(X_j ; X_S)  (bits)")
    ax.set_ylim(-0.05, 1.1); ax.legend(loc="upper left")
    fig.suptitle("Fig. 7 · A local-to-global obstruction: Golay structure is invisible below 7 bits", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    save(fig, "fig07_local_global", "E005. Exact information a k-local observer has about one Golay coordinate (all 4096 codewords "
         "enumerated; 300 random subsets per k). Structure lives only in 8-wise global constraints: why local learners (CTW, MLP) fail.")


# ── Fig 8 · E006: elliptic curves ────────────────────────────────────────────
def fig_elliptic():
    r = load("e006_elliptic_curves/results_full.json")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axs[0]
    th = np.linspace(-1, 1, 200)
    for i, (name, lab) in enumerate([("non-CM 11a3", "non-CM 11a3"), ("CM Z[i]: y^2=x^3-x", "CM y² = x³ − x (a_p ≠ 0)")]):
        v = np.array(r["sato_tate"][name])
        if name.startswith("CM"):
            v = v[v != 0]
        ax.hist(v, bins=40, range=(-1, 1), density=True, histtype="step", lw=2, color=C[i], label=lab)
    ax.plot(th, 2 / np.pi * np.sqrt(1 - th ** 2), color=INK, ls="--", lw=1, label="Sato–Tate semicircle")
    ax.set_xlabel("a_p / 2√p"); ax.set_ylabel("density"); ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("A · Distributions of normalised a_p", loc="left")
    ax = axs[1]
    rows = [("CM ℤ[i]  zero?", "CM Z[i]: y^2=x^3-x", "z (a_p = 0)"), ("CM ℤ[i]  sign", "CM Z[i]: y^2=x^3-x", "s (sign a_p)"),
            ("CM ℤ[ω]  zero?", "CM Z[w]: y^2=x^3+1", "z (a_p = 0)"), ("non-CM  sign", "non-CM: y^2=x^3-x+1", "s (sign a_p)"),
            ("11a3  zero?", "non-CM 11a3", "z (a_p = 0)"), ("11a3  sign", "non-CM 11a3", "s (sign a_p)")]
    for y, (lab, cur, st) in enumerate(rows):
        obs = r["results"][cur][st]["observers"]
        gen = obs["CTW-16"]["bits"]; best_name = min(obs, key=lambda k: obs[k]["bits"]); best = obs[best_name]["bits"]
        ax.scatter(gen, y, color=INK2, marker=MK[2], s=40, zorder=3, label="generic (CTW-16)" if y == 0 else None)
        ax.scatter(best, y, color=C[2], marker=MK[0], s=48, zorder=3, label="best structure-aware observer" if y == 0 else None)
        ax.hlines(y, best, gen, color=GRID, lw=3, zorder=1)
        nm = best_name.replace("Hecke character (a,b mod 4)", "Hecke char. ℤ[i]").replace("residue ", "")
        nm = nm if gen - best > 0.005 else "nothing beats generic"
        ax.text(1.03, y, nm, va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(rows)), [x[0] for x in rows]); ax.set_xlim(-0.03, 1.35); ax.invert_yaxis()
    ax.set_xlabel("bits / symbol (held-out half of primes < 40000)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7.5)
    ax.set_title("B · CM curves are GL(1) in disguise; non-CM signs are fresh", loc="left")
    fig.suptitle("Fig. 8 · Elliptic curves on the ledger", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save(fig, "fig08_elliptic", "E006. a_p by exact Legendre sums (checked vs 11a). Residue moduli chosen by MDL on the first half "
         "(found 4, 3 and, unplanted, 5 for 11a3's rational 5-torsion). The ℤ[i] Hecke-character observer uses p = a² + b².")


# ── Fig 9 · E007: amortised inference ────────────────────────────────────────
def fig_amortized():
    r = load("e007_amortized_inference/analysis_results_modal.json")
    fams = ["iid", "markov", "periodic", "thue", "golay", "lcg", "minkowski", "logistic", "regime"]
    models = list(r["models"])
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axs[0]; x = np.arange(len(fams)); w = 0.8 / (len(models) + 1)
    for i, m in enumerate(models):
        mm = r["models"][m]
        ax.bar(x + (i - len(models) / 2) * w, [mm["families"][f]["bits"] for f in fams], w, color=SEQ[i + 1], edgecolor=SURF,
               lw=0.5, label=f"{m} ({mm['params'] / 1e6:.1f}M params, {mm['tokens'] / 1e6:.0f}M tok)")
    ax.scatter(x + (len(models) / 2) * w, [r["references_bits"]["CTW-12"][f] for f in fams], color=C[1], marker="D", s=30,
               zorder=3, label="CTW-12 (exact Bayes over trees)")
    for f, v in r["references_bits"]["Bayes"].items():
        ax.hlines(v, fams.index(f) - 0.45, fams.index(f) + 0.45, color=INK, ls="--", lw=1)
    ax.axvline(4.5, color=INK2, lw=1, ls=":"); ax.text(4.6, 1.08, "held-out families", fontsize=8, color=INK2)
    ax.set_xticks(x, fams); ax.set_ylabel("bits / symbol"); ax.set_ylim(0.4, 1.12)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3); ax.grid(axis="x", visible=False)
    ax.set_title("A · One forward pass vs Bayesian structure inference (dashed: exact Bayes)", loc="left")
    ax = axs[1]
    best = "d256-L6"
    for i, f in enumerate(["thue", "markov", "golay"]):
        cm = np.convolve(r["models"][best]["families"][f]["curve"], np.ones(16) / 16, "valid")
        cc = np.convolve(r["reference_curves"]["CTW-12"][f], np.ones(16) / 16, "valid")
        ax.plot(cm, color=C[i], lw=2, label=f"{f}: transformer")
        ax.plot(cc, color=C[i], lw=1.2, ls=":", label=f"{f}: CTW-12")
    ax.set_xlabel("position in sequence (in-context examples)"); ax.set_ylabel("bits / symbol (16-pt mean)")
    ax.set_title(f"B · In-context learning curves ({best})", loc="left"); ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Fig. 9 · Amortised inference: a transformer meta-trained on a prior does structure inference in context",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    save(fig, "fig09_amortized", "E007. Causal transformers meta-trained on a prior over generators (iid, Markov up to order 5, noisy "
         "periodic, noisy Thue-Morse, Golay), 220 s each on one H100 (compute-matched; about $0.95 total). 64 eval sequences x 512 "
         "bits per family. MCB <= 0.02 bits on every family, held-out included. Golay stays near 1 bit (parity barrier) vs Bayes 0.60.")


if __name__ == "__main__":
    for f in [fig_ledger, fig_jumps, fig_dynamics, fig_golay, fig_arith, fig_e000, fig_local_global, fig_elliptic, fig_amortized]:
        f(); print("ok", f.__name__)
