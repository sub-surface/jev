"""Analyse E007 results: per-family bits/symbol and CORP split for each model; CTW-12 and Bayes references
on the identical evaluation sequences; in-context learning curves. python analyze.py [results_modal.json]"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from calib.metrics import corp_decomposition  # noqa: E402
import sweep  # noqa: E402


def mod(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m


def bits(p, x):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(x * np.log2(p) + (1 - x) * np.log2(1 - p))


def references(ev):
    cache = HERE / "references.json"
    if cache.exists():
        return json.loads(cache.read_text())
    e001 = mod("e001", "experiments/e001_binary_ground_floor/run.py")
    e002 = mod("e002", "experiments/e002_golay_symmetry/run.py")
    ref = {}
    for fam, X in ev.items():
        P = np.stack([e001.run_observer(e001.CTW(12), x) for x in X])
        ref.setdefault("CTW-12", {})[fam] = bits(P, X).mean(0).tolist()
    X = ev["iid"]; n1 = np.concatenate([np.zeros((len(X), 1)), np.cumsum(X, 1)[:, :-1]], 1)
    n = np.arange(sweep.T)[None, :]
    ref.setdefault("Bayes", {})["iid"] = bits((n1 + 1) / (n + 2), X).mean(0).tolist()  # uniform prior: exact
    C = sweep.GOLAY
    gb = []
    for x in ev["golay"]:  # Bayes with the code known and eps known per sequence (oracle-eps lower bound)
        blocks = np.resize(x, (sweep.T // 24 + 1) * 24).reshape(-1, 24)
        best = None
        for eps in (0.0, 0.03):
            p = e002.bayes_stream(C, blocks, eps)[: sweep.T]
            L = bits(p, x)
            best = L if best is None or L.sum() < best.sum() else best
        gb.append(best)
    ref["Bayes"]["golay"] = np.mean(gb, 0).tolist()
    cache.write_text(json.dumps(ref))
    return ref


def main():
    path = HERE / (sys.argv[1] if len(sys.argv) > 1 else "results_modal.json")
    R = json.loads(path.read_text())
    ev = sweep.eval_sets()
    ref = references(ev)
    table = {}
    for run in R["runs"]:
        name = run["cfg"]["name"]
        row = {}
        for fam, X in ev.items():
            p = np.array(run["probs"][fam], dtype=np.float64)
            d = corp_decomposition(p.ravel(), X.ravel().astype(float))
            row[fam] = {"bits": float(bits(p, X).mean()), "MCB": d["MCB"], "DSC": d["DSC"],
                        "curve": bits(p, X).mean(0).tolist()}
        table[name] = {"params": run["params"], "tokens": run["tokens"], "flops": run["approx_train_flops"],
                       "train_seconds": run["train_seconds"], "pool_passes": run.get("pool_passes"), "families": row}
    refs = {k: {f: float(np.mean(v)) for f, v in d.items()} for k, d in ref.items()}
    out = {"source": path.name, "models": table, "references_bits": refs, "reference_curves": ref}
    (HERE / f"analysis_{path.stem}.json").write_text(json.dumps(out))
    fams = sweep.TRAIN_FAMILIES + sweep.HELDOUT_FAMILIES
    print(f"{'model':>10} {'params':>9} {'Mtok':>6} | " + " ".join(f"{f[:8]:>8}" for f in fams))
    for k, v in table.items():
        print(f"{k:>10} {v['params']:>9} {v['tokens'] / 1e6:>6.0f} | " + " ".join(f"{v['families'][f]['bits']:>8.3f}" for f in fams))
    for k, d in refs.items():
        print(f"{k:>10} {'':>9} {'':>6} | " + " ".join(f"{d.get(f, float('nan')):>8.3f}" for f in fams))
    print("MCB (model rows):", {k: {f: round(v['families'][f]['MCB'], 3) for f in fams} for k, v in table.items()})


if __name__ == "__main__":
    main()
