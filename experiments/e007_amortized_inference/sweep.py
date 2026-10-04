"""E007 — Amortised inference: can a transformer do structure inference in one forward pass?

A small causal transformer is meta-trained on bits from a PRIOR over generating processes (iid, variable-order
Markov, noisy periodic, noisy Thue–Morse, Golay codeword streams). It never sees the generator's identity:
everything must be inferred in context. We compare its code length per family with CTW-12 and exact Bayes, and
test held-out families (pseudorandom LCG, Minkowski ?(u), logistic map, regime switch).

Compute plan (napkin math in README): one Modal H100 container runs a model-size sweep with a fixed WALL-TIME
budget per model (compute-matched), hard timeout bounds cost. Local smoke: python sweep.py --local
Cloud sweep (spends money!):         modal run experiments/e007_amortized_inference/sweep.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

T = 512
TRAIN_FAMILIES = ["iid", "markov", "periodic", "thue", "golay"]
HELDOUT_FAMILIES = ["lcg", "minkowski", "logistic", "regime"]


# ── generators (numpy, vectorised over the batch) ───────────────────────────

def golay_words():
    g = [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1]
    W = []
    for m in range(4096):
        c = np.zeros(23, np.int8)
        for i in range(12):
            if m >> i & 1:
                c[i:i + 12] ^= np.array(g, np.int8)
        W.append(np.append(c, c.sum() % 2))
    return np.array(W, np.int8)


GOLAY = golay_words()


def gen(family, B, rng):
    x = np.zeros((B, T), np.int8)
    if family == "iid":
        x = (rng.uniform(size=(B, T)) < rng.uniform(size=(B, 1))).astype(np.int8)
    elif family == "markov":
        ks = rng.integers(1, 6, B)
        for k in np.unique(ks):
            idx = np.nonzero(ks == k)[0]
            P = rng.beta(0.5, 0.5, size=(len(idx), 2 ** k))
            ctx = np.zeros(len(idx), np.int64)
            for t in range(T):
                b = (rng.uniform(size=len(idx)) < P[np.arange(len(idx)), ctx]).astype(np.int8)
                x[idx, t] = b
                ctx = ((ctx << 1) | b) & (2 ** k - 1)
    elif family == "periodic":
        L = rng.integers(2, 25, B); pat = rng.integers(0, 2, (B, 24)).astype(np.int8)
        eps = rng.uniform(0, 0.1, (B, 1))
        x = pat[np.arange(B)[:, None], np.arange(T)[None, :] % L[:, None]] ^ (rng.uniform(size=(B, T)) < eps)
    elif family == "thue":
        off = rng.integers(0, 1 << 20, (B, 1)); t = np.arange(T)[None, :] + off
        pc = np.zeros_like(t)
        for s in range(22):
            pc += (t >> s) & 1
        x = (pc & 1).astype(np.int8) ^ (rng.uniform(size=(B, T)) < rng.uniform(0, 0.05, (B, 1)))
    elif family == "golay":
        blocks = GOLAY[rng.integers(0, 4096, (B, T // 24 + 1))].reshape(B, -1)[:, :T]
        eps = np.where(rng.uniform(size=(B, 1)) < 0.5, 0.0, 0.03)
        x = blocks ^ (rng.uniform(size=(B, T)) < eps)
    elif family == "lcg":
        s = rng.integers(1, 2 ** 32, B, dtype=np.uint64)
        for t in range(T):
            s = (np.uint64(1664525) * s + np.uint64(1013904223)) & np.uint64(0xFFFFFFFF)
            x[:, t] = (s >> np.uint64(31)).astype(np.int8)
    elif family == "logistic":
        v = rng.uniform(0.1, 0.9, B)
        for t in range(T):
            v = 3.99 * v * (1 - v); x[:, t] = v > 0.5
    elif family == "minkowski":
        for b in range(B):
            out, bit = [], 0
            while len(out) < T:
                u = rng.uniform()
                for _ in range(12):
                    u = 1.0 / u; a = int(u); u -= a
                    if a == 0 or u < 1e-9:
                        break
                    out.extend([bit] * min(a, 64)); bit ^= 1
            x[b] = out[:T]
    elif family == "regime":
        a, c = gen("markov", B, rng), gen("markov", B, rng)
        x = np.concatenate([a[:, : T // 2], c[:, T // 2:]], 1)
    return x.astype(np.int8)


def mixture_batch(B, rng):
    fam = rng.integers(0, len(TRAIN_FAMILIES), B)
    out = np.zeros((B, T), np.int8)
    for i, f in enumerate(TRAIN_FAMILIES):
        idx = np.nonzero(fam == i)[0]
        if len(idx):
            out[idx] = gen(f, len(idx), rng)
    return out


def eval_sets(n=64, seed=12345):
    rng = np.random.default_rng(seed)
    return {f: gen(f, n, rng) for f in TRAIN_FAMILIES + HELDOUT_FAMILIES}


# ── model + training (torch) ────────────────────────────────────────────────

def build(d, L, heads):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

        def forward(self, h):
            B_, T_, _ = h.shape
            q, k, v = self.qkv(self.ln1(h)).view(B_, T_, 3, heads, d // heads).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B_, T_, d)
            h = h + self.proj(a)
            return h + self.mlp(self.ln2(h))

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.pos = nn.Embedding(3, d), nn.Parameter(torch.zeros(1, T, d))
            self.blocks = nn.ModuleList([Block() for _ in range(L)])
            self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, 1)

        def forward(self, x):  # x: (B, T) int bits; predict x_t from x_<t
            inp = torch.cat([torch.full_like(x[:, :1], 2), x[:, :-1]], 1)
            h = self.emb(inp) + self.pos
            for b in self.blocks:
                h = b(h)
            return self.head(self.ln(h)).squeeze(-1)

    return GPT()


def make_pool(n, seed=7):
    rng = np.random.default_rng(seed)
    return np.concatenate([mixture_batch(1024, rng) for _ in range(n // 1024)])


def train_and_eval(cfg, budget_s, device, evals, pool, log=print):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(cfg.get("seed", 0)); rng = np.random.default_rng(cfg.get("seed", 0))
    model = build(cfg["d"], cfg["L"], cfg["heads"]).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01, betas=(0.9, 0.95))
    use_amp = device == "cuda"
    t0, step, tokens = time.time(), 0, 0
    P = torch.from_numpy(pool).to(device)  # resident on the GPU; batches are sampled there (no CPU bottleneck)
    gen_t = torch.Generator(device=device).manual_seed(cfg.get("seed", 0))
    while time.time() - t0 < budget_s:
        frac = (time.time() - t0) / budget_s
        lr = cfg["lr"] * min(1.0, (step + 1) / 200) * 0.5 * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = lr
        xb = P[torch.randint(0, len(P), (cfg["batch"],), device=device, generator=gen_t)].long()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits.float(), xb.float())
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        step += 1; tokens += xb.numel()
        if step % 200 == 0:
            log(f"  [{cfg['name']}] step {step} loss {loss.item() / math.log(2):.3f} bits  {time.time() - t0:.0f}s")
    train_s = time.time() - t0
    model.eval(); out = {}
    with torch.no_grad():
        for fam, X in evals.items():
            xb = torch.from_numpy(X).to(device).long()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                p = torch.sigmoid(model(xb).float()).cpu().numpy()
            out[fam] = p.astype(np.float32).tolist()
    return {"cfg": cfg, "params": nparam, "steps": step, "tokens": tokens, "train_seconds": train_s,
            "approx_train_flops": 6 * nparam * tokens, "pool_passes": tokens / pool.size, "probs": out}


SWEEP = [
    {"name": "d64-L4", "d": 64, "L": 4, "heads": 2, "lr": 2e-3, "batch": 256},
    {"name": "d128-L4", "d": 128, "L": 4, "heads": 4, "lr": 1.5e-3, "batch": 256},
    {"name": "d256-L6", "d": 256, "L": 6, "heads": 8, "lr": 1e-3, "batch": 256},
    {"name": "d384-L8", "d": 384, "L": 8, "heads": 12, "lr": 6e-4, "batch": 256},
]

# ── Modal entrypoint (only imported when running under `modal run`) ──────────
try:
    import modal
    app = modal.App("jev-e007-amortized-inference")
    image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.6.0", "numpy")

    @app.function(gpu="H100", image=image, timeout=1300)  # hard cap: 1300 s * $0.001097/s = $1.43
    def sweep_remote(budget_per_model: float = 200.0):
        import torch
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
        ev = eval_sets(); t0 = time.time(); pool = make_pool(122880)
        print(f"pool {pool.shape} in {time.time() - t0:.0f}s", flush=True)
        return [train_and_eval(cfg, budget_per_model, "cuda", ev, pool, log=lambda s: print(s, flush=True)) for cfg in SWEEP]

    @app.local_entrypoint()
    def main(budget_per_model: float = 200.0):
        t0 = time.time()
        res = sweep_remote.remote(budget_per_model)
        out = Path(__file__).parent / "results_modal.json"
        out.write_text(json.dumps({"wall_seconds": time.time() - t0, "runs": res}))
        print("saved", out, "wall", round(time.time() - t0), "s")
except ImportError:
    pass


if __name__ == "__main__" and "--local" in sys.argv:
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = eval_sets()
    budget = float(sys.argv[sys.argv.index("--local") + 1]) if len(sys.argv) > sys.argv.index("--local") + 1 else 60
    pool = make_pool(16384)
    res = [train_and_eval({**SWEEP[0], "batch": 64}, budget, dev, ev, pool)]
    (Path(__file__).parent / "results_local.json").write_text(json.dumps({"device": dev, "runs": res}))
    print("local done:", res[0]["steps"], "steps", res[0]["tokens"], "tokens")
