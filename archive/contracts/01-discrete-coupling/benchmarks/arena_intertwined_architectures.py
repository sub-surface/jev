"""
Architectural Arena: Intertwined Jev-LLM Symbiosis (Stage A)
============================================================
Authors: Jevformer Research Collective (Leon & Ilya Sutskever persona)

Benchmarking 4 Intertwined Dual-Process Architectures:
  1. Baseline Decoupled: Standard Jevformer (S1 trunk -> Noul early exit / cascade).
  2. Jev-Gated Residual (JGR): Epistemic Noul head directly gates the layer residual update.
     x_{l+1} = x_l + (1 - Noul(x_l)) * Layer_l(x_l). High confidence collapses layer to identity.
  3. Dual-Stream Epistemic Attention (DSEA): Coupled Semantic Stream (S) and Epistemic Stream (E).
     Attention logits are modulated by epistemic urgency: A_ij = (Q_S K_S^T)/sqrt(d) + beta * (Q_E K_E^T).
  4. Epistemic Token-Gated (ETG): Dynamic per-token confidence routing.

Task: Dyck-3 Hierarchical Grammar & Stack Depth Benchmark
  - Bracket types: (), [], {}
  - Balanced vs Unbalanced classification with variable nesting depths D in [1, 5].
  - Shallow nesting (D=1, 2) is solvable in 1-2 layers.
  - Deep nesting (D=4, 5) strictly requires deeper composition.
"""

import os
import sys
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# 1. Dyck-3 Hierarchical Bracket Generator (Provable Circuit Hierarchy)
# ----------------------------------------------------------------------

VOCAB = {
    "<PAD>": 0, "(": 1, ")": 2, "[": 3, "]": 4, "{": 5, "}": 6, "<QUERY>": 7
}
OPEN_TO_CLOSE = {1: 2, 3: 4, 5: 6}
CLOSE_TO_OPEN = {2: 1, 4: 3, 6: 5}

def generate_dyck_sequence(max_depth: int = 5, target_len: int = 24) -> tuple[list[int], bool, int]:
    """Generates a balanced or slightly corrupted Dyck-3 sequence and returns (tokens, is_balanced, max_depth_reached)."""
    # Generate balanced skeleton recursively
    def gen_balanced(depth_left: int) -> tuple[list[int], int]:
        if depth_left <= 0 or random.random() < 0.25:
            return [], 0
        b_type = random.choice([1, 3, 5])
        c_type = OPEN_TO_CLOSE[b_type]
        inner, inner_d = gen_balanced(depth_left - 1)
        outer, outer_d = gen_balanced(depth_left)
        seq = [b_type] + inner + [c_type] + outer
        return seq, max(inner_d + 1, outer_d)

    seq, depth = gen_balanced(max_depth)
    while len(seq) < 4:
        seq, depth = gen_balanced(max_depth)

    # Truncate or pad to fit target_len
    is_balanced = random.random() < 0.5
    if not is_balanced:
        # Corrupt balancedness by flipping one closing bracket
        seq = list(seq)
        closing_indices = [i for i, tok in enumerate(seq) if tok in [2, 4, 6]]
        if closing_indices:
            idx = random.choice(closing_indices)
            corruptions = [c for c in [2, 4, 6] if c != seq[idx]]
            seq[idx] = random.choice(corruptions)
        else:
            is_balanced = True

    # Clip / Pad to target_len - 1
    seq = seq[: target_len - 1]
    tokens = seq + [VOCAB["<QUERY>"]]
    while len(tokens) < target_len:
        tokens.insert(0, VOCAB["<PAD>"])

    # Re-verify exact balancedness and nesting depth
    stack = []
    actual_balanced = True
    cur_d = 0
    max_d = 0
    for tok in tokens:
        if tok in OPEN_TO_CLOSE:
            stack.append(tok)
            cur_d += 1
            max_d = max(max_d, cur_d)
        elif tok in CLOSE_TO_OPEN:
            if not stack or stack[-1] != CLOSE_TO_OPEN[tok]:
                actual_balanced = False
                break
            stack.pop()
            cur_d -= 1

    if stack:
        actual_balanced = False

    return tokens, actual_balanced, max_d


def generate_dyck_dataset(num_samples: int, max_depth: int = 5, seq_len: int = 24):
    tokens = torch.zeros(num_samples, seq_len, dtype=torch.long)
    labels = torch.zeros(num_samples, dtype=torch.long)
    depths = torch.zeros(num_samples, dtype=torch.long)

    for i in range(num_samples):
        seq, is_bal, d = generate_dyck_sequence(max_depth, seq_len)
        tokens[i] = torch.tensor(seq, dtype=torch.long)
        labels[i] = 1 if is_bal else 0
        depths[i] = min(d, max_depth)

    return tokens, labels, depths


# ----------------------------------------------------------------------
# 2. Intertwined Architectures
# ----------------------------------------------------------------------

# ----------------- Architecture 1: Decoupled Baseline -----------------
class DecoupledJevformer(nn.Module):
    def __init__(self, vocab_size=16, d_model=128, num_heads=4, total_layers=4):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 32, d_model) * 0.02)
        
        # System 1: Layer 0
        self.s1_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4, batch_first=True, norm_first=True)
        # Jev Head at exit of S1
        self.noul_proj = nn.Sequential(nn.Linear(d_model, d_model//2), nn.Tanh(), nn.Linear(d_model//2, 1))
        self.choice_s1 = nn.Linear(d_model, 2)
        
        # System 2: Remaining layers (recurrent or stacked)
        self.s2_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4, batch_first=True, norm_first=True)
            for _ in range(total_layers - 1)
        ])
        self.choice_s2 = nn.Linear(d_model, 2)

    def forward(self, x, threshold=0.5):
        b, t = x.shape
        h = self.token_emb(x) + self.pos_emb[:, :t, :]
        h = self.s1_layer(h)
        h_q = h[:, -1, :]

        s1_logits = self.choice_s1(h_q)
        noul_conf = torch.sigmoid(self.noul_proj(h_q)).squeeze(-1)
        exit_mask = noul_conf >= threshold

        cur = h
        for layer in self.s2_layers:
            cur = layer(cur)
        s2_logits = self.choice_s2(cur[:, -1, :])

        final_logits = torch.where(exit_mask.unsqueeze(-1), s1_logits, s2_logits)
        return {
            "final_logits": final_logits,
            "s1_logits": s1_logits,
            "s2_logits": s2_logits,
            "noul_conf": noul_conf,
            "exit_mask": exit_mask,
            "residual_gates": None,
        }


# ---------------- Architecture 2: Jev-Gated Residual (JGR) -------------
class JevGatedResidualTransformer(nn.Module):
    """
    Jev-Gated Residual (JGR):
    At each layer l, an epistemic Noul valve computes:
      pi_l = Noul(x_l)
      gamma_l = 1 - pi_l
      x_{l+1} = x_l + gamma_l * Layer_l(x_l)
    When epistemic certainty is reached, gamma_l -> 0 and layers collapse to the identity operator!
    """
    def __init__(self, vocab_size=16, d_model=128, num_heads=4, total_layers=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 32, d_model) * 0.02)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4, batch_first=True, norm_first=True)
            for _ in range(total_layers)
        ])
        self.noul_gates = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model//4), nn.Tanh(), nn.Linear(d_model//4, 1))
            for _ in range(total_layers)
        ])
        self.out_head = nn.Linear(d_model, 2)

    def forward(self, x, threshold=0.5):
        b, t = x.shape
        h = self.token_emb(x) + self.pos_emb[:, :t, :]
        residual_gates = []
        noul_confs = []

        for l_idx, (layer, noul_gate) in enumerate(zip(self.layers, self.noul_gates)):
            # Epistemic evaluation of current state at query token
            q_rep = h[:, -1, :]
            conf = torch.sigmoid(noul_gate(q_rep)).squeeze(-1)  # [B]
            gamma = 1.0 - conf  # gate in [0, 1]
            
            # Modulate residual perturbation
            delta_h = layer(h) - h
            h = h + gamma.view(b, 1, 1) * delta_h

            residual_gates.append(gamma)
            noul_confs.append(conf)

        final_logits = self.out_head(h[:, -1, :])
        primary_conf = noul_confs[0]  # first layer confidence
        exit_mask = primary_conf >= threshold

        return {
            "final_logits": final_logits,
            "s1_logits": final_logits,
            "s2_logits": final_logits,
            "noul_conf": primary_conf,
            "all_confs": noul_confs,
            "exit_mask": exit_mask,
            "residual_gates": residual_gates,
        }


# ---------- Architecture 3: Dual-Stream Epistemic Attention (DSEA) ----
class DualStreamEpistemicAttention(nn.Module):
    """
    Dual-Stream Epistemic Attention (DSEA):
    Two coupled streams:
      Semantic Stream S in R^{B x T x D}
      Epistemic Stream E in R^{B x T x D_e}
    Semantic self-attention logits are steered by epistemic query-key dot products!
    """
    def __init__(self, vocab_size=16, d_model=128, d_epi=32, num_heads=4, total_layers=4):
        super().__init__()
        self.d_model = d_model
        self.d_epi = d_epi
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.epi_emb = nn.Embedding(vocab_size, d_epi)
        self.pos_emb = nn.Parameter(torch.randn(1, 32, d_model) * 0.02)
        self.pos_epi = nn.Parameter(torch.randn(1, 32, d_epi) * 0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "q_sem": nn.Linear(d_model, d_model),
                "k_sem": nn.Linear(d_model, d_model),
                "v_sem": nn.Linear(d_model, d_model),
                "out_sem": nn.Linear(d_model, d_model),
                "norm_sem": nn.LayerNorm(d_model),
                "ff_sem": nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model)),
                
                # Epistemic steering heads
                "q_epi": nn.Linear(d_epi, d_epi),
                "k_epi": nn.Linear(d_epi, d_epi),
                "update_epi": nn.Sequential(nn.Linear(d_model + d_epi, d_epi), nn.Tanh()),
            })
            for _ in range(total_layers)
        ])
        self.beta = nn.Parameter(torch.ones(1) * 0.5)  # epistemic steering strength
        self.noul_head = nn.Sequential(nn.Linear(d_epi, 1), nn.Sigmoid())
        self.out_head = nn.Linear(d_model, 2)

    def forward(self, x, threshold=0.5):
        b, t = x.shape
        s = self.token_emb(x) + self.pos_emb[:, :t, :]
        e = self.epi_emb(x) + self.pos_epi[:, :t, :]
        attention_maps = []

        for layer in self.layers:
            # Multi-head semantic Q, K, V
            q_s = layer["q_sem"](s).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
            k_s = layer["k_sem"](s).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
            v_s = layer["v_sem"](s).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

            # Epistemic Q, K
            q_e = layer["q_epi"](e).unsqueeze(1)  # broadcast over heads
            k_e = layer["k_epi"](e).unsqueeze(1)

            # Modulate attention logits
            attn_scores = torch.matmul(q_s, k_s.transpose(-2, -1)) / math.sqrt(self.head_dim)
            epi_scores = torch.matmul(q_e, k_e.transpose(-2, -1)) / math.sqrt(self.d_epi)
            total_scores = attn_scores + self.beta * epi_scores

            attn_weights = F.softmax(total_scores, dim=-1)
            attention_maps.append(attn_weights.detach().cpu())

            out_s = torch.matmul(attn_weights, v_s).transpose(1, 2).contiguous().view(b, t, self.d_model)
            s = layer["norm_sem"](s + layer["out_sem"](out_s))
            s = s + layer["ff_sem"](s)

            # Update epistemic stream with semantic feedback
            fused = torch.cat([s, e], dim=-1)
            e = e + layer["update_epi"](fused)

        final_logits = self.out_head(s[:, -1, :])
        noul_conf = self.noul_head(e[:, -1, :]).squeeze(-1)
        exit_mask = noul_conf >= threshold

        return {
            "final_logits": final_logits,
            "s1_logits": final_logits,
            "s2_logits": final_logits,
            "noul_conf": noul_conf,
            "exit_mask": exit_mask,
            "attention_maps": attention_maps,
            "residual_gates": None,
        }


# ----------------------------------------------------------------------
# 3. Training & Evaluation Engine
# ----------------------------------------------------------------------

def train_and_eval_architecture(arch_name: str, model: nn.Module, train_data, test_data, epochs: int = 15, lr: float = 1e-3, device="cuda"):
    train_x, train_y, train_d = train_data
    test_x, test_y, test_d = test_data

    train_x, train_y, train_d = train_x.to(device), train_y.to(device), train_d.to(device)
    test_x, test_y, test_d = test_x.to(device), test_y.to(device), test_d.to(device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    batch_size = 64
    num_batches = len(train_x) // batch_size

    print(f"\n--- Training {arch_name} on {device} ({sum(p.numel() for p in model.parameters()):,} params) ---", flush=True)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(train_x))
        epoch_loss = 0.0

        for b_idx in range(num_batches):
            idx = perm[b_idx * batch_size : (b_idx + 1) * batch_size]
            x, y, d = train_x[idx], train_y[idx], train_d[idx]

            out = model(x)
            logits = out["final_logits"]
            noul_conf = out["noul_conf"]

            loss_task = F.cross_entropy(logits, y)
            is_correct = (torch.argmax(logits, dim=-1) == y).float().detach()
            loss_brier = F.mse_loss(noul_conf, is_correct)

            total_loss = loss_task + 2.0 * loss_brier

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += total_loss.item()

    elapsed = time.time() - t0
    print(f"[{arch_name}]: Training completed in {elapsed:.1f}s | Final Train Loss: {epoch_loss/num_batches:.3f}", flush=True)

    # Holdout evaluation & MechInterp
    model.eval()
    with torch.no_grad():
        out_eval = model(test_x)
        preds = torch.argmax(out_eval["final_logits"], dim=-1)
        total_acc = (preds == test_y).float().mean().item() * 100.0

        # Per-depth accuracy & calibration
        depth_metrics = {}
        for depth_val in range(1, 6):
            mask = test_d == depth_val
            if not mask.any():
                continue
            acc_d = (preds[mask] == test_y[mask]).float().mean().item() * 100.0
            conf_d = out_eval["noul_conf"][mask].mean().item()
            depth_metrics[depth_val] = {"acc": acc_d, "conf": conf_d, "count": mask.sum().item()}

        # ECE calculation
        def compute_ece(confs, corrects, num_bins=10):
            bins = np.linspace(0.0, 1.0, num_bins + 1)
            ece = 0.0
            n = len(confs)
            for i in range(num_bins):
                m = (confs >= bins[i]) & (confs < bins[i + 1])
                if np.any(m):
                    bin_acc = np.mean(corrects[m])
                    bin_conf = np.mean(confs[m])
                    ece += (np.sum(m) / n) * abs(bin_acc - bin_conf)
            return float(ece)

        confs_np = out_eval["noul_conf"].cpu().numpy()
        corrects_np = (preds == test_y).float().cpu().numpy()
        ece = compute_ece(confs_np, corrects_np)

        # MechInterp: check residual gates if available
        gate_summary = None
        if out_eval.get("residual_gates") is not None:
            # Mean gate value per layer
            gate_summary = [g.mean().item() for g in out_eval["residual_gates"]]

    return {
        "name": arch_name,
        "acc": total_acc,
        "ece": ece,
        "depth_metrics": depth_metrics,
        "gate_summary": gate_summary,
        "model": model,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Architectural Arena: Intertwined Jev-LLM Symbiosis ===", flush=True)
    print(f"Execution Device: {device}", flush=True)

    print("Generating Dyck-3 Hierarchical Grammar Dataset...", flush=True)
    train_data = generate_dyck_dataset(4000, max_depth=5, seq_len=24)
    test_data = generate_dyck_dataset(1000, max_depth=5, seq_len=24)
    print("Dataset Ready (Train: 4,000, Test: 1,000).", flush=True)

    results = []

    # 1. Baseline Decoupled
    m1 = DecoupledJevformer(vocab_size=16, d_model=128, num_heads=4, total_layers=4)
    r1 = train_and_eval_architecture("Baseline Decoupled", m1, train_data, test_data, epochs=15, device=device)
    results.append(r1)

    # 2. Jev-Gated Residual (JGR)
    m2 = JevGatedResidualTransformer(vocab_size=16, d_model=128, num_heads=4, total_layers=4)
    r2 = train_and_eval_architecture("Jev-Gated Residual (JGR)", m2, train_data, test_data, epochs=15, device=device)
    results.append(r2)

    # 3. Dual-Stream Epistemic Attention (DSEA)
    m3 = DualStreamEpistemicAttention(vocab_size=16, d_model=128, d_epi=32, num_heads=4, total_layers=4)
    r3 = train_and_eval_architecture("Dual-Stream Epistemic Attn (DSEA)", m3, train_data, test_data, epochs=15, device=device)
    results.append(r3)

    # Summary table
    print("\n" + "=" * 75, flush=True)
    print(f"{'Architecture':<32} | {'Overall Acc':<11} | {'ECE (%)':<9} | {'Depth 1-2':<10} | {'Depth 4-5':<10}", flush=True)
    print("=" * 75, flush=True)

    for r in results:
        dm = r["depth_metrics"]
        acc_shallow = (dm.get(1, {}).get("acc", 0) + dm.get(2, {}).get("acc", 0)) / 2.0
        acc_deep = (dm.get(4, {}).get("acc", 0) + dm.get(5, {}).get("acc", 0)) / 2.0
        print(f"{r['name']:<32} | {r['acc']:>6.2f}%    | {r['ece']*100:>6.2f}%  | {acc_shallow:>6.1f}%    | {acc_deep:>6.1f}%", flush=True)

    print("=" * 75, flush=True)

    # Save results dictionary for plotting
    torch.save(
        {
            "results": [
                {
                    "name": r["name"],
                    "acc": r["acc"],
                    "ece": r["ece"],
                    "depth_metrics": r["depth_metrics"],
                    "gate_summary": r["gate_summary"],
                }
                for r in results
            ]
        },
        "arena_results.pt",
    )
    print("Results saved to arena_results.pt", flush=True)


if __name__ == "__main__":
    main()
