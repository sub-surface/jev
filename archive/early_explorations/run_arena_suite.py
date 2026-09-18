"""
Architectural Arena & Mechanistic Interpretability Suite (Stage A)
==================================================================
Authors: Jevformer Research Collective (Leon & Ilya Sutskever persona)

Benchmarking Intertwined Architectures on Balanced Dyck-3:
  1. Baseline Decoupled Jevformer
  2. Jev-Gated Residual Transformer (JGR)
  3. Dual-Stream Epistemic Attention (DSEA)

Generates:
  - figures/fig8_architectural_arena_comparison.png: Depth-wise accuracy & ECE
  - figures/fig9_activation_biology_and_gates.png: Epistemic residual valve & attention maps
"""

import os
import shutil
import time
import math
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from balanced_dyck import generate_balanced_dyck_dataset

os.makedirs("figures", exist_ok=True)
brain_dir = r"C:\Users\Leon\.gemini\antigravity-cli\brain\8a1de2f2-e75e-4bee-b2b1-e5c4e1d9ba96\figures"
os.makedirs(brain_dir, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Architectures
# ----------------------------------------------------------------------

class DecoupledJevformer(nn.Module):
    def __init__(self, vocab_size=16, d_model=128, num_heads=4, total_layers=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 32, d_model) * 0.02)
        self.s1_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4, batch_first=True, norm_first=True)
        self.noul_proj = nn.Sequential(nn.Linear(d_model, d_model//2), nn.Tanh(), nn.Linear(d_model//2, 1))
        self.choice_s1 = nn.Linear(d_model, 2)
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
        }


class JevGatedResidualTransformer(nn.Module):
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

        for layer, noul_gate in zip(self.layers, self.noul_gates):
            q_rep = h[:, -1, :]
            conf = torch.sigmoid(noul_gate(q_rep)).squeeze(-1)
            gamma = 1.0 - conf
            delta_h = layer(h) - h
            h = h + gamma.view(b, 1, 1) * delta_h

            residual_gates.append(gamma)
            noul_confs.append(conf)

        final_logits = self.out_head(h[:, -1, :])
        primary_conf = noul_confs[0]
        return {
            "final_logits": final_logits,
            "noul_conf": primary_conf,
            "all_confs": noul_confs,
            "residual_gates": residual_gates,
        }


class DualStreamEpistemicAttention(nn.Module):
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
                
                "q_epi": nn.Linear(d_epi, d_epi),
                "k_epi": nn.Linear(d_epi, d_epi),
                "update_epi": nn.Sequential(nn.Linear(d_model + d_epi, d_epi), nn.Tanh()),
            })
            for _ in range(total_layers)
        ])
        self.beta = nn.Parameter(torch.ones(1) * 0.5)
        self.noul_head = nn.Sequential(nn.Linear(d_epi, 1), nn.Sigmoid())
        self.out_head = nn.Linear(d_model, 2)

    def forward(self, x, threshold=0.5):
        b, t = x.shape
        s = self.token_emb(x) + self.pos_emb[:, :t, :]
        e = self.epi_emb(x) + self.pos_epi[:, :t, :]
        attention_maps = []

        for layer in self.layers:
            q_s = layer["q_sem"](s).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
            k_s = layer["k_sem"](s).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
            v_s = layer["v_sem"](s).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

            q_e = layer["q_epi"](e).unsqueeze(1)
            k_e = layer["k_epi"](e).unsqueeze(1)

            attn_scores = torch.matmul(q_s, k_s.transpose(-2, -1)) / math.sqrt(self.head_dim)
            epi_scores = torch.matmul(q_e, k_e.transpose(-2, -1)) / math.sqrt(self.d_epi)
            total_scores = attn_scores + self.beta * epi_scores

            attn_weights = F.softmax(total_scores, dim=-1)
            attention_maps.append(attn_weights.detach())

            out_s = torch.matmul(attn_weights, v_s).transpose(1, 2).contiguous().view(b, t, self.d_model)
            s = layer["norm_sem"](s + layer["out_sem"](out_s))
            s = s + layer["ff_sem"](s)

            fused = torch.cat([s, e], dim=-1)
            e = e + layer["update_epi"](fused)

        final_logits = self.out_head(s[:, -1, :])
        noul_conf = self.noul_head(e[:, -1, :]).squeeze(-1)

        return {
            "final_logits": final_logits,
            "noul_conf": noul_conf,
            "attention_maps": attention_maps,
        }


# ----------------------------------------------------------------------
# 2. Execution & Benchmarking
# ----------------------------------------------------------------------

def run_suite():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Execution]: CUDA Hardware Acceleration: {device}", flush=True)

    print("Generating balanced Dyck-3 dataset (800/depth train, 200/depth test)...", flush=True)
    train_x, train_y, train_d = generate_balanced_dyck_dataset(samples_per_depth=800)
    test_x, test_y, test_d = generate_balanced_dyck_dataset(samples_per_depth=200)

    train_x, train_y, train_d = train_x.to(device), train_y.to(device), train_d.to(device)
    test_x, test_y, test_d = test_x.to(device), test_y.to(device), test_d.to(device)

    models = {
        "Baseline Decoupled": DecoupledJevformer(vocab_size=16, d_model=128, num_heads=4, total_layers=4),
        "Jev-Gated Residual (JGR)": JevGatedResidualTransformer(vocab_size=16, d_model=128, num_heads=4, total_layers=4),
        "Dual-Stream Epistemic Attn (DSEA)": DualStreamEpistemicAttention(vocab_size=16, d_model=128, d_epi=32, num_heads=4, total_layers=4),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...", flush=True)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        batch_size = 64
        num_batches = len(train_x) // batch_size
        t0 = time.time()

        for epoch in range(1, 16):
            model.train()
            perm = torch.randperm(len(train_x))
            for b_idx in range(num_batches):
                idx = perm[b_idx * batch_size : (b_idx + 1) * batch_size]
                x, y, d = train_x[idx], train_y[idx], train_d[idx]
                out = model(x)
                logits = out["final_logits"]
                conf = out["noul_conf"]
                loss = F.cross_entropy(logits, y) + 2.0 * F.mse_loss(conf, (torch.argmax(logits, dim=-1) == y).float().detach())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        print(f"Completed in {time.time() - t0:.1f}s", flush=True)

        model.eval()
        with torch.no_grad():
            out_eval = model(test_x)
            preds = torch.argmax(out_eval["final_logits"], dim=-1)
            acc = (preds == test_y).float().mean().item() * 100.0

            depth_accs = []
            depth_confs = []
            for d_val in range(1, 6):
                m = test_d == d_val
                depth_accs.append((preds[m] == test_y[m]).float().mean().item() * 100.0)
                depth_confs.append(out_eval["noul_conf"][m].mean().item())

            # ECE
            confs_np = out_eval["noul_conf"].cpu().numpy()
            corr_np = (preds == test_y).float().cpu().numpy()
            bins = np.linspace(0.0, 1.0, 11)
            ece = 0.0
            for i in range(10):
                mb = (confs_np >= bins[i]) & (confs_np < bins[i + 1])
                if np.any(mb):
                    ece += (np.sum(mb) / len(confs_np)) * abs(np.mean(corr_np[mb]) - np.mean(confs_np[mb]))

            results[name] = {
                "overall_acc": acc,
                "ece": ece,
                "depth_accs": depth_accs,
                "depth_confs": depth_confs,
                "raw_output": out_eval,
                "model": model,
            }
            print(f"[{name}] Overall Acc: {acc:.2f}% | ECE: {ece*100:.2f}% | Depth Accs: {[round(a, 1) for a in depth_accs]}", flush=True)

    # ------------------------------------------------------------------
    # 3. Figure 8: Architectural Arena Comparison
    # ------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig8, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    depth_labels = [f"D={d}" for d in range(1, 6)]
    x_idx = np.arange(len(depth_labels))
    colors = {"Baseline Decoupled": "#1f77b4", "Jev-Gated Residual (JGR)": "#2ca02c", "Dual-Stream Epistemic Attn (DSEA)": "#ff7f0e"}

    for name, r in results.items():
        ax1.plot(x_idx, r["depth_accs"], "o-", label=f"{name} ({r['overall_acc']:.1f}%)", color=colors[name], linewidth=2.4, markersize=7)

    ax1.set_title("A. Dyck-3 Hierarchical Accuracy by Nesting Depth")
    ax1.set_xlabel("Nesting Stack Depth (D)")
    ax1.set_ylabel("Holdout Accuracy (%)")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(depth_labels)
    ax1.set_ylim(40, 105)
    ax1.legend(loc="lower left", framealpha=0.9, fontsize=9.5)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # ECE bar plot
    names = list(results.keys())
    eces = [results[n]["ece"] * 100.0 for n in names]
    bar_colors = [colors[n] for n in names]
    bars = ax2.bar(names, eces, color=bar_colors, alpha=0.85, width=0.5)
    ax2.set_title("B. Expected Calibration Error (ECE)")
    ax2.set_ylabel("ECE (%) [Lower is Better]")
    ax2.grid(True, linestyle="--", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h + 0.05), ha="center", fontweight="bold", fontsize=10)

    fig8.tight_layout()
    f8_local = "figures/fig8_architectural_arena_comparison.png"
    f8_brain = os.path.join(brain_dir, "fig8_architectural_arena_comparison.png")
    fig8.savefig(f8_local, dpi=300)
    shutil.copyfile(f8_local, f8_brain)
    print(f"Saved: {f8_local} and {f8_brain}", flush=True)

    # ------------------------------------------------------------------
    # 4. Figure 9: Mechanistic Interpretability & Activation Biology
    # ------------------------------------------------------------------
    # Inspect JGR residual gates across depths
    jgr_out = results["Jev-Gated Residual (JGR)"]["raw_output"]
    gates = jgr_out["residual_gates"] # list of 4 tensors [B]
    
    # Gate values for shallow (D=1) vs deep (D=5)
    m_d1 = test_d == 1
    m_d5 = test_d == 5

    gate_d1 = [g[m_d1].mean().item() for g in gates]
    gate_d5 = [g[m_d5].mean().item() for g in gates]

    fig9, (ax_g, ax_a) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    layer_idx = np.arange(len(gate_d1))
    ax_g.plot(layer_idx, gate_d1, "s--", color="#1f77b4", linewidth=2.4, markersize=8, label="Shallow Nesting (D=1)")
    ax_g.plot(layer_idx, gate_d5, "o-", color="#d62728", linewidth=2.4, markersize=8, label="Deep Nesting (D=5)")
    ax_g.set_title("A. Epistemic Residual Gate ($\gamma_l = 1 - \pi_l$) across Layers")
    ax_g.set_xlabel("Transformer Layer Index $l$")
    ax_g.set_ylabel("Residual Perturbation Weight $\gamma_l$")
    ax_g.set_xticks(layer_idx)
    ax_g.set_xticklabels([f"Layer {l}" for l in layer_idx])
    ax_g.legend(loc="upper left", framealpha=0.9)
    ax_g.grid(True, linestyle="--", alpha=0.5)

    # Inspect DSEA attention matrix for a test sample
    dsea_out = results["Dual-Stream Epistemic Attn (DSEA)"]["raw_output"]
    attn_layer3 = dsea_out["attention_maps"][3][0, 0].cpu().numpy() # layer 3, head 0, sample 0
    im = ax_a.imshow(attn_layer3, cmap="viridis", aspect="auto")
    ax_a.set_title("B. Epistemic-Steered Attention Map (DSEA Layer 3)")
    ax_a.set_xlabel("Key Token Index")
    ax_a.set_ylabel("Query Token Index")
    fig9.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)

    fig9.tight_layout()
    f9_local = "figures/fig9_activation_biology_and_gates.png"
    f9_brain = os.path.join(brain_dir, "fig9_activation_biology_and_gates.png")
    fig9.savefig(f9_local, dpi=300)
    shutil.copyfile(f9_local, f9_brain)
    print(f"Saved: {f9_local} and {f9_brain}", flush=True)

    print("\nSuite completed successfully. All figures generated and synchronized.", flush=True)


if __name__ == "__main__":
    run_suite()
