"""
Fast Local PCPR & Epiplexity Validation
=======================================
Pre-generates tensor datasets to avoid python loop overhead.
Evaluates:
  1. Epiplexity / Computational Complexity Gap: Delta_epi(x) = H_{S1}(x) - H_{S2}(x)
  2. Bounded Observer Solvability: S1 (2 layers) vs S2 (Recurrent Deliberator, 4 unrolls)
  3. Epistemic Calibration: Jev Noul (RLCD) vs CALM (Softmax Entropy)
"""

import sys
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_pcpr_dataset(num_samples: int, num_nodes: int = 16, max_hops: int = 3, num_edges: int = 8):
    seq_len = 2 * num_edges + 2
    tokens = torch.zeros(num_samples, seq_len, dtype=torch.long)
    labels = torch.zeros(num_samples, dtype=torch.long)
    true_hops = torch.randint(0, max_hops + 1, (num_samples,))

    for i in range(num_samples):
        k = int(true_hops[i].item())
        perm = list(range(num_nodes))
        random.shuffle(perm)
        path = perm[: k + 1]
        start_node = path[0]
        end_node = path[-1]
        labels[i] = end_node % 8

        path_edges = [(path[h], path[h + 1]) for h in range(k)]
        used_u = {p[0] for p in path_edges}

        distractors = []
        for _ in range(num_edges - k):
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            distractors.append((u, v))

        all_edges = path_edges + distractors
        random.shuffle(all_edges)

        idx = 0
        for u, v in all_edges:
            tokens[i, idx] = u + 10
            tokens[i, idx + 1] = v + 10
            idx += 2

        tokens[i, -2] = start_node + 10
        tokens[i, -1] = 1

    return tokens, labels, true_hops


class JevHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int = 8, max_steps: int = 3):
        super().__init__()
        self.choice_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes),
        )
        self.noul_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        self.score_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, max_steps + 1),
        )

    def forward(self, h_query: torch.Tensor):
        choice_logits = self.choice_proj(h_query)
        noul_prob = torch.sigmoid(self.noul_proj(h_query))
        score_logits = self.score_proj(h_query)
        return choice_logits, noul_prob, score_logits


class RecurrentDeliberator(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_classes: int = 8):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, num_classes)

    def forward(self, h_seq: torch.Tensor, steps: int):
        cur = h_seq
        for _ in range(steps):
            cur = self.layer(cur) + cur
        h_query = self.norm(cur[:, -1, :])
        return self.out_proj(h_query)


class LocalJevformer(nn.Module):
    def __init__(self, vocab_size: int = 64, d_model: int = 128, num_heads: int = 4, s1_layers: int = 2, max_hops: int = 3):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(encoder_layer, num_layers=s1_layers)
        self.jev_head = JevHead(d_model, num_classes=8, max_steps=max_hops)
        self.deliberator = RecurrentDeliberator(d_model, num_heads, num_classes=8)
        self.max_hops = max_hops

    def forward(self, x: torch.Tensor, mode: str = "jev", threshold: float = 0.5):
        b, seq_len = x.shape
        h = self.token_emb(x) + self.pos_emb[:, :seq_len, :]
        h_seq = self.trunk(h)
        h_query = h_seq[:, -1, :]

        s1_logits, noul_p, score_logits = self.jev_head(h_query)
        noul_conf = noul_p.squeeze(-1)

        probs = F.softmax(s1_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        calm_conf = 1.0 - (entropy / math.log(probs.size(-1)))

        if mode == "jev":
            exit_s1 = noul_conf >= threshold
        elif mode == "calm":
            exit_s1 = calm_conf >= threshold
        elif mode == "uniform_s1":
            exit_s1 = torch.ones(b, dtype=torch.bool, device=x.device)
        elif mode == "uniform_s2":
            exit_s1 = torch.zeros(b, dtype=torch.bool, device=x.device)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        s2_logits = None
        if not exit_s1.all():
            s2_logits = self.deliberator(h_seq, steps=self.max_hops)
            final_logits = torch.where(exit_s1.unsqueeze(-1), s1_logits, s2_logits)
        else:
            final_logits = s1_logits

        return {
            "final_logits": final_logits,
            "s1_logits": s1_logits,
            "s2_logits": s2_logits,
            "noul_conf": noul_conf,
            "calm_conf": calm_conf,
            "exit_s1": exit_s1,
            "score_logits": score_logits,
        }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device]: {device}", flush=True)

    # 1. Pre-generate data
    print("Generating pre-computed PCPR graphs...", flush=True)
    t0 = time.time()
    train_x, train_y, train_hops = generate_pcpr_dataset(4000)
    test_x, test_y, test_hops = generate_pcpr_dataset(1000)
    print(f"Data ready in {time.time() - t0:.2f}s (Train: 4000, Test: 1000)", flush=True)

    train_x, train_y, train_hops = train_x.to(device), train_y.to(device), train_hops.to(device)
    test_x, test_y, test_hops = test_x.to(device), test_y.to(device), test_hops.to(device)

    model = LocalJevformer(vocab_size=64, d_model=128, num_heads=4, s1_layers=2, max_hops=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)

    batch_size = 64
    num_batches = len(train_x) // batch_size
    epochs = 15

    print(f"\n--- Training for {epochs} epochs ({epochs * num_batches} steps) ---", flush=True)
    start_train = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(train_x))
        epoch_loss = 0.0

        for b_idx in range(num_batches):
            idx = perm[b_idx * batch_size : (b_idx + 1) * batch_size]
            x, y, hops = train_x[idx], train_y[idx], train_hops[idx]

            b, seq_len = x.shape
            h = model.token_emb(x) + model.pos_emb[:, :seq_len, :]
            h_seq = model.trunk(h)
            h_query = h_seq[:, -1, :]

            s1_logits, noul_p, score_logits = model.jev_head(h_query)
            noul_conf = noul_p.squeeze(-1)
            s2_logits = model.deliberator(h_seq, steps=model.max_hops)

            # Epiplexity & Task Loss formulation:
            # Loss S1 and Loss S2 represent time-bounded cross-entropy H_{S1} and H_{S2}
            loss_s1 = F.cross_entropy(s1_logits, y)
            loss_s2 = F.cross_entropy(s2_logits, y)

            # Calibration target: z = 1 if S1 correct, else 0
            s1_correct = (torch.argmax(s1_logits, dim=-1) == y).float().detach()
            loss_brier = F.mse_loss(noul_conf, s1_correct)
            loss_depth = F.cross_entropy(score_logits, hops)

            total_loss = loss_s1 + loss_s2 + (2.0 * loss_brier) + (0.3 * loss_depth)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += total_loss.item()

        if epoch % 3 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                out_eval = model(test_x, mode="uniform_s1")
                acc_s1 = (torch.argmax(out_eval["final_logits"], dim=-1) == test_y).float().mean().item() * 100.0
                out_s2 = model(test_x, mode="uniform_s2")
                acc_s2 = (torch.argmax(out_s2["final_logits"], dim=-1) == test_y).float().mean().item() * 100.0
                conf_easy = out_eval["noul_conf"][test_hops <= 1].mean().item()
                conf_hard = out_eval["noul_conf"][test_hops >= 2].mean().item()

                print(
                    f"Epoch {epoch:02d}/{epochs} | Avg Loss: {epoch_loss/num_batches:.3f} | "
                    f"S1 Acc: {acc_s1:.1f}% | S2 Acc: {acc_s2:.1f}% | "
                    f"Noul(Hop<=1): {conf_easy:.3f} | Noul(Hop>=2): {conf_hard:.3f}",
                    flush=True,
                )

    print(f"\nTraining completed in {time.time() - start_train:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Rigorous Evaluation & Epiplexity Analysis
    # ------------------------------------------------------------------
    print("\n=======================================================", flush=True)
    print(" Rigorous Empirical Holdout Evaluation (1,000 Graphs)", flush=True)
    print("=======================================================", flush=True)

    model.eval()
    with torch.no_grad():
        # 1. Per-Hop Accuracy & Epiplexity Gap
        # Delta_epi(k) = CrossEntropy(S1) - CrossEntropy(S2)
        print("\n[Per-Hop Circuit Complexity Breakdown]:", flush=True)
        for h in range(4):
            mask = test_hops == h
            xh, yh = test_x[mask], test_y[mask]
            out1 = model(xh, mode="uniform_s1")
            out2 = model(xh, mode="uniform_s2")

            acc1 = (torch.argmax(out1["final_logits"], dim=-1) == yh).float().mean().item() * 100.0
            acc2 = (torch.argmax(out2["final_logits"], dim=-1) == yh).float().mean().item() * 100.0
            h_s1 = F.cross_entropy(out1["final_logits"], yh).item()
            h_s2 = F.cross_entropy(out2["final_logits"], yh).item()
            delta_epi = h_s1 - h_s2  # Structural epiplexity unlocked by S2 compute

            mean_noul = out1["noul_conf"].mean().item()
            mean_calm = out1["calm_conf"].mean().item()

            print(
                f"  Hop {h} (N={mask.sum().item():3d}): "
                f"S1 Acc={acc1:5.1f}% | S2 Acc={acc2:5.1f}% | "
                f"H_S1={h_s1:.2f} | H_S2={h_s2:.2f} | "
                f"Delta_epi={delta_epi:+.2f} | "
                f"Noul Conf={mean_noul:.3f} | CALM Conf={mean_calm:.3f}",
                flush=True,
            )

        # 2. Calibration (ECE)
        def calc_ece(confs, corrects, num_bins=10):
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

        out_all = model(test_x, mode="uniform_s1")
        s1_acc_all = (torch.argmax(out_all["final_logits"], dim=-1) == test_y).float().cpu().numpy()
        noul_confs = out_all["noul_conf"].cpu().numpy()
        calm_confs = out_all["calm_conf"].cpu().numpy()

        ece_jev = calc_ece(noul_confs, s1_acc_all)
        ece_calm = calc_ece(calm_confs, s1_acc_all)

        print(f"\n[Expected Calibration Error]:", flush=True)
        print(f"  * Jevformer (Noul RLCD): {ece_jev * 100:.2f}%", flush=True)
        print(f"  * CALM (Softmax Entropy): {ece_calm * 100:.2f}%", flush=True)

        # 3. Pareto Tradeoff
        print("\n[Pareto Frontier Sweep]:", flush=True)
        print("  Jevformer (tau sweep):", flush=True)
        for tau in [0.20, 0.40, 0.60, 0.80]:
            out_j = model(test_x, mode="jev", threshold=tau)
            acc = (torch.argmax(out_j["final_logits"], dim=-1) == test_y).float().mean().item() * 100.0
            exit_rate = out_j["exit_s1"].float().mean().item() * 100.0
            flops = exit_rate * 0.20 + (100.0 - exit_rate) * 1.0
            print(f"    tau={tau:.2f} -> Accuracy: {acc:5.1f}% | S1 Exits: {exit_rate:5.1f}% | FLOPs: {flops:5.1f}%", flush=True)

        print("  CALM (entropy tau sweep):", flush=True)
        for tau in [0.20, 0.40, 0.60, 0.80]:
            out_c = model(test_x, mode="calm", threshold=tau)
            acc = (torch.argmax(out_c["final_logits"], dim=-1) == test_y).float().mean().item() * 100.0
            exit_rate = out_c["exit_s1"].float().mean().item() * 100.0
            flops = exit_rate * 0.20 + (100.0 - exit_rate) * 1.0
            print(f"    tau={tau:.2f} -> Accuracy: {acc:5.1f}% | S1 Exits: {exit_rate:5.1f}% | FLOPs: {flops:5.1f}%", flush=True)


if __name__ == "__main__":
    main()
