"""
Rigorous Local PCPR Experiment (First-Principles Verification)
============================================================
Author: Jevformer Research Team (Leon & Ilya Sutskever persona)

Hypothesis:
  When graph path edges are guaranteed present in the scrambled sequence:
  1. A shallow System 1 (2 layers) can solve k=0 and k=1 hops, but fails on k >= 2.
  2. A recurrent System 2 deliberator (Universal Transformer recurrence) solves k >= 2.
  3. The Jev epistemic head (Noul), calibrated via RLCD proper scoring, learns to
     confidently exit (p ~ 1.0) on k in {0, 1} and route (p ~ 0.0) on k >= 2.
  4. CALM (softmax entropy) suffers from overconfidence on unsolvable depths.

Tested locally on CUDA before considering any cloud scaling.
"""

import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# 1. Mathematically Sound Permutation Path Tracing Dataset
# ----------------------------------------------------------------------

def generate_pcpr_batch(
    batch_size: int,
    num_nodes: int = 16,
    max_hops: int = 3,
    num_edges: int = 10,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Guarantees:
      - Sequence length: 2 * num_edges + 2 tokens.
      - Exactly k path edges (S -> u_1 -> ... -> u_k) are generated and guaranteed included.
      - Remaining (num_edges - k) slots are filled with random distractor edges.
      - Edges are randomly shuffled so token positions contain zero order hints.
      - Query (S, ?) is placed at sequence end.
      - Target is the true path endpoint: u_k (or S if k=0).
    """
    seq_len = 2 * num_edges + 2
    tokens = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    true_hops = torch.randint(0, max_hops + 1, (batch_size,), device=device)

    for i in range(batch_size):
        k = int(true_hops[i].item())
        # Choose start node
        nodes = list(range(num_nodes))
        random.shuffle(nodes)
        
        path = nodes[: k + 1]  # path of length k: path[0] -> path[1] -> ... -> path[k]
        start_node = path[0]
        end_node = path[-1]
        labels[i] = end_node % 8  # 8 classes

        path_edges = []
        used_sources = set()
        for h in range(k):
            path_edges.append((path[h], path[h + 1]))
            used_sources.add(path[h])

        # Distractor edges
        distractors = []
        attempts = 0
        while len(path_edges) + len(distractors) < num_edges and attempts < 100:
            attempts += 1
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            if u != v and u not in used_sources:
                distractors.append((u, v))
                used_sources.add(u)

        # Fill any remaining slots if needed
        while len(path_edges) + len(distractors) < num_edges:
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            distractors.append((u, v))

        all_edges = path_edges + distractors
        random.shuffle(all_edges)

        # Write to tokens (nodes mapped to vocab: 10 + node_id)
        idx = 0
        for u, v in all_edges:
            tokens[i, idx] = u + 10
            tokens[i, idx + 1] = v + 10
            idx += 2

        # Query token at end: [Start Node, Query Symbol (1)]
        tokens[i, -2] = start_node + 10
        tokens[i, -1] = 1

    return tokens, labels, true_hops


# ----------------------------------------------------------------------
# 2. Architectures: S1 Trunk, Jev Epistemic Head, S2 Deliberator
# ----------------------------------------------------------------------

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

    def forward(self, h_query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        choice_logits = self.choice_proj(h_query)
        noul_prob = torch.sigmoid(self.noul_proj(h_query))
        score_logits = self.score_proj(h_query)
        return choice_logits, noul_prob, score_logits


class RecurrentDeliberator(nn.Module):
    """Universal Transformer style recurrent deliberation block with residual connection."""
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

    def forward(self, h_seq: torch.Tensor, steps: int) -> torch.Tensor:
        cur = h_seq
        for _ in range(steps):
            cur = self.layer(cur) + cur  # recurrent residual
        h_query = self.norm(cur[:, -1, :])
        return self.out_proj(h_query)


class LocalJevformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 64,
        d_model: int = 128,
        num_heads: int = 4,
        s1_layers: int = 2,
        max_hops: int = 3,
        num_classes: int = 8,
    ):
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
        self.jev_head = JevHead(d_model, num_classes=num_classes, max_steps=max_hops)
        self.deliberator = RecurrentDeliberator(d_model, num_heads, num_classes)
        self.max_hops = max_hops

    def forward(self, x: torch.Tensor, mode: str = "jev", threshold: float = 0.5):
        b, seq_len = x.shape
        h = self.token_emb(x) + self.pos_emb[:, :seq_len, :]
        h_seq = self.trunk(h)
        h_query = h_seq[:, -1, :]

        s1_logits, noul_p, score_logits = self.jev_head(h_query)
        noul_conf = noul_p.squeeze(-1)

        # CALM baseline confidence: 1 - normalized entropy
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


# ----------------------------------------------------------------------
# 3. Local Training & Empirical Validation
# ----------------------------------------------------------------------

def run_local_validation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Local First-Principles Verification on {device} ===")
    
    model = LocalJevformer(d_model=128, num_heads=4, s1_layers=2, max_hops=3).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1500, eta_min=1e-5)

    print("\n--- Training (1500 Steps) ---")
    start_t = time.time()
    
    for step in range(1, 1501):
        model.train()
        x, y, true_hops = generate_pcpr_batch(64, num_nodes=16, max_hops=3, num_edges=10, device=device)
        
        # S1 forward
        b, seq_len = x.shape
        h = model.token_emb(x) + model.pos_emb[:, :seq_len, :]
        h_seq = model.trunk(h)
        h_query = h_seq[:, -1, :]
        s1_logits, noul_p, score_logits = model.jev_head(h_query)
        noul_conf = noul_p.squeeze(-1)

        # S2 forward (full deliberation)
        s2_logits = model.deliberator(h_seq, steps=model.max_hops)

        # Task losses
        loss_s1 = F.cross_entropy(s1_logits, y)
        loss_s2 = F.cross_entropy(s2_logits, y)

        # RLCD proper scoring target: z = 1 if S1 is correct, 0 otherwise
        s1_correct = (torch.argmax(s1_logits, dim=-1) == y).float().detach()
        loss_brier = F.mse_loss(noul_conf, s1_correct)

        # Step prediction loss
        loss_depth = F.cross_entropy(score_logits, true_hops)

        # Compute efficiency regularization
        loss_compute = torch.mean(1.0 - noul_conf) * 0.05

        total_loss = loss_s1 + loss_s2 + (2.0 * loss_brier) + (0.3 * loss_depth) + loss_compute

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 300 == 0 or step == 1500:
            model.eval()
            with torch.no_grad():
                acc_s1 = (torch.argmax(s1_logits, dim=-1) == y).float().mean().item() * 100.0
                acc_s2 = (torch.argmax(s2_logits, dim=-1) == y).float().mean().item() * 100.0
                k01_mask = true_hops <= 1
                k23_mask = true_hops >= 2
                conf_easy = noul_conf[k01_mask].mean().item() if k01_mask.any() else 0.0
                conf_hard = noul_conf[k23_mask].mean().item() if k23_mask.any() else 0.0
                print(
                    f"Step {step:04d} | Loss: {total_loss.item():.3f} | "
                    f"S1 Acc: {acc_s1:.1f}% | S2 Acc: {acc_s2:.1f}% | "
                    f"Noul(k<=1): {conf_easy:.3f} | Noul(k>=2): {conf_hard:.3f}"
                )

    print(f"Training finished in {time.time() - start_t:.1f}s")

    # ------------------------------------------------------------------
    # 4. Rigorous Holdout Evaluation (1,000 Unseen Graphs)
    # ------------------------------------------------------------------
    print("\n=== Holdout Evaluation on 1,000 Unseen Graphs ===")
    model.eval()
    with torch.no_grad():
        x_val, y_val, hops_val = generate_pcpr_batch(1000, num_nodes=16, max_hops=3, num_edges=10, device=device)
        
        # Per-Hop Accuracy Breakdown
        print("\n[Per-Hop Accuracy Analysis (S1 vs S2)]:")
        for h in range(4):
            mask = hops_val == h
            if not mask.any():
                continue
            x_h = x_val[mask]
            y_h = y_val[mask]
            
            # S1
            out_s1 = model(x_h, mode="uniform_s1")
            acc_s1_h = (torch.argmax(out_s1["final_logits"], dim=-1) == y_h).float().mean().item() * 100.0
            
            # S2
            out_s2 = model(x_h, mode="uniform_s2")
            acc_s2_h = (torch.argmax(out_s2["final_logits"], dim=-1) == y_h).float().mean().item() * 100.0
            
            mean_noul = out_s1["noul_conf"].mean().item()
            mean_calm = out_s1["calm_conf"].mean().item()
            
            print(f"  Hop {h}: Count={mask.sum().item():3d} | S1: {acc_s1_h:5.1f}% | S2: {acc_s2_h:5.1f}% | Noul Conf: {mean_noul:.3f} | CALM Conf: {mean_calm:.3f}")

        # Calibration (ECE)
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

        out_all = model(x_val, mode="uniform_s1")
        s1_acc_all = (torch.argmax(out_all["final_logits"], dim=-1) == y_val).float().cpu().numpy()
        noul_confs = out_all["noul_conf"].cpu().numpy()
        calm_confs = out_all["calm_conf"].cpu().numpy()

        ece_jev = calc_ece(noul_confs, s1_acc_all)
        ece_calm = calc_ece(calm_confs, s1_acc_all)

        print(f"\n[Expected Calibration Error on Holdout]:")
        print(f"  * Jevformer (Noul RLCD): {ece_jev * 100:.2f}%")
        print(f"  * CALM (Softmax Entropy): {ece_calm * 100:.2f}%")

        # Threshold Pareto Sweep
        print("\n[Pareto Frontier: Jevformer vs CALM]")
        print("  Jevformer Routing (RLCD Proper Scoring):")
        for tau in [0.3, 0.5, 0.7, 0.9]:
            out_j = model(x_val, mode="jev", threshold=tau)
            acc = (torch.argmax(out_j["final_logits"], dim=-1) == y_val).float().mean().item() * 100.0
            exit_rate = out_j["exit_s1"].float().mean().item() * 100.0
            rel_flops = exit_rate * 0.20 + (100.0 - exit_rate) * 1.0
            print(f"    tau={tau:.2f} -> Accuracy: {acc:5.1f}% | S1 Exits: {exit_rate:5.1f}% | Relative FLOPs: {rel_flops:5.1f}%")

        print("  CALM Routing (Softmax Entropy Threshold):")
        for tau in [0.3, 0.5, 0.7, 0.9]:
            out_c = model(x_val, mode="calm", threshold=tau)
            acc = (torch.argmax(out_c["final_logits"], dim=-1) == y_val).float().mean().item() * 100.0
            exit_rate = out_c["exit_s1"].float().mean().item() * 100.0
            rel_flops = exit_rate * 0.20 + (100.0 - exit_rate) * 1.0
            print(f"    tau={tau:.2f} -> Accuracy: {acc:5.1f}% | S1 Exits: {exit_rate:5.1f}% | Relative FLOPs: {rel_flops:5.1f}%")

    print("\nLocal experiment completed successfully with ZERO cloud compute cost.")


if __name__ == "__main__":
    run_local_validation()
