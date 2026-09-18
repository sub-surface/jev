"""
Uniform-Depth Dyck-3 Generator
==============================
Guarantees equal representation across nesting depths D in {1, 2, 3, 4, 5}.
"""

import random
import torch

OPEN_TO_CLOSE = {1: 2, 3: 4, 5: 6}
CLOSE_TO_OPEN = {2: 1, 4: 3, 6: 5}

def make_exact_depth_dyck(target_depth: int, seq_len: int = 24) -> tuple[list[int], bool, int]:
    # Build a spine of exact target_depth
    # e.g. ( [ { ... } ] )
    b_types = [random.choice([1, 3, 5]) for _ in range(target_depth)]
    c_types = [OPEN_TO_CLOSE[b] for b in reversed(b_types)]
    
    # Fill remaining space with shallow pairs () or [] of depth 1
    rem = seq_len - 1 - (2 * target_depth)
    extra = []
    while len(extra) < rem - 1:
        b = random.choice([1, 3, 5])
        extra.extend([b, OPEN_TO_CLOSE[b]])
    
    # Assemble
    seq = b_types + c_types + extra
    seq = seq[: seq_len - 1]
    
    is_balanced = random.random() < 0.5
    if not is_balanced:
        # Corrupt one bracket
        idx = random.randint(0, len(seq) - 1)
        seq[idx] = random.choice([1, 2, 3, 4, 5, 6])

    tokens = seq + [7] # query
    while len(tokens) < seq_len:
        tokens.insert(0, 0)

    # Compute actual max depth and balancedness
    stack = []
    bal = True
    cur_d = 0
    max_d = 0
    for tok in tokens:
        if tok in OPEN_TO_CLOSE:
            stack.append(tok)
            cur_d += 1
            max_d = max(max_d, cur_d)
        elif tok in CLOSE_TO_OPEN:
            if not stack or stack[-1] != CLOSE_TO_OPEN[tok]:
                bal = False
                break
            stack.pop()
            cur_d -= 1
    if stack:
        bal = False

    return tokens, bal, max_d

def generate_balanced_dyck_dataset(samples_per_depth: int = 800, max_depth: int = 5, seq_len: int = 24):
    all_tokens = []
    all_labels = []
    all_depths = []

    for d in range(1, max_depth + 1):
        for _ in range(samples_per_depth):
            toks, bal, actual_d = make_exact_depth_dyck(d, seq_len)
            all_tokens.append(toks)
            all_labels.append(1 if bal else 0)
            all_depths.append(d)

    # Shuffle
    idx = list(range(len(all_tokens)))
    random.shuffle(idx)
    return (
        torch.tensor([all_tokens[i] for i in idx], dtype=torch.long),
        torch.tensor([all_labels[i] for i in idx], dtype=torch.long),
        torch.tensor([all_depths[i] for i in idx], dtype=torch.long),
    )

if __name__ == "__main__":
    t, l, d = generate_balanced_dyck_dataset(samples_per_depth=200)
    print("Dataset shape:", t.shape)
    for depth in range(1, 6):
        print(f"Depth {depth}: {(d == depth).sum().item()} samples | Balanced: {(l[d == depth] == 1).float().mean().item()*100:.1f}%")
