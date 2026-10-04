"""
Crux experiment for the epiplexity self-play proposal.

Question being tested (the falsifiable core of the document under review):
does an epiplexity proxy, combined with mechanistic relatedness between
families, predict cross-family transfer better than either (a) raw
difficulty/final-loss alone, or (b) typedness/validity alone?

Method:
  1. Build a small zoo of 7 token-stream families spanning controls
     (iid, shuffled-structured) and structured generators (periodic,
     modular-add, modular-mul, a finite-state machine, and the typed
     stack-VM from the earlier toy).
  2. Train a fresh tiny learner from scratch on each family, track its
     held-out loss curve, and compute a per-family epiplexity proxy
     (area of loss above its own asymptote, normalised by steps) --
     the same "inexpensive heuristic" used in Finzi et al. (2026).
  3. For six family pairs (related and unrelated), continue-train the
     family-A model on family B and compare against a fresh model
     trained on B from scratch, for a matched step budget. This gives
     a transfer score R(A->B).
  4. Compare the epiplexity ranking, the final-loss ("difficulty")
     ranking, and the transfer matrix against each other.

All families share one vocabulary so losses are directly comparable.
Numpy only (no torch in this sandbox) -- same tiny order-K Markov MLP
architecture as the first toy, refactored into a class so multiple
independent models can be trained and continue-trained.
"""
import sys
import numpy as np
import random

# Rule 1: Windows Unicode encoding hygiene
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

P = 17          # prime modulus for the arithmetic families
SEP = 17
VOCAB = 18
EPISODE_LEN = 24
K = 6           # learner context window
EMB, HID = 8, 48
LR = 0.05

# ---------------------------------------------------------------------------
# Family generators. Each returns ONE episode (list[int]); a stream is many
# episodes concatenated with SEP between them, each episode drawing fresh
# random instance parameters -- so the learner must infer structure
# in-context, not memorise a single global mapping.
# ---------------------------------------------------------------------------
def ep_iid(rng):
    return [int(rng.integers(0, P)) for _ in range(EPISODE_LEN)]

def ep_periodic(rng):
    p = int(rng.integers(2, 5))
    pattern = [int(rng.integers(0, P)) for _ in range(p)]
    return [pattern[i % p] for i in range(EPISODE_LEN)]

def ep_modadd(rng):
    step = int(rng.integers(1, P))
    x = int(rng.integers(0, P))
    out = []
    for _ in range(EPISODE_LEN):
        out.append(x); x = (x + step) % P
    return out

def ep_modmul(rng):
    step = int(rng.integers(1, P))
    x = int(rng.integers(1, P))
    out = []
    for _ in range(EPISODE_LEN):
        out.append(x); x = (x * step) % P
    return out

def ep_modadd_shuffled(rng):
    ep = ep_modadd(rng)
    random.Random(int(rng.integers(0, 1 << 30))).shuffle(ep)
    return ep

def ep_fsm(rng, n_states=6):
    trans = [int(rng.integers(0, n_states)) for _ in range(n_states)]
    emit = [int(rng.integers(0, P)) for _ in range(n_states)]
    s = int(rng.integers(0, n_states))
    out = []
    for _ in range(EPISODE_LEN):
        out.append(emit[s]); s = trans[s]
    return out

# typed stack VM (int 0-9 / bool) from the earlier toy, values reused as-is
def _op_add(v): return [(v[0][1] + v[1][1]) % 10]
def _op_sub(v): return [abs(v[0][1] - v[1][1]) % 10]
def _op_mul(v): return [(v[0][1] * v[1][1]) % 10]
def _op_and(v): return [v[0][1] and v[1][1]]
def _op_or(v):  return [v[0][1] or v[1][1]]
def _op_not(v): return [not v[0][1]]
def _op_eq(v):  return [v[0][1] == v[1][1]]
def _op_gt(v):  return [v[0][1] > v[1][1]]
_OPS = [("ADD", ["I", "I"], ["I"], _op_add), ("SUB", ["I", "I"], ["I"], _op_sub),
        ("MUL", ["I", "I"], ["I"], _op_mul), ("AND", ["B", "B"], ["B"], _op_and),
        ("OR", ["B", "B"], ["B"], _op_or), ("NOT", ["B"], ["B"], _op_not),
        ("EQ", ["I", "I"], ["B"], _op_eq), ("GT", ["I", "I"], ["B"], _op_gt)]

def ep_stack(rng):
    stack_t, stack_v, trace = [], [], []
    for _ in range(EPISODE_LEN):
        choices = [("PUSHI", None), ("PUSHB", None)]
        if stack_t: choices.append(("DUP", None))
        for (name, ins, outs, fn) in _OPS:
            kk = len(ins)
            if len(stack_t) >= kk and stack_t[-kk:] == ins:
                choices.append((name, (ins, outs, fn)))
        name, meta = choices[int(rng.integers(0, len(choices)))]
        if name == "PUSHI":
            v = int(rng.integers(0, 10)); stack_t.append("I"); stack_v.append(("I", v))
        elif name == "PUSHB":
            v = bool(rng.integers(0, 2)); stack_t.append("B"); stack_v.append(("B", v))
        elif name == "DUP":
            stack_t.append(stack_t[-1]); stack_v.append(stack_v[-1])
        else:
            ins, outs, fn = meta; kk = len(ins)
            popped = stack_v[-kk:]; del stack_v[-kk:]; del stack_t[-kk:]
            for t, v in zip(outs, fn(popped)):
                stack_t.append(t); stack_v.append((t, v))
        top_t, top_v = stack_v[-1]
        trace.append(int(top_v) if top_t == "I" else (10 if top_v else 11))
    return trace

FAMILIES = {
    "iid": ep_iid, "periodic": ep_periodic, "modadd": ep_modadd,
    "modmul": ep_modmul, "modadd_shuf": ep_modadd_shuffled,
    "fsm": ep_fsm, "stack": ep_stack,
}

def make_stream(family, n_tokens, rng):
    toks = []
    fn = FAMILIES[family]
    while len(toks) < n_tokens:
        toks.extend(fn(rng)); toks.append(SEP)
    return np.array(toks[:n_tokens])

# ---------------------------------------------------------------------------
# Tiny order-K Markov MLP, as an object so multiple independent instances
# (and continue-training of a copy) are easy to manage.
# ---------------------------------------------------------------------------
class MarkovMLP:
    def __init__(self, rng):
        self.E = rng.normal(0, 0.1, (VOCAB, EMB))
        self.W1 = rng.normal(0, 0.1, (K * EMB, HID)); self.b1 = np.zeros(HID)
        self.W2 = rng.normal(0, 0.1, (HID, VOCAB)); self.b2 = np.zeros(VOCAB)

    def copy(self):
        m = MarkovMLP.__new__(MarkovMLP)
        m.E, m.W1, m.b1 = self.E.copy(), self.W1.copy(), self.b1.copy()
        m.W2, m.b2 = self.W2.copy(), self.b2.copy()
        return m

    def _forward(self, ctx):
        emb = self.E[ctx].reshape(-1)
        h = np.tanh(emb @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        logits -= logits.max()
        p = np.exp(logits); p /= p.sum()
        return emb, h, p

    def loss_only(self, ctx, target):
        _, _, p = self._forward(ctx)
        return -np.log(p[target] + 1e-9)

    def train_step(self, ctx, target):
        emb, h, p = self._forward(ctx)
        loss = -np.log(p[target] + 1e-9)
        dlogits = p.copy(); dlogits[target] -= 1
        dW2 = np.outer(h, dlogits); db2 = dlogits
        dh = (dlogits @ self.W2.T) * (1 - h ** 2)
        dW1 = np.outer(emb, dh); db1 = dh
        demb = dh @ self.W1.T
        self.W2 -= LR * dW2; self.b2 -= LR * db2
        self.W1 -= LR * dW1; self.b1 -= LR * db1
        self.E[ctx] -= (LR * demb.reshape(K, EMB))
        return loss

    def eval_loss(self, stream, rng, n_samples=200):
        idx = rng.integers(K, len(stream) - 1, size=n_samples)
        return float(np.mean([self.loss_only(stream[i-K:i], stream[i]) for i in idx]))


TRAIN_STEPS = 800
LOG_EVERY = 20
STREAM_LEN = 6000
FT_STEPS = 300
SEEDS = [0, 1, 2]


def train_from_scratch(family, seed):
    rng = np.random.default_rng(seed)
    model = MarkovMLP(rng)
    stream = make_stream(family, STREAM_LEN, rng)
    curve = []
    for step in range(TRAIN_STEPS):
        i = rng.integers(K, len(stream) - 1)
        model.train_step(stream[i-K:i], stream[i])
        if step % LOG_EVERY == 0 or step == TRAIN_STEPS - 1:
            eval_stream = make_stream(family, 1000, rng)
            curve.append(model.eval_loss(eval_stream, rng))
    return model, curve


def continue_train(model, family, seed, steps=FT_STEPS):
    rng = np.random.default_rng(seed + 1000)
    stream = make_stream(family, STREAM_LEN, rng)
    curve = []
    for step in range(steps):
        i = rng.integers(K, len(stream) - 1)
        model.train_step(stream[i-K:i], stream[i])
        if step % LOG_EVERY == 0 or step == steps - 1:
            eval_stream = make_stream(family, 1000, rng)
            curve.append(model.eval_loss(eval_stream, rng))
    return curve


def epiplexity_proxy(curve):
    curve = np.array(curve)
    final = curve[-1]
    excess = np.clip(curve - final, 0, None)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(excess) / len(excess))


# ---------------------------------------------------------------------------
# 1. Per-family epiplexity + difficulty (final loss), averaged over seeds
# ---------------------------------------------------------------------------
print("=== Per-family epiplexity proxy and final loss (mean +/- std over 3 seeds) ===")
family_models = {}   # (family, seed) -> trained model, for reuse in transfer test
family_stats = {}
for fam in FAMILIES:
    epis, finals = [], []
    for seed in SEEDS:
        model, curve = train_from_scratch(fam, seed)
        family_models[(fam, seed)] = model
        epis.append(epiplexity_proxy(curve))
        finals.append(curve[-1])
    family_stats[fam] = (np.mean(epis), np.std(epis), np.mean(finals), np.std(finals))
    print(f"{fam:12s}  epiplexity={np.mean(epis):6.3f}+/-{np.std(epis):.3f}   "
          f"final_loss={np.mean(finals):6.3f}+/-{np.std(finals):.3f}")

print(f"\n(uniform-random baseline loss = ln({VOCAB}) = {np.log(VOCAB):.3f} nats)\n")

# ---------------------------------------------------------------------------
# 2. Cross-family transfer test
# ---------------------------------------------------------------------------
PAIRS = [
    ("modadd", "modmul"),          # related: shared affine-recurrence mechanism
    ("modmul", "modadd"),          # related, reverse direction
    ("iid", "modmul"),             # unrelated / null: nothing to transfer
    ("modadd_shuf", "modmul"),     # same marginal stats as modadd, no order structure
    ("stack", "fsm"),              # both "structural" but different mechanism
    ("fsm", "stack"),              # reverse direction
]

print("=== Transfer test: R(A->B) = mean_loss(scratch on B) - mean_loss(pretrained-on-A, then B) ===")
print(f"{'A->B':16s} {'R (mean+/-std)':>18s}   positive = pretraining on A helped on B")
transfer_results = {}
for (A, B) in PAIRS:
    Rs = []
    for seed in SEEDS:
        pretrained = family_models[(A, seed)].copy()
        pretrained_curve = continue_train(pretrained, B, seed)
        scratch = MarkovMLP(np.random.default_rng(seed + 2000))
        scratch_curve = continue_train(scratch, B, seed)
        R = float(np.mean(scratch_curve) - np.mean(pretrained_curve))
        Rs.append(R)
    transfer_results[(A, B)] = (np.mean(Rs), np.std(Rs))
    print(f"{A+'->'+B:16s} {np.mean(Rs):8.3f} +/- {np.std(Rs):.3f}")

# ---------------------------------------------------------------------------
# 3. Compare rankings: does epiplexity (not just low final loss / "difficulty")
#    predict which pretraining source actually transfers?
# ---------------------------------------------------------------------------
print("\n=== Ranking comparison ===")
print("Families ranked by epiplexity (desc):",
      sorted(FAMILIES, key=lambda f: -family_stats[f][0]))
print("Families ranked by final loss / 'ease' (asc = easiest first):",
      sorted(FAMILIES, key=lambda f: family_stats[f][2]))
