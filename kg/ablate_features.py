"""Feature-representation ablation for the node-relevance objective. No LLM calls.

Question: the trained policy puts 99.2% of its first-layer weight energy on 2048
embedding dims and 0.8% on the 4 structural features -- the very features the
heuristic wins with. Does rebalancing the representation beat heuristic_prune's
node-F1 of 0.599 on the same held-out questions?

Pure supervised (the auxiliary loss alone), so this answers the representation
question in minutes instead of a 2-hour RL run.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from sentence_transformers import SentenceTransformer
from retrieve import load_graph, load_embeddings, retrieve, MODEL_NAME, default_device
from compression_policy import build_state_features, EXTRA_FEATURES
from train_compression_policy import load_node_embedding_lookup
from generate_pruning_baselines import heuristic_prune

OUT = Path("output"); CACHE = OUT / "ablation_features.npz"
ck = torch.load(OUT/"compression_policy.pt", weights_only=False)
train_qs, val_qs = set(ck["train_questions"]), set(ck["val_questions"])

if CACHE.exists():
    z = np.load(CACHE, allow_pickle=True)
    X, Y, Hh, Q = z["X"], z["Y"], z["H"], z["Q"]
    print(f"Loaded cached features: {X.shape}")
else:
    qs = json.load(open(OUT/"subset_questions.json", encoding="utf-8"))
    G = load_graph(); lookup = load_node_embedding_lookup(); emb, ids = load_embeddings()
    dev = default_device(); print(f"Embedding {len(qs)} questions on {dev}...")
    model = SentenceTransformer(MODEL_NAME, device=dev)
    Xs, Ys, Hs, Qs = [], [], [], []
    t0 = time.time()
    for i, q in enumerate(qs, 1):
        needed = {t for t, _ in q.get("supporting_facts", [])}
        qe = model.encode([q["question"]], normalize_embeddings=True)[0]
        Gq, seeds = retrieve(q["question"], G, emb, ids, model, top_k=5, hops=2)
        if Gq.number_of_nodes() == 0: continue
        sids = {s for s, _ in seeds}
        nids, feats = build_state_features(Gq, qe, lookup, sids, include_embeddings=True)
        hk = set(heuristic_prune(Gq, sids, 1).nodes())
        Xs.append(feats.astype(np.float32))
        Ys.append(np.array([1 if set(Gq.nodes[n].get("sources", [])) & needed else 0 for n in nids], np.float32))
        Hs.append(np.array([1 if n in hk else 0 for n in nids], np.float32))
        Qs.append(np.array([q["question"]] * len(nids), object))
        if i % 25 == 0:
            el = time.time() - t0
            print(f"  [{i}/{len(qs)}] {el:.0f}s elapsed, ETA {el/i*(len(qs)-i):.0f}s")
    X, Y, Hh, Q = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Hs), np.concatenate(Qs)
    np.savez_compressed(CACHE, X=X, Y=Y, H=Hh, Q=Q)
    print(f"Cached -> {CACHE}  {X.shape}")

tr = np.array([q in train_qs for q in Q]); va = np.array([q in val_qs for q in Q])
E = X.shape[1] - EXTRA_FEATURES
print(f"\ntrain nodes={tr.sum()} ({Y[tr].mean():.1%} relevant) | val nodes={va.sum()} ({Y[va].mean():.1%} relevant)")

def metrics(score, label, keep_rate=None, thr=None):
    if keep_rate is not None:
        thr = np.quantile(score, 1 - keep_rate)
    pred = score >= thr
    tp = int((pred & (label == 1)).sum()); fp = int((pred & (label == 0)).sum())
    fn = int(((~pred) & (label == 1)).sum())
    p = tp/(tp+fp) if tp+fp else 0.0; r = tp/(tp+fn) if tp+fn else 0.0
    return p, r, (2*p*r/(p+r) if p+r else 0.0), pred.mean()

hp, hr, hf, hk = metrics(Hh[va], Y[va], thr=0.5)
print(f"\nTARGET  heuristic_prune (val): precision={hp:.3f} recall={hr:.3f} node-F1={hf:.3f} KEEP-rate={hk:.3f}\n")

class Net(nn.Module):
    def __init__(self, proj_dim):
        super().__init__()
        self.proj_dim = proj_dim
        d = (proj_dim if proj_dim else 0) + EXTRA_FEATURES
        if proj_dim: self.proj = nn.Linear(E, proj_dim)
        self.norm = nn.LayerNorm(d)
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x):
        s = x[:, E:]
        h = torch.cat([self.proj(x[:, :E]), s], 1) if self.proj_dim else s
        return self.net(self.norm(h)).squeeze(-1)

class Full(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(E + EXTRA_FEATURES)
        self.net = nn.Sequential(nn.Linear(E + EXTRA_FEATURES, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(self.norm(x)).squeeze(-1)

Xtr = torch.tensor(X[tr]); Ytr = torch.tensor(Y[tr]); Xva = torch.tensor(X[va])
pw = torch.tensor((Y[tr] == 0).sum() / max((Y[tr] == 1).sum(), 1))
variants = [("full 2052 (control)", Full()), ("emb->64 + struct4", Net(64)),
            ("emb->16 + struct4", Net(16)), ("struct4 only", Net(0))]
print(f"{'representation':<24}{'dims':>6}{'prec':>8}{'recall':>8}{'node-F1':>9}{'  vs heuristic':>15}")
results = []
for name, m in variants:
    torch.manual_seed(42)
    for mod in m.modules():
        if isinstance(mod, nn.Linear): nn.init.xavier_uniform_(mod.weight); nn.init.zeros_(mod.bias)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(400):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(m(Xtr), Ytr, pos_weight=pw)
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): sv = torch.sigmoid(m(Xva)).numpy()
    p, r, f, k = metrics(sv, Y[va], keep_rate=hk)   # matched to the heuristic's operating point
    d = (m.proj_dim + EXTRA_FEATURES) if isinstance(m, Net) else X.shape[1]
    print(f"{name:<24}{d:>6}{p:8.3f}{r:8.3f}{f:9.3f}{f-hf:+15.3f}")
    results.append((name, f))
print(f"\n(all evaluated at the heuristic's own KEEP-rate {hk:.3f}, so the comparison is like-for-like)")
best = max(results, key=lambda x: x[1])
print(f"best: {best[0]} node-F1={best[1]:.3f} vs heuristic {hf:.3f}")
