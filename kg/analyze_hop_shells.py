"""Hop-shell analysis: where do relevant nodes live, and are the ones the 1-hop
heuristic discards recoverable at all? No LLM calls. Reads output/ablation_features.npz
(written by ablate_features.py -- run that first).
"""
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from compression_policy import EXTRA_FEATURES
z=np.load("output/ablation_features.npz",allow_pickle=True)
X,Y,H,Q=z["X"],z["Y"],z["H"],z["Q"]
ck=torch.load("output/compression_policy.pt",weights_only=False)
tr=np.array([q in set(ck["train_questions"]) for q in Q]); va=np.array([q in set(ck["val_questions"]) for q in Q])
E=X.shape[1]-EXTRA_FEATURES; hop=X[:,E+2]
# the 2-hop shell only -- the region the heuristic throws away entirely
t2, v2 = tr&(hop==1.0), va&(hop==1.0)
print(f"2-hop shell: train={t2.sum()} ({Y[t2].mean():.2%} relevant)  val={v2.sum()} ({Y[v2].mean():.2%} relevant)")
print(f"relevant nodes stranded at 2 hops (val): {int(Y[v2].sum())} of {int(Y[va].sum())} total relevant "
      f"({100*Y[v2].sum()/Y[va].sum():.0f}% -- the heuristic misses ALL of these)\n")
sim = X[:,E+0]
print(f"{'signal':<28}{'val AUC':>9}{'P@k':>8}   (chance P = %.3f)" % Y[v2].mean())
print(f"{'query-node cosine sim':<28}{roc_auc_score(Y[v2],sim[v2]):9.3f}", end="")
k=int(Y[v2].sum()); idx=np.argsort(-sim[v2])[:k]; print(f"{Y[v2][idx].mean():8.3f}")
def run(name, cols, epochs=400):
    Xt=torch.tensor(X[t2][:,cols]); Yt=torch.tensor(Y[t2]); Xv=torch.tensor(X[v2][:,cols])
    torch.manual_seed(42)
    m=nn.Sequential(nn.LayerNorm(len(cols)),nn.Linear(len(cols),256),nn.ReLU(),
                    nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
    for mod in m.modules():
        if isinstance(mod,nn.Linear): nn.init.xavier_uniform_(mod.weight); nn.init.zeros_(mod.bias)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    pw=torch.tensor((Y[t2]==0).sum()/max((Y[t2]==1).sum(),1))
    for _ in range(epochs):
        opt.zero_grad(); nn.functional.binary_cross_entropy_with_logits(m(Xt).squeeze(-1),Yt,pos_weight=pw).backward(); opt.step()
    m.eval()
    with torch.no_grad(): s=torch.sigmoid(m(Xv).squeeze(-1)).numpy()
    idx=np.argsort(-s)[:k]
    print(f"{name:<28}{roc_auc_score(Y[v2],s):9.3f}{Y[v2][idx].mean():8.3f}")
run("struct4 (learned)", list(range(E,E+4)))
run("full 2052 (learned)", list(range(X.shape[1])))
run("embeddings only (2048)", list(range(E)))
