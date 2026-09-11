"""
Node-level classification evaluation of the trained compression policy.

Frames each KEEP/REMOVE decision as binary classification: ground truth
"relevant" = 1 if the node's source passage is one of HotpotQA's own
`supporting_facts` for that question (the same ground truth
`evaluate_retrieval.py` uses for Stage 2), else 0. The trained policy's
deterministic KEEP probability is the classifier score.

This is a DIFFERENT accuracy axis than the paper's EM/F1 (which score the
final generated *answer* text): this script scores the policy's per-node
*decisions* directly against which nodes actually mattered, independent of
whatever the downstream LLM did with them. No LLM calls -- pure local
inference over the already-trained policy, fast (seconds, not minutes).

The fixed 1-hop heuristic (`generate_pruning_baselines.heuristic_prune`) is scored on the
SAME nodes in the same pass, and both are broken down by hop distance from the nearest
retrieval seed. That comparison has to be computed here rather than borrowed from another
run: an earlier write-up compared the policy against heuristic numbers measured on a
different (pre-stratification) validation split and reported a node-level "win" that
disappears on the same split. `generate_report.py` draws the paper's confusion-matrix
figure from the `heuristic` block written here, and the per-hop block is where the
"recovers N relevant two-hop nodes" claim comes from.

Usage:
    python evaluate_policy_classification.py
    python evaluate_policy_classification.py --policy output/compression_policy.pt
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score as classification_f1, roc_curve, roc_auc_score,
)
from sentence_transformers import SentenceTransformer

from retrieve import load_graph, load_embeddings, retrieve, MODEL_NAME, default_device
from compression_policy import build_state_features, CompressionPolicy
from train_compression_policy import load_node_embedding_lookup, POLICY_PATH
from generate_pruning_baselines import heuristic_prune

OUT_DIR = Path(__file__).resolve().parent / "output"
HOP_NAMES = {0: "seed", 1: "1-hop", 2: "2+-hop"}


def decision_block(labels: np.ndarray, pred: np.ndarray) -> dict:
    """Precision/recall/F1/KEEP-rate/confusion for one set of hard KEEP decisions."""
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    return {
        "precision": precision_score(labels, pred, zero_division=0),
        "recall": recall_score(labels, pred, zero_division=0),
        "f1": classification_f1(labels, pred, zero_division=0),
        "keep_rate": float(pred.mean()) if len(pred) else 0.0,
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--policy", default=str(POLICY_PATH))
    ap.add_argument("--device", default=None, help="cuda or cpu for the embedding model")
    ap.add_argument("--split", choices=["val", "train", "all"], default="val",
                     help="which questions to evaluate on (default: val). Since "
                          "train_compression_policy.py's auxiliary loss now trains on these exact "
                          "supporting_facts labels, scoring on trained-on questions is circular -- "
                          "'val' uses only the held-out questions recorded in the policy "
                          "checkpoint. Falls back to all questions for a checkpoint trained before "
                          "the split existed (with a warning).")
    args = ap.parse_args()
    if args.device is None:
        args.device = default_device()

    if not Path(args.policy).exists():
        raise SystemExit(f"No trained policy found at {args.policy} -- run train_compression_policy.py first.")

    with open(OUT_DIR / "subset_questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    checkpoint = torch.load(args.policy, weights_only=False)
    policy = CompressionPolicy(input_dim=checkpoint["input_dim"])
    policy.load_state_dict(checkpoint["state_dict"])
    # Feature mode must match how the policy was TRAINED. Checkpoints written before
    # --features existed have no key and were all trained with embeddings.
    use_emb = checkpoint.get("features", "full") == "full"
    policy.eval()

    split_names = checkpoint.get(f"{args.split}_questions")
    if args.split == "all":
        print(f"Evaluating on ALL {len(questions)} questions (includes questions the auxiliary "
              f"loss trained on -- not a generalization measure).")
    elif split_names is None:
        print(f"[WARN] This checkpoint has no '{args.split}' split recorded (trained before the "
              f"train/val split existed) -- falling back to all {len(questions)} questions. "
              f"These numbers are NOT a generalization measure if --aux-coef was > 0.")
    else:
        wanted = set(split_names)
        questions = [q for q in questions if q["question"] in wanted]
        print(f"Evaluating on the {args.split} split: {len(questions)} held-out question(s).")

    G = load_graph()
    node_emb_lookup = load_node_embedding_lookup()
    embeddings, ids = load_embeddings()
    print(f"Loading {MODEL_NAME} on {args.device}...")
    embed_model = SentenceTransformer(MODEL_NAME, device=args.device)

    all_probs, all_actions, all_labels = [], [], []
    all_heur, all_hop = [], []
    per_question = []

    for i, q in enumerate(questions, 1):
        question = q["question"]
        needed_titles = {t for t, _ in q.get("supporting_facts", [])}

        query_emb = embed_model.encode([question], normalize_embeddings=True)[0]
        Gq, seeds = retrieve(question, G, embeddings, ids, embed_model, top_k=args.top_k, hops=args.hops)
        seed_ids = {s for s, _ in seeds}

        if Gq.number_of_nodes() == 0:
            print(f"  [{i}/{len(questions)}] '{question[:45]}' -> empty G_q, skipping")
            continue

        node_ids, features = build_state_features(Gq, query_emb, node_emb_lookup, seed_ids,
                                                   include_embeddings=use_emb)
        with torch.no_grad():
            logits = policy(torch.tensor(features))
            probs = torch.sigmoid(logits).numpy()
        actions = (probs >= 0.5).astype(int)

        labels = np.array([
            1 if set(Gq.nodes[nid].get("sources", [])) & needed_titles else 0
            for nid in node_ids
        ])

        # Heuristic decision and hop bucket for exactly the same nodes. "2+-hop" is everything
        # the 1-hop rule discards, which with the default --hops 2 means exactly two hops.
        within_one = set(heuristic_prune(Gq, seed_ids, 1).nodes())
        heur = np.array([1 if nid in within_one else 0 for nid in node_ids])
        hop = np.array([0 if nid in seed_ids else (1 if nid in within_one else 2) for nid in node_ids])

        all_probs.extend(probs.tolist())
        all_actions.extend(actions.tolist())
        all_labels.extend(labels.tolist())
        all_heur.extend(heur.tolist())
        all_hop.extend(hop.tolist())

        q_acc = accuracy_score(labels, actions) if len(set(labels)) else float("nan")
        per_question.append({
            "question": question, "n_nodes": len(node_ids),
            "n_relevant": int(labels.sum()), "node_accuracy": q_acc,
        })
        print(f"  [{i}/{len(questions)}] '{question[:45]}' nodes={len(node_ids)} "
              f"relevant={int(labels.sum())} node_acc={q_acc:.2f}")

    all_probs = np.array(all_probs)
    all_actions = np.array(all_actions)
    all_labels = np.array(all_labels)
    all_heur = np.array(all_heur)
    all_hop = np.array(all_hop)

    acc = accuracy_score(all_labels, all_actions)
    prec = precision_score(all_labels, all_actions, zero_division=0)
    rec = recall_score(all_labels, all_actions, zero_division=0)
    f1 = classification_f1(all_labels, all_actions, zero_division=0)
    cm = confusion_matrix(all_labels, all_actions, labels=[0, 1])

    has_both_classes = len(set(all_labels.tolist())) == 2
    if has_both_classes:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        auc = roc_auc_score(all_labels, all_probs)
    else:
        fpr, tpr, auc = np.array([]), np.array([]), float("nan")

    heuristic = decision_block(all_labels, all_heur)
    relevant = all_labels == 1
    by_hop = []
    for hv in (0, 1, 2):
        m = all_hop == hv
        by_hop.append({
            "hop": hv, "name": HOP_NAMES[hv], "n_nodes": int(m.sum()), "n_relevant": int((relevant & m).sum()),
            "policy_keep_rate": float(all_actions[m].mean()) if m.any() else 0.0,
            "policy_tp": int(((all_actions == 1) & relevant & m).sum()),
            "policy_fp": int(((all_actions == 1) & ~relevant & m).sum()),
            "heuristic_keep_rate": float(all_heur[m].mean()) if m.any() else 0.0,
            "heuristic_tp": int(((all_heur == 1) & relevant & m).sum()),
            "heuristic_fp": int(((all_heur == 1) & ~relevant & m).sum()),
        })

    print("\n=== Node-Level KEEP/REMOVE Classification (policy decision vs. supporting_facts ground truth) ===")
    print(f"Total nodes evaluated: {len(all_labels)}  (relevant={int(all_labels.sum())}, "
          f"irrelevant={int((1 - all_labels).sum())})")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1 (classification): {f1:.3f}  "
          f"KEEP-rate: {all_actions.mean():.3f}")
    print(f"AUC:       {auc:.3f}" if has_both_classes else "AUC: n/a (only one class present)")
    print(f"Confusion matrix [rows=true, cols=pred, order=(REMOVE,KEEP)]:\n{cm}")
    print(f"\nSame nodes, fixed 1-hop heuristic: precision {heuristic['precision']:.3f}  "
          f"recall {heuristic['recall']:.3f}  F1 {heuristic['f1']:.3f}  KEEP-rate {heuristic['keep_rate']:.3f}")
    print("By hop distance from the nearest retrieval seed:")
    for b in by_hop:
        print(f"  {b['name']:<7} nodes={b['n_nodes']:5d} relevant={b['n_relevant']:4d} | "
              f"policy keeps {100 * b['policy_keep_rate']:5.1f}% (TP {b['policy_tp']}, FP {b['policy_fp']}) | "
              f"heuristic keeps {100 * b['heuristic_keep_rate']:5.1f}% (TP {b['heuristic_tp']}, FP {b['heuristic_fp']})")

    # --- confusion matrix heatmap ---
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["REMOVE", "KEEP"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["REMOVE", "KEEP"])
    ax.set_xlabel("Policy decision"); ax.set_ylabel("Ground truth (supporting_facts)")
    ax.set_title("Node-level confusion matrix")
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                     color="white" if cm[r, c] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # --- ROC curve ---
    if has_both_classes:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(fpr, tpr, label=f"Policy (AUC={auc:.3f})", color="#2563eb", linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random (AUC=0.500)")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC curve -- node KEEP decision vs. supporting_facts")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "roc_curve.png", dpi=150)
        plt.close(fig)

    result = {
        "split": args.split,
        "n_questions": len(per_question), "n_nodes": len(all_labels),
        "n_relevant": int(all_labels.sum()), "accuracy": acc,
        "precision": prec, "recall": rec, "f1": f1,
        "keep_rate": float(all_actions.mean()) if len(all_actions) else 0.0,
        "auc": None if not has_both_classes else auc,
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "heuristic": heuristic,
        "by_hop": by_hop,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()} if has_both_classes else None,
        "per_question": per_question,
    }
    with open(OUT_DIR / "policy_classification_eval.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {OUT_DIR / 'policy_classification_eval.json'}")
    print(f"Saved -> {OUT_DIR / 'confusion_matrix.png'}")
    if has_both_classes:
        print(f"Saved -> {OUT_DIR / 'roc_curve.png'}")


if __name__ == "__main__":
    main()
