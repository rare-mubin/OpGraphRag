"""
Stage 4, part 3 (methodology Section III-E/III-F): run a TRAINED compression
policy's decisions (deterministic inference, no exploration/sampling) across
questions, generate compressed answers A_c, and compare against Stage 3's
uncompressed baseline (A_o, T_o) -- the actual Delta_EM / Delta_F1 /
token-reduction comparison the paper reports.

Usage:
    python generate_compressed_answers.py
    python generate_compressed_answers.py --top-k 5 --hops 2
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from retrieve import load_graph, load_embeddings, retrieve, MODEL_NAME, default_device
from generate_baseline_answers import (
    build_context, generate_answer, shorten_answer, exact_match, f1_score, load_token_counter,
)
from compression_policy import build_state_features, CompressionPolicy
from train_compression_policy import load_node_embedding_lookup, POLICY_PATH

OUT_DIR = Path(__file__).resolve().parent / "output"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--device", default=None, help="cuda or cpu for the embedding model")
    ap.add_argument("--policy", default=str(POLICY_PATH), help="path to a trained policy .pt file")
    ap.add_argument("--n-questions", type=int, default=None,
                     help="limit to the first N questions (default: all) -- for a quick spot-check "
                          "of a policy without waiting on the full subset")
    ap.add_argument("--split", choices=["val", "train", "all"], default="val",
                     help="which questions to answer (default: val). train_compression_policy.py's "
                          "auxiliary loss trains on the supporting_facts labels for its train "
                          "split, so Delta_EM/Delta_F1 measured on those questions is optimistic; "
                          "'val' uses only the held-out questions recorded in the checkpoint. "
                          "Falls back to all questions (with a warning) for a checkpoint trained "
                          "before the split existed.")
    args = ap.parse_args()
    if args.device is None:
        args.device = default_device()

    if not Path(args.policy).exists():
        raise SystemExit(f"No trained policy found at {args.policy} -- run train_compression_policy.py first.")

    with open(OUT_DIR / "subset_questions.json", encoding="utf-8") as f:
        questions = json.load(f)
    with open(OUT_DIR / "subset_passages.json", encoding="utf-8") as f:
        passages = json.load(f)
    passages_by_title = {p["title"]: p["sentences"] for p in passages}
    with open(OUT_DIR / "baseline_answers.json", encoding="utf-8") as f:
        baseline = {r["question"]: r for r in json.load(f)}

    checkpoint = torch.load(args.policy, weights_only=False)

    split_names = checkpoint.get(f"{args.split}_questions")
    if args.split == "all":
        print(f"Answering ALL {len(questions)} questions (includes the policy's own training "
              f"questions -- not a generalization measure).")
    elif split_names is None:
        print(f"[WARN] This checkpoint has no '{args.split}' split recorded (trained before the "
              f"train/val split existed) -- falling back to all {len(questions)} questions.")
    else:
        wanted = set(split_names)
        questions = [q for q in questions if q["question"] in wanted]
        print(f"Answering the {args.split} split: {len(questions)} held-out question(s).")
    if args.n_questions:
        questions = questions[: args.n_questions]
    policy = CompressionPolicy(input_dim=checkpoint["input_dim"])
    policy.load_state_dict(checkpoint["state_dict"])
    # Feature mode must match how the policy was TRAINED. Checkpoints written before
    # --features existed have no key and were all trained with embeddings.
    use_emb = checkpoint.get("features", "full") == "full"
    policy.eval()
    print(f"Loaded policy from {args.policy} (trained with: {checkpoint.get('args', {})})\n")

    G = load_graph()
    node_emb_lookup = load_node_embedding_lookup()
    embeddings, ids = load_embeddings()
    print(f"Loading {MODEL_NAME} on {args.device}...")
    embed_model = SentenceTransformer(MODEL_NAME, device=args.device)
    count_tokens = load_token_counter()

    results = []
    t0 = time.time()
    n = len(questions)

    for i, q in enumerate(questions, 1):
        question = q["question"]
        gt = q["answer"]
        base = baseline.get(question)
        if base is None:
            print(f"  [{i}/{n}] '{question[:45]}' -> no Stage 3 baseline, skipping")
            continue
        T_o, EM_o, F1_o = base["T_o"], base["em"], base["f1"]

        query_emb = embed_model.encode([question], normalize_embeddings=True)[0]
        Gq, seeds = retrieve(question, G, embeddings, ids, embed_model, top_k=args.top_k, hops=args.hops)
        seed_ids = {s for s, _ in seeds}

        if Gq.number_of_nodes() == 0:
            print(f"  [{i}/{n}] '{question[:45]}' -> empty G_q, skipping")
            continue

        node_ids, features = build_state_features(Gq, query_emb, node_emb_lookup, seed_ids,
                                                   include_embeddings=use_emb)
        with torch.no_grad():
            logits = policy(torch.tensor(features))
            keep_mask = torch.sigmoid(logits) >= 0.5  # deterministic inference, no sampling

        keep_ids = [nid for nid, k in zip(node_ids, keep_mask.tolist()) if k]
        Gc = Gq.subgraph(keep_ids).copy()
        context = build_context(Gc, passages_by_title)
        T_c = count_tokens(context)

        t_start = time.time()
        raw = generate_answer(question, context)
        A_c = shorten_answer(question, raw)
        t_this = time.time() - t_start

        EM_c = exact_match(A_c, gt)
        F1_c = f1_score(A_c, gt)
        CR = 1.0 - (T_c / T_o) if T_o > 0 else 0.0
        token_reduction_pct = 100.0 * (T_o - T_c) / T_o if T_o > 0 else 0.0
        delta_em = EM_o - EM_c   # methodology Eq.: positive = baseline was better
        delta_f1 = F1_o - F1_c

        results.append({
            "question": question, "ground_truth": gt,
            "A_o": base["generated_answer"], "A_c": A_c,
            "T_o": T_o, "T_c": T_c, "CR": CR, "token_reduction_pct": token_reduction_pct,
            "EM_o": EM_o, "EM_c": EM_c, "F1_o": F1_o, "F1_c": F1_c,
            "delta_em": delta_em, "delta_f1": delta_f1,
            "gq_nodes": Gq.number_of_nodes(), "gc_nodes": Gc.number_of_nodes(),
        })

        print(f"  [{i}/{n}] '{question[:45]}' G_q={Gq.number_of_nodes()}->G_c={Gc.number_of_nodes()} nodes  "
              f"CR={CR:.0%}  EM_o={EM_o} EM_c={EM_c}  F1_o={F1_o:.2f} F1_c={F1_c:.2f}  ({t_this:.1f}s)")

    n_done = len(results)
    if n_done == 0:
        print("No questions evaluated.")
        return

    avg_em_o = sum(r["EM_o"] for r in results) / n_done
    avg_em_c = sum(r["EM_c"] for r in results) / n_done
    avg_f1_o = sum(r["F1_o"] for r in results) / n_done
    avg_f1_c = sum(r["F1_c"] for r in results) / n_done
    avg_cr = sum(r["CR"] for r in results) / n_done
    avg_token_reduction = sum(r["token_reduction_pct"] for r in results) / n_done
    avg_delta_em = sum(r["delta_em"] for r in results) / n_done
    avg_delta_f1 = sum(r["delta_f1"] for r in results) / n_done

    print("\n=== Compressed vs. Uncompressed Comparison ===")
    print(f"Questions:              {n_done}")
    print(f"EM:   baseline {100*avg_em_o:.1f}%  ->  compressed {100*avg_em_c:.1f}%  (Delta_EM = {100*avg_delta_em:+.1f}pp)")
    print(f"F1:   baseline {100*avg_f1_o:.1f}%  ->  compressed {100*avg_f1_c:.1f}%  (Delta_F1 = {100*avg_delta_f1:+.1f}pp)")
    print(f"Avg compression ratio (CR):     {100*avg_cr:.1f}%")
    print(f"Avg token reduction:            {avg_token_reduction:.1f}%")
    print(f"(Delta_EM/Delta_F1 > 0 means the baseline outperformed the compressed answer;")
    print(f" <= 0 means compression preserved or improved QA quality while shrinking context)")

    with open(OUT_DIR / "compressed_answers.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {OUT_DIR / 'compressed_answers.json'}")


if __name__ == "__main__":
    main()
