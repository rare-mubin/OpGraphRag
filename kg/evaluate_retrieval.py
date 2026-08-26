"""
Stage 2 evaluation: run retrieval over every question in subset_questions.json
and check whether the retrieved subgraph G_q covers the passages HotpotQA says
are actually needed (supporting_facts) -- a direct, quantitative check of
whether Stage 2 retrieval works before building Stage 3 (answer generation)
on top of it. No LLM calls -- just BGE-M3 encoding + graph traversal.
Uses the GPU by default when available (pass --device cpu to force CPU).

Usage:
    python evaluate_retrieval.py
    python evaluate_retrieval.py --top-k 5 --hops 2
"""
import os
# Must be set before numpy/torch are imported (directly below, and transitively via the
# `retrieve` import) -- see embed_nodes.py for why.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from retrieve import load_graph, load_embeddings, retrieve, MODEL_NAME, default_device

OUT_DIR = Path(__file__).resolve().parent / "output"


def passage_titles_in_subgraph(Gq) -> set:
    """Union of every node's `sources` (passage titles) inside G_q."""
    titles = set()
    for _, attrs in Gq.nodes(data=True):
        titles.update(attrs.get("sources", []))
    return titles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available, else cpu)")
    args = ap.parse_args()
    if args.device is None:
        args.device = default_device()

    with open(OUT_DIR / "subset_questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    G = load_graph()
    embeddings, ids = load_embeddings()
    print(f"Loading {MODEL_NAME} on {args.device}...")
    model = SentenceTransformer(MODEL_NAME, device=args.device)
    print(f"Evaluating retrieval (--top-k {args.top_k} --hops {args.hops}) over {len(questions)} questions...\n")

    full_recall = partial_recall = zero_recall = 0
    results = []

    for i, q in enumerate(questions, 1):
        needed_titles = {t for t, _ in q["supporting_facts"]}
        Gq, seeds = retrieve(q["question"], G, embeddings, ids, model, top_k=args.top_k, hops=args.hops)
        covered_titles = passage_titles_in_subgraph(Gq) & needed_titles

        recall = len(covered_titles) / len(needed_titles) if needed_titles else 0.0
        if recall == 1.0:
            full_recall += 1
            status = "FULL   "
        elif recall > 0:
            partial_recall += 1
            status = "PARTIAL"
        else:
            zero_recall += 1
            status = "MISS   "

        results.append({
            "question": q["question"],
            "type": q.get("type", ""),
            "needed": sorted(needed_titles),
            "covered": sorted(covered_titles),
            "missing": sorted(needed_titles - covered_titles),
            "recall": recall,
            "gq_nodes": Gq.number_of_nodes(),
            "gq_edges": Gq.number_of_edges(),
        })

        print(f"[{i}/{len(questions)}] {status} recall={recall:.0%}  ({q.get('type','')})  '{q['question'][:65]}'")

    n = len(questions)
    avg_recall = sum(r["recall"] for r in results) / n if n else 0.0

    print("\n=== Retrieval Evaluation Summary ===")
    print(f"Questions evaluated:            {n}")
    print(f"Full supporting-fact recall:    {full_recall} ({100*full_recall/n:.1f}%)")
    print(f"Partial supporting-fact recall: {partial_recall} ({100*partial_recall/n:.1f}%)")
    print(f"Zero supporting-fact recall:    {zero_recall} ({100*zero_recall/n:.1f}%)")
    print(f"Average recall across all questions: {avg_recall:.1%}")

    with open(OUT_DIR / "retrieval_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved detailed per-question results -> {OUT_DIR / 'retrieval_eval.json'}")


if __name__ == "__main__":
    main()
