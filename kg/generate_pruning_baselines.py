"""
Stage 6 addendum (methodology Section III-F): "Additional comparisons with
similarity-based pruning and fixed/heuristic pruning baselines" -- two
NON-ADAPTIVE (no learning, no query-conditioned policy) compression rules,
run over the same retrieved G_q as the GRPO policy, so the paper can show
whether the learned policy earns its complexity over naive alternatives.

Two baselines:
  - similarity_prune: keep the top KEEP_RATIO fraction of G_q's nodes ranked
    by query-node cosine similarity (Eq. 1) -- a fixed threshold rule, not
    adaptive to graph structure or per-query difficulty.
  - heuristic_prune: keep only nodes within HEURISTIC_HOPS hops of a
    retrieval seed within G_q (vs. the 2-hop graph the retrieval stage
    already built) -- a fixed structural rule, not adaptive to query
    semantics at all.

Both reuse the same generate_answer/shorten_answer/EM/F1 path as A_o (Stage
3) and A_c (Stage 4/5) so all four methods -- uncompressed, RL-compressed,
similarity-pruned, heuristic-pruned -- are directly comparable.

Cost warning: 2 pruning methods x 2 LLM calls (generate + shorten) per
question, ~20 questions => ~80 calls, several seconds each. Expect several
minutes total -- this is NOT a quick script, run it in your own terminal
rather than waiting on it inline.

Usage:
    python generate_pruning_baselines.py
    python generate_pruning_baselines.py --keep-ratio 0.5 --heuristic-hops 1
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import math
import time
from pathlib import Path

import networkx as nx
from sentence_transformers import SentenceTransformer

from retrieve import load_graph, load_embeddings, retrieve, MODEL_NAME, default_device
from generate_baseline_answers import (
    build_context, generate_answer, shorten_answer, exact_match, f1_score, load_token_counter,
)
from train_compression_policy import load_node_embedding_lookup

OUT_DIR = Path(__file__).resolve().parent / "output"


def similarity_prune(Gq: nx.MultiDiGraph, query_emb, node_emb_lookup: dict, keep_ratio: float) -> nx.MultiDiGraph:
    """Fixed-threshold rule: keep the top `keep_ratio` fraction of G_q's nodes ranked by
    query-node cosine similarity. No structure, no learning -- purely a similarity cutoff."""
    node_ids = list(Gq.nodes())
    if not node_ids:
        return Gq.subgraph([]).copy()
    scored = []
    for nid in node_ids:
        vec = node_emb_lookup.get(nid)
        sim = float(vec @ query_emb) if vec is not None else -1.0
        scored.append((nid, sim))
    scored.sort(key=lambda x: -x[1])
    n_keep = max(1, math.ceil(keep_ratio * len(node_ids)))
    keep_ids = [nid for nid, _ in scored[:n_keep]]
    return Gq.subgraph(keep_ids).copy()


def heuristic_prune(Gq: nx.MultiDiGraph, seed_ids: set, hops: int) -> nx.MultiDiGraph:
    """Fixed structural rule: keep only nodes within `hops` hops of a retrieval seed within
    G_q (retrieval itself already expanded 2 hops -- this just re-clips it tighter,
    uniformly, with no regard to query semantics or node content)."""
    Gu = Gq.to_undirected(as_view=True)
    frontier = set(seed_ids) & set(Gq.nodes())
    visited = set(frontier)
    for _ in range(hops):
        next_frontier = set()
        for n in frontier:
            next_frontier |= set(Gu.neighbors(n))
        visited |= next_frontier
        frontier = next_frontier
    return Gq.subgraph(visited).copy()


def run_method(name: str, Gc: nx.MultiDiGraph, question: str, gt: str, T_o: int, EM_o, F1_o,
               passages_by_title: dict, count_tokens) -> dict:
    context = build_context(Gc, passages_by_title)
    T_c = count_tokens(context)
    raw = generate_answer(question, context)
    A_c = shorten_answer(question, raw)
    EM_c = exact_match(A_c, gt)
    F1_c = f1_score(A_c, gt)
    CR = 1.0 - (T_c / T_o) if T_o > 0 else 0.0
    return {
        "method": name, "A": A_c, "T": T_c, "CR": CR,
        "EM": EM_c, "F1": F1_c,
        "delta_em": EM_o - EM_c, "delta_f1": F1_o - F1_c,
        "n_nodes": Gc.number_of_nodes(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--hops", type=int, default=2, help="retrieval hops for G_q (must match other stages)")
    ap.add_argument("--keep-ratio", type=float, default=0.5,
                     help="similarity_prune: fraction of G_q nodes kept (~matches the RL policy's "
                          "observed avg compression ratio, for a fair token-budget comparison)")
    ap.add_argument("--heuristic-hops", type=int, default=1,
                     help="heuristic_prune: hop cutoff within G_q (tighter than retrieval's --hops)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.device is None:
        args.device = default_device()

    with open(OUT_DIR / "subset_questions.json", encoding="utf-8") as f:
        questions = json.load(f)
    with open(OUT_DIR / "subset_passages.json", encoding="utf-8") as f:
        passages = json.load(f)
    passages_by_title = {p["title"]: p["sentences"] for p in passages}
    with open(OUT_DIR / "baseline_answers.json", encoding="utf-8") as f:
        baseline = {r["question"]: r for r in json.load(f)}

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

        t_start = time.time()
        Gc_sim = similarity_prune(Gq, query_emb, node_emb_lookup, args.keep_ratio)
        sim_res = run_method("similarity", Gc_sim, question, gt, T_o, EM_o, F1_o, passages_by_title, count_tokens)

        Gc_heur = heuristic_prune(Gq, seed_ids, args.heuristic_hops)
        heur_res = run_method("heuristic", Gc_heur, question, gt, T_o, EM_o, F1_o, passages_by_title, count_tokens)
        t_this = time.time() - t_start

        results.append({
            "question": question, "ground_truth": gt, "T_o": T_o, "EM_o": EM_o, "F1_o": F1_o,
            "gq_nodes": Gq.number_of_nodes(),
            "similarity": sim_res, "heuristic": heur_res,
        })

        avg_step = (time.time() - t0) / i
        eta = avg_step * (n - i)
        print(f"  [{i}/{n}] '{question[:40]}' "
              f"sim: {sim_res['n_nodes']}n CR={sim_res['CR']:.0%} F1={sim_res['F1']:.2f}  |  "
              f"heur: {heur_res['n_nodes']}n CR={heur_res['CR']:.0%} F1={heur_res['F1']:.2f}  "
              f"({t_this:.1f}s, ETA {eta:.0f}s)")

    n_done = len(results)
    if n_done == 0:
        print("No questions evaluated.")
        return

    def agg(key):
        return {
            "EM": sum(r[key]["EM"] for r in results) / n_done,
            "F1": sum(r[key]["F1"] for r in results) / n_done,
            "CR": sum(r[key]["CR"] for r in results) / n_done,
            "delta_em": sum(r[key]["delta_em"] for r in results) / n_done,
            "delta_f1": sum(r[key]["delta_f1"] for r in results) / n_done,
        }

    avg_em_o = sum(r["EM_o"] for r in results) / n_done
    avg_f1_o = sum(r["F1_o"] for r in results) / n_done
    sim_agg = agg("similarity")
    heur_agg = agg("heuristic")

    print("\n=== Non-Adaptive Pruning Baselines vs. Uncompressed (Section III-F) ===")
    print(f"Questions: {n_done}")
    print(f"{'Method':<14}{'EM':>8}{'F1':>8}{'CR':>8}{'Delta_EM':>12}{'Delta_F1':>12}")
    print(f"{'baseline':<14}{100*avg_em_o:>7.1f}%{100*avg_f1_o:>7.1f}%{'0.0%':>8}{'--':>12}{'--':>12}")
    print(f"{'similarity':<14}{100*sim_agg['EM']:>7.1f}%{100*sim_agg['F1']:>7.1f}%"
          f"{100*sim_agg['CR']:>7.1f}%{100*sim_agg['delta_em']:>+11.1f}pp{100*sim_agg['delta_f1']:>+11.1f}pp")
    print(f"{'heuristic':<14}{100*heur_agg['EM']:>7.1f}%{100*heur_agg['F1']:>7.1f}%"
          f"{100*heur_agg['CR']:>7.1f}%{100*heur_agg['delta_em']:>+11.1f}pp{100*heur_agg['delta_f1']:>+11.1f}pp")
    print("(Compare these rows against compressed_answers.json's RL-policy row -- same baseline, same metrics.)")

    with open(OUT_DIR / "pruning_baselines.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {OUT_DIR / 'pruning_baselines.json'}")


if __name__ == "__main__":
    main()
