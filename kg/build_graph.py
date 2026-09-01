"""
Step 3: Turn raw per-passage extractions into a single NetworkX knowledge
graph, merging duplicate entity mentions and keeping provenance.

By default also runs a stronger entity-resolution pass (see
entity_resolution.py) that catches coreferent entities missed by simple
string normalization (e.g. "Ed Wood" vs "Edward D. Wood Jr."), verified via
LLM calls on cheaply-blocked candidate pairs and cached to disk. Use
--no-resolve to skip it for a faster iteration cycle.

Usage:
    python build_graph.py
    python build_graph.py --no-resolve
    python build_graph.py --workers 8   # more concurrent entity-resolution LLM calls (default: 4)
"""
import argparse
import json
from pathlib import Path

import networkx as nx

from graph_lib import build_graph_from_extractions, to_node_link
from entity_resolution import resolve_entities

OUT_DIR = Path(__file__).resolve().parent / "output"
RESOLUTION_CACHE_PATH = OUT_DIR / "entity_resolution_cache.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-resolve", action="store_true",
                     help="skip the LLM-based entity resolution pass (faster, string-normalization merge only)")
    ap.add_argument("--workers", type=int, default=4,
                     help="concurrent entity-resolution LLM calls against Ollama (default: 4). "
                          "Each call is a small, cheap generation -- raise this if your GPU has "
                          "headroom (nvidia-smi while it runs), lower it if requests start "
                          "queueing up with no speedup or Ollama errors under load.")
    args = ap.parse_args()

    with open(OUT_DIR / "extractions.json", encoding="utf-8") as f:
        extractions = json.load(f)

    G = build_graph_from_extractions(extractions)
    skipped_relations = G.graph.get("skipped_relations", 0)
    skipped_malformed = G.graph.get("skipped_malformed", 0)
    n_nodes_before_resolution = G.number_of_nodes()

    resolution_stats = None
    if not args.no_resolve:
        print("Running entity resolution (blocking + LLM verification, cached)...")
        G, resolution_stats = resolve_entities(G, RESOLUTION_CACHE_PATH, workers=args.workers)

    # --- Save ---
    # node-link JSON (easy to reload / inspect / feed into later phases)
    data = to_node_link(G)
    with open(OUT_DIR / "knowledge_graph.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # --- Stats ---
    n_passages = len(extractions)
    n_empty = sum(1 for p in extractions if not p.get("entities"))
    degrees = dict(G.degree())
    isolated = [n for n, d in degrees.items() if d == 0]
    components = list(nx.weakly_connected_components(G))
    largest_cc = max(components, key=len) if components else set()

    print("\n=== Knowledge Graph Summary ===")
    print(f"Passages processed:      {n_passages} (empty extractions: {n_empty})")
    if resolution_stats:
        print(f"Nodes before resolution: {n_nodes_before_resolution}")
        print(f"Entity resolution:       {resolution_stats['candidate_pairs']} candidate pairs "
              f"({resolution_stats['cached_hits']} cached, {resolution_stats['llm_calls']} new LLM calls), "
              f"{resolution_stats['groups_merged']} groups merged")
    print(f"Nodes (entities):        {G.number_of_nodes()}")
    print(f"Edges (relations):       {G.number_of_edges()} (skipped missing src/tgt: {skipped_relations}, "
          f"skipped non-dict entries: {skipped_malformed})")
    print(f"Avg degree:              {sum(degrees.values())/max(len(degrees),1):.2f}")
    print(f"Isolated nodes:          {len(isolated)}")
    print(f"Connected components:    {len(components)}")
    print(f"Largest component size:  {len(largest_cc)} nodes "
          f"({100*len(largest_cc)/max(G.number_of_nodes(),1):.1f}% of graph)")
    print(f"\nSaved -> {OUT_DIR / 'knowledge_graph.json'}")


if __name__ == "__main__":
    main()
