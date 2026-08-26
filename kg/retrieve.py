"""
Stage 2, part 2: GraphRAG retrieval -- given a query, find top-k semantic
seed nodes (BGE-M3 cosine similarity, Eq. 1 in the methodology) then expand
k hops along graph edges to build the candidate subgraph G_q.

Top-k-seeds + k-hop expansion (rather than pure top-k similarity) is what
actually captures multi-hop chains: the second fact a multi-hop question
needs is often not semantically similar to the query at all -- it's only
reachable by walking the graph from a shared entity.

Usage:
    python retrieve.py "Were Scott Derrickson and Ed Wood of the same nationality?"
    python retrieve.py "..." --top-k 5 --hops 2
"""
import os
# Must be set before numpy/torch are imported -- see embed_nodes.py for why.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

OUT_DIR = Path(__file__).resolve().parent / "output"
MODEL_NAME = "BAAI/bge-m3"


def default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_graph() -> nx.MultiDiGraph:
    with open(OUT_DIR / "knowledge_graph.json", encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data, edges="edges")


def load_embeddings():
    embeddings = np.load(OUT_DIR / "node_embeddings.npy")
    with open(OUT_DIR / "node_embedding_ids.json", encoding="utf-8") as f:
        ids = json.load(f)
    return embeddings, ids


def retrieve(query: str, G: nx.MultiDiGraph, embeddings: np.ndarray, ids: list,
             model: SentenceTransformer, top_k: int = 5, hops: int = 1):
    """Returns (G_q, seeds) where seeds is [(node_id, similarity), ...]."""
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    sims = embeddings @ q_emb  # both L2-normalized -> dot product == cosine similarity
    top_idx = np.argsort(-sims)[:top_k]
    seeds = [(ids[i], float(sims[i])) for i in top_idx]

    # k-hop expansion, undirected -- a relation can be relevant to answer the
    # query regardless of which direction it was extracted in
    Gu = G.to_undirected(as_view=True)
    subgraph_nodes = set()
    for seed_id, _ in seeds:
        if seed_id not in Gu:
            continue
        frontier = {seed_id}
        subgraph_nodes.add(seed_id)
        for _ in range(hops):
            next_frontier = set()
            for n in frontier:
                next_frontier |= set(Gu.neighbors(n))
            subgraph_nodes |= next_frontier
            frontier = next_frontier

    Gq = G.subgraph(subgraph_nodes).copy()
    return Gq, seeds


def print_result(query: str, G: nx.MultiDiGraph, Gq: nx.MultiDiGraph, seeds: list, top_k: int, hops: int):
    print(f"\nQuery: {query}")
    print(f"\nTop-{top_k} seed nodes (cosine similarity):")
    for node_id, sim in seeds:
        name = G.nodes[node_id].get("name", node_id) if node_id in G.nodes else f"{node_id} (not in graph)"
        print(f"  {sim:.3f}  {name}")

    print(f"\nRetrieved G_q: {Gq.number_of_nodes()} nodes, {Gq.number_of_edges()} edges "
          f"({hops}-hop expansion from seeds)")

    print("\nEntities in G_q:")
    for n, attrs in Gq.nodes(data=True):
        print(f"  - {attrs.get('name', n)} ({attrs.get('type', '?')})")

    print("\nRelations in G_q:")
    for u, v, attrs in Gq.edges(data=True):
        uname = Gq.nodes[u].get("name", u)
        vname = Gq.nodes[v].get("name", v)
        print(f"  - {uname} --[{attrs.get('relation', '')}]--> {vname}  "
              f"(source: {attrs.get('source_passage', '')})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available, else cpu)")
    args = ap.parse_args()
    if args.device is None:
        args.device = default_device()

    G = load_graph()
    embeddings, ids = load_embeddings()

    print(f"Loading {MODEL_NAME} on {args.device}...")
    model = SentenceTransformer(MODEL_NAME, device=args.device)

    Gq, seeds = retrieve(args.query, G, embeddings, ids, model, top_k=args.top_k, hops=args.hops)
    print_result(args.query, G, Gq, seeds, args.top_k, args.hops)


if __name__ == "__main__":
    main()
