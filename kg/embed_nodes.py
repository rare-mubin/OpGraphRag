"""
Stage 2, part 1: Embed every knowledge-graph node with BGE-M3 so queries can
be matched against it via cosine similarity (methodology Eq. 1).

Uses the GPU by default when available. If Ollama is also using the GPU
(e.g. extract_kg.py / build_graph.py running concurrently) and VRAM is
tight, pass --device cpu -- CPU embedding of a few thousand short node
texts still takes well under a minute.

Usage:
    python embed_nodes.py
    python embed_nodes.py --device cpu   # force CPU, e.g. to avoid VRAM contention with Ollama
"""
import os
# Must be set before numpy/torch are imported. On Windows, NumPy/MKL and PyTorch can each
# bundle their own OpenMP runtime DLL; if both load, they conflict and crash the process
# with STATUS_ACCESS_VIOLATION -- silently, no Python traceback, nothing to catch. This is
# the official (if blunt) workaround: https://github.com/pytorch/pytorch/issues/37377
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

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


def node_text(node: dict) -> str:
    """Text representation of a node to embed: name + up to 3 descriptions."""
    name = node.get("name", "")
    descs = " ".join(node.get("descriptions", [])[:3])
    return f"{name}: {descs}" if descs else name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available, else cpu)")
    args = ap.parse_args()
    if args.device is None:
        args.device = default_device()

    with open(OUT_DIR / "knowledge_graph.json", encoding="utf-8") as f:
        data = json.load(f)
    nodes = data["nodes"]

    print(f"Loading {MODEL_NAME} on {args.device}...")
    model = SentenceTransformer(MODEL_NAME, device=args.device)

    texts = [node_text(n) for n in nodes]
    ids = [n["id"] for n in nodes]

    print(f"Embedding {len(texts)} nodes...")
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )

    np.save(OUT_DIR / "node_embeddings.npy", embeddings)
    with open(OUT_DIR / "node_embedding_ids.json", "w", encoding="utf-8") as f:
        json.dump(ids, f)

    print(f"\nSaved {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]})")
    print(f"  -> {OUT_DIR / 'node_embeddings.npy'}")
    print(f"  -> {OUT_DIR / 'node_embedding_ids.json'}")


if __name__ == "__main__":
    main()
