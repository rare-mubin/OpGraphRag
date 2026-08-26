"""
Stage 4, part 1 (methodology Section III-D): state representation and policy
network for query-adaptive graph compression.

The policy operates at node granularity: for every node in the retrieved
subgraph G_q, it outputs a KEEP/REMOVE probability. State per node =
[node embedding, query embedding, query-node cosine similarity, normalized
degree within G_q, normalized hop-distance from the nearest retrieval seed,
is-a-seed flag] -- directly matching the methodology's "query embedding,
node representations, query-node semantic relevance scores, and graph-context
features" description. A simple per-node MLP (not a GNN) is used deliberately:
it matches the paper's description without the added complexity/training-data
requirements of learned message passing, which the ~20-question proof-of-
concept dataset can't support anyway.

No LLM calls in this file -- pure NumPy/PyTorch, fast and independently
testable before wiring up the (expensive) reward computation.
"""
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli

EXTRA_FEATURES = 4  # similarity, degree_norm, hop_norm, is_seed


def build_state_features(Gq: nx.MultiDiGraph, query_emb: np.ndarray,
                          node_emb_lookup: dict, seed_ids: set) -> tuple:
    """Returns (node_ids: list[str], features: np.ndarray [N, D]) for every
    node in Gq, in a fixed deterministic order (sorted node id)."""
    node_ids = sorted(Gq.nodes())
    degrees = dict(Gq.degree())
    max_degree = max(degrees.values()) if degrees else 1

    # BFS hop-distance from the nearest seed, within G_q only (undirected)
    Gu = Gq.to_undirected(as_view=True)
    hop_dist = {}
    frontier = set(seed_ids) & set(node_ids)
    for n in frontier:
        hop_dist[n] = 0
    dist = 0
    visited = set(frontier)
    while frontier:
        dist += 1
        next_frontier = set()
        for n in frontier:
            for nb in Gu.neighbors(n):
                if nb not in visited:
                    visited.add(nb)
                    hop_dist[nb] = dist
                    next_frontier.add(nb)
        frontier = next_frontier
    max_hop = max(hop_dist.values()) if hop_dist else 1
    fallback_hop = max_hop + 1  # for any node somehow unreachable from a seed

    emb_dim = len(query_emb)
    features = np.zeros((len(node_ids), 2 * emb_dim + EXTRA_FEATURES), dtype=np.float32)
    for i, nid in enumerate(node_ids):
        node_vec = node_emb_lookup.get(nid)
        if node_vec is None:
            node_vec = np.zeros(emb_dim, dtype=np.float32)
        sim = float(np.dot(node_vec, query_emb))  # both L2-normalized -> cosine similarity
        deg_norm = degrees.get(nid, 0) / max(max_degree, 1)
        hop_norm = hop_dist.get(nid, fallback_hop) / max(max_hop, 1)
        is_seed = 1.0 if nid in seed_ids else 0.0
        features[i, :emb_dim] = node_vec
        features[i, emb_dim:2 * emb_dim] = query_emb
        features[i, 2 * emb_dim:] = [sim, deg_norm, hop_norm, is_seed]

    return node_ids, features


class CompressionPolicy(nn.Module):
    """Per-node KEEP/REMOVE policy. Input: state feature vector per node.
    Output: KEEP logit per node (apply sigmoid for probability)."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        # LayerNorm on the input is important, not cosmetic: the feature vector mixes
        # raw L2-normalized embeddings (~0.03 std per dim) with scalar features on very
        # different scales (similarity in [-1,1], degree/hop in [0,1]) concatenated
        # directly. Without normalizing first, a freshly-initialized linear layer barely
        # differentiates between nodes at all -- verified directly: an untrained policy's
        # logits had std ~0.001 across 16 different nodes, all landing on the same side
        # of the 0.5 threshold by coincidence and producing a degenerate all-keep or
        # all-remove policy before any training happened.
        self.input_norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # The final layer's bias is otherwise drawn near-zero at random and, combined with
        # the small residual logit spread even after LayerNorm, can land the WHOLE policy
        # on the "remove" side of the 0.5 threshold for every node in every graph purely by
        # chance at init -- verified directly: with the default seed this reproduced
        # identically across reruns and survived a couple of training steps untouched.
        # Start biased toward KEEP instead: removing a genuinely useless node costs little,
        # but starting from "remove everything" gives zero usable context and zero reward
        # signal to learn from. sigmoid(1.0) ~= 0.73.
        nn.init.constant_(self.net[-1].bias, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.input_norm(x)).squeeze(-1)  # [N] logits


def sample_trajectory(policy: CompressionPolicy, features: torch.Tensor):
    """Samples one KEEP/REMOVE action per node. Returns (actions [N] bool-ish
    float tensor, log_prob of the whole trajectory (sum over nodes, scalar),
    entropy of the per-node action distribution (sum over nodes, scalar) --
    an entropy bonus in the training loss discourages the policy from
    collapsing to an always-keep/always-remove degenerate strategy)."""
    logits = policy(features)
    probs = torch.sigmoid(logits)
    dist = Bernoulli(probs=probs)
    actions = dist.sample()
    log_prob = dist.log_prob(actions).sum()
    entropy = dist.entropy().sum()
    return actions, log_prob, entropy


def compress_graph(Gq: nx.MultiDiGraph, node_ids: list, actions) -> nx.MultiDiGraph:
    """Builds G_c: the induced subgraph on nodes with action == 1 (KEEP)."""
    keep_ids = [nid for nid, a in zip(node_ids, actions) if float(a) >= 0.5]
    return Gq.subgraph(keep_ids).copy()
