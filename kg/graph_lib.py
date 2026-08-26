"""
Shared graph-assembly logic, used by both build_graph.py (one-shot export)
and live_server.py (continuously rebuilt while extract_kg.py is running).

Kept in one place so the two never drift out of sync with each other.
"""
import re
import networkx as nx

TYPE_COLORS = {
    "PERSON": "#e07a5f",
    "ORG": "#3d5a80",
    "LOCATION": "#81b29a",
    "WORK": "#f2cc8f",
    "EVENT": "#9d4edd",
    "OTHER": "#adb5bd",
}

# Types where two DIFFERENT real-world entities sharing the exact same normalized
# name is a real, demonstrated risk (e.g. five unrelated works all literally
# titled "Black Book"; a 1927 film and a 1999 film both called "American Beauty").
# For these, an exact name match across different passages is NOT auto-merged --
# it's kept as a separate node and flagged for verification (see entity_resolution.py).
# LOCATION/DATE/EVENT/OTHER are left on the cheap auto-merge path: a repeated
# country/nationality/date is essentially always the same referent.
COLLISION_PRONE_TYPES = {"PERSON", "ORG", "WORK"}


def normalize(name: str) -> str:
    """Simple normalization key for entity merging (exact/near-duplicate match)."""
    name = name.strip().lower()
    name = re.sub(r"^(the|a|an)\s+", "", name)
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def build_graph_from_extractions(extractions: list) -> nx.MultiDiGraph:
    """Merge raw per-passage entity/relation extractions into one graph.

    NOTE: the edge attribute for provenance is named 'source_passage', not
    'source' -- networkx's node_link_data() reserves 'source'/'target' for
    the edge's actual endpoints (node ids) and will silently clobber any
    edge attribute with that same name otherwise.

    Same-normalized-name entities of a COLLISION_PRONE_TYPES type, appearing
    in a passage that hasn't already contributed to the existing node of that
    name, are NOT blindly merged -- they become their own passage-scoped node,
    and the pair is recorded in G.graph['same_name_pairs'] for entity_resolution.py
    to verify (relation-based / LLM-checked) rather than assuming identity from
    the name string alone.
    """
    G = nx.MultiDiGraph()
    # normalized name -> node id of the most recently created node for that name
    # (chaining comparisons against the latest occurrence keeps verification
    # roughly linear instead of all-pairs for an entity mentioned many times)
    name_to_latest_node = {}
    same_name_pairs = []

    skipped_relations = 0
    skipped_malformed = 0

    for passage in extractions:
        title = passage["title"]
        local_map = {}  # exact entity name (as given in THIS passage) -> node id

        for ent in passage.get("entities", []):
            if not isinstance(ent, dict):
                skipped_malformed += 1
                continue
            name = ent.get("name", "").strip()
            if not name:
                continue
            etype = ent.get("type", "OTHER")
            desc = ent.get("description", "")
            key = normalize(name)
            if not key:
                continue

            existing_id = name_to_latest_node.get(key)
            if existing_id is None:
                node_id = key
            elif etype in COLLISION_PRONE_TYPES and title not in G.nodes[existing_id]["sources"]:
                # same name, different passage, collision-prone type -- don't assume
                # it's the same entity; give it its own node and flag for verification
                node_id = f"{key}::{title}"
                if node_id not in G.nodes:
                    same_name_pairs.append(tuple(sorted((existing_id, node_id))))
            else:
                # either a safe type, or the same passage already contributed to
                # this node (definitely the same mention, safe to merge directly)
                node_id = existing_id

            if node_id not in G.nodes:
                G.add_node(node_id, name=name, type=etype, descriptions=[], sources=[])
            node = G.nodes[node_id]
            if desc and desc not in node["descriptions"]:
                node["descriptions"].append(desc)
            if title not in node["sources"]:
                node["sources"].append(title)

            name_to_latest_node[key] = node_id
            local_map[name] = node_id

        for rel in passage.get("relations", []):
            if not isinstance(rel, dict):
                skipped_malformed += 1
                continue
            src_name = rel.get("source", "").strip()
            tgt_name = rel.get("target", "").strip()
            if not src_name or not tgt_name:
                skipped_relations += 1
                continue

            # Prefer this passage's own entity list (exact string match) so a
            # relation correctly points at the passage-scoped node when its
            # entity was homonym-disambiguated above; fall back to whatever
            # node currently owns that normalized name, or create one.
            src_key = local_map.get(src_name) or name_to_latest_node.get(normalize(src_name))
            tgt_key = local_map.get(tgt_name) or name_to_latest_node.get(normalize(tgt_name))
            if src_key is None:
                src_key = normalize(src_name)
                if src_key and src_key not in G.nodes:
                    G.add_node(src_key, name=src_name, type="OTHER", descriptions=[], sources=[title])
                    name_to_latest_node[src_key] = src_key
            if tgt_key is None:
                tgt_key = normalize(tgt_name)
                if tgt_key and tgt_key not in G.nodes:
                    G.add_node(tgt_key, name=tgt_name, type="OTHER", descriptions=[], sources=[title])
                    name_to_latest_node[tgt_key] = tgt_key
            if not src_key or not tgt_key:
                skipped_relations += 1
                continue

            G.add_edge(
                src_key, tgt_key,
                relation=rel.get("relation", ""),
                description=rel.get("description", ""),
                source_passage=title,
            )

    G.graph["skipped_relations"] = skipped_relations
    G.graph["skipped_malformed"] = skipped_malformed
    G.graph["same_name_pairs"] = same_name_pairs
    return G


def to_node_link(G: nx.MultiDiGraph) -> dict:
    return nx.node_link_data(G, edges="edges")
