"""
Stronger entity resolution: catches coreferent entities that graph_lib's
normalize()-based merge misses because they don't share a normalized string
(e.g. "Ed Wood" vs "Edward D. Wood Jr.").

Deliberately NOT based on raw embedding similarity -- names like "Sergei
Roshchin" / "Sergei Kornilenko" / "Sergei Chikildin" (different real
footballers, all present in this dataset) would score dangerously similar
by embedding alone. Instead:

  1. Block candidate pairs cheaply (same type, shares a distinctive name token)
  2. Verify each candidate with an LLM call (given both entities' descriptions)
  3. Cache every verification decision to disk (re-running build_graph.py
     during ongoing extraction never re-pays LLM cost for pairs already checked)
  4. Union-Find merge confirmed pairs into one canonical node

Only used by build_graph.py -- NOT live_server.py, which stays on cheap
string-normalization for responsiveness during frequent polling.
"""
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import networkx as nx
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b-instruct"

RESOLVABLE_TYPES = {"PERSON", "ORG", "LOCATION", "WORK"}
MIN_TOKEN_LEN = 4
MAX_BLOCK_SIZE = 40  # skip tokens shared by more than this many entities (e.g. common words)
STOPWORDS = {"the", "and", "for", "with", "from", "this", "that", "were", "have"}

VERIFY_SYSTEM_PROMPT = """You classify the relationship between two entity mentions from a knowledge \
graph into exactly one category. Pick the single best-fitting category -- do not output "same" as a \
separate judgment, the category alone determines that.

Categories:
- "identical": true aliases/coreferences of the exact same thing -- a full legal name vs short name, \
  an old name vs renamed version, an official name vs nickname, a translated title vs original, OR \
  both descriptions independently state the same concrete fact about one specific thing (e.g. same \
  year + same category). One description explicitly stating a rename/alias relationship to the other \
  ("was renamed to X", "originally called Y", "also known as Z", "full name is W") is strong evidence \
  for this category.
- "part_of": one is a chamber, branch, subsidiary, division, member, or founded/created offshoot of the \
  other larger body (e.g. a legislature's chamber is part_of the legislature; a subsidiary is part_of \
  its parent company; a school is part_of the religious order or organization that founded it). A shared \
  parent name mentioned in both descriptions is what MAKES this category apply, not evidence against it \
  -- do not reclassify as "identical" just because a name is shared.
- "different_individual": same category/type (e.g. two footballers, two universities, two counties, \
  two legislative chambers of different states, two co-producers/co-authors/collaborators on the same \
  work, two different genres/styles/categories associated with the same person or artist) but clearly \
  distinct specific entities with no stated identity or containment relationship. This applies to \
  places, institutions, and abstract labels just as much as people: two counties/chambers/organizations \
  with a shared generic word ("County", "House of Representatives") but a DIFFERENT specific proper-noun \
  qualifier (Cheshire vs. Fayette, Alaska vs. Florida) are different entities; two different genre/style \
  tags (e.g. "electronic music" and "noise music") that both happen to describe the same artist's work \
  are still different genres, not one; two named people who collaborated on or co-created the same thing \
  are still two different people. None of this becomes "identical" just because a description says "same \
  name," "same category," or "same context" -- check whether the two THINGS are actually one thing, not \
  whether they're related or co-occurring.
- "related_work": sequel, remake, adaptation, or another separate work in the same series/franchise \
  (a novel is a related_work of its film adaptation, not identical to it).
- "coincidental_same_name": two unrelated things that merely happen to share a title or name, with \
  neither description giving any concrete fact tying them together -- common for films/books/albums \
  that reuse the same title. If a description is purely self-referential with no concrete fact at all \
  (e.g. "the film", "this player", "English title of the film" -- wording that would fit any work) and \
  the two come from differently-titled passages with nothing else linking them, this is the category, \
  even if the names match exactly.
- "unclear": not enough information to tell.

Worked examples:
  A "American Beauty" / "a 1927 American silent film" / passage "American Beauty (1927 film)" \
  vs B "The American Beauty" / "English title of the film" (no concrete fact) / passage "La Belle Américaine" \
  -> "coincidental_same_name" (B states nothing concrete; B's own passage is an unrelated film).
  A "American Beauty" / "1999 American drama film" / passage "American Beauty (1999 film)" \
  vs B "American Beauty" / "1999 film" / passage "List of accolades received by American Beauty" \
  -> "identical" (both state the concrete fact "1999 film", consistent, exact same name).
  A "Lomonosov Moscow State University" / "a public research university in Moscow" \
  vs B "Moscow State University" / "the original name before it was renamed after Lomonosov" \
  -> "identical" (B's description explicitly states the rename relationship to A).
  A "Alaska Legislature" / "the bicameral state legislature of Alaska" \
  vs B "Alaska Senate" / "the upper chamber of the Alaska Legislature" \
  -> "part_of" (B is explicitly one chamber of A, not the whole legislature).
  A "Cheshire County" / "a county" vs B "Fayette County" / "a county" \
  -> "different_individual" (different specific counties; sharing the word "County" is not identity).
  A "American Jewish community" / "collaborated to document the Holocaust" \
  vs B "Jewish Anti-Fascist Committee (JAC)" / "collaborated to document the Holocaust" \
  -> "different_individual" (a broad community and a specific named committee that worked on the same \
  project are two different entities, not one -- collaborating on the same effort is not identity).
  A "Christian Brothers" / "the religious order that founded the college" \
  vs B "Christian Brothers College" / "a Roman Catholic secondary college" \
  -> "part_of" (B was founded by A; a founder is not identical to what it founded).
  A "electronic music" / "a genre associated with this sound artist" \
  vs B "noise music" / "a genre associated with this sound artist" \
  -> "different_individual" (both are genres the same artist works in -- that doesn't make the genres \
  themselves the same genre).
  A "Alliance of Liberals and Democrats for Europe" / "a European political party alliance" \
  vs B "National Wrestling Alliance" / "a professional wrestling promotion" \
  -> "different_individual" (both named "Alliance", completely unrelated fields; a shared generic \
  category word -- "Alliance", "Mosque", "School", "Conference" -- is never evidence of identity on \
  its own; check the SPECIFIC qualifying name, not the shared category noun).
  A "Cars 2" / "the second film in the Cars franchise" vs B "Cars 3" / "the third film in the Cars \
  franchise" -> "related_work" (sequential numbered entries in the same franchise/series are separate \
  works, not the same film, even though the title stem repeats).
  A "Republic of China" / "the state commonly known as Taiwan" \
  vs B "People's Republic of China" / "the mainland Chinese state" \
  -> "different_individual" (distinct sovereign states despite the overlapping name; "Republic of" vs \
  "People's Republic of" are different qualifiers denoting different political entities, not aliases \
  for the same country).

When unsure between "identical" and any other category, pick the other category -- classifying two \
distinct entities as identical silently fuses them in the graph, which is worse than leaving two real \
aliases unmerged. Base your decision only on the names, descriptions, and passage titles given, not on \
real-world knowledge you assume but that isn't stated.

Output ONLY a single JSON object: \
{"category": "identical|part_of|different_individual|related_work|coincidental_same_name|unclear", \
"reason": "short reason, <15 words"}"""


ALIAS_RELATION_KEYWORDS = [
    "known as", "aka", "a.k.a", "formerly", "previous name", "original name", "renamed",
    "alias", "née", "real name", "birth name", "pseudonym", "stage name",
    "also called", "nicknamed", "born as", "true name", "given name at birth",
    "legal name", "credited as", "professionally known as",
]


def find_relation_based_pairs(G: nx.MultiDiGraph) -> list:
    """High-confidence merge candidates: entities directly connected by an
    alias-indicating relation extracted from the source text itself (e.g.
    'was known as', 'formerly named'). No LLM verification needed -- this
    is already stated as fact in the passage, and checking it against
    unrelated same-name-token entities via blocking is what caused a real
    false-positive merge (a vague 'previous name of the player' description
    got matched against the wrong footballer instead of the one it actually
    referred to).

    Relation strings are normalized (underscores/hyphens -> spaces) before
    matching against ALIAS_RELATION_KEYWORDS -- extract_kg.py's model often
    outputs relation types as SCREAMING_SNAKE_CASE (e.g. "ALSO_KNOWN_AS"),
    and without normalization "known as" never matches "also_known_as"
    (space vs underscore), so the keyword check silently misses it. This bit
    us for real: one node extracted with description "Alternate name for
    Evan Thomas" had its genuine ALSO_KNOWN_AS relation missed by the
    unnormalized check, stayed independently blockable, and its vague
    description then fooled the LLM verifier into "identical" against ~50
    unrelated entities sharing a name token -- which Union-Find transitivity
    chained into a single 141-entity merge group (Shakespeare, Mark
    Zuckerberg, Tom Clancy, and Marvel's Peter Parker, among many others, all
    fused into one node). Audit any existing entity_resolution_cache.json
    for pre-existing damage from this after applying the fix -- it only
    prevents new instances, it doesn't undo merges already in the cache."""
    pairs = set()
    for u, v, attrs in G.edges(data=True):
        rel = (attrs.get("relation") or "").lower().replace("_", " ").replace("-", " ")
        if any(kw in rel for kw in ALIAS_RELATION_KEYWORDS):
            pairs.add(tuple(sorted((u, v))))
    return sorted(pairs)


def find_relation_based_canonical_hints(G: nx.MultiDiGraph) -> set:
    """Which side of each alias relation is the CANONICAL entity, per the
    relation's own stated direction (source ALSO_KNOWN_AS target -- source is
    the real entity, target is the alias mention). Used to bias
    `_apply_pair_merges`' representative choice away from picking whichever
    string happens to be longer, which can otherwise select a vague,
    self-referential alias description (e.g. "Alternate name for Evan
    Thomas") as the surviving merged node's identity -- exactly the kind of
    description that then fools the LLM verifier into further false
    positives against unrelated entities during blocking (see CLAUDE.md)."""
    hints = set()
    for u, v, attrs in G.edges(data=True):
        rel = (attrs.get("relation") or "").lower().replace("_", " ").replace("-", " ")
        if any(kw in rel for kw in ALIAS_RELATION_KEYWORDS):
            hints.add(u)  # the edge's source is the canonical side
    return hints


def find_description_based_alias_pairs(G: nx.MultiDiGraph) -> tuple:
    """Catches vague 'alias' entities whose own DESCRIPTION states an alias
    relationship (e.g. "Original name of Stephen Marcus", "Alternate name
    for Evan Thomas") in cases where extract_kg.py failed to also produce a
    matching relation edge for it -- find_relation_based_pairs() has nothing
    to catch there, and the entity's own vague description is exactly what
    fools the LLM verifier into false positives during blocking. This is a
    real, recurring gap, not a one-off: it produced three separate ~31-36
    entity merge-groups in one 200-question build (Scott/Spurrier,
    Richard/Sherman, Marie-Joseph/Lafayette clusters), not just the single
    Evan Thomas / Peter Evan Thomas case the relation-based fix caught.

    Heuristic: such an entity's true referent is reliably the entity in the
    SAME source passage whose name exactly matches that passage's title --
    extract_kg.py's own system prompt instructs the model to always extract
    the passage's title as an entity when it's a coherent named entity, and
    the two real cases audited so far both fit this pattern exactly (Evan
    Thomas / "Evan Thomas (actor)", Stephen Marcus / "Stephen Marcus").
    Returns (pairs, canonical_hints) -- same shape as the relation-based
    functions above, for the same reason (representative selection needs to
    know which side is canonical, not just which name is longer)."""
    title_nodes = {}  # passage title (lowercased) -> node id, for nodes whose name matches it
    for n, attrs in G.nodes(data=True):
        name = (attrs.get("name") or "").strip().lower()
        if not name:
            continue
        for src in attrs.get("sources", []):
            if name == src.strip().lower():
                title_nodes[src.strip().lower()] = n

    pairs = set()
    canonical_hints = set()
    for n, attrs in G.nodes(data=True):
        desc = " | ".join(attrs.get("descriptions", [])).lower()
        if not any(kw in desc for kw in ALIAS_RELATION_KEYWORDS):
            continue
        for src in attrs.get("sources", []):
            target = title_nodes.get(src.strip().lower())
            if target and target != n:
                pairs.add(tuple(sorted((n, target))))
                canonical_hints.add(target)
    return sorted(pairs), canonical_hints


def _tokenize(name: str) -> list:
    name = re.sub(r"[^\w\s]", " ", name.lower())
    return [t for t in name.split() if len(t) >= MIN_TOKEN_LEN and t not in STOPWORDS]


def find_candidate_pairs(G: nx.MultiDiGraph) -> list:
    """Cheap local blocking: same-type node pairs sharing a distinctive name token."""
    from collections import defaultdict

    by_type = defaultdict(list)
    for n, attrs in G.nodes(data=True):
        if attrs.get("type") in RESOLVABLE_TYPES:
            by_type[attrs["type"]].append(n)

    pairs = set()
    for etype, node_ids in by_type.items():
        index = defaultdict(set)
        for n in node_ids:
            name = G.nodes[n].get("name", "")
            for tok in _tokenize(name):
                index[tok].add(n)
        for tok, ids in index.items():
            if len(ids) > MAX_BLOCK_SIZE:
                continue  # too generic a token to be useful signal
            ids = sorted(ids)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pairs.add((ids[i], ids[j]))
    return sorted(pairs)


def _load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache_path: Path, cache: dict) -> None:
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _verify_pair(name_a: str, desc_a: str, sources_a: list, name_b: str, desc_b: str, sources_b: list,
                  same_name: bool = False, retries: int = 2) -> tuple:
    """Returns (same: bool, reason: str)."""
    src_a = ", ".join(sources_a[:3]) or "unknown"
    src_b = ", ".join(sources_b[:3]) or "unknown"
    note = (
        "\n\nNote: both entities have the exact same name -- this raises the prior that they're "
        "the same thing, but they could still be two unrelated entities that happen to share a "
        "title (this genuinely happens, e.g. several unrelated works titled \"Black Book\"). Check "
        "the descriptions before deciding."
        if same_name else ""
    )
    user_prompt = (
        f"Entity A: {name_a}\nDescription A: {desc_a}\nAppears in passage(s): {src_a}\n\n"
        f"Entity B: {name_b}\nDescription B: {desc_b}\nAppears in passage(s): {src_b}{note}"
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0},
    }
    for _ in range(retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            parsed = json.loads(content)
            category = str(parsed.get("category", "unclear")).strip().lower()
            reason = str(parsed.get("reason", ""))
            same = category == "identical"  # derived from the category itself, not a separate
            # model-stated boolean -- avoids the model naming the correct category (e.g. "part_of")
            # in its own reasoning and then contradicting it with same=true anyway
            return same, f"[{category}] {reason}"
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return False, "[unclear] verification failed (network/parse error) -- defaulted to different"


# NOTE: an order-swap self-consistency check (re-verify an "identical" verdict with A/B
# swapped, only trust it if both agree) was tried here and removed -- it introduced more
# false negatives than the false positives it caught (e.g. it rejected the genuine
# Lomonosov Moscow State University / Moscow State University rename alias). This model's
# category output appears sensitive to which entity is framed as "A" vs "B" independent of
# correctness, so order-swapping isn't a reliable consistency signal for it. The prompt
# fixes above (broader "different_individual" category, explicit county example) turned out
# to be sufficient on their own. If reintroducing a self-consistency mechanism later, verify
# it against a wide regression set first -- it's easy for a check like this to look like a
# safety net while actually trading false positives for a comparable rate of false negatives.


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _apply_pair_merges(G: nx.MultiDiGraph, confirmed_pairs: list, verbose: bool = False, label: str = "",
                        canonical_hints: set = None):
    """Union-Find merge a set of confirmed-same node pairs into G, returning
    (new_graph, id_map) where id_map covers EVERY original node id (mapped to
    itself if unchanged, or to its group's representative if merged).

    `canonical_hints` (from find_relation_based_canonical_hints) is preferred
    for representative selection when any group member is in it -- otherwise
    falls back to the longest-name heuristic, which can otherwise pick a
    vague alias description as the surviving node's identity (see that
    function's docstring)."""
    canonical_hints = canonical_hints or set()
    uf = UnionFind(list(G.nodes()))
    for a, b in confirmed_pairs:
        uf.union(a, b)

    from collections import defaultdict
    groups = defaultdict(list)
    for n in G.nodes():
        groups[uf.find(n)].append(n)

    id_map = {}
    merge_examples = []
    for root, members in groups.items():
        if len(members) == 1:
            id_map[members[0]] = members[0]
            continue
        canonical_members = [m for m in members if m in canonical_hints]
        rep_pool = canonical_members or members
        rep = max(rep_pool, key=lambda k: len(G.nodes[k].get("name", "")))
        others = [m for m in members if m != rep]
        rep_node = G.nodes[rep]
        for m in members:
            id_map[m] = rep
        for m in others:
            node = G.nodes[m]
            for d in node.get("descriptions", []):
                if d not in rep_node.setdefault("descriptions", []):
                    rep_node["descriptions"].append(d)
            for s in node.get("sources", []):
                if s not in rep_node.setdefault("sources", []):
                    rep_node["sources"].append(s)
        merge_examples.append((rep_node.get("name"), [G.nodes[m].get("name") for m in others], len(members)))

    if verbose and merge_examples:
        print(f"\n-- {label} merges --")
        for rep_name, other_names, _ in merge_examples[:20]:
            print(f"  '{rep_name}'  <=  {other_names}")
        if len(merge_examples) > 20:
            print(f"  ... and {len(merge_examples) - 20} more")

    # Always surface abnormally large merge groups in full, regardless of the 20-example cap
    # above -- a healthy build's groups are almost all size 2-3 (a handful of legitimately
    # very-common entities like "National Basketball Association" run into the double digits
    # and are fine); anything past LARGE_GROUP_WARNING_SIZE has, in every case audited in this
    # project so far, turned out to be transitive-bridging damage from one bad pairwise link
    # (see CLAUDE.md) -- and it was invisible in this exact truncated printout until a manual
    # audit found it. Don't let that happen silently again.
    LARGE_GROUP_WARNING_SIZE = 10
    large_groups = [(rep, others, size) for rep, others, size in merge_examples if size > LARGE_GROUP_WARNING_SIZE]
    if verbose and large_groups:
        print(f"\n-- WARNING: {len(large_groups)} abnormally large merge group(s) in {label or 'this'} "
              f"(size > {LARGE_GROUP_WARNING_SIZE}) -- inspect these before trusting the graph --")
        for rep_name, other_names, size in sorted(large_groups, key=lambda x: -x[2]):
            print(f"  size={size}: '{rep_name}'  <=  {other_names}")

    G2 = nx.MultiDiGraph(**G.graph)
    for n, attrs in G.nodes(data=True):
        if id_map[n] != n:
            continue
        G2.add_node(n, **attrs)
    for u, v, attrs in G.edges(data=True):
        u2, v2 = id_map[u], id_map[v]
        if u2 == v2:
            continue  # drop self-loop created by merging both endpoints together
        G2.add_edge(u2, v2, **attrs)

    return G2, id_map, len(merge_examples)


def resolve_entities(G: nx.MultiDiGraph, cache_path: Path, verbose: bool = True, workers: int = 4) -> tuple:
    """Returns (merged_graph, stats_dict)."""
    relation_pairs = find_relation_based_pairs(G)
    if verbose and relation_pairs:
        print(f"Relation-based merges (no LLM needed, already stated in source text): {len(relation_pairs)}")
        for a, b in relation_pairs:
            print(f"  '{G.nodes[a].get('name', a)}' <-> '{G.nodes[b].get('name', b)}'")

    # Exact-name homonym pairs flagged by graph_lib.py, captured BEFORE the graph changes below.
    same_name_pairs_orig = [tuple(sorted(p)) for p in G.graph.get("same_name_pairs", [])]

    # Apply relation-based merges to the graph FIRST, before any blocking/LLM verification.
    # This is important: a node with a vague, context-free description (e.g. "previous name
    # of the player") that's already known (for free) to alias one specific other entity must
    # not remain separately blockable -- otherwise it gets compared against every OTHER
    # same-name-token candidate too, and its vagueness can fool the LLM into false positives
    # against unrelated entities, which then bridges them all together via Union-Find
    # transitivity (this happened: it fused 6 different real footballers into one node).
    desc_pairs, desc_canonical_hints = find_description_based_alias_pairs(G)
    if verbose and desc_pairs:
        print(f"Description-based merges (vague alias description, no matching relation extracted): {len(desc_pairs)}")
        for a, b in desc_pairs:
            print(f"  '{G.nodes[a].get('name', a)}' <-> '{G.nodes[b].get('name', b)}'")

    n_relation_merged = len(relation_pairs) + len(desc_pairs)
    all_free_pairs = sorted(set(relation_pairs) | set(desc_pairs))
    canonical_hints = find_relation_based_canonical_hints(G) | desc_canonical_hints
    G, id_map, _ = _apply_pair_merges(G, all_free_pairs, canonical_hints=canonical_hints)

    same_name_pairs = sorted({
        tuple(sorted((id_map[a], id_map[b])))
        for a, b in same_name_pairs_orig
        if id_map[a] != id_map[b]
    })
    same_name_set = set(same_name_pairs)

    blocked_pairs = [p for p in find_candidate_pairs(G) if p not in same_name_set]
    pairs = same_name_pairs + blocked_pairs
    cache = _load_cache(cache_path)
    total = len(pairs)

    # Split into "already cached" (no LLM call needed, near-instant) and "needs
    # verification" (the actual cost) up front, so the concurrent work below is
    # only ever the real LLM calls.
    to_verify = []
    for a, b in pairs:
        key = "||".join(sorted((a, b)))
        if key not in cache:
            to_verify.append((a, b, key))

    if verbose:
        print(f"Entity resolution: {total} candidate pairs to check "
              f"({total - len(to_verify)} already cached, {len(to_verify)} need an LLM call, "
              f"{workers} concurrent worker(s))")

    llm_calls = 0
    t0 = time.time()
    cache_lock = threading.Lock()  # guards both the in-memory dict and the on-disk checkpoint

    def verify_task(a, b, key):
        name_a = G.nodes[a].get("name", a)
        name_b = G.nodes[b].get("name", b)
        desc_a = " | ".join(G.nodes[a].get("descriptions", []))[:300]
        desc_b = " | ".join(G.nodes[b].get("descriptions", []))[:300]
        sources_a = G.nodes[a].get("sources", [])
        sources_b = G.nodes[b].get("sources", [])
        t_start = time.time()
        same, reason = _verify_pair(name_a, desc_a, sources_a, name_b, desc_b, sources_b,
                                     same_name=((a, b) in same_name_set))
        t_this = time.time() - t_start
        return key, name_a, name_b, same, reason, t_this

    if to_verify:
        # Threads, not processes: each call is a blocking HTTP request waiting on
        # Ollama, which releases the GIL during that wait -- `workers` concurrent
        # requests can genuinely be in flight against Ollama at once. How much
        # that actually speeds things up depends on Ollama's own OLLAMA_NUM_PARALLEL
        # setting and available VRAM; if you see requests queueing up with no
        # speedup (or Ollama erroring under load), lower --workers.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(verify_task, a, b, key) for a, b, key in to_verify]
            for future in as_completed(futures):
                key, name_a, name_b, same, reason, t_this = future.result()
                with cache_lock:
                    cache[key] = {"same": same, "reason": reason, "name_a": name_a, "name_b": name_b}
                    llm_calls += 1
                    # checkpoint after every new LLM call -- safe to interrupt/resume anytime
                    _save_cache(cache_path, cache)
                    done = llm_calls
                if verbose:
                    avg = (time.time() - t0) / done
                    eta = avg * (len(to_verify) - done)
                    verdict = "SAME" if same else "different"
                    print(f"  [{done}/{len(to_verify)}] '{name_a}' vs '{name_b}' -> {verdict} ({reason}) "
                          f"({t_this:.1f}s, ETA ~{eta/60:.1f}m)")

    checked = 0
    confirmed_same = []
    for a, b in pairs:
        key = "||".join(sorted((a, b)))
        same = cache[key]["same"]
        checked += 1
        if same:
            confirmed_same.append((a, b))

    G2, _, n_groups_merged = _apply_pair_merges(G, confirmed_same, verbose=verbose, label="Entity resolution")

    stats = {
        "relation_based_merges": n_relation_merged,
        "candidate_pairs": len(pairs),
        "cached_hits": checked - llm_calls,
        "llm_calls": llm_calls,
        "confirmed_merges": len(confirmed_same),
        "groups_merged": n_groups_merged,
        "nodes_before": G.number_of_nodes(),
        "nodes_after": G2.number_of_nodes(),
    }

    return G2, stats
