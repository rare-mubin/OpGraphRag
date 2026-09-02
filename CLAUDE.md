# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This repo builds a knowledge graph from HotpotQA (`Dataset/formatted_output.json`) using a local LLM
(Qwen2.5-7B-Instruct via Ollama) for entity/relation extraction, retrieves query-specific subgraphs from
it, generates uncompressed baseline answers, and trains a GRPO-based policy to compress those subgraphs.
It implements **Stages 1–4** of a larger research project (see the team's methodology draft): Stage 1 is
KG construction (`select_subset.py` → `extract_kg.py` → `build_graph.py`), Stage 2 is query embedding +
GraphRAG retrieval (`embed_nodes.py` → `retrieve.py`, evaluated by `evaluate_retrieval.py` against
HotpotQA's own `supporting_facts`), Stage 3 is uncompressed baseline answer generation
(`generate_baseline_answers.py`, methodology Section III-C) — produces $A_o$/$T_o$/EM/F1 that later
compression stages get compared against. Stage 4 is the GRPO-trained RL graph-compression policy
(`compression_policy.py` + `train_compression_policy.py`, Section III-D/III-E) and its evaluation
(`generate_compressed_answers.py`) — **mechanically validated but not yet really trained**: the current
~20-question subset is a proof-of-concept scale, not enough to learn a generalizing policy.
`generate_pruning_baselines.py` implements the methodology's mandatory non-adaptive comparisons
(Section III-F: fixed similarity-threshold pruning and fixed hop-cutoff pruning) against the same
retrieved `G_q` and baseline. (Community detection, mentioned in `main.tex`'s Related Work only as
part of Edge et al.'s GraphRAG, is not part of this paper's own methodology and isn't planned here.)

All active code lives under `kg/`. **`Dataset/` is gitignored** (~97MB, freely redownloadable) --
see README's "Getting the dataset" for the Kaggle/HF links and the prepare step.

`Dataset/formatted_output.json` is what `select_subset.py` reads;
`Dataset/hotpot_dev_distractor_v1.json` is the download and isn't read by any script. **They are
content-identical** -- same 7,405 records in the same order, every field equal (verified by
comparing a canonical re-dump of each). `formatted_output.json` was produced by a PowerShell
`ConvertFrom-Json | ConvertTo-Json -Depth 100` round-trip, which only re-indents; the 52MB vs 45MB
difference is entirely whitespace. So despite the name, there is no cleaning or validation step
here and nothing is lost by just copying the raw file to that name. (An earlier version of this
file described `formatted_output.json` as "the cleaned, validated dataset" -- that was wrong.)

**Full setup/run/troubleshooting instructions are in [README.md](README.md) — read that before
making changes to the pipeline.** This file covers architecture and gotchas that aren't obvious from
reading a single file.

## Reproducing the full result package

After any pipeline rerun (e.g. on a bigger dataset), regenerate everything under `result/`
(images, per-question data, and `result/report.html`) with, in order:

```bash
python generate_pruning_baselines.py       # Stage 6 non-adaptive baselines (similarity/heuristic)
python evaluate_policy_classification.py   # node-level confusion matrix / ROC / AUC (no LLM calls, fast)
python generate_report.py                  # aggregates everything above into result/ -- no LLM calls, fast
```

This assumes `generate_baseline_answers.py`, `train_compression_policy.py`, and
`generate_compressed_answers.py` have already been rerun for the new dataset size (see Commands
below). `generate_report.py`'s ablation section is a **fixed historical reference** from the original
20-question pilot (documenting the reward-gating fix) -- it does not regenerate from the new run, since
the bug it demonstrates is fixed in the code and a fresh run only ever reproduces the "after" row. That's
expected and labeled as such in the report; it's not something to chase into matching the new dataset.

## Commands

No build/lint/test tooling — this is a data pipeline. Run from `kg/`, in order:

```bash
python select_subset.py --n 20 --seed 42 --max-passages 50   # sample HotpotQA questions -> unique passages
python extract_kg.py                                          # LLM entity/relation extraction (resumable, Ctrl+C-safe)
python extract_kg.py --gpu-temp-guard --gpu-temp-high 80 --gpu-temp-low 70   # same, but pauses
    # between passages if the GPU (via nvidia-smi) hits --gpu-temp-high, resumes at --gpu-temp-low
python build_graph.py --workers 4                               # merge into final graph + entity resolution
    # (--workers: concurrent LLM verification calls against Ollama; tune to your GPU's headroom)
python retry_failed_extractions.py                              # recover any empty extractions (see below)
python live_server.py                                          # live graph view at http://localhost:8765
python embed_nodes.py                                          # Stage 2: BGE-M3 embed every node (CPU by default)
python retrieve.py "a question" --top-k 5 --hops 2             # Stage 2: query -> candidate subgraph G_q
python evaluate_retrieval.py                                   # Stage 2: recall vs. supporting_facts, all subset questions
python generate_baseline_answers.py                             # Stage 3: uncompressed A_o, T_o, EM/F1 (needs Ollama)
python train_compression_policy.py --group-size 4 --epochs 3 --workers 4   # Stage 4: GRPO training (needs Ollama)
python generate_compressed_answers.py                           # Stage 4: A_c, T_c, Delta_EM/Delta_F1 vs. baseline
```

Dependencies: `pip install networkx requests pyvis sentence-transformers numpy matplotlib scikit-learn`
(no requirements.txt/pyproject.toml exists). `matplotlib`/`scikit-learn` are Stage 6 only
(`evaluate_policy_classification.py`, `generate_report.py`).

**Ollama does not survive closing the terminal that started it.** Before running `extract_kg.py` or
`build_graph.py` in a new session, check `ollama list` responds; if not, run `ollama serve` first.
Models are stored in `kg/ollama_models/` (relocated off the OS default via the `OLLAMA_MODELS` env var,
set persistently at the user level) — not a fresh `ollama pull` unless that directory is actually empty.

## Architecture

**Pipeline data flow:** `select_subset.py` → `extract_kg.py` → `build_graph.py` → `live_server.py`,
with intermediate state in `kg/output/` (`subset_passages.json`, `extractions.json`,
`entity_resolution_cache.json`, `knowledge_graph.json`). Every stage is designed to be re-run
idempotently against whatever's already on disk rather than assuming a clean run.

**`graph_lib.py` is the single source of truth for graph assembly**, imported by both `build_graph.py`
(one-shot export to `knowledge_graph.json`) and `live_server.py` (rebuilt from scratch on every poll
while extraction is running). Edit merge logic there, not in either caller, or the two will drift.

**Two-stage entity merging** (the non-obvious part of this codebase):
1. `graph_lib.py` auto-merges same-normalized-name entities *unless* the type is in
   `COLLISION_PRONE_TYPES` (`PERSON`/`ORG`/`WORK`) *and* they come from a different passage than any
   prior mention. This guard exists because blind name-merging silently fused unrelated real-world
   entities that happen to share a title (e.g. five different works all literally called "Black Book").
   Flagged same-name pairs are recorded in `G.graph['same_name_pairs']` instead of merged outright.
2. `entity_resolution.py` (used only by `build_graph.py` — deliberately **not** by `live_server.py`,
   which stays on cheap string-normalization so frequent polling stays fast) does the actual
   verification: relation-based merges are free (an alias relation like "was known as" already
   extracted from the text is trusted directly), while `same_name_pairs` and fuzzy-blocked candidates
   (same type, shared distinctive name token) each cost one LLM call. Every verification decision is
   cached in `output/entity_resolution_cache.json`, checkpointed after each individual call — safe to
   interrupt and resume without re-paying for pairs already checked.

   **Relation-based merges are applied to the graph *before* blocking runs** (`_apply_pair_merges`
   in `entity_resolution.py`), not just recorded alongside it. This matters: a node with a vague,
   context-free description (e.g. "previous name of the player") that's already known for free to
   alias one specific other entity must not remain separately blockable — otherwise it gets compared
   against every *other* same-name-token candidate too, and its vagueness can fool the LLM into false
   positives against unrelated entities, which then bridges them all together via Union-Find
   transitivity. (This happened once: it fused six different real footballers into one node.)

   **The relation-based free-merge safeguard itself had a real string-matching bug that let this
   exact failure mode through at 200-question scale, discovered via a full audit of
   `entity_resolution_cache.json` (per the audit method below).** `find_relation_based_pairs()`
   matched `ALIAS_RELATION_KEYWORDS` (natural-language phrases like `"known as"`) against relation
   strings *without normalizing* them first -- but `extract_kg.py`'s model frequently outputs
   relation types as `SCREAMING_SNAKE_CASE` (`"ALSO_KNOWN_AS"`, ~49% of all relations in one 200q
   run), and `"known as" in "also_known_as"` is `False` (space vs. underscore) even though it's
   the exact same relation semantically. Quantified impact: 89 of 151 genuine alias relations
   (59%) were silently missed. One casualty: a node extracted with the vague description
   `"Alternate name for Evan Thomas"` (from a real `ALSO_KNOWN_AS` relation that got missed) stayed
   independently blockable, and its vagueness fooled the LLM verifier into "identical" against
   ~50 unrelated same-name-token entities -- Union-Find transitivity then chained that into a
   single 141-entity merge group spanning William Shakespeare, Mark Zuckerberg, Tom Clancy,
   Richard Nixon, and even Marvel's Peter Parker, all fused into one node. **Fixed** by
   normalizing relation strings (`.replace("_", " ").replace("-", " ")`) before the keyword check.
   This fix only prevents *new* instances -- it does not retroactively undo merges already baked
   into an existing `entity_resolution_cache.json`; a graph built before this fix needs
   `build_graph.py` rerun (which will re-verify affected pairs, since the merge-group's
   cached "identical" verdicts are now pre-empted by the corrected free merge) to actually
   recover, not just the fix landing in the file.

   **Lesson for auditing this pipeline**: don't just check the ~0.76% pairwise LLM-verification
   error rate below in isolation -- also check merge-*group* sizes via Union-Find over confirmed
   `"same": true` pairs. A healthy build's groups should almost all be size 2-3; a small number of
   much larger groups (double digits+) is a strong, cheap-to-compute signal of transitive-bridging
   damage worth tracing back to its root pairwise link before trusting the graph.

   **When auditing group sizes, recompute Union-Find over only the LIVE candidate pairs a fresh
   `resolve_entities()` run would actually generate -- not blindly over every `"same": true"`
   entry ever written to `entity_resolution_cache.json`.** The cache accumulates across every run
   forever and is never pruned; after a code fix changes which node IDs exist (e.g. a
   representative-selection fix below), stale entries for node IDs that no longer exist as live
   candidates just sit there unused. A blind full-cache audit will keep "finding" a bug that's
   already fixed. Reconstruct `same_name_pairs + blocked_pairs` fresh from the current code and
   graph, then filter the cache to just those keys, before doing group-size analysis.

   **A representative-selection bug compounded the problem above, independent of the string-match
   bug itself.** `_apply_pair_merges()` picked the *longest name* as a merged group's surviving
   identity/description, with no regard for which side was the real entity. For the Evan Thomas
   case, `"Peter Evan Thomas"` (17 chars, the vague alias mention) beat `"Evan Thomas"` (11 chars,
   the real actor) purely on string length -- so even after the pair was correctly detected, the
   *wrong* side survived, keeping the poisonous vague description attached to the merged node and
   leaving it just as blockable as before. **Fixed** by `find_relation_based_canonical_hints()`,
   which uses the alias relation's own stated direction (`source ALSO_KNOWN_AS target` -- source is
   canonical) to bias representative selection, falling back to longest-name only when no
   directional hint exists.

   **A third, separate root cause: extraction sometimes creates a vague-alias-description entity
   with NO matching relation edge at all**, so `find_relation_based_pairs()` (even fully fixed)
   has nothing to catch -- e.g. an entity literally named `"Stephen Mark Scott"` with description
   `"Original name of Stephen Marcus"` and zero relations connecting it to anything. This alone
   produced its own ~30-entity merge group (Mark Zuckerberg, Stephen King, Scott Glenn, ...), same
   mechanism as the footballer/Shakespeare cases, just missing the relation-edge trigger entirely.
   **Fixed** by `find_description_based_alias_pairs()`: scan entity DESCRIPTIONS themselves for
   `ALIAS_RELATION_KEYWORDS` (now also including "original name", "true name", "legal name",
   "credited as", "professionally known as" -- the original list was missing "original name",
   which is exactly what caused this specific case to slip through even after the function
   existed), and treat the entity in the SAME source passage whose name exactly matches that
   passage's title as the canonical referent -- reliable because `extract_kg.py`'s own system
   prompt instructs the model to always extract the passage title as its own entity when coherent.

   **After all three fixes, a substantial residual remained: several independent ~15-35 entity
   merge groups persisted** (Scott/Spurrier cluster, Richard/Sherman/Shakespeare cluster, the
   Marie-Joseph/Lafayette cluster from the audit above), all via the SAME underlying mechanism --
   the LLM verifier over-trusting a shared common name TOKEN or generic category word (e.g.
   "Stephen", "Richard", "Joseph", "Alliance", "Mosque") within a much longer compound name as
   evidence of aliasing, not an extraction-completeness gap any of the three fixes above can catch.
   In the meantime, `_apply_pair_merges()` now always prints any merge group above
   `LARGE_GROUP_WARNING_SIZE` (10) in full, regardless of the standard 20-example console cap --
   this is what actually caused this whole class of bug to go undetected for as long as it did
   (the console output silently truncated to "... and N more"). This is a permanent, general
   safeguard: it doesn't fix the residual, but it guarantees the *next* large group, whatever new
   pattern causes it, shows up in every build's own console output instead of requiring another
   manual cache audit to discover. A follow-up full audit at 200-question scale found this pattern
   is broader than people's names alone -- it also fused `China`/`Taiwan`/`Iran` into one node,
   merged a European political-party alliance with the Yoga Alliance and National Wrestling
   Alliance, and merged the three separate Pixar `Cars`/`Cars 2`/`Cars 3` films into one.

   **Fix attempted: three targeted worked examples added to `VERIFY_SYSTEM_PROMPT`** covering
   generic-category-word sharing ("Alliance", "Mosque" are not evidence of identity on their own),
   sequential franchise entries (`Cars 2` vs `Cars 3` is `related_work`, not `identical`), and
   political entities with different qualifiers (`Republic of China` vs `People's Republic of
   China`). **Tested empirically, not just assumed to work**: re-verified the ~334 cached pairs
   inside the confirmed-bad groups found in the audit. Result: 111/334 (33%) corrected outright
   (e.g. `Chinese Taipei`/`Republic of China (Taiwan)` now correctly `different_individual`,
   `Ashley Tisdale`/`William Ashley Freehan` no longer conflated). Of the 134 non-self-referential
   pairs still verified "identical," manual spot-check found many are actually legitimate correct
   merges the audit had over-flagged (Alexander II/III of Russia title variants, `Nazi Party`
   `(NSDAP)`/`NSDAP` -- literally the same org, `Stephen Orr Spurrier`/`Steve Spurrier` -- his real
   name) -- but a real, smaller residual remains genuinely wrong: the `Alliance 90/The Greens
   Hamburg` cluster still merges with Yoga Alliance and National Wrestling Alliance (the exact
   pattern the new worked example targeted, in a structurally similar but not identical instance --
   the fix does not reliably generalize within a cluster), several `Sánchez` pairs, `Nazi Party
   (NSDAP)`/`Partido Nazi de Costa Rica`, `Burns and Allen`/`Gracie Allen` conflated with `Carlos
   Gracie Jr.`, and several cross-mosque merges including `Hassan II Mosque`.

   **Lesson: prompt-only iteration for this class of LLM-judgment weakness has real, demonstrated
   diminishing returns** -- a worked example reliably fixes the exact case it targets and close
   variants, but does not generalize to every structurally-similar case even within the same
   merge group. Further chasing individual remaining pairs with more worked examples is not
   recommended; the effort-to-return ratio degrades fast. Current state: net improvement applied
   and kept, residual accepted as documented (same posture as the existing ~0.76% pairwise-error
   baseline below, just with a fuller, quantified characterization now), and the large-group
   warning is the durable safety net for whatever surfaces next.

   An order-swap self-consistency check (re-verify an "identical" verdict with entities A/B swapped,
   only trust it if both agree) was tried and **removed** — it rejected genuine true positives about
   as often as it caught real errors, because this model's category output is sensitive to which
   entity is framed as A vs. B independent of correctness. Don't re-add it without validating against
   a wide regression set first; see the comment left in `entity_resolution.py` at that spot.

**Verification prompt outputs a category, not a boolean.** `entity_resolution.py`'s LLM call asks for
`category` (`identical`/`part_of`/`different_individual`/`related_work`/`coincidental_same_name`/
`unclear`), and `same` is derived from that category in code — never trust a model-stated `same: true/false`
directly here. This was a deliberate fix: the smaller local model's free-text reasoning would sometimes
correctly identify a part-of relationship (e.g. "Alaska Senate" as a chamber of "Alaska Legislature")
and then still output a contradictory `same: true` anyway.

**Entity resolution has a non-zero, documented residual error rate even after prompt fixes.** A full
manual audit of one build (1,975 pairs checked) found 15 wrong `identical` verdicts (~0.76%) across
several categories: transitive bridging via one bad pairwise link (the footballer case above),
same-name-different-thing pairs the model insists are related ("Cairn na Burgh Beag"/"Mòr" -- two
different islands), collaborators-on-the-same-work conflated as one person/org (co-producers, a
religious order and the school it founded), and genre/style labels merged just because they describe
the same artist ("electronic music"/"noise music"). Targeted prompt fixes for each pattern resolved
11 of 15; 4 residual errors persisted even after the fix was added specifically for their pattern --
treat this as a real, non-zero error ceiling for this model at this task, not a bug to keep chasing.
When auditing a build, don't just spot-check the first N merges printed to console -- pull every
`"same": true` entry from `output/entity_resolution_cache.json` and eyeball the full list; several of
the errors found here were well past the default 20-entry console preview.

**Stage 2 retrieval (`embed_nodes.py` → `retrieve.py`) reads `output/knowledge_graph.json` and
`output/node_embeddings.npy` independently of each other** — embeddings are a separate on-disk
artifact, not regenerated automatically when the graph changes. Rerun `embed_nodes.py` after any
`build_graph.py` run whose output you want reflected in retrieval. Retrieval is top-k cosine-similarity
seeds (BGE-M3, L2-normalized so dot product = cosine similarity) + k-hop undirected graph expansion —
deliberately not pure top-k similarity, since a multi-hop answer is often nowhere near the query in
embedding space and only reachable by walking the graph from a shared entity. `evaluate_retrieval.py`
checks this quantitatively: whether a retrieved $G_q$'s node `sources` cover the passages HotpotQA's
`supporting_facts` say are actually needed for each subset question.

**Edge attribute naming gotcha:** an edge's source-passage provenance is always `source_passage`, never
`source` — `source`/`target` are reserved by `networkx.node_link_data()` for the edge's actual endpoint
node ids, and a custom attribute literally named `source` gets silently overwritten on export.

**Stage 3 answer generation is two LLM calls per question, deliberately.** `generate_answer()` uses a
plain prompt; a second call, `shorten_answer()`, extracts the short direct answer from that response.
This isn't a stylistic choice -- tested strict brevity instructions, few-shot format examples, and a
JSON-schema constraint directly in the main prompt, and qwen2.5:7b-instruct ignored all of them for
"who"/"which" style questions, reliably wrapping a correct answer in a full explanatory sentence
regardless. Asking it to shorten its *own already-written* answer works reliably where asking it to
*answer tersely from scratch* doesn't. Both `generated_answer` (shortened) and `raw_answer` (original)
are saved in `output/baseline_answers.json`.

**Pipeline propagation gotcha: nothing downstream regenerates automatically.** The chain is
`extractions.json` → `build_graph.py` → `knowledge_graph.json` → `embed_nodes.py` →
`node_embeddings.npy` → `retrieve.py` / `generate_baseline_answers.py`. A fix or manual patch at any
point (e.g. editing `extractions.json` directly) has zero effect on retrieval/answer-generation results
until everything after it in the chain is rerun. This bit us once: a real extraction fix (a missing
numeric fact) was verified correct in `extractions.json`, but a baseline answer run happened to get that
same question right anyway before `build_graph.py` had actually been rerun to incorporate it -- LLM
inference isn't perfectly deterministic even at temperature 0, so a single passing result is not
confirmation a fix propagated. Always check the actual downstream artifact (e.g. query
`knowledge_graph.json` for the expected new node/edge) before crediting a fix, not just a re-run's output.

**Stage 4's policy had two real initialization bugs, not RL/reward-logic bugs, and both looked
exactly like "policy collapse" until diagnosed.** `compression_policy.py`'s state vector concatenates
raw L2-normalized node/query embeddings (~0.03 std per dim) directly with scalar graph-context features
on very different scales (similarity in [-1,1], degree/hop-norm in [0,1]). Two consequences, found by
directly inspecting an *untrained* policy's output (not by staring at training curves):
1. Without input normalization, a freshly-initialized linear layer produced logits with essentially no
   spread (std ~0.001 across 16 different real nodes) -- every node got nearly the same probability
   regardless of its actual features. Fixed with `nn.LayerNorm` as the first op in `CompressionPolicy.forward`.
2. Even after that, the final layer's randomly-initialized bias can still land the *entire* policy on
   the "remove" side of the 0.5 threshold for every node in every graph, purely by chance -- and since
   `train_compression_policy.py` uses a fixed default seed, this reproduced identically on every rerun
   and survived real GRPO gradient steps untouched (a couple of steps on 2 questions isn't nearly enough
   signal to move a systematically-biased init). This produced a fully degenerate "compress everything to
   nothing" policy (`generate_compressed_answers.py` showed `G_c=0 nodes` on 19/20 questions, EM/F1
   cratering) that looked exactly like classic small-sample RL collapse but wasn't -- retraining with a
   different reward balance or more entropy regularization would NOT have fixed it, because the actual
   cause was upstream of any training happening at all. Fixed by explicitly initializing the final
   layer's bias toward KEEP (`nn.init.constant_(self.net[-1].bias, 1.0)`, giving `sigmoid(1.0) ≈ 0.73`):
   an untrained policy should default to behaving like the Stage 3 baseline (keep ~everything), not to
   discarding all context and getting zero learnable reward signal to ever escape from.

   **Lesson for next time**: before trusting a training run's results (especially on a tiny dataset),
   directly print an *untrained* policy's output distribution on a real input first. A collapsed
   distribution at initialization is an init bug, not something more training or a different reward
   shape will fix -- verify this before spending 45+ minutes on a training run.

**Stage 4 also had a reward-shaping exploit that reproduces the exact same "compress everything to
nothing" symptom as the init bugs above, even with both init fixes in place** -- diagnosed by rerunning
training and directly reading `compression_training_log.json`'s per-step `mean_CR`/`entropy`, not by
staring at final EM/F1. `compute_reward()` in `train_compression_policy.py` originally gave the CR
(compression-ratio) bonus unconditionally: `R = Q_c + lambda_cr*CR - gamma_penalty*P`. Two properties
specific to this LLM+dataset made "remove everything" a dominant strategy under that formula: (1)
`ANSWER_SYSTEM_PROMPT` (`generate_baseline_answers.py`) tells the model to guess rather than refuse, and
Qwen2.5-7B-Instruct often answers HotpotQA's Wikipedia-famous entities correctly from parametric
knowledge alone even with `G_c` = 0 nodes -- so an empty graph could score `Q_c ~= Q_o` (`P~=0`) *and*
still collect the full CR bonus, strictly beating keeping full context and getting the same answer
right. (2) When the baseline was already wrong (`Q_o` low), `P` stayed ~0 even if the compressed answer
was also wrong, making an empty graph free positive reward regardless of correctness. Across a group of
GRPO trajectories this made "empty the graph" the group-relative-advantage-favored move on most
questions, and entropy collapsed to ~0 by epoch 3 (a genuinely trained, converged policy -- not noise)
that empties every graph. **Fix:** gate the CR bonus on not losing quality --
`cr_bonus = lambda_cr*CR if Q_c >= Q_o else 0.0` -- so compression only pays off when it didn't cost
accuracy. After the fix, entropy no longer collapses (stays double-digit through epoch 3) and the
policy's behavior on the 20-question subset is no longer uniform: some questions keep the full graph
(`CR=0%`, matches baseline), a few compress and *improve* over baseline, most that compress heavily
still lose the answer. Net EM/F1 delta is still negative post-fix (this is expected at 20-question/
3-epoch scale per the proof-of-concept caveat above, not a sign of a remaining bug) but the earlier
uniform 20/20 "G_c=0 nodes" collapse is specifically the exploit above, not the model failing to learn
-- don't re-diagnose it as an init issue if it resurfaces; check `compute_reward` and the training log's
per-step entropy first.

**On this same 20-question subset, both non-adaptive pruning baselines currently beat the RL policy
outright** (`output/pruning_baselines.json`, generated by `generate_pruning_baselines.py`): fixed
1-hop heuristic pruning gets 33% compression at ~0 F1 cost (Delta_F1 -0.1pp, i.e. slightly *better*
than baseline), fixed similarity-threshold pruning gets 44% compression at a 7.2pp F1 cost, while the
RL policy gets 53% compression at a 36.5pp F1 cost. This is the expected result of a policy trained
on 20 questions for 3 epochs (still in the bimodal "keep everything or lose the answer" regime noted
above), not a regression or a sign the framework is broken -- re-evaluate this comparison after
training on meaningfully more data before treating it as a real result.

**The `Q_c >= Q_o` reward gate above only closed HALF the exploit it was meant to fix -- confirmed
by scaling to 200 questions, where the policy collapsed to `CR=99.9%` (0 KEEP predictions across all
15,447 evaluated nodes in `evaluate_policy_classification.py`) even faster than before (entropy hit
~0 by the END of epoch 1, not epoch 3).** The gate's own case-(2) description above ("when the
baseline was already wrong... emptying the graph was free positive reward regardless of correctness")
was never actually fixed by `Q_c >= Q_o`: when `Q_o == 0` (baseline totally wrong) and the compressed
answer is also wrong (`Q_c == 0`), `0 >= 0` is trivially true, so the CR bonus was still granted in
full -- the fix only ever closed case (1) (the "lucky empty-context guess" exploit). This barely
mattered on the 20-question pilot (few questions had `Q_o == 0`), which is exactly why it looked
fixed there, but at 200-question scale a more representative/harder question mix pushed the average
baseline down to EM 27.0%/F1 33.7% (from 60.0%/73.5% at n=20 -- confirmed NOT a retrieval regression,
`retrieval_eval.json` still shows 93.2% avg recall), and 59.5% of training steps had `Q_o == 0`,
making the unclosed loophole the *dominant* case rather than an edge case. **Fix:** additionally
require `Q_c > 0`, not just `Q_c >= Q_o` --
`cr_bonus = lambda_cr*CR if (Q_c > 1e-6 and Q_c >= Q_o - 1e-6) else 0.0` -- so the bonus can only be
earned by preserving *actual, nonzero* correctness, never by merely matching an already-worthless
baseline. **Lesson:** when a reward-gate's own inline comment names a specific failure case, verify
the actual gate condition excludes that case algebraically (what happens when both sides are exactly
0?) -- don't trust that describing the exploit in a comment means the code next to it actually closes
it. **That gate was necessary but NOT sufficient -- retraining with it still collapsed to CR=99.9%.**
See the next section for what actually fixed it.

**The collapse was ultimately THREE independent causes, and the two reward gates above were only
the third of them.** Retraining with `Q_c > 0` in place still produced `CR=99.9%`, `G_c=0` on
200/200 questions. Full diagnosis, in order of how much each mattered:

1. **A gradient-scale bug in `sample_trajectory()` (`compression_policy.py`) -- the actual
   killer.** It returned the SUM of per-node log-probs and entropy, not the mean. `|G_q|` here
   averages ~114 nodes and reaches 610, so a trajectory's log_prob scaled with graph size and
   `-advantage * log_prob` produced pg_loss magnitudes up to ~15 for an advantage of ~1. With Adam
   at lr 1e-3 that drove the final-layer logits to saturation within ~30 steps. Once saturated,
   every trajectory in a GRPO group samples identically -> reward std 0 -> advantage 0 -> **92% of
   all 600 steps produced literally zero policy gradient**, and the run was decided by step 39.
   The entropy bonus cannot rescue this either: at saturated logits its own gradient
   (`dH/dlogit = log((1-p)/p) * p(1-p)`) vanishes too. **Fixed** by returning per-node means, which
   makes gradient scale independent of `|G_q|` -- essential when graph size varies by 100x across
   questions. Also switched to `Bernoulli(logits=...)` so log_prob stays finite when saturated.
2. **GRPO cannot express per-node credit at all.** One scalar advantage is applied uniformly to
   every node's log-prob, so it can never say "keep THIS node, drop THAT one" -- only "that whole
   subset was good/bad." **Fixed** by adding an auxiliary per-node supervised BCE loss
   (`--aux-coef`, default 1.0) against the same `supporting_facts`-derived relevance labels
   `evaluate_policy_classification.py` scores against, with `pos_weight` correcting the ~10:1
   class imbalance. Even after fix 1, ~49% of steps still have zero reward spread; the auxiliary
   loss is what trains on those. **Because this trains on the eval's own labels, a train/val
   split (`--val-frac`, default 0.2) is now mandatory** and is saved into the policy checkpoint;
   `evaluate_policy_classification.py` and `generate_compressed_answers.py` both default to
   `--split val`. Reporting node metrics without it is circular.
3. **The reward's global maximum was "empty graph + correct answer" (R=1.5), which this LLM hits
   constantly from parametric knowledge.** 30 training steps had `Q_o == 0` (full 30k-token
   context answered WRONG) while a near-empty trajectory scored `R > 1.2` by answering correctly
   from memory -- e.g. "Which 'Roseanne' star is in Scream 2?" -> full context F1 0.00, near-empty
   context -> "Laurie Metcalf", F1 1.00. **No gate on `Q_c` can catch this**: `Q_c` really is 1.0,
   and the reward was honestly crediting the COMPRESSION for an answer the model already knew.
   **Fixed** by gating the CR bonus on beating a per-question no-context control `Q_empty` -- the
   F1 this same model scores on this same question with an EMPTY context, measured once and cached
   in `output/no_context_answers.json` (2 LLM calls/question, checkpointed, reused across runs).
   An empty `G_c` scores `Q_c == Q_empty` by construction and so can never earn the bonus.
   Measured `mean Q_empty = 0.247` across all 200 questions -- a large parametric-knowledge floor,
   though note full context (F1 0.336) does still beat no context on average, so "no context beats
   full context" is true only per-question, not in aggregate. **This is a real deviation from
   main.tex's stated reward equation (Section III-E) and needs reflecting there.**

**SUPERSEDED by the struct4 retrain below -- the numbers in this section were measured with the
old full-2052-feature state vector and a skewed random split. Kept because the stability evidence
(entropy, CR spread) is what confirmed the three collapse fixes worked.**

**Post-fix results (200q, group-size 4, 3 epochs, 160 train / 40 val).** Training is stable: mean
entropy settles at 0.61 -> 0.53 -> 0.50 per epoch (and is flat WITHIN epoch 3), `mean_CR` holds at
0.45 -> 0.48 -> 0.51 instead of running to 1.0, and per-question CR spans 0.00-0.98 (stdev 0.218)
-- i.e. genuinely query-adaptive, keeping entire graphs for some questions and 2% for others.
Held-out node classification: **AUC 0.871** (was 0.517), accuracy 0.934, precision 0.501 at a 6.6%
base rate = **7.6x lift over chance**, recall 0.464.

Answer-level, all on the same 40 held-out questions:

| method | EM | F1 | CR |
|---|---|---|---|
| Uncompressed baseline | 25.0% | 29.7% | 0% |
| **RL policy (GRPO + aux)** | **40.0%** | **49.2%** | **72.6%** |
| Similarity pruning | 45.0% | 52.0% | 60.6% |
| Heuristic pruning | 47.5% | 60.4% | 74.0% |

**Read this honestly: the collapse is fixed and the policy is now genuinely useful (+15.0pp EM /
+19.5pp F1 over the uncompressed baseline at 72.6% compression), but it still does NOT beat the
non-adaptive baselines -- `heuristic_prune` strictly dominates it on all three axes at once.**
That remains the open problem, not something the fixes above resolved. Also note both pruners beat
the uncompressed baseline by a wide margin, so "pruning helps this generator focus" is the robust
finding here, independent of learning.

**Known caveats in the current run, worth fixing before this is paper-final:**
- The random train/val split is **skewed 3.1 sigma**: `G_q` sizes are heavy-tailed (median 48,
  mean 99, max 610) and val drew the big graphs (175.5 nodes/q vs train's 98.6), giving val a
  lower relevant base rate (6.6% vs 11.6%). Val is therefore the HARDER split, so the
  generalization numbers are conservative -- but train and val precision are not directly
  comparable to each other. Use a stratified-by-graph-size split on any rerun.
- Val node-F1 actually peaked at epoch 1 (0.494) vs epoch 3 (0.482) while train kept improving
  (3.0x -> 4.0x lift): mild overfitting, so 3 epochs may be slightly past the sweet spot.
- Epoch 2 was unstable (val KEEP-rate spiked to 0.366, val F1 fell to 0.263) before recovering.
- ~10% of questions have NO `supporting_facts` node in their `G_q` at all, so the auxiliary loss
  is skipped there (`n_pos == 0` guard) and only the RL term applies.

**WHY heuristic pruning still beats the policy -- diagnosed, and it is NOT the decision
threshold.** Compared both methods' KEEP sets against `supporting_facts` on the 39 val questions
that have at least one relevant node (no LLM calls; per-question detail in
`output/keepset_comparison.json`):

| | nodes kept | precision | recall | answer F1 |
|---|---|---|---|---|
| RL policy | 10.8 | 0.529 | 0.499 | 0.505 |
| heuristic (1-hop) | 15.8 | 0.564 | **0.727** | 0.620 |

**The gap is recall, not precision** -- near-identical precision, but the policy keeps ~2/3 as
many nodes and loses 23pp of recall. Its KEEP set is also almost a strict SUBSET of the
heuristic's: of 10.8 nodes kept, 9.2 (85%) are also kept by the heuristic and only 1.6 are unique,
while the heuristic keeps 6.7 relevant-ish nodes the policy drops. In other words the policy
learned a conservative subset of "1 hop from a retrieval seed" and nothing the heuristic doesn't
already know. Answer-level, 31 of 39 questions are ties; the whole F1 gap comes from 8 questions.

**A KEEP-threshold sweep rules out the easy fix, and corrects a misreading of the AUC.** Since
AUC is 0.871, the obvious hypothesis was that the ranking is good and only the 0.5 operating point
is misplaced. It is not:

| threshold | precision | recall | node-F1 | KEEP-rate |
|---|---|---|---|---|
| heuristic | 0.521 | 0.704 | **0.599** | 0.090 |
| 0.50 (current) | 0.501 | 0.464 | 0.482 | 0.061 |
| 0.40 | 0.391 | 0.588 | 0.470 | 0.100 |
| 0.30 | 0.255 | 0.706 | 0.375 | 0.184 |
| 0.20 | 0.159 | 0.873 | 0.269 | 0.365 |

At the threshold that MATCHES the heuristic's recall (0.30), the policy's precision is 0.255 vs
the heuristic's 0.521 -- half, at the same recall. **No threshold anywhere on the curve reaches
the heuristic's node-F1 of 0.599; the policy's best is 0.482, at the 0.5 it already uses.** The
heuristic dominates the policy's entire precision-recall curve.

**So do not quote AUC 0.871 as evidence the policy learned the task.** With a 6.6% base rate,
7,019 nodes and only 466 relevant, AUC is dominated by correctly ranking the mass of obviously
irrelevant nodes, which is easy. Precision at the top of the ranking -- the only region a
compression policy operates in -- is where it is actually weak. Report precision/recall at a
stated KEEP-rate against the heuristic's operating point instead; AUC alone flatters this task
badly.

**The important structural implication**: hop-distance-from-nearest-seed is ALREADY an input
feature in `build_state_features()`, so the policy could represent `heuristic_prune` exactly and
simply does not learn it. That makes this an optimization/objective problem, not a representational
one, and points at making the structural prior the policy's starting point rather than hoping it
rediscovers it -- e.g. feed the heuristic's own KEEP decision in as a feature and learn a residual
on top of it (which floors performance at the heuristic), or initialize/regularize toward it.
Worth trying before reaching for a different RL algorithm.

**ROOT CAUSE FOUND: the raw node/query embeddings in the state vector are actively
HARMFUL, not merely diluting.** The trained policy puts 99.2% of its first-layer weight energy
on the 2048 embedding dims and 0.8% on the 4 structural features -- though per-dimension it
correctly weights structural features 1.85x higher, with `hop_norm` the single highest-weighted
feature it has. A supervised-only feature ablation (`ablate_features.py`, no LLM calls, minutes)
settles what that costs, all scored on the val split at the heuristic's own KEEP-rate (0.090) so
the comparison is like-for-like:

| representation | dims | precision | recall | node-F1 |
|---|---|---|---|---|
| full 2052 (current) | 2052 | 0.349 | 0.472 | 0.401 |
| emb->64 + struct4 | 68 | 0.360 | 0.487 | 0.414 |
| emb->16 + struct4 | 20 | 0.348 | 0.470 | 0.400 |
| **struct4 only** | **4** | **0.527** | **0.712** | **0.606** |
| *heuristic_prune (target)* | -- | *0.521* | *0.704* | *0.599* |

**A 4-feature model beats the 2052-feature one by +0.205 node-F1 and is the only variant that
clears the heuristic.** Note this is NOT a dilution/rebalancing problem: projecting the
embeddings down to 64 or even 16 dims barely helps (0.414 / 0.400). They have to go entirely.
With only ~15.7k training nodes against 2048 dims, the embeddings are overfitting fuel.

**But `struct4` is 97.7% identical to `heuristic_prune` -- it relearned the 1-hop rule**, and
its +0.007 margin is parity, not a win. `analyze_hop_shells.py` shows why the structure is so
dominant, and where the only real headroom is (val split):

| hop from seed | nodes | % relevant | heuristic KEEP | struct4 KEEP |
|---|---|---|---|---|
| 0 (seed) | 200 | 50.0% | 100% | 84.5% |
| 1 | 430 | 53.0% | 100% | 88.1% |
| 2 | 6,389 | **2.2%** | **0%** | 1.3% |

**138 relevant nodes (30% of all 466) are stranded in the 2-hop shell that the heuristic
discards wholesale.** That is the entire remaining headroom, and it caps the heuristic's recall
at 0.704. Within that shell they ARE recoverable:

| signal (2-hop shell only) | val AUC | P@k (chance = 0.022) |
|---|---|---|
| query-node cosine similarity ALONE | 0.867 | 0.304 (14x chance) |
| struct4 (learned) | **0.913** | 0.275 |
| full 2052 (learned) | 0.718 | 0.116 |
| raw embeddings only (2048) | 0.604 | 0.029 (~chance) |

**The headline lesson: the derived scalar beats the representation it came from, badly.** One
hand-computed cosine similarity gets AUC 0.867 in the 2-hop shell; the 2048 raw dims it is
computed FROM get 0.604, barely above chance -- and including them drags the learned model from
0.913 down to 0.718. At this data scale, feature engineering beats representation learning here,
and that is a reportable finding rather than just a bug.

**Caution when acting on this**: naively adding the top-138 2-hop nodes at ~30% precision trades
precision for recall almost evenly (node-F1 0.599 -> ~0.600). Node-F1 is NOT the objective --
answer quality is, and a missing supporting fact costs far more than an extra node's tokens
(the reward already encodes this: `gamma_penalty` 2.0 vs `lambda_cr` 0.5). So the retrain should
deliberately operate at higher recall than the node-F1-optimal point, and be judged on the
answer-level table, not on node-F1.

**`generate_report.py` scopes the pruning baselines to whatever question set
`compressed_answers.json` covers.** Since the policy now defaults to the 40-question val split
while `generate_pruning_baselines.py` has no policy to hold out from and covers all 200, an
unscoped table silently compared a 40-question policy result against a 200-question baseline
result (which flattered heuristic pruning by ~14pp F1). Don't remove that scoping.

**Baseline EM/F1 dropping substantially from n=20 to n=200 (60.0%/73.5% -> 27.0%/33.7%) is expected
sample-composition variance, not a bug** -- a small 20-question pilot is more likely to land on an
easier-than-average slice of HotpotQA than a 200-question sample; retrieval quality itself held up
fine at scale (93.2% avg recall, 87.5% full recall vs. 97.5% avg at n=20). Worth noting: both
non-adaptive pruning baselines *beat* this weaker n=200 baseline outright (similarity-pruning: EM
39.5%/F1 49.2% at 63% compression; heuristic-pruning: EM 34.0%/F1 43.7% at 66% compression) -- a
genuinely interesting result suggesting pruning irrelevant retrieved context helps the generator focus
at this scale, independent of whatever the RL policy is doing.

**Windows native-crash gotcha:** if `embed_nodes.py`/`retrieve.py`/`evaluate_retrieval.py` exit
instantly with no output/traceback (exit code `-1073741819` = `STATUS_ACCESS_VIOLATION`), check for an
orphan `datasets` package first (`pip show datasets` — if `Required-by:` is empty, nothing in this
project needs it). `datasets` pulls in `pyarrow`, and `pyarrow`'s `arrow.dll` intermittently crashes
when loaded into the same process as `torch`'s CUDA DLLs. `pip uninstall datasets pyarrow` fixes it.
Diagnosed via Windows Event Viewer → Application log → "Application Error" (shows the faulting module
name directly) — worth checking there first for any similar unexplained silent-exit crash rather than
guessing at OpenMP/MKL conflicts, which looked plausible but was a red herring here.

## Struct4 retrain: current state of Stage 4 (supersedes the tables above)

**Acting on the ablation worked.** `build_state_features()` now defaults to
`include_embeddings=False` (the 4 engineered scalars only); `--features full`
reproduces the old behaviour. The feature mode is stored in the checkpoint and both
eval scripts read it back, defaulting to `"full"` for pre-flag checkpoints.

Two supporting changes landed with it, and both immediately earned their place:
- **Stratified train/val split by `|G_q|`** (sort by size, take every 5th). The old
  random split was 3.1 sigma skewed; val is now 4,154 nodes rather than 7,019, i.e.
  the giant graphs are spread across both splits instead of piling into val.
- **Best-by-val-F1 checkpointing.** Val node-F1 peaked at epoch 1 in *both* runs
  (0.647 -> 0.618 -> 0.586 here) while train kept improving, so a fixed 3 epochs was
  shipping an overfit policy. The saved checkpoint is now epoch 1.
- Retrieval is also cached per question now (it was recomputed every epoch plus twice
  more per epoch for validation). Under struct4 the cache is ~4 floats per node.

**Node level, held-out split -- a clear win, and the recall gap is closed and reversed:**

| | precision | recall | node-F1 | AUC | KEEP-rate |
|---|---|---|---|---|---|
| heuristic_prune | 0.521 | 0.704 | 0.599 | -- | 0.090 |
| old policy (2052 feat) | 0.501 | 0.464 | 0.482 | 0.871 | 0.061 |
| **new policy (struct4)** | **0.534** | **0.820** | **0.647** | **0.932** | 0.170 |

Recall went 0.464 -> 0.820, past the heuristic, and precision rose at the same time
despite keeping ~2x as many nodes. That is genuine dominance, not a threshold slide.

**Answer level -- competitive, but the margin is NOT significant. Do not report it as a win:**

| method | EM | F1 | CR |
|---|---|---|---|
| Uncompressed baseline | 32.5% | 36.7% | 0% |
| RL policy (struct4) | 42.5% | **53.5%** | 59.1% |
| Similarity pruning | **45.0%** | 51.9% | 59.8% |
| Heuristic pruning | **45.0%** | 51.1% | 61.7% |

Paired per-question against the heuristic: **+0.023 F1 (SE 0.041, 0.6 sigma)** and
**-0.025 EM (-0.6 sigma)**, with **35 of 40 questions tied**. Against similarity
pruning: 0.3 sigma and -0.4 sigma. None of this is distinguishable from noise at
n=40. The defensible claim is that the policy is **no longer strictly dominated** by
the non-adaptive baselines (last run `heuristic_prune` beat it on all three axes at
once); it is not evidence that it beats them.

**Why node-level gains are not reaching the answers**: on 35/40 questions the
generator returns the same answer from either node set, so a large node-level
improvement compresses into a tie. Separating the methods needs more questions
(n=40 gives SE ~0.04, so only a ~8pp+ gap would register) or a subset where the
retrieved context actually decides the answer.

**Do not compare these numbers to the section above.** Stratification changed the val
split, so the uncompressed baseline itself moved (F1 29.7% -> 36.7%). Only within-run
comparisons are valid.

## The real ceiling: Ollama was silently truncating every long context

**READ THIS FIRST -- it invalidates the section below and most numbers recorded above.**

`generate_baseline_answers.py` passed only `{"temperature": 0.0}` to Ollama and never set
`num_ctx`. **Ollama 0.32's default is 4096 tokens and it silently truncates anything longer,
keeping the END of the message.** The context sits at the start of the user message and the
question at the end, so the question always survived and the evidence was discarded -- with
no error, no warning, and nothing in the response to indicate it happened. `T_o` averages
**30,352 tokens** here, so nearly every question was scored against a model that had never
seen its supporting passages.

Verified directly, not inferred: a unique fact placed at the START of a ~34k-token context
returned *"The information provided in the given text does not contain any detail..."* under
the default, and was answered correctly with `num_ctx=32768` on byte-identical input. Then,
re-asking 6 real previously-refused questions with the fix: **5 of 6 improved**, several from
F1 0.00 to 1.00 (`'No'` -> `'Sherwood Stewart'`, `'No year'` -> `'1961'`).

**Fixed** by `ANSWER_NUM_CTX = 32768` (this model's max trained context) and
`SHORTEN_NUM_CTX = 4096` in `generate_baseline_answers.py`. All other stages import
`generate_answer`/`shorten_answer` from there, so the fix propagates to pruning baselines,
GRPO training, and compressed-answer evaluation automatically.

**VRAM interaction (8GB card):** KV cache for Qwen2.5-7B is ~56 KB/token -- 0.45 GB at 8k,
0.88 GB at 16k, **1.75 GB at 32k** -- on top of ~4.7 GB of Q4_K_M weights. 32k fits in 8 GB
only if BGE-M3 (~2.2 GB) is NOT also resident, so pass `--device cpu` to anything that
embeds and calls Ollama in the same process. Raising `num_ctx` is not free.

**What this invalidates.** Everything below in this section, and every EM/F1 number recorded
anywhere above, was measured under truncation. In particular the "context size actively
degrades the generator" claim was measuring Ollama's truncation, not the model -- and the
6.4 sigma "pruning beats the uncompressed baseline" result is confounded, because pruning's
main effect was getting contexts under 4096 so they survived truncation intact. **Re-run the
full pipeline from `generate_baseline_answers.py` onward before trusting or reporting any
comparison.** The observations below are kept only as the record of how this was found.

---

## (SUPERSEDED, kept for the diagnostic trail) Refusal-rate analysis that led to the above

**This looked like the most reportable finding in the repo. It was an artifact.** Retrieval recall is 92.0% and the uncompressed baseline still only
scores F1 0.336. The gap is not retrieval and not reasoning -- it is that the generator
cannot LOCATE the supporting facts in a large context, and then honestly says they are
absent:

| context quartile | mean `T_o` | refusal rate | mean F1 |
|---|---|---|---|
| Q1 smallest | 2,246 | **12.0%** | **0.590** |
| Q2 | 8,659 | 22.0% | 0.302 |
| Q3 | 25,830 | 36.0% | 0.196 |
| Q4 largest | 84,672 | 36.0% | 0.255 |

- **53/200 (26.5%)** of raw answers contain a "the information provided does not
  contain..." style refusal. Mean F1 on those is 0.176 vs 0.393 on the rest.
- **46 of those 53 (87%) had FULL retrieval recall** -- every supporting passage was
  present in the serialized context and the model still did not find it.
- Refusal rate **triples** (12% -> 36%) and F1 falls from 0.590 to ~0.20 as context grows.

This single mechanism explains a lot that was previously filed as separate puzzles:
why both non-adaptive pruners beat the uncompressed baseline by ~15pp (pruning is
repairing a generation failure, not just saving tokens); why `mean Q_empty` is a large
0.247; and why 15 of 40 val questions are answered wrong by *every* method (they cluster
in the big-context quartiles where the generator refuses regardless of pruning).

**Implication for the paper**: "retrieved-context size actively degrades a 7B generator,
and pruning recovers ~15pp F1" is a clean, well-evidenced claim on n=200 with a real
mechanism. The RL-vs-heuristic answer-level margin is 0.6 sigma on n=40 and is not
reportable no matter how much more the policy is tuned. Lead with the former.

**Also found: `shorten_answer` discards a correct answer on 4/200 (2.0%)** -- the raw
answer contains the gold string but the shortened form is a null token ("No"). Worth
fixing, but small. **Do not measure this as "gold string appears in raw_answer" alone**:
on comparison questions ("Between X and Y, which...") both candidate names appear in the
raw text, so that test inflates the rate to ~7.5% by counting cases where the model
simply chose the wrong candidate. The script's own check requires the shortened answer to
be a null token.

## Post-`num_ctx`-fix results (the first uncontaminated run)

Everything from `generate_baseline_answers.py` onward was regenerated with
`ANSWER_NUM_CTX = 32768`. **These supersede every EM/F1/CR number above.**

**The truncation artifact is confirmed and gone.** Uncompressed baseline F1
**0.336 -> 0.575**; refusal rate **26.5% -> 8.0%**. The refusal-vs-context-size
trend that the superseded section treated as a real finding has vanished: by
quartile it is now 8% / 2% / 2% / 20%, and Q3 (mean 25.8k tokens) scores the
*highest* F1 (0.693). More context is now better, up to the window limit. Only Q4
(mean 84.7k) still degrades, because it genuinely exceeds 32k.

**Both non-adaptive pruners lost their advantage, exactly as predicted.** Paired
against uncompressed over all 200 questions: heuristic **+0.004 (0.2 sigma)**,
similarity **+0.010 (0.4 sigma)**, both with 23 better / 23 worse / 154 tied. The
earlier 6.4 sigma "pruning beats the uncompressed baseline" result was measuring
pruning's ability to get contexts under 4096 so they survived truncation. Do not
revive it.

**Answer level, 40 held-out val questions:**

| method | EM | F1 | CR |
|---|---|---|---|
| Uncompressed baseline | 57.5% | 63.8% | 0% |
| **RL policy (struct4)** | 57.5% | **68.8%** | **58.6%** |
| Heuristic pruning | 55.0% | 63.1% | 61.7% |
| Similarity pruning | 50.0% | 56.9% | 59.8% |

The policy is now the only method that beats uncompressed, cutting context 77.3%
(26,134 -> 5,942 tokens) while doing so, and it never loses a question to either
pruner (3-0 vs heuristic, 7-2 vs similarity). Paired significance is still modest:
vs similarity **2.3 sigma**, vs heuristic **1.7 sigma**, vs uncompressed **1.1
sigma** (F1 +0.050, EM exactly 0.0), with 33/40 tied against uncompressed. State
the supportable claim -- the policy *preserves* answer quality while removing 77%
of context where the fixed rules do not -- rather than claiming it improves it.

### The result worth leading with: compression matters where context OVERFLOWS

Splitting by whether the full context fits in `num_ctx`:

| | n | baseline F1 | policy F1 | dF1 | policy `T_c` |
|---|---|---|---|---|---|
| context fits | 31 | 0.681 | 0.681 | **-0.000 (0.0 sigma)** | 2,804 |
| context overflows | 9 | 0.489 | 0.710 | **+0.222 (2.0 sigma)** | 16,752 |

Same split for the pruners over all 200 (n=149 fits / 51 overflows):

| method | fits | overflows |
|---|---|---|
| heuristic | **-0.055 (-1.9 sigma)** | +0.177 (3.2 sigma) |
| similarity | -0.027 (-1.0 sigma) | +0.118 (1.8 sigma) |

Two claims come out of this, and they are the strongest the project has:
1. **The learned policy is harmless when compression is unnecessary; the fixed
   rules are not.** On questions whose context already fits, `heuristic_prune`
   costs 5.5pp F1 at -1.9 sigma while the policy is exactly neutral. It also keeps
   ~6x more context on overflow questions (16,752 vs 2,804 tokens) -- i.e. it
   modulates by need, which is the entire point of query-adaptive compression.
2. **26% of questions (51/200) physically cannot be answered uncompressed** --
   `T_o` reaches 159,344 tokens, with up to 79% discarded. Their mean F1 is 0.409
   vs 0.632 for those that fit, and it is NOT retrieval (recall is *higher* on
   them, 0.951 vs 0.909). For those queries compression is a precondition for using
   the evidence at all, not an optimization.

**Remaining headroom**: if the 51 still-truncated questions scored like the ones
that fit, overall baseline F1 would be 0.632 rather than 0.575.

### `shorten_answer`: diagnosed, mostly not worth fixing

The 4/200 losses share one pattern -- when the raw answer leads with a negation
("Naomi Campbell did not appear... instead, Rosie O'Donnell..."), the shortener
collapses the whole thing to "No", because the prompt offers "yes-or-no" as an
output type. A targeted prompt fix (only answer Yes/No when the QUESTION is a
yes/no question) was tested on all four: **1 recovered, 12/12 regression check
clean.** The fix is applied because it is free, but the other three are not
extraction bugs -- the generator itself produced hedged or wrong raw answers
("Neither dog breed is specifically known..."). At ~0.5pp this does **not** justify
re-running the pipeline; it will take effect on the next full run. Same
diminishing-returns pattern as the entity-resolution prompt work.
