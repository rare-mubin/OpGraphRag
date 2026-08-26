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

All active code lives under `kg/`. `Dataset/formatted_output.json` is the cleaned, validated dataset
in use; `Dataset/hotpot_dev_distractor_v1.json` is the original unformatted source and isn't read by
any script — don't confuse the two.

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
python build_graph.py                                          # merge into final graph + entity resolution
python live_server.py                                          # live graph view at http://localhost:8765
python embed_nodes.py                                          # Stage 2: BGE-M3 embed every node (CPU by default)
python retrieve.py "a question" --top-k 5 --hops 2             # Stage 2: query -> candidate subgraph G_q
python evaluate_retrieval.py                                   # Stage 2: recall vs. supporting_facts, all subset questions
python generate_baseline_answers.py                             # Stage 3: uncompressed A_o, T_o, EM/F1 (needs Ollama)
python train_compression_policy.py --group-size 4 --epochs 3   # Stage 4: GRPO training (~45-55 min, needs Ollama)
python generate_compressed_answers.py                           # Stage 4: A_c, T_c, Delta_EM/Delta_F1 vs. baseline
```

Dependencies: `pip install networkx requests pyvis sentence-transformers` (no requirements.txt/pyproject.toml exists).

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

**Windows native-crash gotcha:** if `embed_nodes.py`/`retrieve.py`/`evaluate_retrieval.py` exit
instantly with no output/traceback (exit code `-1073741819` = `STATUS_ACCESS_VIOLATION`), check for an
orphan `datasets` package first (`pip show datasets` — if `Required-by:` is empty, nothing in this
project needs it). `datasets` pulls in `pyarrow`, and `pyarrow`'s `arrow.dll` intermittently crashes
when loaded into the same process as `torch`'s CUDA DLLs. `pip uninstall datasets pyarrow` fixes it.
Diagnosed via Windows Event Viewer → Application log → "Application Error" (shows the faulting module
name directly) — worth checking there first for any similar unexplained silent-exit crash rather than
guessing at OpenMP/MKL conflicts, which looked plausible but was a red herring here.
