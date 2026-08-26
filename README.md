# GraphRAG Knowledge Graph + Retrieval

**Stage 1** builds a knowledge graph from a subset of the HotpotQA dataset
using a local, open-source LLM (Qwen2.5-7B-Instruct via Ollama) to extract
entities and relationships from each context passage, resolve duplicate/
coreferent entities, and render it as a **live view that updates in the
browser while extraction is still running**.

**Stage 2** embeds every graph node with BGE-M3 and retrieves a query-specific
candidate subgraph $G_q$ via top-k semantic seeds + k-hop graph expansion —
validated at 97.5% average recall against HotpotQA's own `supporting_facts`
on the current subset.

**Stage 3** (methodology Section III-C) generates the uncompressed baseline
answer $A_o$ for each question from $G_q$ via Qwen2.5-7B-Instruct, and
records the context token count $T_o$ plus EM/F1 against ground truth —
the reference point later compression stages are compared against.

**Stage 4** (methodology Section III-D/III-E) is the actual research
contribution: a GRPO-trained policy that learns to prune $G_q$ down to a
compressed $G_c$ per query, trading context size against answer quality. The
training loop (`train_compression_policy.py`) and its own inference/eval
script (`generate_compressed_answers.py`) are built and trained end-to-end on
the current ~20-question subset — mechanically sound (a reward-shaping
exploit that once collapsed it to "remove everything" is fixed, see the
Stage 4 section below) but not yet trained on enough data to generalize.

**Stage 6** (methodology Section III-F) evaluates the trained policy two
ways: against the mandatory non-adaptive baselines (fixed similarity-
threshold pruning and fixed hop-cutoff pruning, `generate_pruning_baselines.py`)
and as node-level KEEP/REMOVE classification against HotpotQA's own
`supporting_facts` (`evaluate_policy_classification.py` — confusion matrix,
ROC/AUC). `generate_report.py` consolidates everything (both stages' metrics,
plots, and a per-question tabulation table) into a self-contained
`result/report.html`. Community detection is mentioned in `main.tex`'s
Related Work only as part of prior work (Edge et al.'s GraphRAG) — it is not
part of this paper's own methodology and isn't implemented here.

## Project structure

```
code/
├── README.md                      # this file
├── CLAUDE.md                      # architecture/gotcha notes for Claude Code
├── .gitignore
├── Dataset/
│   └── formatted_output.json      # source HotpotQA-style dataset (must already exist)
├── result/                        # consolidated results package (generate_report.py output)
│   ├── report.html                # self-contained HTML report: KPIs, confusion matrix, ROC,
│   │                              #   comparison heatmap, ablation, per-question tabulation
│   ├── images/                    # confusion_matrix.png, roc_curve.png, comparison_heatmap.png, ablation_chart.png
│   └── data/                      # copies of the underlying JSON results, self-contained
└── kg/
    ├── select_subset.py           # step 1: sample questions, dedupe passages
    ├── extract_kg.py              # step 2: LLM entity/relation extraction (resumable)
    ├── graph_lib.py                # shared graph-assembly logic (used by both step 3 and step 4)
    ├── entity_resolution.py       # step 3 helper: LLM-verified entity de-duplication
    ├── build_graph.py             # step 3: assemble the final NetworkX knowledge graph
    ├── live_server.py             # step 4: live-updating graph visualization
    ├── embed_nodes.py             # stage 2, step 1: BGE-M3 node embeddings
    ├── retrieve.py                # stage 2, step 2: query -> candidate subgraph G_q
    ├── evaluate_retrieval.py      # stage 2, step 3: recall vs. supporting_facts
    ├── generate_baseline_answers.py  # stage 3: uncompressed answer A_o + T_o + EM/F1
    ├── compression_policy.py      # stage 4: state features + per-node MLP policy network
    ├── train_compression_policy.py   # stage 4: GRPO training loop
    ├── generate_compressed_answers.py  # stage 4: run a trained policy, compare vs. Stage 3 baseline
    ├── generate_pruning_baselines.py   # stage 6: non-adaptive similarity/heuristic pruning baselines
    ├── evaluate_policy_classification.py  # stage 6: node-level confusion matrix / ROC / AUC
    ├── generate_report.py         # stage 6: consolidates everything into ../result/
    ├── ollama_models/             # local Ollama model storage (moved off C:, see below)
    └── output/                    # generated files (created by the scripts)
        ├── subset_questions.json
        ├── subset_passages.json
        ├── extractions.json               # raw per-passage extractions (resume checkpoint)
        ├── extraction.log                 # timestamped per-passage extraction log
        ├── entity_resolution_cache.json   # cached LLM merge/no-merge verdicts (resumable)
        ├── knowledge_graph.json           # final merged graph, NetworkX node-link format
        ├── node_embeddings.npy            # BGE-M3 dense embeddings, one row per graph node
        ├── node_embedding_ids.json        # node ids, same order as node_embeddings.npy rows
        ├── retrieval_eval.json            # per-question retrieval recall results
        ├── baseline_answers.json          # per-question A_o, T_o, EM/F1, and full context used
        ├── compression_policy.pt          # trained policy weights + the args it was trained with
        ├── compression_training_log.json  # per-question-per-step training log (rewards, loss, entropy)
        ├── compressed_answers.json        # per-question A_c, T_c, CR, Delta_EM/Delta_F1 vs. baseline
        ├── pruning_baselines.json         # per-question similarity/heuristic pruning results
        ├── policy_classification_eval.json  # node-level accuracy/precision/recall/F1/AUC + confusion matrix
        ├── confusion_matrix.png           # node-level KEEP/REMOVE confusion matrix heatmap
        └── roc_curve.png                  # node-level KEEP-decision ROC curve
```

## Prerequisites

- Python 3.10+ with `pip`
- ~5 GB free disk space (for the model)
- A GPU with 8 GB+ VRAM is recommended (used: RTX 3070 Laptop, 8 GB). CPU-only
  works but extraction will be much slower.
- `Dataset/formatted_output.json` already present (a HotpotQA-formatted JSON:
  a list of records with `_id`, `question`, `answer`, `context`,
  `supporting_facts`, `type`, `level`).

## Setup from scratch

### 1. Install Python dependencies

```bash
pip install networkx requests pyvis sentence-transformers
```

(`pyvis` isn't used for static HTML generation here — `live_server.py` just
reuses its bundled vis-network JS/CSS assets so the live view works fully
offline. `sentence-transformers` is Stage 2 only — it downloads the BGE-M3
model, ~2.2 GB, the first time `embed_nodes.py` or `retrieve.py` runs. It
also pulls in `torch`, which Stage 4's policy network uses directly — no
separate install needed. A CUDA-enabled `torch` build is recommended if you
have a GPU (see Troubleshooting) but not required.)

### 2. Install Ollama (runs the local LLM)

**Windows:**
```bash
winget install --id Ollama.Ollama -e --silent --accept-package-agreements --accept-source-agreements
```

**macOS / Linux:** download from [ollama.com/download](https://ollama.com/download).

### 3. (Optional) Store models inside the project instead of the default user folder

By default Ollama stores models under `~/.ollama/models` (on Windows,
`C:\Users\<you>\.ollama\models`). To keep everything self-contained in this
project (e.g. `kg/ollama_models`), set the `OLLAMA_MODELS` environment
variable to that path **before** pulling the model:

```powershell
[System.Environment]::SetEnvironmentVariable("OLLAMA_MODELS", "D:\path\to\code\kg\ollama_models", "User")
```

Restart any open terminal (and the Ollama server, if already running) after
setting this so it picks up the new variable. Skip this step to just use the
default location. (This directory is already in `.gitignore` — it's several
GB of re-downloadable model weights, not something to commit.)

### 4. Start the Ollama server

The installer usually registers Ollama to start automatically. If `ollama list`
returns a connection error, start it manually:

```bash
ollama serve
```
(leave this running in its own terminal/background process — **the server
does not survive closing that terminal**; restart it at the beginning of
every work session before running `extract_kg.py` or `build_graph.py`)

### 5. Pull the model

```bash
ollama pull qwen2.5:7b-instruct
```

This downloads ~4.7 GB. Verify it worked:

```bash
ollama list
```

You should see `qwen2.5:7b-instruct` listed. (Smaller/faster alternatives
`qwen2.5:3b-instruct` and `qwen2.5:1.5b-instruct` are also supported — see
Troubleshooting.)

## Stage 1: Running the KG construction pipeline

Run these scripts **in order** from the `kg/` directory.

### Step 1 — Sample questions & collect unique passages

```bash
python select_subset.py --n 20 --seed 42 --max-passages 50
```

- `--n`: number of HotpotQA questions to sample (default 20). Each question
  has ~10 context passages, and passages are deduplicated by title, so 20
  questions typically yields ~150–200 unique passages before capping.
- `--seed`: random seed, for reproducible sampling.
- `--max-passages`: caps the number of unique passages kept (randomly
  trimmed down to this count). Use this to control how long Step 2 takes —
  e.g. `50` for a quick run, omit it to keep everything found.

Outputs:
- `output/subset_questions.json` — the sampled questions (with answers and
  `supporting_facts`, kept for later evaluation)
- `output/subset_passages.json` — the unique passages to extract from

### Step 2 — Extract entities & relations with the local LLM

```bash
python extract_kg.py
```

For every passage in `subset_passages.json`, this prompts
`qwen2.5:7b-instruct` (via `http://localhost:11434`, Ollama's default API
port) to return strict JSON: a list of entities (name/type/description) and
relations (source/target/relation/description), grounded only in that
passage's text. The prompt specifically calls out numbers/counts/quantities
("contains three species", "population of 729") as easy to skip but often
exactly what a downstream question asks about — found via a real miss where
a species count present in the source text wasn't extracted as a fact.

**Resumable**: every passage is saved to `output/extractions.json` and
logged to `output/extraction.log` the moment it finishes. Stop it anytime
with **Ctrl+C** — nothing is lost, and rerunning `extract_kg.py` picks up
exactly where you left off instead of starting over.

**Live display**: a scrollback line per finished passage (title, entity/
relation counts, and that passage's actual time — not an average), with a
progress bar pinned at the bottom showing overall percentage and an ETA
(estimated from the last 10 passages' average speed).

**Final statistics**: printed automatically once every passage is done (or
on interrupt) — this-run timing (fastest/slowest/mean passage) plus all-time
totals across everything in `extractions.json` (entity/relation counts,
per-passage averages, empty/failed count).

Output:
- `output/extractions.json` — raw per-passage entity/relation extractions
- `output/extraction.log` — timestamped append-only log of every passage processed

Expect roughly a few seconds per passage on an 8 GB GPU (e.g. ~50 passages
takes several minutes; 200 passages takes on the order of 15–30 minutes).
Use `--max-passages` in step 1 to control the run length, or just stop and
resume across multiple sessions with Ctrl+C.

### Step 3 — Build the knowledge graph (with entity resolution)

```bash
python build_graph.py
```

Merges all extracted triples into a single graph (via the shared logic in
`graph_lib.py`), then runs a second, stronger de-duplication pass
(`entity_resolution.py`) before saving. Every node/edge keeps a record of
which source passage it came from (`sources` on nodes, `source_passage` on
edges — needed later to check retrieval against `supporting_facts`).
Malformed entries occasionally returned by the model are skipped and counted
rather than crashing the run.

**Why two merge passes?** Simple name-normalization alone gets fooled both
ways: it misses real aliases with different wording (`"Ed Wood"` vs
`"Edward D. Wood Jr."`), and it wrongly fuses unrelated entities that happen
to share an exact title (several real, unrelated works in this dataset are
all literally titled `"Black Book"`). `entity_resolution.py` fixes both,
using Ollama for verification:
- **Free, no LLM needed**: pairs already linked by an alias-indicating
  relation extracted from the source text itself (e.g. `"was known as"`,
  `"formerly named"`) are merged directly — it's already a stated fact.
- **LLM-verified**: same-name entities from different passages, and
  fuzzy-matched candidates (same type, sharing a distinctive name token),
  are each checked with one LLM call asking it to classify the relationship
  (`identical` / `part_of` / `different_individual` / `related_work` /
  `coincidental_same_name` / `unclear`) — only `identical` merges. This
  categorical approach (rather than a plain yes/no) is deliberate: it
  correctly separates cases like a legislature from one of its chambers, or
  two different films that happen to share a title.
- **Cached and resumable**: every verification decision is saved to
  `output/entity_resolution_cache.json` immediately after each LLM call, so
  interrupting and rerunning `build_graph.py` never re-pays for a pair
  already checked. Delete this file to force a full fresh re-verification
  (e.g. after a prompt/logic change to `entity_resolution.py`).
- Skip this pass entirely with `python build_graph.py --no-resolve` for a
  faster iteration cycle (falls back to name-normalization merging only).

Output:
- `output/knowledge_graph.json` — full merged graph, NetworkX node-link
  format. This is the artifact later phases (retrieval, RL compression)
  would build on — a one-shot snapshot, run it again anytime after more
  extraction to refresh it.
- Console summary: node/edge counts, average degree, isolated nodes,
  connected components, and entity-resolution stats (candidate pairs
  checked, cache hits vs. new LLM calls, groups merged).

### Step 4 — Visualize the graph (live view)

```bash
python live_server.py
```

Starts a small local web server at **http://localhost:8765**. Open that URL
in a browser and leave it open — it polls every 3 seconds, rebuilding the
graph fresh from `extractions.json` each time (independent of
`build_graph.py`/`knowledge_graph.json` — no need to run step 3 first or
refresh anything manually), and upserts new entities/relations into the
running visualization as they're extracted. Existing node positions are
preserved, only new nodes animate in via physics.

Run this in one terminal and `extract_kg.py` in another to watch the graph
grow in real time. The status bar shows passages extracted / total,
entity/relation counts, and a progress bar; it also works fine as a
read-only viewer when extraction isn't currently running. Fully offline —
vis-network's JS/CSS is served locally from `pyvis`'s bundled copy.

Note: for responsiveness under frequent polling, the live view intentionally
uses only the cheap name-normalization merge (no `entity_resolution.py`
LLM verification) — it's a quick approximate look, not the authoritative
graph. `knowledge_graph.json` from Step 3 is the properly-resolved artifact.

## Stage 2: Query embedding & GraphRAG retrieval

Once `knowledge_graph.json` exists (Stage 1, Step 3), these turn it into
something queryable. No Ollama/GPU needed — BGE-M3 runs fine on CPU for a
graph this size (well under a minute for ~1,500 nodes).

### Step 1 — Embed every graph node

```bash
python embed_nodes.py
```

Embeds each node's name + up to 3 descriptions with `BAAI/bge-m3`
(1024-dim, L2-normalized so dot product == cosine similarity — methodology
Eq. 1). Runs on CPU by default to avoid competing with Ollama for VRAM; pass
`--device cuda` if Ollama isn't using the GPU at the time.

**Re-run this whenever `knowledge_graph.json` changes** (e.g. after
extracting more passages or rerunning `build_graph.py`) — embeddings aren't
regenerated automatically.

Output: `output/node_embeddings.npy` + `output/node_embedding_ids.json`

### Step 2 — Retrieve a candidate subgraph for a query

```bash
python retrieve.py "Were Scott Derrickson and Ed Wood of the same nationality?" --top-k 5 --hops 2
```

Embeds the query, finds the `--top-k` most similar nodes by cosine
similarity as seeds, then expands `--hops` hops along graph edges
(undirected) to build $G_q$. Prints the seeds with their similarity scores,
then every entity and relation pulled into $G_q$, with each relation's
source passage.

**Why expand hops instead of just returning the top-k similar nodes?**
Multi-hop questions often need a fact that isn't semantically similar to the
question at all — it's only reachable by walking the graph from a shared
entity (e.g. a founding date is nowhere near "university" in embedding
space, but is one hop away from the university's node). Pure top-k
similarity would miss it entirely; hop expansion is what actually uses the
graph structure GraphRAG is for.

### Step 3 — Evaluate retrieval quality

```bash
python evaluate_retrieval.py --top-k 5 --hops 2
```

Runs retrieval over every question in `subset_questions.json` and checks
whether $G_q$'s nodes cover the passages HotpotQA's own `supporting_facts`
say are actually needed — a direct, quantitative signal on retrieval
quality before building answer generation on top of it. No LLM calls, just
embedding + graph traversal, so it's fast.

Output: console summary (full/partial/zero recall counts, average recall)
and `output/retrieval_eval.json` with per-question detail (which passages
were needed vs. covered vs. missing).

## Stage 3: Baseline answer generation

```bash
python generate_baseline_answers.py --top-k 5 --hops 2
```

For each question: retrieves $G_q$ (Stage 2), serializes it into a context
(structured graph facts + the raw source-passage text for every passage any
node in $G_q$ came from), sends it to `qwen2.5:7b-instruct` alongside the
question, and records the answer $A_o$, context token count $T_o$ (via the
real Qwen2.5-7B-Instruct tokenizer — downloads a small tokenizer-only file
from Hugging Face the first time), and EM/F1 against the ground-truth
answer. No pruning or learned policy at this step — this is the uncompressed
baseline that later GRPO-compressed answers ($A_c$) get compared against.

**Two-step generation, not one.** This model reliably explains its reasoning
even when told not to — tested strict brevity instructions, few-shot format
examples, and a JSON-schema constraint, none stopped it from wrapping a
correct answer in a full sentence for "who"/"which" style questions (e.g.
producing `"John André was hanged as a spy by the Continental Army..."`
instead of just `"John André"`, which EM scores as a miss even though the
answer is right there). Fighting that in the main prompt didn't work, so
`generate_answer()` uses a simple prompt and a **second call**
(`shorten_answer()`) extracts the short direct answer from the first
response — a much easier task for the model than "answer tersely from
scratch," and it reliably works. Both the shortened `generated_answer` and
the original `raw_answer` are saved for comparison. This roughly doubles
generation time per question (one extra LLM call) but meaningfully improved
results: **EM 40.0% → 60.0%, F1 56.6% → 73.5%** on the current subset
(combined effect of this fix, the extraction prompt fix below, and the
entity-resolution fixes in Stage 1 — see `entity_resolution.py`'s notes in
[CLAUDE.md](CLAUDE.md) for the full story, including a case where fixing
duplicate near-identical nodes directly un-buried a fact a question needed).

Output:
- `output/baseline_answers.json` — per question: `generated_answer`,
  `raw_answer`, `ground_truth`, `em`, `f1`, `context`, `T_o`, `gq_nodes`/`gq_edges`
- Console summary: EM%, F1%, average $T_o$, total time

## Stage 4: GRPO-based query-adaptive graph compression

The research contribution: a policy that learns to prune $G_q$ down to a
compressed $G_c$ per query (Section III-D), trading context size against
answer quality via a trained reward (Section III-E) rather than a fixed
heuristic. **Trained end-to-end, not yet trained on enough data to
generalize** — the current subset is only ~20 questions/3 epochs, which is
enough to validate the whole mechanism (including finding and fixing a real
reward-shaping exploit, see Step 1 below) but not enough to learn real
node-level discrimination; see the note at the end of this section and the
Stage 6 evaluation further down for exactly how that shows up in the numbers.

### Step 1 — Train the policy

```bash
python train_compression_policy.py --group-size 4 --epochs 3
```

For each question: retrieves $G_q$, builds a per-node state vector (node
embedding + query embedding + query-node cosine similarity + graph-context
features — degree, hop-distance from the nearest retrieval seed, is-a-seed
flag), samples `--group-size` KEEP/REMOVE trajectories from the policy
(`compression_policy.py`'s per-node MLP), generates a real compressed answer
$A_c$ for each (reusing Stage 3's two-step generate+shorten), and computes
the joint reward
$R = Q_c + \lambda \cdot CR - \gamma \cdot P$ (`--lambda-cr`/`--gamma-penalty`,
defaults 0.5/2.0) where $Q_c$ reuses Stage 3's F1 metric, $CR = 1 - T_c/T_o$,
and $P = \max(0, Q_o - Q_c)$. Trajectory rewards within a group are turned
into a GRPO advantage
$A_i = (R_i - \mu_R)/(\sigma_R + \epsilon)$ and the policy is updated via
policy-gradient (`-mean(advantage * log_prob)`, plus a small entropy bonus
— see below).

**Cost warning**: each trajectory costs ~2 real LLM calls. `--n-questions
(default: all) x --group-size x --epochs x 2` calls, each several seconds —
the default `--group-size 4 --epochs 3` over the current ~19-question subset
is roughly 45-55 minutes. Use `--n-questions 2 --group-size 2 --epochs 1`
for a ~1 minute mechanism smoke test before committing to a real run.

**Three real bugs were found and fixed here** — two initialization bugs and
one reward-shaping exploit, all of which mattered far more than the RL logic
itself and are worth knowing about before touching this file again:
1. The state vector concatenates raw L2-normalized embeddings (~0.03 std per
   dimension) directly with scalar graph-context features on very different
   scales (similarity in [-1,1], degree/hop in [0,1]). Without normalizing
   first, a freshly-initialized linear layer barely differentiates between
   nodes at all (measured: logit std ~0.001 across 16 different real nodes).
   Fixed with `nn.LayerNorm` on the input inside `CompressionPolicy`.
2. Even after that fix, the final layer's randomly-initialized bias can
   still land the *entire* policy on the "remove" side of the 0.5 threshold
   for every node in every graph, purely by chance — and since the training
   script uses a fixed seed, this reproduced identically every rerun and
   survived a couple of real gradient steps untouched, producing a
   degenerate "remove everything" policy that looked like RL collapse but
   wasn't. Fixed by explicitly initializing that bias toward KEEP
   (`sigmoid(1.0) ≈ 0.73`) — an untrained policy now defaults to
   keeping essentially everything (i.e. behaving like the Stage 3 baseline)
   rather than discarding all context and getting zero learnable signal.
   If you ever see a trained policy collapse to all-KEEP or all-REMOVE
   again, check `output/compression_training_log.json`'s `mean_entropy`
   column first — it should stay well above ~0 throughout training.
3. A **reward-shaping exploit** reproduced the exact same "collapse to
   all-REMOVE" symptom as bug 2 above, even with both init fixes in place —
   only found by rerunning training and reading the per-step `mean_entropy`
   again (it collapsed to ~0 by epoch 3, a genuinely converged policy, not
   noise). `compute_reward()`'s compression-ratio bonus was originally
   unconditional, and two properties specific to this LLM+dataset made
   "empty the graph" a dominant strategy: the answer prompt tells the model
   to guess rather than refuse, and Qwen2.5-7B-Instruct often answers
   HotpotQA's Wikipedia-famous entities correctly from parametric knowledge
   alone even with zero graph context — so an empty graph could score
   $Q_c \approx Q_o$ (no penalty) *and* still collect the full CR bonus,
   beating keeping full context and getting the same answer right. Fixed by
   gating the bonus on not losing quality: `cr_bonus = lambda_cr * CR if
   Q_c >= Q_o else 0.0`. After the fix, entropy no longer collapses and
   compression behavior varies per-question instead of being uniformly zero
   nodes — see Stage 6 below for what that looks like in practice.

Output:
- `output/compression_policy.pt` — trained weights + the args used to train them
- `output/compression_training_log.json` — per-question-per-step rewards,
  mean compression ratio, entropy, loss

### Step 2 — Evaluate the trained policy

```bash
python generate_compressed_answers.py --top-k 5 --hops 2
```

Runs the trained policy's **deterministic** decisions (KEEP if
$\sigma(\text{logit}) \geq 0.5$ — no sampling, this is inference not
exploration) across questions, generates $A_c$, and reports $\Delta EM$/
$\Delta F1$ against Stage 3's baseline plus the achieved compression ratio
and token reduction. Per the methodology's sign convention, $\Delta EM/F1 > 0$
means the baseline outperformed the compressed answer; $\leq 0$ means
compression preserved or improved quality while shrinking context.

Output: `output/compressed_answers.json` (per-question $A_o$/$A_c$, $T_o$/$T_c$,
$CR$, $EM_o$/$EM_c$, $F1_o$/$F1_c$, $\Delta EM$/$\Delta F1$) plus a console summary.

**Current numbers, and why they're not the paper's headline result yet**: a
full run (`--group-size 4 --epochs 3` over all ~20 questions) gives
EM 60.0%→30.0%, F1 73.5%→37.0% at 52.9% average compression — i.e. the
policy compresses meaningfully but still costs real quality. Stage 6's
node-level classification (below) shows why: 51.7% node-decision accuracy
and 0.521 AUC, barely above chance at picking *which* nodes actually matter.
That's consistent with 20 questions/3 epochs being enough to validate the
mechanism (see Step 1's bug 3 above — it no longer collapses to emptying
every graph) but not enough signal to learn real discrimination; there's
also no train/val split at this data scale to check generalization even if
it had learned something. Treat these numbers as a mechanism-correctness
result, not a paper-reportable finding — scale up the dataset (more subset
questions) before trusting $\Delta EM$/$\Delta F1$ as a real comparison.

## Stage 6: Non-adaptive baselines, classification metrics & the results report

Three scripts, run after Stage 4 Step 2, none of which call Ollama (fast —
seconds to low single-digit minutes even on a bigger dataset).

### Step 1 — Non-adaptive pruning baselines

```bash
python generate_pruning_baselines.py --keep-ratio 0.5 --heuristic-hops 1
```

The methodology (Section III-F) mandates comparing the learned policy
against non-adaptive rules. Two are implemented, both over the same
retrieved $G_q$ and same baseline as Stage 4:
- **`similarity_prune`**: keep the top `--keep-ratio` fraction of $G_q$'s
  nodes ranked by query-node cosine similarity — a fixed threshold, no
  learning.
- **`heuristic_prune`**: keep only nodes within `--heuristic-hops` hops of a
  retrieval seed within $G_q$ — a fixed structural rule, no query-adaptivity
  at all.

This does call Ollama (2 methods × 2 LLM calls per question), so it's not
instant — expect several minutes on ~20 questions, scaling with dataset size.

Output: `output/pruning_baselines.json` (per-question $A$/$T$/$CR$/EM/F1 for
both methods) plus a console comparison table.

### Step 2 — Node-level classification metrics

```bash
python evaluate_policy_classification.py
```

Reframes each KEEP/REMOVE decision as binary classification: ground truth
"relevant" is whether a node's source passage is one of HotpotQA's own
`supporting_facts` for that question (the same ground truth
`evaluate_retrieval.py` uses for Stage 2). The trained policy's deterministic
KEEP probability is the classifier score. No LLM calls — pure local
inference over the already-trained policy.

Output: `output/policy_classification_eval.json` (accuracy, precision,
recall, classification F1, confusion matrix, ROC points, AUC) plus
`output/confusion_matrix.png` and `output/roc_curve.png`.

### Step 3 — Build the consolidated results report

```bash
python generate_report.py
```

Aggregates Stage 4's, Step 1's, and Step 2's JSON outputs into a comparison
heatmap, a per-question tabulation table, and a self-contained
`result/report.html` (fonts/images embedded, opens standalone offline) —
also writes copies of all the underlying images/JSON to `result/images/` and
`result/data/`. Safe to rerun anytime after any upstream script changes;
overwrites `result/` fresh each time.

Note: this script's ablation-study section (reward-gating fix, before vs.
after) is a **fixed historical reference** from the original 20-question
pilot run that first found the bug in Step 1's training bug 3 above — a
rerun can't reproduce the "before" row since that bug is now fixed in the
code, so it's kept as a labeled constant rather than silently implying it
came from the current dataset.

## Quick start (all steps)

```bash
cd kg
python select_subset.py --n 20 --seed 42 --max-passages 50
python live_server.py            # in one terminal, leave running -- open http://localhost:8765
python extract_kg.py             # in another terminal
python build_graph.py            # whenever you want a resolved knowledge_graph.json snapshot
python embed_nodes.py            # stage 2: embed the graph
python retrieve.py "your question here"
python evaluate_retrieval.py     # check retrieval recall across the whole subset
python generate_baseline_answers.py   # stage 3: uncompressed baseline A_o, T_o, EM/F1
python train_compression_policy.py --group-size 4 --epochs 3   # stage 4: GRPO training (~25-30 min)
python generate_compressed_answers.py   # stage 4: A_c, T_c, Delta_EM/Delta_F1 vs. baseline
python generate_pruning_baselines.py    # stage 6: non-adaptive similarity/heuristic baselines
python evaluate_policy_classification.py  # stage 6: confusion matrix / ROC / AUC (fast, no LLM calls)
python generate_report.py               # stage 6: consolidates everything into ../result/
```

## Troubleshooting

- **`ConnectionError` / extraction all fails**: Ollama isn't running (it does
  not survive closing the terminal that started it). Run `ollama serve` and
  check `ollama list` shows the model. If `extract_kg.py` already wrote empty
  entries for passages that failed this way, remove those entries from
  `output/extractions.json` before rerunning so they get retried (an
  all-`[]` entities+relations entry is the tell).
- **Out of memory / very slow**: switch to a smaller model in
  `extract_kg.py` (change `MODEL = "qwen2.5:7b-instruct"` to
  `"qwen2.5:3b-instruct"` or `"qwen2.5:1.5b-instruct"` and `ollama pull` it first).
- **Want a bigger/smaller test**: adjust `--n` and `--max-passages` in step
  1. Start small (e.g. `--max-passages 10`) to sanity-check the whole
  pipeline runs before doing a larger batch.
- **Stopping partway through**: Ctrl+C in the `extract_kg.py` or
  `build_graph.py` terminal at any time — the last completed passage/pair is
  always already saved, so it's always safe to resume later with the same
  command.
- **Malformed JSON / non-dict entries from the model**: `extract_kg.py`
  retries twice per passage and logs a warning if it still fails;
  `graph_lib.py` / `live_server.py` skip any malformed entity/relation
  entries and report the count rather than crashing.
- **Entity resolution seems slow**: it's one LLM call per candidate pair
  (typically 1,000–2,000 pairs on a full subset, a few seconds each — expect
  30–60 minutes on a fresh cache). Use `--no-resolve` for a quick iteration,
  or just let it run in the background; it's checkpointed so it's safe to
  interrupt.
- **Suspect a wrong merge in `knowledge_graph.json`**: check
  `output/entity_resolution_cache.json` for the pair (keyed by
  `normalized_id_a||normalized_id_b`) — each entry records the `category`
  the model assigned and its `reason`. Delete a specific bad entry (or the
  whole file, for a full re-verification) and rerun `build_graph.py`.
- **Live view shows "server unreachable"**: `live_server.py` isn't running,
  or was stopped — restart it and refresh the page.
- **`embed_nodes.py`/`retrieve.py` slow to start / HF Hub warning**: the
  first run downloads `BAAI/bge-m3` (~2.2 GB) from Hugging Face; subsequent
  runs load from the local cache and start in seconds. The "unauthenticated
  requests" warning is harmless — set `HF_TOKEN` only if you're hitting rate
  limits.
- **`embed_nodes.py`/`retrieve.py`/`evaluate_retrieval.py` closes instantly
  with no output or error (Windows)**: this is a native crash
  (`STATUS_ACCESS_VIOLATION`, exit code `-1073741819`) below the Python
  level, so there's nothing for Python to catch or print. Root cause found:
  an unused `datasets` package (which nothing in this project's actual
  dependencies requires) drags in `pyarrow`, and `pyarrow`'s `arrow.dll`
  conflicts with `torch`'s CUDA DLLs when both end up loaded in the same
  process — intermittently, since it depends on load timing. Fix:
  `pip uninstall datasets pyarrow`. Confirmed via Windows' own crash report
  (Event Viewer → Application log → "Application Error", faulting module
  `arrow.dll`) if you ever need to diagnose a similar crash from a different
  cause. Verify your environment isn't affected: `pip show datasets` should
  say "Required-by:" is empty (or the package isn't installed at all) before
  trusting Stage 2 runs.
- **Retrieval seems to miss an obviously-relevant entity**: check
  `output/retrieval_eval.json` for that question's `missing` list, then
  confirm the entity actually exists in `knowledge_graph.json` and is within
  `--hops` of a seed — a low top-1 similarity seed or a fragmented (not yet
  fully entity-resolved) graph can both push a relevant node just outside
  the expansion radius. Try increasing `--top-k` and/or `--hops` first.
- **Retrieval results look stale**: `retrieve.py`/`evaluate_retrieval.py`
  read `node_embeddings.npy`, not `knowledge_graph.json` directly — rerun
  `embed_nodes.py` after any change to the graph.
- **A fix to `extract_kg.py` or a manual patch to `extractions.json` doesn't
  show up in retrieval/baseline results**: the full chain is
  `extractions.json` → `build_graph.py` → `knowledge_graph.json` →
  `embed_nodes.py` → `node_embeddings.npy` → `retrieve.py`/
  `generate_baseline_answers.py`. Nothing downstream regenerates
  automatically — a change at any point needs everything after it rerun.
  This bit us once: a real extraction fix was verified in
  `extractions.json`, but a baseline run "coincidentally" got that same
  question right anyway (LLM inference isn't perfectly deterministic even
  at temperature 0) before `build_graph.py` had actually been rerun to
  incorporate the fix — don't trust an isolated pass as confirmation without
  checking the graph itself changed.

## What's next (not in this README)

- Scaling up the dataset (more than ~20 questions) — needed before Stage 4
  training can produce a genuinely generalizing policy that beats the
  non-adaptive baselines (currently it doesn't — see Stage 6), not just a
  mechanism smoke test
- A larger Stage 4 training run and honest $\Delta EM$/$\Delta F1$ /
  node-classification results from it, ideally with a train/val split once
  the dataset is large enough to support one
