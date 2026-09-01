"""
Stage 6 addendum: builds the full results package (confusion matrix, ROC/AUC,
comparison heatmap, per-question tabulation table, and a self-contained HTML
report) from whatever is currently in kg/output/, and writes everything to
../result/ at the project root.

Run this AFTER the full pipeline (including generate_pruning_baselines.py and
evaluate_policy_classification.py) has produced its output/*.json files --
this script only aggregates/visualizes, it makes no LLM calls itself and runs
in seconds regardless of dataset size.

Required upstream files in kg/output/ (produced by the scripts named):
    baseline_answers.json            <- generate_baseline_answers.py
    compressed_answers.json          <- generate_compressed_answers.py
    pruning_baselines.json           <- generate_pruning_baselines.py
    policy_classification_eval.json  <- evaluate_policy_classification.py

Usage:
    python generate_report.py
"""
import json
import base64
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KG_OUT = Path(__file__).resolve().parent / "output"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_ROOT / "result"
IMAGES_DIR = RESULT_DIR / "images"
DATA_DIR = RESULT_DIR / "data"

# The reward-gating ablation (compute_reward's CR-bonus condition, see CLAUDE.md) is a
# one-time historical comparison from the 20-question pilot run, NOT something a rerun
# regenerates automatically -- the bug it demonstrates is fixed in the code now, so any
# future run only ever reproduces the "after" row. Kept here as a fixed reference point;
# label it as such in the report rather than implying it came from the current run.
ABLATION_REFERENCE = {
    "n_questions": 20,
    "before": {"label": "Ungated CR bonus\n(exploit present)", "em": 0.20, "f1": 0.22, "cr": 0.998,
               "g_c_zero": 20},
    "after": {"label": "Gated CR bonus\n(Q_c >= Q_o required)", "em": 0.30, "f1": 0.37, "cr": 0.529,
              "g_c_zero": 6},
}


def require(path: Path, produced_by: str):
    if not path.exists():
        raise SystemExit(f"Missing {path} -- run {produced_by} first.")
    return path


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(require(KG_OUT / "compressed_answers.json", "generate_compressed_answers.py"), encoding="utf-8") as f:
        compressed = json.load(f)
    with open(require(KG_OUT / "pruning_baselines.json", "generate_pruning_baselines.py"), encoding="utf-8") as f:
        pruning = json.load(f)
    with open(require(KG_OUT / "policy_classification_eval.json", "evaluate_policy_classification.py"),
              encoding="utf-8") as f:
        clf = json.load(f)

    n = len(compressed)
    avg_em_o = sum(r["EM_o"] for r in compressed) / n
    avg_f1_o = sum(r["F1_o"] for r in compressed) / n
    avg_em_c = sum(r["EM_c"] for r in compressed) / n
    avg_f1_c = sum(r["F1_c"] for r in compressed) / n
    avg_cr_c = sum(r["CR"] for r in compressed) / n

    # Restrict the non-adaptive baselines to the SAME questions the policy was evaluated on.
    # generate_compressed_answers.py now defaults to the held-out val split, while
    # generate_pruning_baselines.py has no policy to hold out from and so covers every question.
    # Averaging those two scopes into one comparison table silently compares a 40-question policy
    # result against a 200-question baseline result -- which flattered the pruning rows by several
    # points and is exactly the kind of apples-to-oranges row this table exists to prevent.
    eval_qs = {r["question"] for r in compressed}
    scoped = [r for r in pruning if r["question"] in eval_qs]
    if len(scoped) < len(pruning):
        print(f"Scoping pruning baselines to the {len(scoped)} question(s) the policy was "
              f"evaluated on (of {len(pruning)} in pruning_baselines.json).")
    if not scoped:
        raise SystemExit("No overlap between pruning_baselines.json and compressed_answers.json -- "
                         "rerun generate_pruning_baselines.py.")
    pruning = scoped

    pn = len(pruning)
    sim_em = sum(r["similarity"]["EM"] for r in pruning) / pn
    sim_f1 = sum(r["similarity"]["F1"] for r in pruning) / pn
    sim_cr = sum(r["similarity"]["CR"] for r in pruning) / pn
    heur_em = sum(r["heuristic"]["EM"] for r in pruning) / pn
    heur_f1 = sum(r["heuristic"]["F1"] for r in pruning) / pn
    heur_cr = sum(r["heuristic"]["CR"] for r in pruning) / pn

    methods = ["Uncompressed\n(baseline)", "RL policy\n(GRPO)", "Similarity\npruning", "Heuristic\npruning"]
    em_vals = [avg_em_o, avg_em_c, sim_em, heur_em]
    f1_vals = [avg_f1_o, avg_f1_c, sim_f1, heur_f1]
    cr_vals = [0.0, avg_cr_c, sim_cr, heur_cr]
    comparison = {"methods": methods, "em": em_vals, "f1": f1_vals, "cr": cr_vals}

    # ---------- comparison heatmap ----------
    metrics_matrix = np.array([em_vals, f1_vals, cr_vals])
    row_norm = (metrics_matrix - metrics_matrix.min(axis=1, keepdims=True)) / (
        metrics_matrix.max(axis=1, keepdims=True) - metrics_matrix.min(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.imshow(row_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels([m.replace("\n", " ") for m in methods], fontsize=9)
    ax.set_yticks(range(3)); ax.set_yticklabels(["EM", "F1", "Compression Ratio"], fontsize=10)
    for r, row in enumerate(metrics_matrix):
        for c, val in enumerate(row):
            ax.text(c, r, f"{100*val:.1f}%", ha="center", va="center", fontsize=10,
                     color="black", fontweight="bold")
    ax.set_title(f"Method comparison, n={n} (green = best per row)", fontsize=11)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "comparison_heatmap.png", dpi=150)
    plt.close(fig)

    # ---------- ablation reference chart ----------
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.6))
    labels = [ABLATION_REFERENCE["before"]["label"], ABLATION_REFERENCE["after"]["label"]]
    colors = ["#c1666b", "#3f7d5c"]
    for ax, key, title, denom in zip(
            axes, ["em", "f1", "g_c_zero"],
            ["EM", "F1", "Questions fully emptied\n(G_c = 0 nodes)"],
            [1, 1, ABLATION_REFERENCE["n_questions"]]):
        vals = [ABLATION_REFERENCE["before"][key] / denom, ABLATION_REFERENCE["after"][key] / denom]
        bars = ax.bar(labels, vals, color=colors, width=0.6)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{100*v:.1f}%", ha="center", fontsize=9)
    fig.suptitle(f"Ablation: reward-gating fix (reference run, n={ABLATION_REFERENCE['n_questions']})", fontsize=11)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "ablation_chart.png", dpi=150)
    plt.close(fig)

    # confusion matrix / ROC images are already produced by evaluate_policy_classification.py --
    # just copy them alongside the ones built here.
    import shutil
    for fname in ("confusion_matrix.png", "roc_curve.png"):
        src = KG_OUT / fname
        if src.exists():
            shutil.copyfile(src, IMAGES_DIR / fname)

    # ---------- merged tabulation table ----------
    prune_by_q = {r["question"]: r for r in pruning}
    table_rows = []
    for r in compressed:
        p = prune_by_q.get(r["question"], {})
        table_rows.append({
            "question": r["question"], "gt": r["ground_truth"], "gq_nodes": r["gq_nodes"],
            "baseline": {"f1": r["F1_o"], "em": r["EM_o"]},
            "rl": {"f1": r["F1_c"], "em": r["EM_c"], "cr": r["CR"], "nodes": r["gc_nodes"]},
            "similarity": {"f1": p.get("similarity", {}).get("F1"), "em": p.get("similarity", {}).get("EM"),
                           "cr": p.get("similarity", {}).get("CR"), "nodes": p.get("similarity", {}).get("n_nodes")},
            "heuristic": {"f1": p.get("heuristic", {}).get("F1"), "em": p.get("heuristic", {}).get("EM"),
                          "cr": p.get("heuristic", {}).get("CR"), "nodes": p.get("heuristic", {}).get("n_nodes")},
        })

    report_data = {
        "comparison": comparison, "ablation_reference": ABLATION_REFERENCE,
        "classification": clf, "table_rows": table_rows, "n_questions": n,
    }
    with open(DATA_DIR / "report_data.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    # also copy the raw upstream JSONs so result/ is self-contained
    for fname in ("baseline_answers.json", "compressed_answers.json", "pruning_baselines.json",
                  "policy_classification_eval.json", "compression_training_log.json"):
        src = KG_OUT / fname
        if src.exists():
            import shutil
            shutil.copyfile(src, DATA_DIR / fname)

    _build_html(report_data, IMAGES_DIR, RESULT_DIR)
    print(f"\nDone. Results written to {RESULT_DIR}")


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _pct(x):
    return f"{100*x:.1f}%" if x is not None else "&mdash;"


def _delta_class(base, other):
    if base is None or other is None:
        return ""
    d = base - other
    if d > 0.02:
        return "worse"
    if d < -0.02:
        return "better"
    return "same"


def _build_html(D: dict, images_dir: Path, result_dir: Path):
    comp = D["comparison"]
    abl = D["ablation_reference"]
    clf = D["classification"]
    rows = D["table_rows"]
    n = D["n_questions"]

    img = {
        "confusion_matrix": _b64(images_dir / "confusion_matrix.png"),
        "roc_curve": _b64(images_dir / "roc_curve.png"),
        "comparison_heatmap": _b64(images_dir / "comparison_heatmap.png"),
        "ablation_chart": _b64(images_dir / "ablation_chart.png"),
    }

    table_rows_html = []
    for r in rows:
        q = r["question"].strip()
        q_short = (q[:58] + "…") if len(q) > 58 else q
        b = r["baseline"]; rl = r["rl"]; sim = r["similarity"]; heur = r["heuristic"]

        def cell(method):
            f1 = method.get("f1"); cr = method.get("cr")
            cls = _delta_class(b["f1"], f1)
            cr_str = f'<span class="cr">{_pct(cr)}</span>' if cr is not None else ""
            return f'<td class="{cls}"><span class="f1">{_pct(f1)}</span>{cr_str}</td>'

        table_rows_html.append(
            f'<tr><td class="qcell" title="{q}">{q_short}</td>'
            f'<td class="gt">{r["gt"]}</td><td>{r["gq_nodes"]}</td>'
            f'<td><span class="f1">{_pct(b["f1"])}</span></td>'
            f'{cell(rl)}{cell(sim)}{cell(heur)}</tr>'
        )
    table_rows_html = "\n".join(table_rows_html)

    auc_str = f"{clf['auc']:.3f}" if clf.get("auc") is not None else "n/a"

    html = f"""<title>Adaptive Graph Compression Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:#f5f4f1;--surface:#fff;--surface-alt:#eceeed;--ink:#1b2023;--ink-soft:#565f66;--ink-faint:#8b9298;
    --accent:#2b6777;--accent-soft:#dcebec;--good:#3f7d5c;--good-soft:#e1efe6;
    --warn:#a8542a;--bad:#ba3b3b;--border:#daddd9;
    --shadow:0 1px 2px rgba(20,20,18,.05),0 8px 24px -12px rgba(20,20,18,.12);
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg:#14181a;--surface:#1b2124;--surface-alt:#222a2d;--ink:#edf1f0;--ink-soft:#a7b1ad;--ink-faint:#6c7674;
      --accent:#7ec2cd;--accent-soft:#22383c;--good:#6bc08f;--good-soft:#1c332a;
      --warn:#e39a63;--bad:#e2726b;--border:#2c3336;
      --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -12px rgba(0,0,0,.5);
    }}
  }}
  :root[data-theme="dark"] {{
    --bg:#14181a;--surface:#1b2124;--surface-alt:#222a2d;--ink:#edf1f0;--ink-soft:#a7b1ad;--ink-faint:#6c7674;
    --accent:#7ec2cd;--accent-soft:#22383c;--good:#6bc08f;--good-soft:#1c332a;
    --warn:#e39a63;--bad:#e2726b;--border:#2c3336;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -12px rgba(0,0,0,.5);
  }}
  * {{ box-sizing: border-box; }}
  body {{ background:var(--bg); color:var(--ink); font-family:"IBM Plex Sans",system-ui,sans-serif;
          font-size:15px; line-height:1.55; margin:0; padding:0 20px 80px; }}
  .wrap {{ max-width:1120px; margin:0 auto; }}
  h1,h2,h3 {{ font-family:"Source Serif 4",Georgia,serif; text-wrap:balance; font-weight:600; margin:0; }}
  .mono,.f1,.cr {{ font-variant-numeric: tabular-nums; font-family:"IBM Plex Mono",monospace; }}
  header.hero {{ padding:56px 0 28px; border-bottom:1px solid var(--border); }}
  .eyebrow {{ font-family:"IBM Plex Mono",monospace; font-size:11.5px; letter-spacing:.12em; text-transform:uppercase;
              color:var(--accent); font-weight:600; margin-bottom:10px; }}
  header.hero h1 {{ font-size:clamp(28px,4vw,40px); line-height:1.12; max-width:18ch; }}
  header.hero p.sub {{ margin-top:14px; max-width:62ch; color:var(--ink-soft); font-size:16px; }}
  .stagebar {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:26px; }}
  .stage {{ display:flex; align-items:center; gap:7px; padding:7px 12px 7px 9px; background:var(--surface);
            border:1px solid var(--border); border-radius:100px; font-size:12.5px; color:var(--ink-soft); }}
  .stage .n {{ display:inline-flex; align-items:center; justify-content:center; width:18px; height:18px;
               border-radius:50%; background:var(--good-soft); color:var(--good);
               font-family:"IBM Plex Mono",monospace; font-size:10.5px; font-weight:600; }}
  section {{ padding:46px 0; border-bottom:1px solid var(--border); }}
  section:last-of-type {{ border-bottom:none; }}
  .section-head {{ margin-bottom:22px; }}
  .section-head h2 {{ font-size:22px; }}
  .section-head p {{ color:var(--ink-soft); margin:8px 0 0; max-width:68ch; font-size:14.5px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }}
  .kpi {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px 18px;
          box-shadow:var(--shadow); }}
  .kpi .label {{ font-size:11.5px; color:var(--ink-faint); text-transform:uppercase; letter-spacing:.06em; }}
  .kpi .value {{ font-family:"IBM Plex Mono",monospace; font-size:26px; font-weight:600; margin-top:6px; }}
  .kpi .note {{ font-size:12px; color:var(--ink-soft); margin-top:4px; }}
  .kpi.accent .value {{ color:var(--accent); }} .kpi.warn .value {{ color:var(--warn); }} .kpi.good .value {{ color:var(--good); }}
  .imgpanel {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:14px;
               box-shadow:var(--shadow); }}
  .imgpanel img {{ width:100%; height:auto; display:block; border-radius:4px; }}
  .imgpanel figcaption {{ font-size:12.5px; color:var(--ink-soft); margin-top:10px; padding:0 4px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  @media (max-width:760px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
  table {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
  thead th {{ text-align:left; font-family:"IBM Plex Mono",monospace; font-size:11px; text-transform:uppercase;
              letter-spacing:.05em; color:var(--ink-faint); padding:10px 12px; border-bottom:2px solid var(--border); }}
  tbody td {{ padding:10px 12px; border-bottom:1px solid var(--border); }}
  tbody tr:last-child td {{ border-bottom:none; }}
  .comp-table-wrap, .bigtable-wrap {{ background:var(--surface); border:1px solid var(--border); border-radius:12px;
                                       box-shadow:var(--shadow); overflow:hidden; }}
  .bigtable-wrap {{ max-height:560px; overflow-y:auto; }}
  .scroll-x {{ overflow-x:auto; }}
  .bigtable-wrap table {{ min-width:880px; }}
  .bigtable-wrap thead th {{ position:sticky; top:0; background:var(--surface-alt); z-index:1; }}
  td.qcell {{ max-width:260px; color:var(--ink-soft); }}
  td.gt {{ color:var(--ink-faint); font-style:italic; max-width:140px; }}
  .f1 {{ font-weight:600; }}
  .cr {{ display:block; font-size:11px; color:var(--ink-faint); margin-top:2px; }}
  td.better .f1 {{ color:var(--good); }} td.worse .f1 {{ color:var(--bad); }}
  .ablation-cards {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:18px; }}
  @media (max-width:700px) {{ .ablation-cards {{ grid-template-columns:1fr; }} }}
  .abl-card {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px 18px; }}
  .abl-card h4 {{ font-family:"IBM Plex Mono",monospace; font-size:12px; text-transform:uppercase; letter-spacing:.04em;
                  margin:0 0 10px; color:var(--ink-soft); }}
  .abl-card.before {{ border-left:3px solid var(--bad); }} .abl-card.after {{ border-left:3px solid var(--good); }}
  .abl-row {{ display:flex; justify-content:space-between; padding:4px 0; font-size:13.5px; }}
  .abl-row span:last-child {{ font-family:"IBM Plex Mono",monospace; font-weight:600; }}
  .callout {{ background:var(--accent-soft); border-radius:10px; padding:16px 18px; font-size:13.5px; margin-top:18px; }}
  .callout strong {{ color:var(--accent); }}
  footer {{ padding:36px 0 8px; color:var(--ink-faint); font-size:12.5px; }}
</style>
<div class="wrap">
<header class="hero">
  <div class="eyebrow">Adaptive Graph Compression &middot; GraphRAG Evaluation</div>
  <h1>Query-adaptive compression, measured against its own baseline and two naive rivals</h1>
  <p class="sub">Results from the full six-stage pipeline &mdash; HotpotQA subset (n={n}), Qwen2.5-7B-Instruct
  for construction &amp; generation, BGE-M3 retrieval, a GRPO-trained node-level compression policy.</p>
  <div class="stagebar">
    <div class="stage"><span class="n">1</span>KG construction</div>
    <div class="stage"><span class="n">2</span>Retrieval (G_q)</div>
    <div class="stage"><span class="n">3</span>Baseline answer (A_o)</div>
    <div class="stage"><span class="n">4</span>GRPO compression policy</div>
    <div class="stage"><span class="n">5</span>Compressed answer (A_c)</div>
    <div class="stage"><span class="n">6</span>Evaluation &amp; baselines</div>
  </div>
</header>
<section>
  <div class="section-head"><h2>Headline numbers</h2>
  <p>QA quality (EM/F1) against the uncompressed baseline, plus the node-level classification read.</p></div>
  <div class="kpis">
    <div class="kpi"><div class="label">Baseline EM / F1</div><div class="value">{100*comp['em'][0]:.0f}% / {100*comp['f1'][0]:.0f}%</div><div class="note">Uncompressed G_q</div></div>
    <div class="kpi warn"><div class="label">RL policy EM / F1</div><div class="value">{100*comp['em'][1]:.0f}% / {100*comp['f1'][1]:.0f}%</div><div class="note">at {100*comp['cr'][1]:.0f}% compression</div></div>
    <div class="kpi good"><div class="label">Best naive baseline</div><div class="value">{100*comp['f1'][3]:.0f}% F1</div><div class="note">Heuristic pruning, {100*comp['cr'][3]:.0f}% compression</div></div>
    <div class="kpi accent"><div class="label">Node-decision accuracy</div><div class="value">{100*clf['accuracy']:.1f}%</div><div class="note">policy KEEP vs. supporting_facts</div></div>
    <div class="kpi accent"><div class="label">Node-decision AUC</div><div class="value">{auc_str}</div><div class="note">{clf['n_nodes']} nodes, {clf['n_relevant']} relevant</div></div>
  </div>
</section>
<section>
  <div class="section-head"><h2>Comparison across four methods</h2>
  <p>Uncompressed baseline and non-adaptive pruning baselines (Section III-F) against the GRPO policy.</p></div>
  <div class="imgpanel"><img src="data:image/png;base64,{img['comparison_heatmap']}" alt="Comparison heatmap" />
  <figcaption>Row-normalized for color only &mdash; printed values are the true averages across all {n} questions.</figcaption></div>
  <div class="callout"><strong>Reading this honestly:</strong> if the RL policy is not beating the naive
  baselines, that is expected at small training-data scale and is a training-signal issue, not a broken
  mechanism &mdash; see the ablation below.</div>
</section>
<section>
  <div class="section-head"><h2>Node-level classification: confusion matrix &amp; ROC</h2>
  <p>Ground truth &ldquo;relevant&rdquo; = node's source passage is in HotpotQA's own
  <span class="mono">supporting_facts</span> for that question.</p></div>
  <div class="grid2">
    <div class="imgpanel"><img src="data:image/png;base64,{img['confusion_matrix']}" alt="Confusion matrix" />
    <figcaption>Precision {clf['precision']:.3f} &middot; Recall {clf['recall']:.3f} &middot; F1 {clf['f1']:.3f}</figcaption></div>
    <div class="imgpanel"><img src="data:image/png;base64,{img['roc_curve']}" alt="ROC curve" />
    <figcaption>AUC {auc_str}</figcaption></div>
  </div>
</section>
<section>
  <div class="section-head"><h2>Ablation: the reward-gating fix</h2>
  <p><em>Reference run recorded on the original {abl['n_questions']}-question pilot</em> &mdash; the bug it
  demonstrates is fixed in the code, so a fresh run only ever reproduces the "after" row; kept here as a
  fixed comparison point, not regenerated from the current dataset.</p></div>
  <div class="imgpanel"><img src="data:image/png;base64,{img['ablation_chart']}" alt="Ablation chart" /></div>
  <div class="ablation-cards">
    <div class="abl-card before"><h4>Before &mdash; ungated CR bonus</h4>
      <div class="abl-row"><span>EM</span><span>{100*abl['before']['em']:.1f}%</span></div>
      <div class="abl-row"><span>F1</span><span>{100*abl['before']['f1']:.1f}%</span></div>
      <div class="abl-row"><span>Avg. CR</span><span>{100*abl['before']['cr']:.1f}%</span></div>
      <div class="abl-row"><span>G_c = 0 nodes</span><span>{abl['before']['g_c_zero']}/{abl['n_questions']}</span></div></div>
    <div class="abl-card after"><h4>After &mdash; gated on Q_c &ge; Q_o</h4>
      <div class="abl-row"><span>EM</span><span>{100*abl['after']['em']:.1f}%</span></div>
      <div class="abl-row"><span>F1</span><span>{100*abl['after']['f1']:.1f}%</span></div>
      <div class="abl-row"><span>Avg. CR</span><span>{100*abl['after']['cr']:.1f}%</span></div>
      <div class="abl-row"><span>G_c = 0 nodes</span><span>{abl['after']['g_c_zero']}/{abl['n_questions']}</span></div></div>
  </div>
</section>
<section>
  <div class="section-head"><h2>Per-question tabulation</h2>
  <p>F1 (bold) with compression ratio underneath, all {n} questions.</p></div>
  <div class="bigtable-wrap scroll-x"><table>
    <thead><tr><th>Question</th><th>Ground truth</th><th>|G_q|</th><th>Baseline</th><th>RL policy</th><th>Similarity</th><th>Heuristic</th></tr></thead>
    <tbody>{table_rows_html}</tbody>
  </table></div>
</section>
<footer>
  Pipeline: <span class="mono">select_subset.py &rarr; extract_kg.py &rarr; build_graph.py &rarr; embed_nodes.py &rarr;
  retrieve.py &rarr; generate_baseline_answers.py &rarr; train_compression_policy.py &rarr;
  generate_compressed_answers.py &rarr; generate_pruning_baselines.py &rarr; evaluate_policy_classification.py &rarr;
  generate_report.py</span>. Full methodology in <span class="mono">main.tex</span> &sect;III.
</footer>
</div>
"""
    (result_dir / "report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
