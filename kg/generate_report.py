"""
Stage 6 addendum: builds the full results package (confusion matrix, ROC/AUC,
comparison heatmap, per-question tabulation table, the paper's result figures,
significance and context-overflow statistics, and a self-contained HTML report)
from whatever is currently in kg/output/, and writes everything to ../result/ at
the project root.

Run this AFTER the full pipeline (including generate_pruning_baselines.py and
evaluate_policy_classification.py) has produced its output/*.json files --
this script only aggregates/visualizes, it makes no LLM calls itself and runs
in seconds regardless of dataset size.

Required upstream files in kg/output/ (produced by the scripts named):
    baseline_answers.json            <- generate_baseline_answers.py
    compressed_answers.json          <- generate_compressed_answers.py
    pruning_baselines.json           <- generate_pruning_baselines.py
    policy_classification_eval.json  <- evaluate_policy_classification.py

Paper figures (the files main.tex includes by name), each drawn only when its input
exists and otherwise skipped with a message saying which script to run:
    Fig. 2  training_dynamics.png  <- compression_training_log.json  (train_compression_policy.py)
    Fig. 3  node_confusion.png     <- policy_classification_eval.json, its `heuristic` block
    Fig. 4  feature_ablation.png   <- feature_ablation.json          (ablate_features.py)
Nothing in them is hard-coded, so a rerun on new data redraws them consistently. After a
rerun, upload those three PNGs from result/images/ to the Overleaf project.

Usage:
    python generate_report.py
"""
import base64
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath

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

# ---------------------------------------------------------------------------------------
# Paper figures (main.tex Figs. 2-4). IEEE single column: 3.5in wide, 300 dpi.
# Palette: dataviz reference categorical slots 1-2, validated all-pairs on a white print
# surface (worst adjacent CVD dE 24.7, normal-vision dE 33.6), plus its sequential blue ramp
# for the confusion matrices. Colour follows the entity across panels: orange = training,
# blue = validation / the proposed method. Measures on different scales get their own panel,
# never a second y-axis.
# ---------------------------------------------------------------------------------------
PAPER_DPI = 300
P_BLUE, P_ORANGE = "#2a78d6", "#eb6834"
P_INK, P_INK2, P_MUTED = "#0b0b0b", "#52514e", "#898781"
P_GRID, P_AXIS, P_SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
P_BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.edgecolor": P_AXIS, "axes.linewidth": 0.6, "axes.labelcolor": P_INK2,
    "xtick.color": P_MUTED, "ytick.color": P_MUTED, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "text.color": P_INK, "figure.facecolor": P_SURFACE,
    "axes.facecolor": P_SURFACE, "savefig.facecolor": P_SURFACE,
}


def require(path: Path, produced_by: str):
    if not path.exists():
        raise SystemExit(f"Missing {path} -- run {produced_by} first.")
    return path


def _f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def _paired(diffs):
    """Mean paired difference with its standard error and z. None when n < 2."""
    n = len(diffs)
    if n < 2:
        return None
    m = sum(diffs) / n
    se = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1)) / math.sqrt(n)
    return {"n": n, "mean": m, "se": se, "z": (m / se) if se else 0.0,
            "better": sum(1 for x in diffs if x > 1e-9), "worse": sum(1 for x in diffs if x < -1e-9)}


def _style(ax):
    ax.grid(axis="y", color=P_GRID, linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _fig_training(out: Path):
    """Fig. 2: (a) per-step compression ratio, (b) node-level F1 per epoch, train vs val."""
    path = KG_OUT / "compression_training_log.json"
    if not path.exists():
        return "skipped: no compression_training_log.json (run train_compression_policy.py)"
    log = json.load(open(path, encoding="utf-8"))
    if isinstance(log, list) or not log.get("validation"):
        return "skipped: training log predates per-epoch validation (retrain to get it)"
    steps = log["steps"]
    cr = [s["mean_CR"] for s in steps]
    win = 16
    roll = [float(np.mean(cr[max(0, i - win + 1):i + 1])) for i in range(len(cr))]
    n_epochs = len({s["epoch"] for s in steps})
    per_epoch = len([s for s in steps if s["epoch"] == steps[0]["epoch"]])
    epochs = [v["epoch"] for v in log["validation"]]
    tr = [_f1(v["train"]["precision"], v["train"]["recall"]) for v in log["validation"]]
    va = [_f1(v["val"]["precision"], v["val"]["recall"]) for v in log["validation"]]

    fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.65), gridspec_kw={"width_ratios": [1.55, 1]})
    _style(a); _style(b)
    a.plot(range(1, len(roll) + 1), roll, color=P_ORANGE, linewidth=1.3, solid_capstyle="round")
    for k in range(1, n_epochs):
        a.axvline(k * per_epoch + 0.5, color=P_AXIS, linewidth=0.5)
    for k in range(n_epochs):
        a.text((k + 0.5) * per_epoch, 0.97, f"Epoch {k + 1}", ha="center", va="top", fontsize=6, color=P_MUTED)
    a.set_ylim(0, 1); a.set_xlim(1, len(roll))
    a.set_xlabel("Training step"); a.set_ylabel("Compression ratio")
    a.set_title("(a)", fontsize=7, color=P_INK2, loc="left", pad=3)

    mk = dict(marker="o", markersize=4.2, markeredgecolor=P_SURFACE, markeredgewidth=0.8)
    b.plot(epochs, tr, color=P_ORANGE, linewidth=1.3, label="Training", **mk)
    b.plot(epochs, va, color=P_BLUE, linewidth=1.3, label="Validation", **mk)
    lo = math.floor((min(tr + va) - 0.02) / 0.05) * 0.05
    hi = math.ceil((max(tr + va) + 0.03) / 0.05) * 0.05
    # The saved checkpoint is the best epoch by validation F1 (train_compression_policy.py).
    best = int(np.argmax(va))
    b.annotate("saved", xy=(epochs[best], va[best]), xytext=(epochs[best] - 0.24, va[best] - 0.03),
               ha="left", fontsize=6, color=P_INK2,
               arrowprops=dict(arrowstyle="-", color=P_MUTED, linewidth=0.5))
    b.set_xticks(epochs); b.set_xlim(epochs[0] - 0.3, epochs[-1] + 0.3); b.set_ylim(lo, hi)
    b.set_xlabel("Epoch"); b.set_ylabel("Node-level F1")
    b.set_title("(b)", fontsize=7, color=P_INK2, loc="left", pad=3)
    b.legend(frameon=False, fontsize=6, loc="upper right", handlelength=1.4, borderaxespad=0.1)
    fig.tight_layout(pad=0.3, w_pad=0.9)
    fig.savefig(out, dpi=PAPER_DPI)
    plt.close(fig)
    return None


def _fig_confusion(clf: dict, out: Path):
    """Fig. 3: node-level confusion matrices, heuristic vs policy, on one shared colour scale."""
    if not clf.get("heuristic"):
        return "skipped: policy_classification_eval.json has no heuristic block (rerun evaluate_policy_classification.py)"
    def as_array(c):
        return np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    mats = (as_array(clf["heuristic"]["confusion_matrix"]), as_array(clf["confusion_matrix"]))
    cmap = LinearSegmentedColormap.from_list("seqblue", P_BLUE_RAMP)
    vmax = max(m.max() for m in mats)
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75))
    for ax, cm, title in zip(axes, mats, ("(a) Heuristic pruning", "(b) Proposed policy")):
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=vmax)
        for r in range(2):
            for c in range(2):
                val = cm[r, c]
                ax.text(c, r, f"{val:,}", ha="center", va="center", fontsize=7.5,
                        color=P_SURFACE if val > vmax * 0.45 else P_INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["REMOVE", "KEEP"], color=P_INK2)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["REMOVE", "KEEP"], color=P_INK2)
        ax.set_xticks([0.5], minor=True); ax.set_yticks([0.5], minor=True)
        ax.grid(which="minor", color=P_SURFACE, linewidth=2)      # 2px surface gap between cells
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("Decision")
        ax.set_title(title, fontsize=7, color=P_INK2, pad=3)
    axes[0].set_ylabel("Ground truth")
    fig.tight_layout(pad=0.3, w_pad=1.2)
    fig.savefig(out, dpi=PAPER_DPI)
    plt.close(fig)
    return None


def _fig_ablation(out: Path):
    """Fig. 4: node-level F1 by state representation, at the heuristic's KEEP-rate."""
    path = KG_OUT / "feature_ablation.json"
    if not path.exists():
        return "skipped: no feature_ablation.json (run ablate_features.py)"
    abl = json.load(open(path, encoding="utf-8"))
    rows = [(r.get("label", r["name"]), r["f1"], r["name"] == "struct4 only") for r in abl["rows"]]
    heur = abl["heuristic"]["f1"]

    # Figure dpi = save dpi, so the pixel-based corner radius below means what it says.
    fig, ax = plt.subplots(figsize=(3.5, 1.5), dpi=PAPER_DPI)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=P_GRID, linewidth=0.5, linestyle="-")
    ax.set_xlim(0, max([v for _, v, _ in rows] + [heur]) + 0.12)
    ax.set_ylim(-0.6, len(rows) - 0.05)       # headroom so the reference label sits inside
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _, _ in reversed(rows)], color=P_INK2)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Node-level F1 (validation)")
    ax.axvline(heur, color=P_INK2, linewidth=0.8)
    ax.text(heur - 0.008, len(rows) - 0.42, f"Heuristic pruning {heur:.3f}", ha="right", va="center",
            fontsize=6, color=P_INK2)
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()                         # layout final: data-per-pixel is stable now
    bb = ax.get_window_extent()
    px_x = (ax.get_xlim()[1] - ax.get_xlim()[0]) / bb.width
    px_y = (ax.get_ylim()[1] - ax.get_ylim()[0]) / bb.height
    bh, r_px = 0.46, 4 * PAPER_DPI / 96       # the spec's 4px data-end, scaled to print dpi
    for i, (_, val, highlight) in enumerate(rows):
        y0 = len(rows) - 1 - i
        # One path per bar -- square at the baseline, rounded at the data end. Two
        # overlapping patches leave a visible seam where they meet.
        rx, ry = r_px * px_x, min(r_px * px_y, bh / 2)
        lo_, hi_ = y0 - bh / 2, y0 + bh / 2
        verts = [(0, lo_), (val - rx, lo_), (val, lo_), (val, lo_ + ry), (val, hi_ - ry),
                 (val, hi_), (val - rx, hi_), (0, hi_), (0, lo_)]
        codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO,
                 MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor=P_BLUE if highlight else P_AXIS,
                               edgecolor="none", zorder=2))
        ax.text(val + 0.008, y0, f"{val:.3f}", va="center", ha="left", fontsize=6.5, color=P_INK, zorder=3)
    fig.savefig(out, dpi=PAPER_DPI)
    plt.close(fig)
    return None


def _fig_roc(clf: dict, out: Path):
    """ROC for the node-level KEEP decision, with both methods' operating points marked.

    The curve alone would only restate the AUC. Plotting where each method actually operates is
    what explains the near-equal node-F1: the policy sits higher and further right than the fixed
    1-hop rule (more relevant nodes kept, at more irrelevant ones), rather than strictly above it."""
    roc = clf.get("roc_curve")
    if not roc or not roc.get("fpr"):
        return "skipped: no ROC points in policy_classification_eval.json (only one class present?)"

    def point(cm):
        return cm["fp"] / max(cm["fp"] + cm["tn"], 1), cm["tp"] / max(cm["tp"] + cm["fn"], 1)
    p_fpr, p_tpr = point(clf["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    _style(ax)
    ax.grid(axis="x", color=P_GRID, linewidth=0.5, linestyle="-")
    ax.plot([0, 1], [0, 1], color=P_MUTED, linewidth=1, linestyle=(0, (4, 3)), label="Chance")
    ax.plot(roc["fpr"], roc["tpr"], color=P_BLUE, linewidth=1.4, solid_capstyle="round",
            label=f"Policy (AUC = {clf['auc']:.3f})")
    ax.fill_between(roc["fpr"], roc["tpr"], color=P_BLUE, alpha=0.08, linewidth=0)

    mk = dict(markersize=5, markeredgecolor=P_SURFACE, markeredgewidth=0.9, linestyle="none", zorder=3)
    ax.plot([p_fpr], [p_tpr], marker="o", color=P_BLUE, **mk)
    ax.annotate(f"policy at $p \\geq 0.5$\n({p_fpr:.3f}, {p_tpr:.3f})", xy=(p_fpr, p_tpr),
                xytext=(p_fpr + 0.10, p_tpr - 0.10), fontsize=6, color=P_INK2,
                arrowprops=dict(arrowstyle="-", color=P_MUTED, linewidth=0.5))
    if clf.get("heuristic"):
        h_fpr, h_tpr = point(clf["heuristic"]["confusion_matrix"])
        ax.plot([h_fpr], [h_tpr], marker="s", color=P_ORANGE, **mk)
        ax.annotate(f"heuristic pruning\n({h_fpr:.3f}, {h_tpr:.3f})", xy=(h_fpr, h_tpr),
                    xytext=(h_fpr + 0.14, h_tpr - 0.26), fontsize=6, color=P_INK2,
                    arrowprops=dict(arrowstyle="-", color=P_MUTED, linewidth=0.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right", handlelength=1.6, borderaxespad=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(out, dpi=PAPER_DPI)
    plt.close(fig)
    return None


def build_paper_figures(clf: dict) -> list:
    """Draws Figs. 2-5 into result/images/. Returns the file names actually written."""
    written = []
    with plt.rc_context(PAPER_RC):
        for fname, fn in (("training_dynamics.png", lambda p: _fig_training(p)),
                          ("node_confusion.png", lambda p: _fig_confusion(clf, p)),
                          ("node_roc.png", lambda p: _fig_roc(clf, p)),
                          ("feature_ablation.png", lambda p: _fig_ablation(p))):
            reason = fn(IMAGES_DIR / fname)
            if reason:
                print(f"  paper figure {fname}: {reason}")
            else:
                written.append(fname)
                print(f"  paper figure {fname}: written")
    return written


def _overflow_split(compressed: list, pruning_all: list):
    """Compression's effect split by whether the uncompressed context fits the answer model's
    window. The policy is measured on its own evaluation questions; the pruners, which have no
    held-out split, over every question they cover."""
    try:
        from generate_baseline_answers import ANSWER_NUM_CTX
    except Exception as e:  # noqa: BLE001 -- report and carry on, this block is optional
        print(f"  context-overflow split skipped: {e}")
        return None
    out = {"num_ctx": ANSWER_NUM_CTX, "n_pruning_questions": len(pruning_all)}
    for key, keep in (("fits", lambda t: t <= ANSWER_NUM_CTX), ("overflows", lambda t: t > ANSWER_NUM_CTX)):
        rows = [r for r in compressed if keep(r["T_o"])]
        stats = _paired([r["F1_c"] - r["F1_o"] for r in rows]) or {"n": len(rows)}
        if rows:
            stats.update(baseline_f1=sum(r["F1_o"] for r in rows) / len(rows),
                         policy_f1=sum(r["F1_c"] for r in rows) / len(rows),
                         mean_T_c=sum(r["T_c"] for r in rows) / len(rows))
        out[f"policy_{key}"] = stats
        for m in ("heuristic", "similarity"):
            out[f"{m}_{key}"] = _paired([r[m]["F1"] - r["F1_o"] for r in pruning_all if keep(r["T_o"])])
    return out


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(require(KG_OUT / "compressed_answers.json", "generate_compressed_answers.py"), encoding="utf-8") as f:
        compressed = json.load(f)
    with open(require(KG_OUT / "pruning_baselines.json", "generate_pruning_baselines.py"), encoding="utf-8") as f:
        pruning_all = json.load(f)
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
    pruning = [r for r in pruning_all if r["question"] in eval_qs]
    if len(pruning) < len(pruning_all):
        print(f"Scoping pruning baselines to the {len(pruning)} question(s) the policy was "
              f"evaluated on (of {len(pruning_all)} in pruning_baselines.json).")
    if not pruning:
        raise SystemExit("No overlap between pruning_baselines.json and compressed_answers.json -- "
                         "rerun generate_pruning_baselines.py.")

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
    tokens = [sum(r["T_o"] for r in compressed) / n, sum(r["T_c"] for r in compressed) / n,
              sum(r["similarity"]["T"] for r in pruning) / pn, sum(r["heuristic"]["T"] for r in pruning) / pn]
    comparison = {"methods": methods, "em": em_vals, "f1": f1_vals, "cr": cr_vals, "mean_tokens": tokens}

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
    for fname in ("confusion_matrix.png", "roc_curve.png"):
        src = KG_OUT / fname
        if src.exists():
            shutil.copyfile(src, IMAGES_DIR / fname)

    print("Paper figures (main.tex Figs. 2-4):")
    paper_figures = build_paper_figures(clf)

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

    # ---------- paired significance + context-overflow split (the numbers the paper quotes) ----------
    both = [r for r in compressed if r["question"] in prune_by_q]
    paired = {
        "policy_vs_uncompressed_f1": _paired([r["F1_c"] - r["F1_o"] for r in compressed]),
        "policy_vs_uncompressed_em": _paired([float(r["EM_c"]) - float(r["EM_o"]) for r in compressed]),
        "policy_vs_heuristic_f1": _paired([r["F1_c"] - prune_by_q[r["question"]]["heuristic"]["F1"] for r in both]),
        "policy_vs_similarity_f1": _paired([r["F1_c"] - prune_by_q[r["question"]]["similarity"]["F1"] for r in both]),
    }
    overflow = _overflow_split(compressed, pruning_all)

    report_data = {
        "comparison": comparison, "ablation_reference": ABLATION_REFERENCE,
        "classification": clf, "table_rows": table_rows, "n_questions": n,
        "paired": paired, "overflow": overflow, "paper_figures": paper_figures,
    }
    with open(DATA_DIR / "report_data.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    # also copy the raw upstream JSONs so result/ is self-contained
    for fname in ("baseline_answers.json", "compressed_answers.json", "pruning_baselines.json",
                  "policy_classification_eval.json", "compression_training_log.json", "feature_ablation.json"):
        src = KG_OUT / fname
        if src.exists():
            shutil.copyfile(src, DATA_DIR / fname)

    print("\nPaired per question (a difference under ~2 sigma is not distinguishable from noise):")
    for key, p in paired.items():
        if p:
            print(f"  {key:<28} {p['mean']:+.3f}  SE {p['se']:.3f}  ({p['z']:+.1f} sigma)  "
                  f"better {p['better']} / worse {p['worse']} / tied {p['n'] - p['better'] - p['worse']}")
    if overflow:
        print(f"Context-overflow split at num_ctx={overflow['num_ctx']}:")
        for key in ("fits", "overflows"):
            p = overflow[f"policy_{key}"]
            if p.get("mean") is not None:
                print(f"  policy {key:<9} n={p['n']:3d}  F1 {p['baseline_f1']:.3f} -> {p['policy_f1']:.3f}  "
                      f"dF1 {p['mean']:+.3f} ({p['z']:+.1f} sigma)")
            for m in ("heuristic", "similarity"):
                q = overflow[f"{m}_{key}"]
                if q:
                    print(f"  {m:<10} {key:<9} n={q['n']:3d}  dF1 {q['mean']:+.3f} ({q['z']:+.1f} sigma)")

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


def _paired_str(p):
    if not p:
        return "&mdash;"
    tied = p["n"] - p["better"] - p["worse"]
    return (f'<span class="mono">{p["mean"]:+.3f}</span> (SE {p["se"]:.3f}, {p["z"]:+.1f}&sigma;; '
            f'{p["better"]} better / {p["worse"]} worse / {tied} tied)')


def _build_html(D: dict, images_dir: Path, result_dir: Path):
    comp = D["comparison"]
    abl = D["ablation_reference"]
    clf = D["classification"]
    rows = D["table_rows"]
    n = D["n_questions"]
    paired = D.get("paired") or {}
    overflow = D.get("overflow")

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

    # Best naive baseline by F1, not hard-wired to the heuristic.
    bi = 2 if comp["f1"][2] >= comp["f1"][3] else 3
    best_name = "Similarity pruning" if bi == 2 else "Heuristic pruning"

    callout_html = (
        f'<div class="callout"><strong>Significance, paired per question (n={n}):</strong> '
        f'policy &minus; uncompressed F1 {_paired_str(paired.get("policy_vs_uncompressed_f1"))}; '
        f'policy &minus; heuristic F1 {_paired_str(paired.get("policy_vs_heuristic_f1"))}; '
        f'policy &minus; similarity F1 {_paired_str(paired.get("policy_vs_similarity_f1"))}. '
        'A difference under roughly 2&sigma; is not distinguishable from noise at this sample size, '
        'so report it as preserved quality rather than as an improvement.</div>')

    h = clf.get("heuristic")
    heur_caption = (f' &middot; heuristic on the same nodes: precision {h["precision"]:.3f} &middot; '
                    f'recall {h["recall"]:.3f} &middot; F1 {h["f1"]:.3f}') if h else ""
    by_hop_html = ""
    if clf.get("by_hop"):
        hop_rows = "".join(
            f'<tr><td>{b["name"]}</td><td class="mono">{b["n_nodes"]:,}</td><td class="mono">{b["n_relevant"]:,}</td>'
            f'<td class="mono">{100 * b["policy_keep_rate"]:.1f}% &middot; {b["policy_tp"]} TP / {b["policy_fp"]} FP</td>'
            f'<td class="mono">{100 * b["heuristic_keep_rate"]:.1f}% &middot; {b["heuristic_tp"]} TP / {b["heuristic_fp"]} FP</td></tr>'
            for b in clf["by_hop"])
        by_hop_html = (
            '<div class="comp-table-wrap scroll-x" style="margin-top:20px"><table><thead><tr>'
            '<th>Hop from nearest seed</th><th>Nodes</th><th>Relevant</th><th>Policy keeps</th>'
            f'<th>Heuristic keeps</th></tr></thead><tbody>{hop_rows}</tbody></table></div>')

    overflow_html = ""
    if overflow and overflow.get("policy_fits", {}).get("mean") is not None \
            and overflow.get("policy_overflows", {}).get("mean") is not None:
        def orow(label, p):
            return (f'<tr><td>{label}</td><td class="mono">{p["n"]}</td>'
                    f'<td class="mono">{p["baseline_f1"]:.3f}</td><td class="mono">{p["policy_f1"]:.3f}</td>'
                    f'<td class="mono">{p["mean"]:+.3f} ({p["z"]:+.1f}&sigma;)</td></tr>')
        def prow(label, p):
            return (f'<tr><td>{label}</td><td class="mono">{p["n"]}</td><td colspan="2">&mdash;</td>'
                    f'<td class="mono">{p["mean"]:+.3f} ({p["z"]:+.1f}&sigma;)</td></tr>') if p else ""
        overflow_html = f"""<section>
  <div class="section-head"><h2>Where compression matters: context overflow</h2>
  <p>Split by whether the uncompressed context fits the answer model's {overflow['num_ctx']:,}-token window.
  Policy rows are its {n} evaluation questions; pruning rows cover all {overflow['n_pruning_questions']}
  questions, since the pruners have no held-out split.</p></div>
  <div class="comp-table-wrap scroll-x"><table><thead><tr><th>Subset</th><th>n</th><th>Uncompressed F1</th>
  <th>Policy F1</th><th>&Delta;F1 vs uncompressed</th></tr></thead><tbody>
  {orow("Policy &middot; context fits", overflow["policy_fits"])}
  {orow("Policy &middot; context overflows", overflow["policy_overflows"])}
  {prow("Heuristic pruning &middot; context fits", overflow.get("heuristic_fits"))}
  {prow("Heuristic pruning &middot; context overflows", overflow.get("heuristic_overflows"))}
  {prow("Similarity pruning &middot; context fits", overflow.get("similarity_fits"))}
  {prow("Similarity pruning &middot; context overflows", overflow.get("similarity_overflows"))}
  </tbody></table></div>
</section>"""

    paper_captions = {
        "training_dynamics.png": "Fig. 2 &mdash; per-step compression ratio (16-step moving average; questions are "
                                 "visited in order of increasing |G_q| within each epoch) and node-level F1 per epoch.",
        "node_confusion.png": "Fig. 3 &mdash; node-level confusion matrices, heuristic pruning vs. the policy, on the "
                              "same validation nodes and one shared colour scale.",
        "node_roc.png": "Fig. 4 &mdash; ROC for the node-level KEEP decision, with both methods' operating "
                        "points marked.",
        "feature_ablation.png": "Fig. 5 &mdash; node-level F1 by state representation, each scored at the "
                                "heuristic's KEEP-rate.",
    }
    panels = [f'<figure class="imgpanel"><img src="data:image/png;base64,{_b64(images_dir / f)}" alt="{f}" />'
              f'<figcaption>{paper_captions[f]}</figcaption></figure>'
              for f in D.get("paper_figures", []) if (images_dir / f).exists()]
    paper_html = ""
    if panels:
        paper_html = f"""<section>
  <div class="section-head"><h2>Paper figures</h2>
  <p>The exact PNGs <span class="mono">main.tex</span> includes, drawn from this run's data. Upload them from
  <span class="mono">result/images/</span> to the Overleaf project after every rerun.</p></div>
  <div class="grid2">{"".join(panels)}</div>
</section>"""

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
  figure.imgpanel {{ margin:0; }}
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
    <div class="kpi accent"><div class="label">RL policy EM / F1</div><div class="value">{100*comp['em'][1]:.0f}% / {100*comp['f1'][1]:.0f}%</div><div class="note">at {100*comp['cr'][1]:.0f}% compression</div></div>
    <div class="kpi"><div class="label">Best naive baseline</div><div class="value">{100*comp['f1'][bi]:.0f}% F1</div><div class="note">{best_name}, {100*comp['cr'][bi]:.0f}% compression</div></div>
    <div class="kpi accent"><div class="label">Node-decision accuracy</div><div class="value">{100*clf['accuracy']:.1f}%</div><div class="note">policy KEEP vs. supporting_facts</div></div>
    <div class="kpi accent"><div class="label">Node-decision AUC</div><div class="value">{auc_str}</div><div class="note">{clf['n_nodes']} nodes, {clf['n_relevant']} relevant</div></div>
  </div>
</section>
<section>
  <div class="section-head"><h2>Comparison across four methods</h2>
  <p>Uncompressed baseline and non-adaptive pruning baselines (Section III-F) against the GRPO policy.</p></div>
  <div class="imgpanel"><img src="data:image/png;base64,{img['comparison_heatmap']}" alt="Comparison heatmap" />
  <figcaption>Row-normalized for color only &mdash; printed values are the true averages across all {n} questions.</figcaption></div>
  {callout_html}
</section>
{overflow_html}
<section>
  <div class="section-head"><h2>Node-level classification: confusion matrix &amp; ROC</h2>
  <p>Ground truth &ldquo;relevant&rdquo; = node's source passage is in HotpotQA's own
  <span class="mono">supporting_facts</span> for that question.</p></div>
  <div class="grid2">
    <div class="imgpanel"><img src="data:image/png;base64,{img['confusion_matrix']}" alt="Confusion matrix" />
    <figcaption>Precision {clf['precision']:.3f} &middot; Recall {clf['recall']:.3f} &middot; F1 {clf['f1']:.3f}{heur_caption}</figcaption></div>
    <div class="imgpanel"><img src="data:image/png;base64,{img['roc_curve']}" alt="ROC curve" />
    <figcaption>AUC {auc_str}</figcaption></div>
  </div>
  {by_hop_html}
</section>
{paper_html}
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
  ablate_features.py &rarr; generate_report.py</span>. Full methodology in <span class="mono">main.tex</span> &sect;III.
</footer>
</div>
"""
    (result_dir / "report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
