"""Why the uncompressed baseline is weak: context size, not retrieval.

Retrieval recall on this subset is 93.2%, yet the uncompressed baseline scores only
F1 0.336. This script shows where that goes: the generator increasingly REFUSES --
answering "the information provided does not contain..." -- as the serialized context
grows, even when retrieval demonstrably found every supporting passage.

That reframes the whole Stage 4 comparison. Compression is not just a token-budget
optimization here; pruning is repairing a real generation failure, which is why both
non-adaptive pruners beat the uncompressed baseline outright.

No LLM calls -- reads existing artifacts only. Run after generate_baseline_answers.py.

Usage:
    python analyze_context_ceiling.py
"""
import json
import statistics as st
from pathlib import Path

OUT = Path(__file__).resolve().parent / "output"

# Phrases this model uses when it believes the answer is absent from the context.
# Deliberately conservative -- these are refusals about the CONTEXT, not hedged answers.
REFUSAL_MARKERS = (
    "does not include", "does not contain", "does not mention", "does not provide",
    "not provided", "no information", "no relevant information",
    "don't have specific", "cannot provide",
)
# Answers that carry no content, used to catch shorten_answer discarding a real answer.
NULLISH = {"no", "yes", "none", "unknown", "no answer", "n/a", "not provided",
           "no information", "no year", "no information available"}


def is_refusal(record) -> bool:
    raw = str(record.get("raw_answer") or "").lower()
    return any(m in raw for m in REFUSAL_MARKERS)


def main():
    with open(OUT / "baseline_answers.json", encoding="utf-8") as f:
        base = json.load(f)
    rj = json.load(open(OUT / "retrieval_eval.json", encoding="utf-8"))
    if not isinstance(rj, list):
        rj = rj.get("per_question", rj)
    recall = {r["question"]: (r.get("recall") or 0.0) for r in rj}

    n = len(base)
    refusals = [r for r in base if is_refusal(r)]
    answered = [r for r in base if not is_refusal(r)]

    print(f"=== Context ceiling analysis ({n} questions) ===\n")
    print(f"Uncompressed baseline mean F1: {st.mean(r['f1'] for r in base):.3f}")
    print(f"Mean retrieval recall:         {st.mean(recall.get(r['question'], 0) for r in base):.3f}")
    print(f"Mean context size T_o:         {st.mean(r['T_o'] for r in base):,.0f} tokens\n")

    print(f"Answers claiming the context lacks the answer: {len(refusals)}/{n} "
          f"({100 * len(refusals) / n:.1f}%)")
    if refusals and answered:
        print(f"  mean F1 when refusing:     {st.mean(r['f1'] for r in refusals):.3f}")
        print(f"  mean F1 otherwise:         {st.mean(r['f1'] for r in answered):.3f}")
    # The damning part: refusing even though retrieval found everything needed.
    full = [r for r in refusals if recall.get(r["question"], 0) >= 0.999]
    print(f"  ...of which retrieval had FULL recall: {len(full)} "
          f"({100 * len(full) / max(len(refusals), 1):.0f}% of refusals) -- the supporting")
    print(f"     passages were in the context and the model still did not find them.\n")

    print("Refusal rate and quality by context-size quartile:")
    srt = sorted(base, key=lambda r: r["T_o"])
    q = len(srt) // 4
    print(f"  {'quartile':<14}{'mean T_o':>10}{'refusal':>10}{'mean F1':>10}")
    for i, name in enumerate(["Q1 smallest", "Q2", "Q3", "Q4 largest"]):
        chunk = srt[i * q:(i + 1) * q] if i < 3 else srt[3 * q:]
        rr = sum(1 for r in chunk if is_refusal(r)) / len(chunk)
        print(f"  {name:<14}{st.mean(r['T_o'] for r in chunk):>10,.0f}"
              f"{100 * rr:>9.1f}%{st.mean(r['f1'] for r in chunk):>10.3f}")

    lost = [r for r in base
            if r["f1"] < 1e-9
            and str(r["generated_answer"]).strip().lower().rstrip(".") in NULLISH
            and str(r["ground_truth"]).lower() in str(r.get("raw_answer") or "").lower()]
    print(f"\nshorten_answer discarded a correct answer (raw had the gold, shortened to a")
    print(f"null token): {len(lost)}/{n} ({100 * len(lost) / n:.1f}%)")
    for r in lost[:5]:
        print(f"  gold={r['ground_truth']!r:<30} shortened={str(r['generated_answer'])[:24]!r}")
    print("\nNote: do NOT test this as 'gold string appears in raw_answer' alone -- on")
    print("comparison questions ('Between X and Y, which...') both candidates appear in")
    print("the raw text, so that test reports ~7.5% when the real rate is this one.")

    json.dump({
        "n_questions": n,
        "baseline_f1": st.mean(r["f1"] for r in base),
        "mean_recall": st.mean(recall.get(r["question"], 0) for r in base),
        "refusal_rate": len(refusals) / n,
        "refusals_with_full_recall": len(full),
        "shorten_answer_losses": len(lost),
        "by_quartile": [
            {"quartile": i + 1,
             "mean_T_o": st.mean(r["T_o"] for r in (srt[i * q:(i + 1) * q] if i < 3 else srt[3 * q:])),
             "refusal_rate": sum(1 for r in (srt[i * q:(i + 1) * q] if i < 3 else srt[3 * q:]) if is_refusal(r))
                             / len(srt[i * q:(i + 1) * q] if i < 3 else srt[3 * q:]),
             "mean_f1": st.mean(r["f1"] for r in (srt[i * q:(i + 1) * q] if i < 3 else srt[3 * q:]))}
            for i in range(4)],
    }, open(OUT / "context_ceiling.json", "w", encoding="utf-8"), indent=2)
    print(f"\nSaved -> {OUT / 'context_ceiling.json'}")


if __name__ == "__main__":
    main()
