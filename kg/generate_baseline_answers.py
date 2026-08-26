"""
Stage 3 (methodology Section III-C): Baseline (uncompressed) answer
generation from the retrieved subgraph G_q.

For every subset question: retrieve G_q (Stage 2), serialize its entities/
relations/source-passage text into a context, prompt Qwen2.5-7B-Instruct
(same model used for extraction) for answer A_o, and record the context
token count T_o plus EM/F1 against the ground-truth answer.

This is the uncompressed baseline (A_o, T_o) that the later GRPO-based graph
compression policy (Section III-D/III-E) is trained and compared against --
no pruning or learned policy is used at this step.

Usage:
    python generate_baseline_answers.py
    python generate_baseline_answers.py --top-k 5 --hops 2
"""
import os
# Must be set before numpy/torch are imported -- see embed_nodes.py for why.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import re
import string
import time
from collections import Counter
from pathlib import Path

import requests
from sentence_transformers import SentenceTransformer

from retrieve import load_graph, load_embeddings, retrieve, MODEL_NAME, default_device

OUT_DIR = Path(__file__).resolve().parent / "output"
OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "qwen2.5:7b-instruct"
TOKENIZER_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # tokenizer-only download (~small), matches LLM_MODEL's family

ANSWER_SYSTEM_PROMPT = """You answer questions using ONLY the provided context (graph facts and source \
text). If the context doesn't fully contain the answer, give your best guess based on what's there \
rather than refusing."""

# This model reliably explains its reasoning even when told not to (tested: neither strict
# instructions nor few-shot examples nor JSON-format constraints stopped it from wrapping a
# correct answer in a full sentence for "who"/"which" style questions). Rather than fight that,
# a second pass asks it to extract the short answer FROM its own longer response -- a much
# easier task than "answer tersely from scratch", and reliably works.
SHORTEN_SYSTEM_PROMPT = """Extract the short, direct answer from a longer response -- just the \
name/number/date/yes-or-no/short phrase that directly answers the question, nothing else, exactly as \
it would appear filled into a blank. Output ONLY a JSON object: {"answer": "..."}"""


def build_context(Gq, passages_by_title: dict) -> str:
    """Serialize G_q into the 'entities, relations, and their textual evidence'
    context described in Section III-C: structured graph facts plus the raw
    source-passage text for every passage any node in G_q came from."""
    lines = ["Facts:"]
    for u, v, attrs in Gq.edges(data=True):
        uname = Gq.nodes[u].get("name", u)
        vname = Gq.nodes[v].get("name", v)
        rel = attrs.get("relation", "")
        desc = attrs.get("description", "")
        lines.append(f"- {uname} {rel} {vname}. {desc}".strip())

    titles = set()
    for _, attrs in Gq.nodes(data=True):
        titles.update(attrs.get("sources", []))

    if titles:
        lines.append("\nSource passages:")
        for t in sorted(titles):
            sents = passages_by_title.get(t)
            if sents:
                lines.append(f"[{t}] " + " ".join(sents))

    return "\n".join(lines)


def generate_answer(question: str, context: str, retries: int = 2) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    last_err = None
    for _ in range(retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1)
    print(f"  [WARN] answer generation failed: {last_err}")
    return ""


def shorten_answer(question: str, verbose_answer: str, retries: int = 2) -> str:
    """Second pass: extract the short direct answer from a (possibly explanatory)
    generated answer. See SHORTEN_SYSTEM_PROMPT for why this is a separate call."""
    if not verbose_answer:
        return verbose_answer
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SHORTEN_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\nLonger response: {verbose_answer}"},
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
            short = parsed.get("answer")
            if short:
                return str(short).strip()
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return verbose_answer  # fall back to the unshortened answer rather than losing it


# --- standard SQuAD/HotpotQA-style EM/F1 (self-contained, no external eval package) ---

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def exact_match(pred: str, gt: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gt)


def f1_score(pred: str, gt: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gt_toks = normalize_answer(gt).split()
    common = Counter(pred_toks) & Counter(gt_toks)
    num_same = sum(common.values())
    if not pred_toks or not gt_toks:
        return float(pred_toks == gt_toks)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gt_toks)
    return 2 * precision * recall / (precision + recall)


def load_token_counter():
    """Real Qwen tokenizer if available (matches what T_o means for later compression-ratio
    math); falls back to a whitespace word count if the tokenizer can't be fetched."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
        return lambda text: len(tokenizer.encode(text))
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Could not load Qwen tokenizer ({e}); using whitespace word count for T_o instead.")
        return lambda text: len(text.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--device", default=None, help="cuda or cpu for the embedding model")
    args = ap.parse_args()
    if args.device is None:
        args.device = default_device()

    with open(OUT_DIR / "subset_questions.json", encoding="utf-8") as f:
        questions = json.load(f)
    with open(OUT_DIR / "subset_passages.json", encoding="utf-8") as f:
        passages = json.load(f)
    passages_by_title = {p["title"]: p["sentences"] for p in passages}

    count_tokens = load_token_counter()

    G = load_graph()
    embeddings, ids = load_embeddings()
    print(f"Loading {MODEL_NAME} on {args.device}...")
    embed_model = SentenceTransformer(MODEL_NAME, device=args.device)
    print(f"Generating baseline answers for {len(questions)} questions "
          f"(--top-k {args.top_k} --hops {args.hops})...\n")

    results = []
    em_sum = 0.0
    f1_sum = 0.0
    t0 = time.time()

    for i, q in enumerate(questions, 1):
        Gq, seeds = retrieve(q["question"], G, embeddings, ids, embed_model, top_k=args.top_k, hops=args.hops)
        context = build_context(Gq, passages_by_title)
        T_o = count_tokens(context)

        t_start = time.time()
        raw_answer = generate_answer(q["question"], context)
        answer = shorten_answer(q["question"], raw_answer)
        t_this = time.time() - t_start

        em = exact_match(answer, q["answer"])
        f1 = f1_score(answer, q["answer"])
        em_sum += em
        f1_sum += f1

        results.append({
            "question": q["question"],
            "type": q.get("type", ""),
            "ground_truth": q["answer"],
            "generated_answer": answer,
            "raw_answer": raw_answer,
            "em": em,
            "f1": f1,
            "context": context,
            "T_o": T_o,
            "gq_nodes": Gq.number_of_nodes(),
            "gq_edges": Gq.number_of_edges(),
        })

        avg = (time.time() - t0) / i
        eta = avg * (len(questions) - i)
        print(f"[{i}/{len(questions)}] EM={int(em)} F1={f1:.2f} T_o={T_o:4d}tok ({t_this:.1f}s, "
              f"ETA {eta:.0f}s)  pred='{answer[:40]}'  gt='{q['answer'][:40]}'")

    n = len(questions)
    print("\n=== Baseline Answer Generation Summary (uncompressed, A_o) ===")
    print(f"Questions:          {n}")
    print(f"EM:                 {100*em_sum/n:.1f}%")
    print(f"F1:                 {100*f1_sum/n:.1f}%")
    print(f"Avg T_o (context tokens): {sum(r['T_o'] for r in results)/n:.0f}")
    print(f"Total time:         {time.time()-t0:.1f}s")

    with open(OUT_DIR / "baseline_answers.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {OUT_DIR / 'baseline_answers.json'}")


if __name__ == "__main__":
    main()
