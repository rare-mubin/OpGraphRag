"""
Step 1: Select a small subset of HotpotQA questions and pull out their
unique context passages, ready for entity/relation extraction.

Usage:
    python select_subset.py --n 20 --seed 42
    python select_subset.py --n 20 --seed 42 --max-passages 50
"""
import json
import argparse
import random
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "Dataset" / "formatted_output.json"
OUT_DIR = Path(__file__).resolve().parent / "output"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="number of questions to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-passages", type=int, default=None,
                     help="cap the number of unique passages kept (randomly trimmed to this count)")
    args = ap.parse_args()

    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    random.seed(args.seed)
    sample = random.sample(data, args.n)

    # Dedupe passages by title across the sampled questions
    passages = {}  # title -> {"title":..., "sentences":[...]}
    questions_out = []
    for item in sample:
        for title, sents in item["context"]:
            if title not in passages:
                passages[title] = {"title": title, "sentences": sents}
        questions_out.append({
            "_id": item["_id"],
            "question": item["question"],
            "answer": item["answer"],
            "type": item["type"],
            "level": item["level"],
            "supporting_facts": item["supporting_facts"],
            "context_titles": [t for t, _ in item["context"]],
        })

    passage_list = list(passages.values())
    if args.max_passages is not None and len(passage_list) > args.max_passages:
        passage_list = random.sample(passage_list, args.max_passages)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "subset_questions.json", "w", encoding="utf-8") as f:
        json.dump(questions_out, f, indent=2, ensure_ascii=False)

    with open(OUT_DIR / "subset_passages.json", "w", encoding="utf-8") as f:
        json.dump(passage_list, f, indent=2, ensure_ascii=False)

    print(f"Sampled {args.n} questions -> {OUT_DIR / 'subset_questions.json'}")
    print(f"Unique passages found: {len(passages)}; kept {len(passage_list)} "
          f"-> {OUT_DIR / 'subset_passages.json'}")


if __name__ == "__main__":
    main()
