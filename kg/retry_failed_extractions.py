"""
Recover empty (failed) extractions left over from an extract_kg.py run.

extract_kg.py's retries use temperature 0.0 (deterministic) with the same
prompt every attempt -- if the model gets stuck in a runaway/repetitive
generation for a specific passage (a real local-LLM failure mode, not
necessarily related to input length or content), every retry hangs and times
out identically, which is exactly what a same-temperature retry can never
fix. This script targets ONLY passages already recorded as empty in
extractions.json and retries them with a nonzero temperature (breaking the
deterministic repeat) across up to two escalating attempts.

Recovered passages are merged back into extractions.json in place,
checkpointed after each one. Passages that still come back empty are left as
they are -- already correctly recorded, not silently lost.

Usage:
    python retry_failed_extractions.py
"""
import json
import time

import requests

from extract_kg import (
    EXTRACTIONS_PATH, PASSAGES_PATH, OLLAMA_URL, MODEL,
    SYSTEM_PROMPT, save_extractions, append_log,
)

# Two escalating attempts, both away from the temperature-0.0 that produced
# the original hang -- a second, higher temperature gives a genuinely
# different chance to escape a degenerate generation loop if the first
# doesn't, rather than repeating the same (still deterministic) attempt twice.
RETRY_TEMPERATURES = [0.3, 0.7]
TIMEOUT_SECONDS = 180


def extract_one_with_temp(title: str, sentences: list[str], temperature: float) -> dict:
    text = " ".join(sentences)
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Title: {title}\nText: {text}"},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SECONDS)
    resp.raise_for_status()
    parsed = json.loads(resp.json()["message"]["content"])
    parsed.setdefault("entities", [])
    parsed.setdefault("relations", [])
    return parsed


def main():
    with open(EXTRACTIONS_PATH, encoding="utf-8") as f:
        results = json.load(f)
    with open(PASSAGES_PATH, encoding="utf-8") as f:
        passages_by_title = {p["title"]: p["sentences"] for p in json.load(f)}

    failed_indices = [
        i for i, r in enumerate(results)
        if not r["entities"] and not r["relations"]
    ]
    if not failed_indices:
        print("No empty extractions found -- nothing to retry.")
        return

    print(f"Found {len(failed_indices)} empty extraction(s) to retry: "
          f"{[results[i]['title'] for i in failed_indices]}\n")

    recovered, still_empty = [], []
    for i in failed_indices:
        title = results[i]["title"]
        sentences = passages_by_title.get(title)
        if sentences is None:
            print(f"  [SKIP] '{title}' not found in subset_passages.json")
            continue

        fixed = None
        for temp in RETRY_TEMPERATURES:
            t_start = time.time()
            try:
                extracted = extract_one_with_temp(title, sentences, temp)
            except Exception as e:  # noqa: BLE001
                t_this = time.time() - t_start
                print(f"  [temp={temp}] '{title}' -> error after {t_this:.1f}s: {e}")
                continue
            t_this = time.time() - t_start
            if extracted["entities"] or extracted["relations"]:
                print(f"  [temp={temp}] '{title}' -> RECOVERED: "
                      f"{len(extracted['entities'])}e/{len(extracted['relations'])}r in {t_this:.1f}s")
                fixed = extracted
                break
            print(f"  [temp={temp}] '{title}' -> still empty ({t_this:.1f}s)")

        if fixed is not None:
            results[i] = {"title": title, "entities": fixed["entities"], "relations": fixed["relations"]}
            save_extractions(results)  # checkpoint immediately, same as extract_kg.py
            append_log(f"[RECOVERY] '{title}' -> {len(fixed['entities'])}e/{len(fixed['relations'])}r")
            recovered.append(title)
        else:
            append_log(f"[RECOVERY] '{title}' -> still empty after {len(RETRY_TEMPERATURES)} attempts")
            still_empty.append(title)

    print(f"\n=== Recovery summary ===")
    print(f"Recovered:   {len(recovered)}  {recovered}")
    print(f"Still empty: {len(still_empty)}  {still_empty}")
    if recovered:
        print(f"\nSaved -> {EXTRACTIONS_PATH}")
        print("Rerun build_graph.py (and everything downstream) to pick up the recovered passages -- "
              "see CLAUDE.md's pipeline-propagation note.")


if __name__ == "__main__":
    main()
