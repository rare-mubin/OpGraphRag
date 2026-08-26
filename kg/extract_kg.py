"""
Step 2: Run entity/relation extraction over the sampled passages using a
local Qwen2.5 model served by Ollama, and save the raw triples.

Resumable: safe to stop at any time (Ctrl+C, closing the terminal, power
loss) and rerun later -- every passage is saved to output/extractions.json
and logged to output/extraction.log the moment it finishes, so a rerun picks
up exactly where the last run left off instead of starting over.

Requires Ollama running locally with the model pulled:
    ollama pull qwen2.5:7b-instruct

Usage:
    python extract_kg.py
"""
import json
import os
import sys
import time
import requests
from pathlib import Path
from datetime import datetime

OUT_DIR = Path(__file__).resolve().parent / "output"
PASSAGES_PATH = OUT_DIR / "subset_passages.json"
EXTRACTIONS_PATH = OUT_DIR / "extractions.json"
LOG_PATH = OUT_DIR / "extraction.log"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b-instruct"
# MODEL = "qwen2.5:3b-instruct"
# MODEL = "qwen2.5:1.5b-instruct"

ETA_WINDOW = 10  # how many recent passages to average over for the ETA estimate
BAR_WIDTH = 70
TITLE_WIDTH = 45

SYSTEM_PROMPT = """You are an information extraction system building a knowledge graph.
Given a short passage (a title and its text), extract:
1. entities: distinct real-world entities explicitly mentioned in the text (people, organizations, places, works like films/books, events, dates, or other notable concepts).
2. relations: factual relationships between two entities from your entity list, stated or clearly implied by the text.

Rules:
- Only use information present in the passage. Do not invent facts.
- Use the passage's title as an entity if it is a coherent named entity (it usually is).
- Entity names must be exactly as they appear in the text (no paraphrasing).
- Keep descriptions short (<20 words), grounded in the passage.
- Pay special attention to any number, count, or quantity stated about an entity (e.g. "contains
  three species", "won five awards", "population of 729") -- these are easy to skip but are often
  exactly what a downstream question asks about. Extract the number itself as an entity and add a
  relation for it, even if the sentence containing it is short or looks incomplete/truncated.
- Output ONLY a single JSON object, no markdown, no commentary, matching this schema:

{
  "entities": [
    {"name": "string", "type": "PERSON|ORG|LOCATION|WORK|EVENT|OTHER", "description": "string"}
  ],
  "relations": [
    {"source": "string", "target": "string", "relation": "string", "description": "string"}
  ]
}
"""


def extract_one(title: str, sentences: list[str], retries: int = 2) -> dict:
    text = " ".join(sentences)
    user_prompt = f"Title: {title}\nText: {text}"

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0},
    }

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            parsed = json.loads(content)
            parsed.setdefault("entities", [])
            parsed.setdefault("relations", [])
            return parsed
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1)
    print(f"  [WARN] extraction failed for '{title}': {last_err}")
    return {"entities": [], "relations": []}


def load_existing(subset_titles: set) -> list:
    """Load previously-saved extractions, dropping any that no longer belong
    to the current subset_passages.json (e.g. it was re-sampled)."""
    if not EXTRACTIONS_PATH.exists():
        return []
    with open(EXTRACTIONS_PATH, encoding="utf-8") as f:
        existing = json.load(f)
    kept = [r for r in existing if r.get("title") in subset_titles]
    dropped = len(existing) - len(kept)
    if dropped:
        print(f"Note: dropping {dropped} stale extraction(s) not in the current subset.")
    return kept


def save_extractions(results: list) -> None:
    with open(EXTRACTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def append_log(msg: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")


def enable_windows_ansi() -> None:
    """Turn on ANSI/VT escape-sequence support in the Windows console, so the
    pinned progress bar (cursor-up + clear-line) renders correctly."""
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:  # noqa: BLE001
        pass


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def format_passage_line(done: int, total: int, title: str, n_ent: int, n_rel: int, t_this: float) -> str:
    """One scrollback line per finished passage, e.g.:
    (60/200) | 'Tom Clancy's Net Force Explorers: D'      [6 entities] [4 relations] | took   5.0s
    """
    title_disp = title[:TITLE_WIDTH]
    return (f"({done}/{total}) | '{title_disp:<{TITLE_WIDTH}}' "
            f"[{n_ent} entities] [{n_rel} relations] | took {t_this:6.1f}s")


def format_bar_line(done: int, total: int, eta_seconds: float, width: int = BAR_WIDTH) -> str:
    """The single live bar pinned at the bottom, e.g.:
    [=========--------------------------------------------------]  30.5%  ETA   17m46s
    """
    frac = done / total if total else 1.0
    filled = int(width * frac)
    bar = "[" + "=" * filled + "-" * (width - filled) + "]"
    return f"{bar} {frac*100:5.1f}%  ETA {format_duration(eta_seconds):>8}"


def redraw(stdout, log_line: str, bar_line: str) -> None:
    """Clear the pinned bar line, print the finished passage above it
    (scrolls up into history), then redraw the bar pinned at the bottom."""
    stdout.write("\r\x1b[2K")   # carriage return + clear the current (bar) line
    stdout.write(log_line + "\n")
    stdout.write(bar_line)
    stdout.flush()


def print_stats(results: list, run_passage_times: list, run_warn_count: int,
                 run_elapsed: float, total_in_subset: int) -> None:
    n_total = len(results)
    n_entities = sum(len(r["entities"]) for r in results)
    n_relations = sum(len(r["relations"]) for r in results)
    n_empty = sum(1 for r in results if not r["entities"] and not r["relations"])

    print("\n" + "=" * 50)
    print("EXTRACTION STATISTICS")
    print("=" * 50)
    if n_total == total_in_subset:
        print(f"Status:                 COMPLETE ({n_total}/{total_in_subset} passages)")
    else:
        print(f"Status:                 IN PROGRESS ({n_total}/{total_in_subset} passages done, "
              f"{total_in_subset - n_total} remaining)")

    print(f"\n-- This run --")
    print(f"Passages processed:     {len(run_passage_times)}")
    if run_passage_times:
        print(f"Wall time:              {format_duration(run_elapsed)}")
        print(f"Fastest passage:        {min(run_passage_times):.1f}s")
        print(f"Slowest passage:        {max(run_passage_times):.1f}s")
        print(f"Mean time/passage:      {sum(run_passage_times)/len(run_passage_times):.1f}s")
        print(f"Empty/failed:           {run_warn_count}")

    print(f"\n-- All extracted data (output/extractions.json) --")
    print(f"Total passages:         {n_total}")
    print(f"Total entities:         {n_entities}")
    print(f"Total relations:        {n_relations}")
    if n_total:
        print(f"Avg entities/passage:   {n_entities/n_total:.2f}")
        print(f"Avg relations/passage:  {n_relations/n_total:.2f}")
    print(f"Empty extractions:      {n_empty}")
    print("=" * 50)
    print(f"Saved  -> {EXTRACTIONS_PATH}")
    print(f"Log    -> {LOG_PATH}")


def main():
    enable_windows_ansi()

    with open(PASSAGES_PATH, encoding="utf-8") as f:
        passages = json.load(f)

    subset_titles = {p["title"] for p in passages}
    results = load_existing(subset_titles)
    done_titles = {r["title"] for r in results}
    remaining = [p for p in passages if p["title"] not in done_titles]

    total = len(passages)
    already_done = total - len(remaining)

    if already_done:
        print(f"Resuming previous run: {already_done}/{total} passages already extracted, "
              f"{len(remaining)} remaining.\n")

    if not remaining:
        print("All passages already extracted -- nothing to do.")
        print_stats(results, [], 0, 0.0, total)
        return

    print(f"Extracting {len(remaining)} passage(s) using model '{MODEL}'...\n")

    run_passage_times = []
    run_warn_count = 0
    run_t0 = time.time()

    # initial bar, ETA unknown until the first passage finishes
    sys.stdout.write(format_bar_line(already_done, total, eta_seconds=0))
    sys.stdout.flush()

    try:
        for i, p in enumerate(remaining, 1):
            t_start = time.time()
            extracted = extract_one(p["title"], p["sentences"])
            t_this = time.time() - t_start
            run_passage_times.append(t_this)

            failed = not extracted["entities"] and not extracted["relations"]
            if failed:
                run_warn_count += 1

            results.append({
                "title": p["title"],
                "entities": extracted["entities"],
                "relations": extracted["relations"],
            })

            # checkpoint + log after EVERY passage, so a rerun never redoes work
            save_extractions(results)
            append_log(
                f"[{already_done + i}/{total}] '{p['title']}' -> "
                f"{len(extracted['entities'])}e/{len(extracted['relations'])}r in {t_this:.1f}s"
                + (" [WARN empty]" if failed else "")
            )

            # ETA from a rolling average of the last ETA_WINDOW passages (adapts to recent speed)
            window = run_passage_times[-ETA_WINDOW:]
            avg_recent = sum(window) / len(window)
            eta = avg_recent * (len(remaining) - i)

            done_total = already_done + i
            log_str = format_passage_line(
                done_total, total, p["title"],
                len(extracted["entities"]), len(extracted["relations"]), t_this,
            )
            if failed:
                log_str += "  [WARN empty]"
            bar_str = format_bar_line(done_total, total, eta)
            redraw(sys.stdout, log_str, bar_str)

    except KeyboardInterrupt:
        sys.stdout.write("\n")
        print("\nInterrupted -- progress up to the last completed passage is already saved.")
        print("Rerun `python extract_kg.py` any time to resume from here.")
        print_stats(results, run_passage_times, run_warn_count, time.time() - run_t0, total)
        sys.exit(1)

    sys.stdout.write("\n")
    print_stats(results, run_passage_times, run_warn_count, time.time() - run_t0, total)


if __name__ == "__main__":
    main()
