"""
Stage 4, part 2 (methodology Section III-D/III-E): GRPO training loop for the
query-adaptive graph compression policy.

For each question: retrieve G_q (Stage 2), sample a GROUP of --group-size
compression trajectories from the current policy, generate a compressed
answer A_c for each (reusing Stage 3's two-step generate+shorten), compute
the joint reward R = Q_c + lambda*CR - gamma*P (Eq. in Section III-E), turn
group rewards into a GRPO advantage (Eq. 2), and update the policy via
policy-gradient: loss = -mean(advantage_i * log_prob_i).

This is a proof-of-concept training loop on the current small (~20 question)
subset -- deliberately not expected to generalize yet; the point is to
validate the mechanism (state -> policy -> sampled compression -> real LLM
reward -> GRPO update) works correctly end-to-end before scaling up data.

Cost warning: each trajectory costs ~2 real LLM calls (generate + shorten).
--n-questions * --group-size * --epochs * 2 calls, each several seconds. A
question's --group-size trajectories run concurrently (see --workers), but
questions/epochs themselves are still sequential (each ends in a GRPO
gradient step that depends on that question's own group of results).
Defaults are kept small; increase deliberately.

Usage:
    python train_compression_policy.py --n-questions 3 --group-size 2 --epochs 1   # quick mechanism test
    python train_compression_policy.py --group-size 4 --epochs 3                    # fuller (but still small-data) run
    python train_compression_policy.py --group-size 4 --epochs 3 --workers 8        # more concurrent LLM calls per group
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from retrieve import load_graph, load_embeddings, retrieve, MODEL_NAME, default_device
from generate_baseline_answers import (
    build_context, generate_answer, shorten_answer, f1_score, load_token_counter,
)
from compression_policy import (
    build_state_features, CompressionPolicy, sample_trajectory, compress_graph, EXTRA_FEATURES,
)

OUT_DIR = Path(__file__).resolve().parent / "output"
POLICY_PATH = OUT_DIR / "compression_policy.pt"
TRAINING_LOG_PATH = OUT_DIR / "compression_training_log.json"
NO_CONTEXT_PATH = OUT_DIR / "no_context_answers.json"


def relevance_labels(Gq, node_ids: list, supporting_facts) -> np.ndarray:
    """Ground-truth per-node KEEP label: 1 if the node came from one of HotpotQA's own
    supporting_facts passages for this question. Identical to the labelling in
    evaluate_policy_classification.py -- kept in sync deliberately, since the auxiliary
    loss below trains on exactly the labels that script evaluates against (which is why
    a train/val split is mandatory now; see --val-frac)."""
    needed = {t for t, _ in (supporting_facts or [])}
    return np.array([
        1.0 if set(Gq.nodes[nid].get("sources", [])) & needed else 0.0
        for nid in node_ids
    ], dtype=np.float32)


def load_node_embedding_lookup():
    embeddings, ids = load_embeddings()
    return {nid: embeddings[i] for i, nid in enumerate(ids)}


def compute_reward(A_c: str, gt: str, T_c: int, T_o: int, Q_o: float,
                    lambda_cr: float, gamma_penalty: float, Q_empty: float = 0.0) -> dict:
    Q_c = f1_score(A_c, gt)
    CR = 1.0 - (T_c / T_o) if T_o > 0 else 0.0
    P = max(0.0, Q_o - Q_c)
    # The CR bonus is gated on not losing quality (Q_c >= Q_o) AND on the compressed answer
    # having earned some real credit (Q_c > 0) -- not given unconditionally, and not given just
    # for "not losing" an already-zero baseline. Ungated, it made "empty the graph" a dominant
    # strategy for two reasons specific to this LLM+dataset: (1) ANSWER_SYSTEM_PROMPT tells the
    # model to guess rather than refuse, and Qwen2.5-7B often answers HotpotQA's Wikipedia-famous
    # entities correctly from parametric knowledge alone even with zero graph context -- so
    # compressing to nothing could still get Q_c ~= Q_o (P~=0) *and* collect the full CR bonus on
    # top, a strictly better reward than keeping full context and getting the same answer right.
    # (2) when the baseline was already wrong (Q_o low), P stays ~0 even if the compressed answer
    # is also wrong, so emptying the graph was free positive reward regardless of correctness.
    #
    # The first fix for this (Q_c >= Q_o alone, no Q_c > 0 requirement) closed case (1) but NOT
    # case (2): when Q_o == 0 and the compressed answer is also wrong (Q_c == 0), "0 >= 0" is
    # still trivially true, so the bonus was still granted for free. This mattered little on a
    # 20-question pilot (few questions had Q_o == 0) but became the DOMINANT case at 200-question
    # scale (59.5% of training steps had Q_o == 0, since a much more representative/harder
    # question mix has a much weaker baseline on average). Requiring Q_c > 0 closes that
    # "0 >= 0" loophole.
    #
    # But adding Q_c > 0 did NOT stop the collapse either (200q rerun: CR 99.9%, G_c empty on
    # 200/200 questions), because the deepest problem is not a loophole at all -- it is that
    # R = Q_c + lambda*CR has its GLOBAL MAXIMUM (1.5) at exactly "empty graph + correct answer",
    # and this LLM reaches that constantly from parametric knowledge alone. 30 training steps had
    # Q_o == 0 (full 30k-token context answered WRONG) while a near-empty trajectory scored
    # R > 1.2 by answering correctly -- e.g. "Which 'Roseanne' star is in Scream 2?" -> full
    # context F1 0.0, near-empty context -> "Laurie Metcalf", F1 1.0. No gate on Q_c can catch
    # that: Q_c is genuinely 1.0, the reward is honestly reporting that removing the context
    # helped. So the reward was rewarding the right thing for the wrong reason -- crediting the
    # COMPRESSION for an answer the model already knew without any graph at all.
    #
    # Fix: gate the bonus on beating a per-question no-context control Q_empty (the F1 this same
    # model scores on this same question with an EMPTY context, measured once and cached in
    # output/no_context_answers.json). An empty G_c scores Q_c == Q_empty by construction, so it
    # can never earn the bonus -- compression is now only paid for context that demonstrably adds
    # something over what the model already knew. NOTE: this is a real deviation from main.tex's
    # stated reward equation (Section III-E) and needs to be reflected there.
    cr_bonus = (lambda_cr * CR
                if (Q_c >= Q_o - 1e-6 and Q_c > Q_empty + 1e-6)
                else 0.0)
    R = Q_c + cr_bonus - gamma_penalty * P
    return {"Q_c": Q_c, "CR": CR, "P": P, "R": R, "T_c": T_c, "Q_empty": Q_empty}


def load_or_build_no_context_baseline(questions: list, path: Path) -> dict:
    """Q_empty per question: what this model scores with an EMPTY context.

    This is the control the CR bonus is gated against (see compute_reward). Costs 2 LLM
    calls per question, once -- cached to disk and checkpointed after every question, so
    it is safe to interrupt and rerun. Reused across training runs; delete the file to
    force a refresh.
    """
    cache = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            cache = json.load(f)

    todo = [q for q in questions if q["question"] not in cache]
    if not todo:
        print(f"No-context control: all {len(questions)} questions cached ({path.name})")
        return cache

    print(f"No-context control (Q_empty): {len(todo)} question(s) to measure "
          f"({len(cache)} already cached) -- ~2 LLM calls each")
    t0 = time.time()
    for i, q in enumerate(todo, 1):
        question = q["question"]
        raw = generate_answer(question, "")
        A_empty = shorten_answer(question, raw)
        Q_empty = f1_score(A_empty, q["answer"])
        cache[question] = {"A_empty": A_empty, "Q_empty": Q_empty}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        elapsed = time.time() - t0
        eta = (elapsed / i) * (len(todo) - i)
        print(f"  [{i}/{len(todo)}] Q_empty={Q_empty:.2f} '{A_empty[:40]}' "
              f"({elapsed / i:.1f}s/q, ETA {eta:.0f}s)")

    mean_qe = sum(v["Q_empty"] for v in cache.values()) / max(len(cache), 1)
    print(f"No-context control done: mean Q_empty = {mean_qe:.3f} -> {path}\n")
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-questions", type=int, default=None, help="limit to first N questions (default: all)")
    ap.add_argument("--group-size", type=int, default=4, help="GRPO trajectories sampled per question")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-cr", type=float, default=0.5, help="weight on compression ratio in the reward")
    ap.add_argument("--gamma-penalty", type=float, default=2.0, help="weight on the degradation penalty")
    ap.add_argument("--entropy-coef", type=float, default=0.02,
                     help="entropy bonus weight -- discourages the policy from collapsing to an "
                          "always-keep/always-remove degenerate strategy, especially likely on a "
                          "small dataset with few gradient steps")
    ap.add_argument("--aux-coef", type=float, default=1.0,
                     help="weight on the auxiliary per-node supervised loss (BCE against the "
                          "supporting_facts relevance labels). This is the term that actually "
                          "teaches per-node discrimination: GRPO gives ONE scalar advantage for a "
                          "whole multi-node action, and in the previous 600-step run 92%% of steps "
                          "had advantage exactly 0 (every trajectory in the group sampled "
                          "identically), so the RL term alone supplied no gradient at all. Set 0 "
                          "to reproduce the old pure-GRPO behaviour.")
    ap.add_argument("--val-frac", type=float, default=0.2,
                     help="fraction of questions held out for validation. Mandatory once "
                          "--aux-coef > 0: the auxiliary loss trains on the same supporting_facts "
                          "labels evaluate_policy_classification.py scores against, so metrics on "
                          "trained-on questions are circular. The split is saved into the policy "
                          "checkpoint so downstream eval can restrict to held-out questions.")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--device", default=None, help="cuda or cpu for the embedding model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4,
                     help="concurrent generate+shorten LLM calls against Ollama, within a single "
                          "question's group of trajectories (default: 4, matches build_graph.py's "
                          "--workers). Each trajectory's 2 calls are independent of the others in "
                          "its group -- only policy sampling and the GRPO update stay sequential. "
                          "Raise if your GPU has headroom, lower if requests queue up with no "
                          "speedup or Ollama errors under load.")
    args = ap.parse_args()
    if args.device is None:
        args.device = default_device()

    torch.manual_seed(args.seed)

    with open(OUT_DIR / "subset_questions.json", encoding="utf-8") as f:
        questions = json.load(f)
    with open(OUT_DIR / "subset_passages.json", encoding="utf-8") as f:
        passages = json.load(f)
    passages_by_title = {p["title"]: p["sentences"] for p in passages}
    with open(OUT_DIR / "baseline_answers.json", encoding="utf-8") as f:
        baseline = {r["question"]: r for r in json.load(f)}

    if args.n_questions:
        questions = questions[: args.n_questions]

    missing = [q["question"] for q in questions if q["question"] not in baseline]
    if missing:
        print(f"[WARN] {len(missing)} question(s) have no Stage 3 baseline (T_o/Q_o) -- skipping them. "
              f"Rerun generate_baseline_answers.py if this is unexpected.")
        questions = [q for q in questions if q["question"] not in missing]

    # --- train/val split (before any training touches the labels) ---
    shuffled = list(questions)
    random.Random(args.seed).shuffle(shuffled)
    n_val = int(round(len(shuffled) * args.val_frac))
    val_questions = shuffled[:n_val]
    train_questions = shuffled[n_val:]
    val_set = {q["question"] for q in val_questions}
    print(f"Split: {len(train_questions)} train / {len(val_questions)} val "
          f"(--val-frac {args.val_frac}, seed {args.seed})")

    n_calls = len(train_questions) * args.group_size * args.epochs * 2
    print(f"Training: {len(train_questions)} questions x {args.group_size} trajectories x {args.epochs} epoch(s) "
          f"= {len(train_questions) * args.group_size * args.epochs} trajectories (~{n_calls} LLM calls)\n")

    no_context = load_or_build_no_context_baseline(questions, NO_CONTEXT_PATH)

    G = load_graph()
    node_emb_lookup = load_node_embedding_lookup()
    embeddings, ids = load_embeddings()

    print(f"Loading {MODEL_NAME} on {args.device}...")
    embed_model = SentenceTransformer(MODEL_NAME, device=args.device)
    count_tokens = load_token_counter()

    emb_dim = embeddings.shape[1]
    input_dim = 2 * emb_dim + EXTRA_FEATURES
    policy = CompressionPolicy(input_dim=input_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    history = []
    t0 = time.time()
    step = 0
    total_steps = len(train_questions) * args.epochs

    executor = ThreadPoolExecutor(max_workers=args.workers)

    def answer_and_reward(question: str, gt: str, T_o: int, Q_o: float, context: str, T_c: int,
                           Q_empty: float) -> tuple:
        raw = generate_answer(question, context)
        A_c = shorten_answer(question, raw)
        reward_info = compute_reward(A_c, gt, T_c, T_o, Q_o, args.lambda_cr, args.gamma_penalty,
                                      Q_empty)
        return reward_info, A_c

    def node_states(q):
        """Shared by training and validation: (Gq, node_ids, features tensor, labels tensor).
        Returns None when retrieval comes back empty."""
        query_emb = embed_model.encode([q["question"]], normalize_embeddings=True)[0]
        Gq, seeds = retrieve(q["question"], G, embeddings, ids, embed_model,
                              top_k=args.top_k, hops=args.hops)
        if Gq.number_of_nodes() == 0:
            return None
        seed_ids = {s for s, _ in seeds}
        node_ids, features = build_state_features(Gq, query_emb, node_emb_lookup, seed_ids)
        labels = relevance_labels(Gq, node_ids, q.get("supporting_facts"))
        return Gq, node_ids, torch.tensor(features), torch.tensor(labels)

    @torch.no_grad()
    def evaluate_split(split_questions, name):
        """Node-level accuracy / KEEP-rate on a split. No LLM calls -- pure local inference,
        so this is cheap enough to run after every epoch."""
        policy.eval()
        tp = fp = fn = tn = 0
        for q in split_questions:
            st = node_states(q)
            if st is None:
                continue
            _, _, feats, labels = st
            pred = (torch.sigmoid(policy(feats)) >= 0.5).float()
            tp += int(((pred == 1) & (labels == 1)).sum())
            fp += int(((pred == 1) & (labels == 0)).sum())
            fn += int(((pred == 0) & (labels == 1)).sum())
            tn += int(((pred == 0) & (labels == 0)).sum())
        policy.train()
        total = tp + fp + fn + tn
        if total == 0:
            return {}
        acc = (tp + tn) / total
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        keep_rate = (tp + fp) / total
        print(f"    [{name}] nodes={total} acc={acc:.3f} precision={prec:.3f} "
              f"recall={rec:.3f} KEEP-rate={keep_rate:.3f}")
        return {"split": name, "n_nodes": total, "accuracy": acc, "precision": prec,
                "recall": rec, "keep_rate": keep_rate, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    val_history = []

    for epoch in range(1, args.epochs + 1):
        for qi, q in enumerate(train_questions, 1):
            step += 1
            question = q["question"]
            gt = q["answer"]
            base = baseline[question]
            T_o = base["T_o"]
            Q_o = base["f1"]
            Q_empty = no_context.get(question, {}).get("Q_empty", 0.0)

            st = node_states(q)
            if st is None:
                print(f"  [{step}/{total_steps}] '{question[:50]}' -> empty G_q, skipping")
                continue
            Gq, node_ids, features_t, labels_t = st

            # Policy sampling stays sequential (fast, local torch ops, not the bottleneck) --
            # only the group's LLM round-trips are dispatched concurrently below, since each
            # trajectory's compressed graph/context is already fully determined by this point
            # and independent of every other trajectory in the group.
            sampled = []  # (actions, log_prob, entropy, context, T_c)
            for _ in range(args.group_size):
                actions, log_prob, entropy = sample_trajectory(policy, features_t)
                Gc = compress_graph(Gq, node_ids, actions)
                context = build_context(Gc, passages_by_title)
                T_c = count_tokens(context)
                sampled.append((actions, log_prob, entropy, context, T_c))

            t_start = time.time()
            answers = list(executor.map(
                lambda s: answer_and_reward(question, gt, T_o, Q_o, s[3], s[4], Q_empty), sampled))
            trajectories = [  # (actions, log_prob, entropy, reward_info, A_c)
                (actions, log_prob, entropy, reward_info, A_c)
                for (actions, log_prob, entropy, _, _), (reward_info, A_c) in zip(sampled, answers)
            ]

            rewards = np.array([t[3]["R"] for t in trajectories])
            mean_r, std_r = rewards.mean(), rewards.std()
            advantages = (rewards - mean_r) / (std_r + 1e-8)

            pg_loss = torch.stack([
                -adv * log_prob for (_, log_prob, _, _, _), adv in zip(trajectories, advantages)
            ]).mean()
            mean_entropy = torch.stack([t[2] for t in trajectories]).mean()

            # Auxiliary per-node supervised loss. GRPO hands one scalar advantage to a whole
            # multi-node action, which cannot express "keep THIS node, drop THAT one" -- and
            # empirically it often expresses nothing at all: in the previous run 551 of 600 steps
            # had every trajectory in the group sampling identically, so reward std was 0, the
            # advantage was 0, and pg_loss was exactly 0.000. This term is what supplies gradient
            # on those steps. pos_weight rebalances the ~10:1 irrelevant:relevant class ratio so
            # "predict REMOVE for everything" stops being a good solution to the BCE objective too.
            n_pos = float(labels_t.sum())
            if args.aux_coef > 0 and 0 < n_pos < len(labels_t):
                pos_weight = torch.tensor((len(labels_t) - n_pos) / n_pos)
                aux_loss = F.binary_cross_entropy_with_logits(
                    policy(features_t), labels_t, pos_weight=pos_weight)
            else:
                aux_loss = torch.zeros(())

            # subtract an entropy bonus (i.e. reward higher entropy) so the policy doesn't
            # collapse to always-keep/always-remove before it's seen enough reward signal
            loss = pg_loss - args.entropy_coef * mean_entropy + args.aux_coef * aux_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_this = time.time() - t_start
            avg_R = rewards.mean()
            avg_CR = np.mean([t[3]["CR"] for t in trajectories])
            best_idx = int(rewards.argmax())
            history.append({
                "epoch": epoch, "question": question, "Q_o": Q_o, "T_o": T_o, "Q_empty": Q_empty,
                "rewards": rewards.tolist(), "mean_reward": float(avg_R), "mean_CR": float(avg_CR),
                "mean_entropy": float(mean_entropy.item()),
                "best_A_c": trajectories[best_idx][4], "best_R": float(rewards[best_idx]),
                "loss": float(loss.item()), "pg_loss": float(pg_loss.item()),
                "aux_loss": float(aux_loss.item()), "advantage_std": float(std_r),
                "n_nodes": len(node_ids), "n_relevant": int(n_pos),
            })

            avg_step_time = (time.time() - t0) / step
            eta = avg_step_time * (total_steps - step)
            print(f"  [{step}/{total_steps}] epoch {epoch} '{question[:45]}' "
                  f"mean_R={avg_R:.3f} mean_CR={avg_CR:.2f} entropy={mean_entropy.item():.2f} "
                  f"aux={aux_loss.item():.3f} loss={loss.item():.3f} ({t_this:.1f}s, ETA {eta:.0f}s)")

        print(f"  --- end of epoch {epoch}: node-level decisions (no LLM calls) ---")
        ep_train = evaluate_split(train_questions, f"epoch{epoch} train")
        ep_val = evaluate_split(val_questions, f"epoch{epoch} val")
        val_history.append({"epoch": epoch, "train": ep_train, "val": ep_val})

    executor.shutdown(wait=True)

    torch.save({"state_dict": policy.state_dict(), "input_dim": input_dim,
                "args": vars(args),
                # Saved so downstream eval can restrict to questions the auxiliary loss never
                # trained on -- without this, evaluate_policy_classification.py scores the policy
                # against the very labels it was trained on and the numbers mean nothing.
                "train_questions": [q["question"] for q in train_questions],
                "val_questions": [q["question"] for q in val_questions]}, POLICY_PATH)
    with open(TRAINING_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump({"steps": history, "validation": val_history}, f, indent=2, ensure_ascii=False)

    print(f"\nSaved policy -> {POLICY_PATH}")
    print(f"Saved training log -> {TRAINING_LOG_PATH}")
    if history:
        print(f"Mean reward, first step: {history[0]['mean_reward']:.3f}  "
              f"last step: {history[-1]['mean_reward']:.3f}")
        dead = sum(1 for h in history if h["advantage_std"] < 1e-9)
        print(f"Steps with zero reward spread (no RL gradient): {dead}/{len(history)} "
              f"({100 * dead / len(history):.0f}%) -- the auxiliary loss is what trains on these.")
    if val_history and val_history[-1].get("val"):
        v = val_history[-1]["val"]
        print(f"Final held-out val: acc={v['accuracy']:.3f} precision={v['precision']:.3f} "
              f"recall={v['recall']:.3f} KEEP-rate={v['keep_rate']:.3f}")


if __name__ == "__main__":
    main()
