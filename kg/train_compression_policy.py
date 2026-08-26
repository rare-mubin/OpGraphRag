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
--n-questions * --group-size * --epochs * 2 calls, each several seconds.
Defaults are kept small; increase deliberately.

Usage:
    python train_compression_policy.py --n-questions 3 --group-size 2 --epochs 1   # quick mechanism test
    python train_compression_policy.py --group-size 4 --epochs 3                    # fuller (but still small-data) run
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
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


def load_node_embedding_lookup():
    embeddings, ids = load_embeddings()
    return {nid: embeddings[i] for i, nid in enumerate(ids)}


def compute_reward(A_c: str, gt: str, T_c: int, T_o: int, Q_o: float,
                    lambda_cr: float, gamma_penalty: float) -> dict:
    Q_c = f1_score(A_c, gt)
    CR = 1.0 - (T_c / T_o) if T_o > 0 else 0.0
    P = max(0.0, Q_o - Q_c)
    # The CR bonus is gated on not losing quality (Q_c >= Q_o), not given unconditionally.
    # Ungated, it made "empty the graph" a dominant strategy for two reasons specific to this
    # LLM+dataset: (1) ANSWER_SYSTEM_PROMPT tells the model to guess rather than refuse, and
    # Qwen2.5-7B often answers HotpotQA's Wikipedia-famous entities correctly from parametric
    # knowledge alone even with zero graph context -- so compressing to nothing could still get
    # Q_c ~= Q_o (P~=0) *and* collect the full CR bonus on top, a strictly better reward than
    # keeping full context and getting the same answer right. (2) when the baseline was already
    # wrong (Q_o low), P stays ~0 even if the compressed answer is also wrong, so emptying the
    # graph was free positive reward regardless of correctness. Both made "remove everything"
    # the group-relative-advantage-favored move under GRPO, which is what collapsed the policy.
    cr_bonus = lambda_cr * CR if Q_c >= Q_o - 1e-6 else 0.0
    R = Q_c + cr_bonus - gamma_penalty * P
    return {"Q_c": Q_c, "CR": CR, "P": P, "R": R, "T_c": T_c}


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
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--device", default=None, help="cuda or cpu for the embedding model")
    ap.add_argument("--seed", type=int, default=42)
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

    n_calls = len(questions) * args.group_size * args.epochs * 2
    print(f"Training: {len(questions)} questions x {args.group_size} trajectories x {args.epochs} epoch(s) "
          f"= {len(questions) * args.group_size * args.epochs} trajectories (~{n_calls} LLM calls)\n")

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
    total_steps = len(questions) * args.epochs

    for epoch in range(1, args.epochs + 1):
        for qi, q in enumerate(questions, 1):
            step += 1
            question = q["question"]
            gt = q["answer"]
            base = baseline[question]
            T_o = base["T_o"]
            Q_o = base["f1"]

            query_emb = embed_model.encode([question], normalize_embeddings=True)[0]
            Gq, seeds = retrieve(question, G, embeddings, ids, embed_model, top_k=args.top_k, hops=args.hops)
            seed_ids = {s for s, _ in seeds}

            if Gq.number_of_nodes() == 0:
                print(f"  [{step}/{total_steps}] '{question[:50]}' -> empty G_q, skipping")
                continue

            node_ids, features = build_state_features(Gq, query_emb, node_emb_lookup, seed_ids)
            features_t = torch.tensor(features)

            trajectories = []  # (actions, log_prob, entropy, reward_info, A_c)
            t_start = time.time()
            for _ in range(args.group_size):
                actions, log_prob, entropy = sample_trajectory(policy, features_t)
                Gc = compress_graph(Gq, node_ids, actions)
                context = build_context(Gc, passages_by_title)
                T_c = count_tokens(context)
                raw = generate_answer(question, context)
                A_c = shorten_answer(question, raw)
                reward_info = compute_reward(A_c, gt, T_c, T_o, Q_o, args.lambda_cr, args.gamma_penalty)
                trajectories.append((actions, log_prob, entropy, reward_info, A_c))

            rewards = np.array([t[3]["R"] for t in trajectories])
            mean_r, std_r = rewards.mean(), rewards.std()
            advantages = (rewards - mean_r) / (std_r + 1e-8)

            pg_loss = torch.stack([
                -adv * log_prob for (_, log_prob, _, _, _), adv in zip(trajectories, advantages)
            ]).mean()
            mean_entropy = torch.stack([t[2] for t in trajectories]).mean()
            # subtract an entropy bonus (i.e. reward higher entropy) so the policy doesn't
            # collapse to always-keep/always-remove before it's seen enough reward signal
            loss = pg_loss - args.entropy_coef * mean_entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_this = time.time() - t_start
            avg_R = rewards.mean()
            avg_CR = np.mean([t[3]["CR"] for t in trajectories])
            best_idx = int(rewards.argmax())
            history.append({
                "epoch": epoch, "question": question, "Q_o": Q_o, "T_o": T_o,
                "rewards": rewards.tolist(), "mean_reward": float(avg_R), "mean_CR": float(avg_CR),
                "mean_entropy": float(mean_entropy.item()),
                "best_A_c": trajectories[best_idx][4], "best_R": float(rewards[best_idx]),
                "loss": float(loss.item()), "pg_loss": float(pg_loss.item()),
            })

            avg_step_time = (time.time() - t0) / step
            eta = avg_step_time * (total_steps - step)
            print(f"  [{step}/{total_steps}] epoch {epoch} '{question[:45]}' "
                  f"mean_R={avg_R:.3f} mean_CR={avg_CR:.2f} entropy={mean_entropy.item():.2f} "
                  f"loss={loss.item():.3f} ({t_this:.1f}s, ETA {eta:.0f}s)")

    torch.save({"state_dict": policy.state_dict(), "input_dim": input_dim,
                "args": vars(args)}, POLICY_PATH)
    with open(TRAINING_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\nSaved policy -> {POLICY_PATH}")
    print(f"Saved training log -> {TRAINING_LOG_PATH}")
    if history:
        print(f"Mean reward, first step: {history[0]['mean_reward']:.3f}  "
              f"last step: {history[-1]['mean_reward']:.3f}")


if __name__ == "__main__":
    main()
