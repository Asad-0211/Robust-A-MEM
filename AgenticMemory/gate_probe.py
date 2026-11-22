"""Can retrieval confidence separate adversarial (category 5) questions from answerable ones? No LLM needed.

    python gate_probe.py results_full_noevo.json cached_memories_robust_ollama_qwen2.5-3b_noevo

Uses each question's text (not the LLM-generated keywords) against the cached note embeddings of its conversation.
AUROC 0.5 = no separation; above 0.6 there is some signal for a gate.
"""

import json
import sys
from collections import defaultdict

import numpy as np


def auroc(pos, neg):
    """Probability that a random score from `pos` exceeds one from `neg` (ties count half)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    srt = allv[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))


def main():
    if sys.argv[1] == "--selftest":
        assert abs(auroc([3, 4, 5], [1, 2, 3]) - 8.5 / 9) < 1e-9
        assert auroc([1, 1], [1, 1]) == 0.5 and auroc([2], [1]) == 1.0
        print("auroc self-test passed")
        return
    results, cache = sys.argv[1], sys.argv[2]
    from sentence_transformers import SentenceTransformer
    rows = json.load(open(results, encoding="utf-8"))["individual_results"]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    feats = defaultdict(lambda: defaultdict(list))        # feature -> category -> scores
    by_sample = defaultdict(list)
    for r in rows:
        by_sample[r["sample_id"]].append(r)
    for sid, rs in sorted(by_sample.items()):
        emb = np.load(f"{cache}/retriever_cache_embeddings_sample_{sid}.npy")
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        q = model.encode([r["question"] for r in rs])
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        sims = np.sort(q @ emb.T, axis=1)[:, ::-1]
        for r, s in zip(rs, sims):
            c = int(r["category"])
            feats["top1"][c].append(s[0])
            feats["top3_mean"][c].append(s[:3].mean())
            feats["top1_minus_top5"][c].append(s[0] - s[4])
    print(f"{'feature':<18}{'cat5 mean':>10}{'cats1-4':>9}{'AUROC(answerable > cat5)':>26}   per category vs cat 5")
    for f, byc in feats.items():
        ans = [x for c in (1, 2, 3, 4) for x in byc[c]]
        parts = "  ".join(f"c{c}:{auroc(byc[c], byc[5]):.2f}" for c in (1, 2, 3, 4) if byc[c])
        print(f"{f:<18}{np.mean(byc[5]):>10.3f}{np.mean(ans):>9.3f}{auroc(ans, byc[5]):>26.3f}   {parts}")


if __name__ == "__main__":
    main()
