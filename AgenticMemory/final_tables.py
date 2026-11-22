"""Print every results table of the study from the result files (scale 0-100).

    python final_tables.py

Configurations (a missing file is skipped with a note):
    E2        results_full_qwen3b.json                                      baseline, paper pipeline in our setup
    NoEvo     results_full_noevo.json                                       no evolution
    +datefix  results_full_noevo_datefix.json                               no evolution + date resolution
    +hybrid   results_pilot_hybrid50.json,results_hybrid50_rest.json        final system (pilot + confirmation files)
Open-domain at k=50:
    E3        results_cat3_k50.json
    NoEvo+DF  results_cat3_k50_noevo_datefix.json
    +hybrid   results_cat3_k50_hybrid.json
"""

import json
import os
from collections import defaultdict

import numpy as np

PAPER = {  # base paper, Qwen2.5-3B, A-MEM: category -> (F1, BLEU-1)
    1: (12.57, 9.01), 2: (27.59, 25.07), 3: (7.12, 7.28), 4: (17.23, 13.12), 5: (27.91, 25.15)}
CATS = {1: "Multi-hop", 2: "Temporal", 3: "Open-domain", 4: "Single-hop", 5: "Adversarial*"}

MAIN = [("E2", "results_full_qwen3b.json"),
        ("NoEvo", "results_full_noevo.json"),
        ("+datefix", "results_full_noevo_datefix.json"),
        ("+hybrid", "results_pilot_hybrid50.json,results_hybrid50_rest.json")]
K50 = [("E3", "results_cat3_k50.json"),
       ("NoEvo+DF", "results_cat3_k50_noevo_datefix.json"),
       ("+hybrid", "results_cat3_k50_hybrid.json")]


def load(spec):
    rows = []
    for p in spec.split(","):
        if not os.path.exists(p):
            return None
        rows += json.load(open(p, encoding="utf-8"))["individual_results"]
    return rows


def mean(rows, metric, cats=None, sample=None):
    v = [float(r["metrics"][metric]) for r in rows
         if metric in r["metrics"] and (cats is None or int(r["category"]) in cats)
         and (sample is None or r["sample_id"] == sample)]
    return 100 * float(np.mean(v)) if v else float("nan")


def count(rows, cat):
    return sum(1 for r in rows if int(r["category"]) == cat)


def table(title, configs, metric, extra_paper=None):
    data = {n: load(s) for n, s in configs}
    for n, d in data.items():
        if d is None:
            print(f"(skipped {n}: file not found)")
    names = [n for n in data if data[n] is not None]
    print(f"\n=== {title} ===")
    head = f"{'category':<14}{'n':>6}" + (f"{'paper':>9}" if extra_paper else "") + "".join(f"{n:>11}" for n in names)
    print(head)
    print("-" * len(head))
    for c in (1, 2, 3, 4, 5):
        n = count(data[names[0]], c) if names else 0
        if n == 0:
            continue
        row = f"{CATS[c]:<14}{n:>6}"
        if extra_paper:
            row += f"{extra_paper[c]:>9.2f}"
        print(row + "".join(f"{mean(data[nm], metric, {c}):>11.2f}" for nm in names))
    cat_means = lambda nm: np.mean([mean(data[nm], metric, {c}) for c in (1, 2, 3, 4) if count(data[nm], c)])
    row = f"{'Mean cats 1-4':<14}{'':>6}"
    if extra_paper:
        row += f"{np.mean([extra_paper[c] for c in (1, 2, 3, 4)]):>9.2f}"
    print(row + "".join(f"{cat_means(nm):>11.2f}" for nm in names) + "   (mean of category means)")
    row = f"{'Pooled 1-4':<14}{'':>6}" + (f"{'':>9}" if extra_paper else "")
    print(row + "".join(f"{mean(data[nm], metric, {1, 2, 3, 4}):>11.2f}" for nm in names) + "   (all questions pooled)")


def abstention():
    print("\n=== Correct abstention, category 5 (%) ===")
    for n, s in MAIN:
        d = load(s)
        if d is None:
            print(f"(skipped {n}: file not found)")
            continue
        v = [float(r["metrics"]["adversarial_correct"]) for r in d if "adversarial_correct" in r["metrics"]]
        print(f"{n:<12}{100 * np.mean(v):>8.2f}   (n={len(v)})")


def per_conversation():
    data = {n: load(s) for n, s in MAIN}
    names = [n for n in data if data[n] is not None]
    print("\n=== Cats 1-4 F1 per conversation ===")
    print(f"{'conv':>5}" + "".join(f"{n:>11}" for n in names))
    for sid in range(10):
        print(f"{sid:>5}" + "".join(f"{mean(data[n], 'f1', {1, 2, 3, 4}, sid):>11.2f}" for n in names))


def main():
    table("F1 (paper-matched k=10 except where noted)", MAIN, "f1", {c: PAPER[c][0] for c in PAPER})
    table("BLEU-1", MAIN, "bleu1", {c: PAPER[c][1] for c in PAPER})
    table("BERTScore F1", MAIN, "bert_f1")
    abstention()
    per_conversation()
    table("Open-domain at k=50: F1", K50, "f1", {3: 7.12, 1: 0, 2: 0, 4: 0, 5: 0})
    table("Open-domain at k=50: BLEU-1", K50, "bleu1", {3: 7.28, 1: 0, 2: 0, 4: 0, 5: 0})
    print("\n* Category-5 F1 and BLEU-1 follow the base-paper protocol; use correct abstention for behaviour.")
    print("Open-domain in the main tables is at k=10; the paper used k=50 for that category.")


if __name__ == "__main__":
    main()
