"""Paired comparison of two result files on the same questions, with bootstrap confidence intervals.

    python compare_runs.py REFERENCE.json VARIANT.json [--samples 1] [--metrics f1,bleu1,bert_f1]
    python compare_runs.py REF_a.json,REF_b.json VAR_a.json,VAR_b.json      (pools conversations from several files)

Delta = VARIANT - REFERENCE (scale 0-100). A '*' marks a 95% interval that excludes zero.
"""

import argparse
import json
from collections import defaultdict

import numpy as np


def load(path):
    """One result file, or several separated by commas (their questions are pooled)."""
    rows, seen, first = {}, defaultdict(int), None
    for p in path.split(","):
        d = json.load(open(p, encoding="utf-8"))
        first = first or d
        for r in d["individual_results"]:
            base = (r["sample_id"], r["question"])
            seen[base] += 1
            rows[base + (seen[base],)] = r
    return first, rows


def boot_ci(x, n_boot=10000, seed=0):
    x = np.asarray(x, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for s in range(0, n_boot, 1000):
        m = min(1000, n_boot - s)
        idx = rng.integers(0, n, size=(m, n))
        means[s:s + m] = x[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def value(r, metric):
    v = r["metrics"].get(metric)
    return None if v is None else float(v)


def report_group(title, keys, ref_rows, var_rows, metric):
    a, b = [], []
    for k in keys:
        x, y = value(ref_rows[k], metric), value(var_rows[k], metric)
        if x is not None and y is not None:
            a.append(x)
            b.append(y)
    if not a:
        return
    a, b = np.array(a) * 100, np.array(b) * 100
    delta = b - a
    lo, hi = boot_ci(delta)
    star = "*" if (lo > 0 or hi < 0) else " "
    print(f"{title:<16}{metric:<20}{len(a):>5}{a.mean():>9.2f}{b.mean():>9.2f}{delta.mean():>+9.2f}"
          f"   [{lo:+.2f}, {hi:+.2f}] {star}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("variant")
    ap.add_argument("--samples", default=None, help="comma-separated conversation indices")
    ap.add_argument("--metrics", default="f1,bleu1,bert_f1")
    ap.add_argument("--only_categories", default=None, help="comma-separated categories to keep, e.g. 2 or 1,2,4")
    ap.add_argument("--exclude_failed", action="store_true",
                    help="drop question pairs where either side has bert_f1 < 0.5 (empty or failed answers)")
    ap.add_argument("--per_conversation", action="store_true",
                    help="also print the cats 1-4 delta of every conversation separately")
    args = ap.parse_args()

    dref, ref_rows = load(args.reference)
    dvar, var_rows = load(args.variant)
    keys = sorted(set(ref_rows) & set(var_rows))
    if args.samples:
        wanted = {int(x) for x in args.samples.split(",")}
        keys = [k for k in keys if k[0] in wanted]
    if args.only_categories:
        wanted_cats = {int(x) for x in args.only_categories.split(",")}
        keys = [k for k in keys if int(ref_rows[k]["category"]) in wanted_cats]

    def failed(rows, k):
        v = value(rows[k], "bert_f1")
        return v is not None and v < 0.5

    ref_bad = sum(failed(ref_rows, k) for k in keys)
    var_bad = sum(failed(var_rows, k) for k in keys)
    dropped = 0
    if args.exclude_failed:
        keep = [k for k in keys if not failed(ref_rows, k) and not failed(var_rows, k)]
        dropped = len(keys) - len(keep)
        keys = keep
    print(f"reference: {args.reference}  variant={dref.get('variant')}")
    print(f"variant  : {args.variant}  variant={dvar.get('variant')}")
    print(f"questions in common: {len(keys)} (reference {len(ref_rows)}, variant {len(var_rows)})")
    print(f"questions with bert_f1 < 0.5 (likely empty or failed answers): reference {ref_bad}, variant {var_bad}"
          + (f"; {dropped} pairs excluded" if args.exclude_failed else "; use --exclude_failed to drop them"))
    for name, d in (("reference", dref), ("variant", dvar)):
        print(f"{name} llm_failures={d.get('llm_failures')}  variant_stats={d.get('variant_stats')}")
    print()
    print(f"{'group':<16}{'metric':<20}{'n':>5}{'ref':>9}{'var':>9}{'delta':>9}   95% CI (paired bootstrap)")
    print("-" * 90)

    by_cat = defaultdict(list)
    for k in keys:
        by_cat[int(ref_rows[k]["category"])].append(k)
    groups = [("cats 1-4", [k for c in (1, 2, 3, 4) for k in by_cat[c]])]
    groups += [(f"category {c}", by_cat[c]) for c in (1, 2, 3, 4, 5) if by_cat[c]]
    groups += [("all", keys)]
    for metric in args.metrics.split(","):
        for title, ks in groups:
            if title == "category 5" and metric == "f1":
                title = "category 5 (*)"
            report_group(title, ks, ref_rows, var_rows, metric)
        print()
    if args.per_conversation:
        print("per conversation, cats 1-4 (metric: f1):")
        print(f"{'conv':>6}{'n':>6}{'ref':>9}{'var':>9}{'delta':>9}")
        positive = total = 0
        for sid in sorted({k[0] for k in keys}):
            ks = [k for k in keys if k[0] == sid and int(ref_rows[k]["category"]) != 5]
            a = [value(ref_rows[k], "f1") for k in ks if value(ref_rows[k], "f1") is not None]
            b = [value(var_rows[k], "f1") for k in ks if value(var_rows[k], "f1") is not None]
            if a:
                d = (np.mean(b) - np.mean(a)) * 100
                positive += d > 0
                total += 1
                print(f"{sid:>6}{len(a):>6}{np.mean(a) * 100:>9.2f}{np.mean(b) * 100:>9.2f}{d:>+9.2f}")
        print(f"conversations with a positive delta: {positive} of {total}\n")
    if by_cat[5]:
        report_group("category 5", by_cat[5], ref_rows, var_rows, "adversarial_correct")
        print("\n(*) category-5 F1 follows the base-paper protocol; adversarial_correct is the behavioural measure.")


if __name__ == "__main__":
    main()
