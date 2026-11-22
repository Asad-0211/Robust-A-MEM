"""What did the answering model actually see? Reads an evaluation log written by the driver (logs/eval_robust_*.log).

    python exposure_audit.py logs\\eval_robust_....log [context_window_tokens]

Token counts are estimated as characters / 4 and are only indicative (template text and punctuation use more tokens
per character, so the estimate is a lower bound).
"""

import re
import sys
from collections import defaultdict

TEMPLATE = re.compile(r"updated context sentence|a single sentence summarizing|\b(updated[ _])?tag ?\d\b", re.I)
STAMP = re.compile(r"(?m)^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} - \w+ - ")


def main():
    path = sys.argv[1]
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    text = open(path, encoding="utf-8", errors="replace").read()
    messages = STAMP.split(text)
    rows, cur = [], {}
    for m in messages:
        if m.startswith("Question "):
            cur = {}
        elif m.startswith("User Prompt: "):
            cur["prompt_chars"] = len(m) - len("User Prompt: ")
        elif m.startswith("Category: "):
            cur["category"] = int(m[len("Category: "):].strip()[:1])
        elif m.startswith("Raw Context: "):
            ctx = m[len("Raw Context: "):]
            segs = [s.strip() for s in ctx.split("talk start time:") if s.strip()]
            cur["notes"] = len(segs)
            cur["template_notes"] = sum(1 for s in segs if TEMPLATE.search(s))
            cur["duplicates"] = len(segs) - len(set(segs))
            if "category" in cur and "prompt_chars" in cur:
                rows.append(cur)
            cur = {}
    if not rows:
        print("No questions found in the log.")
        return
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    by_cat[0] = rows
    print(f"{path}\ncontext window assumed: {window} tokens\n")
    print(f"{'category':<10}{'questions':>10}{'notes/ctx':>11}{'template%':>11}{'dup notes':>11}"
          f"{'~tokens':>9}{'max ~tok':>10}{'>80% win':>10}{'>100% win':>11}")
    for c in sorted(by_cat):
        rs = by_cat[c]
        n = len(rs)
        notes = sum(r["notes"] for r in rs)
        tmpl = sum(r["template_notes"] for r in rs)
        toks = [r["prompt_chars"] / 4 for r in rs]
        print(f"{('all' if c == 0 else c):<10}{n:>10}{notes / n:>11.1f}{100 * tmpl / max(notes, 1):>10.1f}%"
              f"{sum(r['duplicates'] for r in rs) / n:>11.2f}{sum(toks) / n:>9.0f}{max(toks):>10.0f}"
              f"{100 * sum(t > 0.8 * window for t in toks) / n:>9.1f}%{100 * sum(t > window for t in toks) / n:>10.1f}%")


if __name__ == "__main__":
    main()
