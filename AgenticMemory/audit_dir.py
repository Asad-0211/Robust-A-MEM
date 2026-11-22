"""Template-text and link statistics of the memory caches in one cache folder.

    python audit_dir.py cached_memories_robust_ollama_qwen2.5-3b_bt0
"""

import collections
import glob
import pickle
import re
import sys

import amem_variants as V

_FMT = re.compile(r"[\[\]]|^['\"]|['\"]$")   # stray brackets or quote marks at the ends of a tag
d = sys.argv[1]
files = sorted(glob.glob(d + "/memory_cache_sample_*.pkl"))
if not files:
    raise SystemExit("no memory_cache_sample_*.pkl in " + d)
tot, hist = collections.Counter(), collections.Counter()
print(f"{'file':<28}{'notes':>7}{'tmpl ctx':>10}{'tmpl tags':>11}{'any %':>8}{'fmt art.':>10}{'history':>9}")
for p in files:
    notes = list(pickle.load(open(p, "rb")).values())
    c = collections.Counter()
    for nt in notes:
        ctx = nt.context if isinstance(nt.context, str) else " ".join(nt.context)
        tags = [str(t) for t in (nt.tags or [])]
        bad_ctx = V.is_template(ctx)
        bad_tags = bool(tags) and any(V.is_template(t) for t in tags)
        c["notes"] += 1
        c["ctx"] += bad_ctx
        c["tags"] += bad_tags
        c["any"] += (bad_ctx or bad_tags)
        c["fmt"] += bool(tags) and any(_FMT.search(t.strip()) for t in tags)
        c["hist"] += bool(nt.evolution_history)
        hist[min(len(nt.links or []), 5)] += 1
    tot.update(c)
    print(f"{p.split('/')[-1].split(chr(92))[-1]:<28}{c['notes']:>7}{c['ctx']:>10}{c['tags']:>11}"
          f"{100 * c['any'] / c['notes']:>7.1f}%{c['fmt']:>10}{c['hist']:>9}")
print(f"\nTOTAL notes={tot['notes']}  template context={tot['ctx']}  template tags={tot['tags']}  "
      f"any={100 * tot['any'] / tot['notes']:.1f}%  tag format artifacts={tot['fmt']}  notes with history={tot['hist']}")
print("links per note (5 = 5 or more):", dict(sorted(hist.items())))
