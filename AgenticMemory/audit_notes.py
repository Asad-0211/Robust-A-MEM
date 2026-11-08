import pickle, re, collections

PATTERNS = [r"updated\s+context", r"updated\s+tag", r"\btag\s?\d\b",
            r"context sentence", r"new context", r"keyword\s?\d\b"]
rx = [re.compile(p) for p in PATTERNS]

def flagged(texts):
    return any(r.search(t.lower()) for t in texts for r in rx)

tot = collections.Counter()
tagsets = collections.Counter()
print(f"{'sample':>6}{'notes':>7}{'tmpl':>7}{'w/links':>9}{'avg_lnk':>9}"
      f"{'dup':>6}{'self':>6}{'fwd':>6}{'inbound%':>10}{'hist':>6}{'General':>9}")
for i in range(10):
    p = f'cached_memories_robust_ollama_qwen2.5-3b/memory_cache_sample_{i}.pkl'
    notes = list(pickle.load(open(p, 'rb')).values())
    n = len(notes)
    c = collections.Counter()
    inbound = set()
    nlinks = 0
    for pos, nt in enumerate(notes):
        ctx = nt.context if isinstance(nt.context, str) else " ".join(nt.context)
        tags = [str(t) for t in (nt.tags or [])]
        kws = [str(k) for k in (nt.keywords or [])]
        if flagged([ctx] + tags + kws):
            c['tmpl'] += 1
        links = [l for l in (nt.links or []) if isinstance(l, int)]
        if links:
            c['w_links'] += 1
        nlinks += len(links)
        c['dup'] += len(links) - len(set(links))
        c['self'] += int(pos in links)
        c['fwd'] += int(any(l > pos for l in links))
        inbound.update(l for l in links if 0 <= l < n)
        c['hist'] += int(bool(nt.evolution_history))
        c['general'] += int(ctx.strip() == "General")
        tagsets[tuple(tags)] += 1
    for k, v in c.items():
        tot[k] += v
    tot['n'] += n
    tot['nlinks'] += nlinks
    tot['inbound'] += len(inbound)
    print(f"{i:>6}{n:>7}{c['tmpl']:>7}{c['w_links']:>9}{nlinks/n:>9.2f}"
          f"{c['dup']:>6}{c['self']:>6}{c['fwd']:>6}{100*len(inbound)/n:>9.1f}%"
          f"{c['hist']:>6}{c['general']:>9}")

n = tot['n']
print(f"\nTOTAL notes={n}  template-text notes={tot['tmpl']} ({100*tot['tmpl']/n:.1f}%)"
      f"  avg links/note={tot['nlinks']/n:.2f}  duplicate links={tot['dup']}"
      f"  notes with inbound link={100*tot['inbound']/n:.1f}%")
print("\nMost common tag sets:")
for ts, cnt in tagsets.most_common(5):
    print(f"  {cnt:>5}  {list(ts)}")