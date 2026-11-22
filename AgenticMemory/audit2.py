import pickle, re, collections
rx_tag = re.compile(r"^\W*(updated[\s_]*)?tag\s?_?\d\W*$", re.I)
rx_ctx = re.compile(r"updated context sentence|a single sentence summarizing", re.I)
rx_kw = re.compile(r"^\W*keyword\s?\d\W*$", re.I)

by_field = collections.Counter()
link_hist = collections.Counter()
total = 0
for i in range(10):
    p = f'cached_memories_robust_ollama_qwen2.5-3b/memory_cache_sample_{i}.pkl'
    for nt in pickle.load(open(p, 'rb')).values():
        total += 1
        ctx = nt.context if isinstance(nt.context, str) else " ".join(nt.context)
        t = bool(nt.tags) and all(rx_tag.match(str(x)) for x in nt.tags)
        c = bool(rx_ctx.search(ctx))
        k = bool(nt.keywords) and any(rx_kw.match(str(x)) for x in nt.keywords)
        by_field['tags_placeholder'] += t
        by_field['context_placeholder'] += c
        by_field['keywords_placeholder'] += k
        by_field['any'] += (t or c or k)
        by_field['tags_and_context'] += (t and c)
        link_hist[min(len(nt.links or []), 5)] += 1
print(total, dict(by_field))
print("links per note (5 = 5 or more):", dict(sorted(link_hist.items())))