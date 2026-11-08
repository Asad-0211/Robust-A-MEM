import pickle, json

d = json.load(open('results_full_qwen3b.json'))
print("cached_samples:", d.get('cached_samples'))
print("stale_cache_samples:", d.get('stale_cache_samples'))
print("llm_failures:", d.get('llm_failures'))
print()

total = 0
for i in range(10):
    p = f'cached_memories_robust_ollama_qwen2.5-3b/memory_cache_sample_{i}.pkl'
    with open(p, 'rb') as f:
        obj = pickle.load(f)
    desc = type(obj).__name__
    n = None
    if isinstance(obj, (list, dict, set)):
        n = len(obj)
    elif isinstance(obj, tuple):
        desc += " of " + ", ".join(f"{type(x).__name__}({len(x) if hasattr(x, '__len__') else '-'})" for x in obj)
    print(f"sample {i}: {desc}  len={n}")
    if n:
        total += n
print("\ntotal notes (if len applies):", total)