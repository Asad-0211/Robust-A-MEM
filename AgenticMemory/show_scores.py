import json

d = json.load(open('results_smoke_qwen3b.json'))

for k in ['model', 'dataset', 'memory_layer', 'total_questions',
          'category_distribution', 'ratio', 'seed', 'legacy_c5_reference',
          'refresh_index', 'memory_scope', 'pooled_notes',
          'cached_samples', 'stale_cache_samples', 'llm_failures']:
    print(f"{k}: {d.get(k)}")

print("\n--- aggregate metric means ---")
printed = 0

def walk(x, path=''):
    global printed
    if isinstance(x, dict):
        if 'mean' in x and not isinstance(x['mean'], dict):
            try:
                line = f"{path:60s} mean={float(x['mean']):.4f}"
                if 'count' in x:
                    line += f"  n={float(x['count']):.0f}"
                print(line)
                printed += 1
            except (TypeError, ValueError):
                pass
        else:
            for k, v in x.items():
                walk(v, path + '/' + str(k))

walk(d['aggregate_metrics'])

if printed == 0:
    print("No 'mean' fields found. Raw structure (truncated):")
    print(json.dumps(d['aggregate_metrics'], indent=1)[:3000])