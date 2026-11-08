import json

paper = {  # Paper Tables 1, 5, 6: Qwen2.5-3b, A-MEM, Open Domain (k=50 per Table 8)
    'f1': 7.12, 'bleu1': 7.28, 'rouge2_f': 1.51, 'rougeL_f': 7.51,
    'meteor': 6.56, 'sbert_similarity': 30.60,
}

def load(path):
    d = json.load(open(path))
    a = d['aggregate_metrics']
    return d, a.get('category_3') or a.get('3')

d10, c10 = load('results_full_qwen3b.json')
d50, c50 = load('results_cat3_k50.json')

def m(c, k):
    try:
        return float(c[k]['mean']) * 100
    except Exception:
        return float('nan')

def n(c):
    try:
        return int(float(c['f1']['count']))
    except Exception:
        return -1

print(f"k=10 run: n={n(c10)}   k=50 run: n={n(c50)}   (paper: k=50)")
print(f"llm_failures in k=50 run: {d50.get('llm_failures')}\n")
print(f"{'metric':<18}{'k=10':>9}{'k=50':>9}{'paper':>9}{'k50-k10':>10}{'k50-paper':>11}")
print('-' * 66)
for k, p in paper.items():
    a, b = m(c10, k), m(c50, k)
    print(f"{k:<18}{a:>9.2f}{b:>9.2f}{p:>9.2f}{b-a:>+10.2f}{b-p:>+11.2f}")