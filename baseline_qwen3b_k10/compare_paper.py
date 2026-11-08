import json

d = json.load(open('results_full_qwen3b.json'))
agg = d['aggregate_metrics']

# Paper Table 1, Qwen2.5-3b, A-MEM: (F1, BLEU-1)
paper = {
    'category_1': ('Multi-hop',   12.57, 9.01),
    'category_2': ('Temporal',    27.59, 25.07),
    'category_3': ('Open-domain',  7.12, 7.28),
    'category_4': ('Single-hop',  17.23, 13.12),
    'category_5': ('Adversarial', 27.91, 25.15),
}

def mean(cat, metric):
    try:
        return float(agg[cat][metric]['mean']) * 100
    except Exception:
        return float('nan')

def count(cat):
    try:
        return int(float(agg[cat]['f1']['count']))
    except Exception:
        return -1

print(f"model={d.get('model')}  questions={d.get('total_questions')}  "
      f"ratio={d.get('ratio')}  retrieve_k=10  legacy_c5={d.get('legacy_c5_reference')}")
print(f"llm_failures={d.get('llm_failures')}\n")

hdr = f"{'Category':<12}{'n':>5}  {'F1 you':>8}{'F1 paper':>10}{'diff':>8}   {'B1 you':>8}{'B1 paper':>10}{'diff':>8}"
print(hdr)
print('-' * len(hdr))

f1s, b1s = [], []
for cat, (name, pf1, pb1) in paper.items():
    f1, b1 = mean(cat, 'f1'), mean(cat, 'bleu1')
    f1s.append(f1); b1s.append(b1)
    print(f"{name:<12}{count(cat):>5}  {f1:>8.2f}{pf1:>10.2f}{f1-pf1:>+8.2f}   "
          f"{b1:>8.2f}{pb1:>10.2f}{b1-pb1:>+8.2f}")

print('-' * len(hdr))
pf = sum(v[1] for v in paper.values()) / 5
pb = sum(v[2] for v in paper.values()) / 5
mf, mb = sum(f1s) / 5, sum(b1s) / 5
print(f"{'Mean of 5':<12}{'':>5}  {mf:>8.2f}{pf:>10.2f}{mf-pf:>+8.2f}   {mb:>8.2f}{pb:>10.2f}{mb-pb:>+8.2f}")

try:
    print(f"\nadversarial_correct (cat 5): {float(agg['category_5']['adversarial_correct']['mean']):.4f}")
except Exception:
    pass