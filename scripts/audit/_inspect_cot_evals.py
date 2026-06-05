"""Inspect ceselder/cot-oracle-evals datasets: columns, a sample, label distribution.
Run in container to decide how to map each eval to our activation-oracle AUROC."""
import json
from datasets import load_dataset

EVALS = [
    "ceselder/cot-oracle-eval-decorative-cot",
    "ceselder/cot-oracle-eval-rot13-reconstruction",
    "ceselder/cot-oracle-truthfulqa-hint-admission-verbalized",
    "ceselder/cot-oracle-truthfulqa-hint-admission-unverbalized",
]

for repo in EVALS:
    print("\n" + "=" * 80 + f"\n{repo}")
    try:
        ds = load_dataset(repo)
    except Exception as e:
        print("  LOAD FAIL:", e); continue
    for split in ds:
        d = ds[split]
        print(f"  split={split} n={len(d)} cols={d.column_names}")
        r0 = d[0]
        for k, v in r0.items():
            sv = str(v).replace("\n", " ")
            print(f"    {k:24} = {sv[:140]}")
        # guess a label column + distribution
        for cand in ["label", "labels", "is_faithful", "faithful", "hint_admitted",
                     "admitted", "decorative", "correct", "answer", "target", "y"]:
            if cand in d.column_names:
                import collections
                c = collections.Counter(d[cand])
                print(f"    [dist {cand}] {dict(list(c.items())[:8])}")
        break  # first split only
