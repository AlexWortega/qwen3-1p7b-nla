"""Ingest ceselder/cot-oracle-corpus-v5 as a cross-model AO task ('cot_oracle').

External corpus (Neel/MATS "Building Better Activation Oracles" line): 40.5k Qwen3-8B
chain-of-thought MATH rollouts with a per-rollout `cot_correct` bool (did the reasoning
reach the right answer). We fold it into OUR cross-model Activation-Oracle pipeline as a
binary correctness/faithfulness detector: replay (question, cot_response) through the
universal readers, mean-pool the assistant span, and learn "does this reasoning reach the
correct answer? Yes/No" from the activation — the same recipe as the lie detector.

Output (extract_v18_xmodel --dialogue-files compatible):
  <out>  rows {bias, user, assistant, cot_correct, source}
    bias = "cot_correct" (label 1) | "cot_incorrect" (label 0)  — balanced
    user = question, assistant = cot_response (truncated to --max-chars)

Then extract cross-model:
  bash scripts/audit/run_v19_extract.sh <out> /big/audit/cot_oracle_xmodel 1000

Run:
  python -m scripts.audit.prep_cot_oracle --out /big/audit/cot_oracle/cot_transcripts.jsonl \
      --per-class 1500 --max-chars 4000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/big/audit/cot_oracle/cot_transcripts.jsonl")
    ap.add_argument("--repo", default="ceselder/cot-oracle-corpus-v5")
    ap.add_argument("--per-class", type=int, default=1500,
                    help="max rows per class (cot_correct True/False); balanced")
    ap.add_argument("--max-chars", type=int, default=4000,
                    help="truncate cot_response (extract caps tokens anyway)")
    ap.add_argument("--min-chars", type=int, default=120)
    ap.add_argument("--correct-as-neutral", action="store_true",
                    help="v19 merge: label correct-reasoning rows bias='neutral' (legit "
                         "negatives) and incorrect rows bias='cot_incorrect' (the positive "
                         "class). Lets the cot task fold into the v19 detect mix as-is.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.repo, split="train")
    rng = random.Random(args.seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)

    kept = {True: [], False: []}
    for i in idxs:
        if len(kept[True]) >= args.per_class and len(kept[False]) >= args.per_class:
            break
        r = ds[i]
        cot = (r.get("cot_response") or r.get("cot_content") or "").strip()
        q = (r.get("question") or "").strip()
        cc = r.get("cot_correct")
        if cc is None or not q or len(cot) < args.min_chars:
            continue
        cc = bool(cc)
        if len(kept[cc]) >= args.per_class:
            continue
        if args.correct_as_neutral:
            bias = "neutral" if cc else "cot_incorrect"
        else:
            bias = "cot_correct" if cc else "cot_incorrect"
        kept[cc].append({
            "bias": bias,
            "user": q,
            "assistant": cot[:args.max_chars],
            "cot_correct": cc,
            "source": str(r.get("source", "cot-oracle-v5")),
        })

    rows = kept[True] + kept[False]
    rng.shuffle(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"[cot_oracle] wrote {len(rows)} rows -> {out} "
          f"(correct={len(kept[True])}, incorrect={len(kept[False])})")


if __name__ == "__main__":
    main()
