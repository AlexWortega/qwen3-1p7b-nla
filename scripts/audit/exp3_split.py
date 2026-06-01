"""Exp 3 — instance-level held-out split of the v13 supervised biases.

Floor test: can the AO name the RIGHT (trained) bias on an UNSEEN INSTANCE?
Reuses ao_rows_v13.jsonl (free-form answers) + acts_ao_{org,base}_mean (TR order).
Per bias, hold out --frac of its transcripts → eval; train on the rest.

Writes:
  ao_rows_exp3.jsonl              training rows (transcript_idx in train split)
  eval_heldout_exp3.jsonl        eval transcripts (held-out instances)
  acts_exp3_eval_org_mean.safetensors / _base_mean   eval-aligned act slices
"""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ao-dir", default="artifacts/audit/ao")
    ap.add_argument("--rows", default="ao_rows_v13.jsonl")
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    d = Path(args.ao_dir)
    rows = [json.loads(l) for l in (d / args.rows).read_text().splitlines() if l.strip()]
    TR = [json.loads(l) for l in (d / "transcripts_base.jsonl").read_text().splitlines() if l.strip()]
    Horg = load_file(str(d / "acts_ao_org_mean.safetensors"))["h"]
    Hbase = load_file(str(d / "acts_ao_base_mean.safetensors"))["h"]

    # transcript_idx -> bias (from biased TR rows). Build per-bias index lists.
    idx_bias = {}
    for t in TR:
        idx_bias[int(t["id"][2:]) if str(t["id"]).startswith("tr") else None] = t.get("bias")
    # TR id is "tr<global_idx>"; map by order instead (robust):
    by_bias = defaultdict(list)
    for gi, t in enumerate(TR):
        b = t.get("bias", "neutral")
        if b != "neutral":
            by_bias[b].append(gi)

    eval_idx = set()
    for b, idxs in by_bias.items():
        rng.shuffle(idxs)
        k = max(1, int(len(idxs) * args.frac))
        eval_idx.update(idxs[:k])
    print(f"[exp3] {len(eval_idx)} eval instances across {len(by_bias)} biases "
          f"(train transcripts {len(TR)-len(eval_idx)})")

    # training rows: drop any row whose transcript is in eval split
    train_rows = [r for r in rows if int(r["transcript_idx"]) not in eval_idx]
    (d / "ao_rows_exp3.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train_rows) + "\n")

    # eval battery + aligned act slices (org enacts the bias)
    eval_sorted = sorted(eval_idx)
    bat = []
    for gi in eval_sorted:
        t = TR[gi]
        bat.append({"id": f"ev{gi}", "user": t["user"], "assistant": t["assistant"],
                    "bias": t.get("bias"), "category": t.get("bias")})
    (d / "eval_heldout_exp3.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in bat) + "\n")
    save_file({"h": Horg[torch.tensor(eval_sorted)]}, str(d / "acts_exp3_eval_org_mean.safetensors"))
    save_file({"h": Hbase[torch.tensor(eval_sorted)]}, str(d / "acts_exp3_eval_base_mean.safetensors"))
    print(f"[exp3] train rows={len(train_rows)} eval battery={len(bat)} -> {d}")


if __name__ == "__main__":
    main()
