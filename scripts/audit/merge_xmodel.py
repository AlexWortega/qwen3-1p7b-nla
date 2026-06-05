"""v20: merge v18_xmodel (17 quirk concepts) + v19_xmodel (6 social + cot + neutral) into a
single broad detect corpus, so the trunk trains on ~23 concepts (their 'classification' breadth)
instead of a narrow vocab. Both dirs share the same 8 tags at the same per-tag layers
(adapters_v9_serve_llama enc), so per-tag acts concatenate directly.

Output /big/audit/v20_xmodel/: rows.jsonl (v18 ++ v19, reindexed) + <tag>/acts.safetensors
(cat along dim 0). neutral classes from both are merged.

Run: python -m scripts.audit.merge_xmodel --out /big/audit/v20_xmodel
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TAGS = ["qwen3-1p7b", "phi-1p5", "smollm3-3b", "qwen2p5-7b", "gemma2",
        "qwen2p5-0p5b", "qwen3-4b", "llama3-8b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", default=["/big/audit/v18_xmodel", "/big/audit/v19_xmodel"])
    ap.add_argument("--out", default="/big/audit/v20_xmodel")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # rows: concat in src order; reindex idx
    rows = []
    per_src_counts = []
    for s in args.src:
        r = [json.loads(l) for l in (Path(s) / "rows.jsonl").read_text().splitlines() if l.strip()]
        per_src_counts.append(len(r))
        rows.extend(r)
    for i, r in enumerate(rows):
        r["idx"] = i
    (out / "rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"[merge] rows: {' + '.join(map(str, per_src_counts))} = {len(rows)}")

    for tag in TAGS:
        parts = []
        ok = True
        for s in args.src:
            p = Path(s) / tag / "acts.safetensors"
            if not p.exists():
                print(f"[merge] WARN {tag}: missing {p}, skipping tag"); ok = False; break
            parts.append(load_file(str(p))["h"].float())
        if not ok:
            continue
        h = torch.cat(parts, dim=0)
        assert h.shape[0] == len(rows), f"{tag}: {h.shape[0]} != {len(rows)} rows"
        (out / tag).mkdir(parents=True, exist_ok=True)
        save_file({"h": h}, str(out / tag / "acts.safetensors"))
        print(f"[merge] {tag}: {[tuple(p.shape) for p in parts]} -> {tuple(h.shape)}")

    # per-bias counts
    import collections
    c = collections.Counter(r["bias"] for r in rows)
    print(f"[merge] {len(c)} concepts: {dict(sorted(c.items(), key=lambda x:-x[1]))}")


if __name__ == "__main__":
    main()
