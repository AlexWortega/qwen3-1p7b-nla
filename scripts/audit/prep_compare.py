"""Prep for the v8/v9/KitFT comparison:
  * Build pool dirs (extract_multi format) that run_kitft_av.py can read, from the
    already-extracted battery L20 acts + the battery transcripts.
  * Write the v9 run_av_explain plan (reuses the same battery acts).

Usage (in container):
  python scripts/audit/prep_compare.py --acts-dir artifacts/audit/acts \
    --battery scripts/audit/prompts_battery.json --out-root artifacts/audit
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def write_pool(out_dir, src_shard, passages):
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_shard, out_dir / "qwen2p5-7b.safetensors")
    (out_dir / "qwen2p5-7b.meta.json").write_text(json.dumps(
        {"tag": "qwen2p5-7b", "model": "Qwen/Qwen2.5-7B-Instruct(+organism LoRA)",
         "d_model": 3584, "layer_index": 20, "n_passages": len(passages),
         "shard": "qwen2p5-7b.safetensors"}, indent=2))
    (out_dir / "passages.jsonl").write_text(
        "\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in passages) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--battery", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()
    acts = Path(args.acts_dir)
    root = Path(args.out_root)
    battery = json.loads(Path(args.battery).read_text())
    passages = [f"User: {b['user']}\nAssistant: {b['assistant']}" for b in battery]

    # KitFT pools (org + base) at L20
    write_pool(root / "pool_kitft_org", acts / "acts_org-L20_mean.safetensors", passages)
    write_pool(root / "pool_kitft_base", acts / "acts_base-L20_mean.safetensors", passages)
    print(f"[prep] wrote KitFT pools ({len(passages)} passages)")
    print("[prep] passage-ids:", ",".join(str(i) for i in range(len(passages))))

    # v9 plan (reuse battery acts through v9 bundle/AV)
    v9_plan = [
        {"label": "v9-organism-L20-mean", "shard": "acts_org-L20_mean.safetensors", "enc_tag": "qwen2p5-7b"},
        {"label": "v9-base-L20-mean",     "shard": "acts_base-L20_mean.safetensors", "enc_tag": "qwen2p5-7b"},
        {"label": "v9-organism-L14-mean", "shard": "acts_org-L14_mean.safetensors", "enc_tag": "qwen2p5-7b"},
        {"label": "v9-base-L14-mean",     "shard": "acts_base-L14_mean.safetensors", "enc_tag": "qwen2p5-7b"},
    ]
    (Path("scripts/audit/plan_v9.json")).write_text(json.dumps(v9_plan, indent=2))
    print("[prep] wrote scripts/audit/plan_v9.json")


if __name__ == "__main__":
    main()
