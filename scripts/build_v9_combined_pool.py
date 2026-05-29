"""Build a v9 combined activation pool from two sources:

  1. The existing FineWeb pool: artifacts/activations_pool_300m/
     - passages.jsonl + per-tag <tag>.safetensors (mean-pool, depth 0.5)
  2. The multilingual L_mid mean-pool rows from
     artifacts/activations_multilayer_multi_v1/

For each tag we want a single combined shard `<tag>.safetensors` and a single
`passages.jsonl` with all (passage_id, lang, text, z) rows, so the existing
`scripts/train_av_multi.py` + `nla/data_multi.py` pipeline can train on it
without modification.

Output layout (matches activations_pool_300m):
  passages.jsonl             — concatenated, re-numbered so multilingual rows
                                land at passage_ids [N_fineweb, N_fineweb+500).
  <tag>.safetensors          — [N_fineweb + 500, d_M] mean-pool h.
  <tag>.meta.json            — { d_model, shard, layer, n_passages, ... }
  index.json                 — { tag → { shard, d_model, layer, n_passages } }

We pull the mean-pool row from each multilayer_multi_v1 shard using its
meta.json (rows[i] with `char_offset == -1` AND `layer_idx == int(n_layers*0.5)`).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fineweb-pool", default="artifacts/activations_pool_300m")
    ap.add_argument("--multi-pool",   default="artifacts/activations_pool_multi")
    ap.add_argument("--multi-mlmp",   default="artifacts/activations_multilayer_multi_v1")
    ap.add_argument("--out-dir",      default="artifacts/activations_pool_v9")
    ap.add_argument("--tags", required=True,
                    help="Comma-separated tags to combine (must exist in both sources).")
    args = ap.parse_args()

    fw_pool = Path(args.fineweb_pool)
    multi_pool = Path(args.multi_pool)
    multi_mlmp = Path(args.multi_mlmp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    # 1. Concatenate passages.jsonl.
    fw_passages = [json.loads(l) for l in (fw_pool / "passages.jsonl").read_text().splitlines() if l.strip()]
    multi_passages = [json.loads(l) for l in (multi_pool / "passages.jsonl").read_text().splitlines() if l.strip()]
    n_fw, n_multi = len(fw_passages), len(multi_passages)
    print(f"[v9-pool] FineWeb passages: {n_fw}, multilingual: {n_multi}")

    combined = list(fw_passages)
    for i, p in enumerate(multi_passages):
        p2 = dict(p); p2["passage_id"] = n_fw + i
        combined.append(p2)
    with (out_dir / "passages.jsonl").open("w") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[v9-pool] wrote {out_dir/'passages.jsonl'}  total {n_fw+n_multi}")

    # 2. Per tag: load FineWeb shard + pick L_mid mean-pool rows from multi shard.
    fw_idx = json.loads((fw_pool / "index.json").read_text())
    out_idx = {}
    for tag in tags:
        fw_entry = fw_idx.get(tag)
        if fw_entry is None:
            print(f"  [{tag}] no FineWeb shard — skip")
            continue
        d_M = int(fw_entry["d_model"])
        h_fw = load_file(str(fw_pool / fw_entry["shard"]))["h"].float()    # [N_fw, d_M]
        assert h_fw.shape == (n_fw, d_M), f"FineWeb shard shape {h_fw.shape} ≠ ({n_fw},{d_M})"

        meta_path = multi_mlmp / f"{tag}_meta.json"
        if not meta_path.exists():
            print(f"  [{tag}] no multilingual meta — copying FineWeb only")
            h_combined = h_fw
        else:
            meta = json.loads(meta_path.read_text())
            n_layers = int(meta["n_layers"])
            l_mid = max(0, min(n_layers - 1, int(n_layers * 0.5)))
            # Pick the L_mid shard.
            l_mid_shard = multi_mlmp / f"{tag}_L{l_mid}.safetensors"
            h_multi_all = load_file(str(l_mid_shard))["h"].float()           # [N_rows, d_M]
            # Each per-layer shard holds only rows for that layer, in order.
            # Build the within-L_mid index for mean-pool rows.
            within_layer_idx = 0
            mean_pool_local = []
            for r in meta["rows"]:
                if r["layer_idx"] != l_mid:
                    continue
                if r["char_offset"] == -1:
                    mean_pool_local.append(within_layer_idx)
                within_layer_idx += 1
            assert len(mean_pool_local) == n_multi, (
                f"{tag}: expected {n_multi} mean-pool rows at L{l_mid}, got {len(mean_pool_local)}"
            )
            h_multi = h_multi_all[torch.tensor(mean_pool_local, dtype=torch.long)]
            assert h_multi.shape == (n_multi, d_M), f"{tag}: multi shape {h_multi.shape}"
            h_combined = torch.cat([h_fw, h_multi], dim=0)
            print(f"  [{tag}] FineWeb {n_fw} + multi {n_multi} = {h_combined.shape[0]} rows  d={d_M}")

        shard_name = f"{tag}.safetensors"
        save_file({"h": h_combined}, str(out_dir / shard_name))
        meta_out = {
            "tag": tag, "d_model": d_M, "shard": shard_name,
            "layer": int(fw_entry.get("layer", -1)),
            "n_passages": int(h_combined.shape[0]), "pool": "mean_content_tokens",
        }
        (out_dir / f"{tag}.meta.json").write_text(json.dumps(meta_out, indent=1))
        out_idx[tag] = {"shard": shard_name, "d_model": d_M,
                        "layer": meta_out["layer"], "n_passages": meta_out["n_passages"]}

    (out_dir / "index.json").write_text(json.dumps(out_idx, indent=2, sort_keys=True))
    print(f"[v9-pool] DONE  {len(out_idx)} tags  → {out_dir}")


if __name__ == "__main__":
    main()
