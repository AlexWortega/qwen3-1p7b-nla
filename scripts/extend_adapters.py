"""Add NEW per-model adapters to an existing ModelPoolAdapters bundle without
re-training the existing ones.

Used for the held-out / zero-shot demonstration: the AV trunk + the 5 pool
models' (enc, dec) pairs were jointly trained in `train_av_multi.py`. To plug
in a sixth, never-seen model M', we want to:
  - keep AV LoRA and all existing enc/dec FROZEN at their trained weights
  - fit (enc_M', dec_M') via closed-form lstsq against the anchor's activations
  - save the augmented bundle so eval_universal can verbalise M''s activations
    with the frozen trunk

Reads:
  --base-adapters      directory of an existing ModelPoolAdapters (from train_av_multi)
  --pool-dir           directory with all shards including the new model
  --new-tags           comma-separated tags to add (must have shards + meta.json)
  --anchor-tag         tag in pool-dir whose d_model equals d_shared

Writes a new ModelPoolAdapters bundle at --out-dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nla.enc_dec_adapters import LinearAdapter, ModelPoolAdapters
from scripts.init_adapters import load_shard, split_train_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-adapters", required=True, help="existing ModelPoolAdapters dir")
    ap.add_argument("--pool-dir", required=True, help="activations pool dir (extract_multi output)")
    ap.add_argument("--new-tags", required=True, help="comma-separated tags to add")
    ap.add_argument("--anchor-tag", required=True, help="reference tag, d == d_shared")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool_dir = Path(args.pool_dir)
    pool_index = json.loads((pool_dir / "index.json").read_text())
    new_tags = [t.strip() for t in args.new_tags.split(",") if t.strip()]
    for t in new_tags:
        assert t in pool_index, f"new tag {t!r} not present in {pool_dir}/index.json"

    H_anchor_all, anchor_meta = load_shard(pool_dir, args.anchor_tag)
    N = H_anchor_all.shape[0]
    train_idx, eval_idx = split_train_eval(N, args.train_frac, args.seed)
    H_anchor_tr = H_anchor_all[train_idx]
    H_anchor_ev = H_anchor_all[eval_idx]

    base = ModelPoolAdapters.load(args.base_adapters)
    assert anchor_meta["d_model"] == base.d_shared, (
        f"anchor d={anchor_meta['d_model']} ≠ base.d_shared={base.d_shared}"
    )
    print(f"[extend] base has {len(base.tags)} tags: {base.tags}")
    print(f"[extend] adding {new_tags}")

    report: dict[str, dict] = {}
    for tag in new_tags:
        if tag in base.tags:
            print(f"[extend] tag {tag!r} already in base bundle — overwriting")
        H_m_all, meta_m = load_shard(pool_dir, tag)
        assert H_m_all.shape[0] == N, (
            f"row count for {tag} ({H_m_all.shape[0]}) ≠ anchor ({N})"
        )
        H_m_tr = H_m_all[train_idx]
        H_m_ev = H_m_all[eval_idx]
        d_m = meta_m["d_model"]
        enc = LinearAdapter.lstsq_init(H_m_tr, H_anchor_tr)
        dec = LinearAdapter.lstsq_init(H_anchor_tr, H_m_tr)
        fve_enc = enc.lstsq_fve(H_m_ev, H_anchor_ev)
        fve_dec = dec.lstsq_fve(H_anchor_ev, H_m_ev)
        if tag not in base.tags:
            base.add_model(tag, d_m)
        base.set_pair(tag, enc, dec)
        report[tag] = {"d_model": d_m,
                       "fve_enc_eval": round(fve_enc, 4),
                       "fve_dec_eval": round(fve_dec, 4)}
        print(f"[extend] {tag} d={d_m}  FVE_enc={fve_enc:.4f}  FVE_dec={fve_dec:.4f}")

    out_dir = Path(args.out_dir)
    base.save(out_dir)
    (out_dir / "extend_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"[extend] saved → {out_dir}")


if __name__ == "__main__":
    main()
