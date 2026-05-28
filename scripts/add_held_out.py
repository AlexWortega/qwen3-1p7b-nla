"""One-shot held-out tag addition via `ModelPoolAdapters.add_held_out_tag()`.

Reads an SFT bundle that already has a serve cache (`serve_cache.safetensors`),
loads the held-out model's raw activations from the pool dir, calls the
closed-form lstsq path, and writes a new bundle with the tag included.

No GPU, no training. Replaces the two-step (extend_adapters →
refit_heldout_enc) recipe with a single self-contained call.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from nla.enc_dec_adapters import ModelPoolAdapters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-adapters", required=True,
                    help="Adapter bundle with serve_cache.safetensors present.")
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--tags", required=True,
                    help="Comma-separated held-out tags to add.")
    ap.add_argument("--out-adapters", required=True)
    args = ap.parse_args()

    pool_dir = Path(args.pool_dir)
    pool = ModelPoolAdapters.load(args.in_adapters)
    if not pool.has_serve_cache:
        raise SystemExit(
            f"{args.in_adapters} has no serve_cache.safetensors. "
            f"Run scripts/build_serve_cache.py first."
        )

    new_tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    report = {}
    for tag in new_tags:
        meta_path = pool_dir / f"{tag}.meta.json"
        if not meta_path.exists():
            print(f"  [{tag}] no .meta.json in pool — skip")
            continue
        meta = json.loads(meta_path.read_text())
        h = load_file(str(pool_dir / meta["shard"]))["h"].float()
        info = pool.add_held_out_tag(tag, h)
        report[tag] = info
        print(f"  + {tag:18s} d_M={info['d_M']:5d}  enc_fve={info['enc_fve']:+.4f}  dec_fve={info['dec_fve']:+.4f}")

    out = Path(args.out_adapters)
    pool.save(out)
    (out / "add_held_out_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"[add-held-out] saved → {out}")


if __name__ == "__main__":
    main()
