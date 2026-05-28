"""Bake the serve-time auto-refit target into an existing adapter bundle.

Reads an SFT-trained `ModelPoolAdapters` bundle and the activation pool dir
it was trained on, computes the mean of post-SFT encoder projections across
trained tags, and writes `serve_cache.safetensors` next to the bundle's
`adapters.safetensors`. Once a bundle has a cache, any held-out model can be
added via `pool.add_held_out_tag(tag, h_new_raw)` — no training step.

Usage:
  python scripts/build_serve_cache.py \
    --in-adapters artifacts/av_v8_mixed/adapters \
    --pool-dir artifacts/activations_pool_300m \
    --trained-tags gpt-neo-1p3b,...,yagpt-5-8b \
    --inject-scale 45.25 \
    --out-adapters artifacts/av_v8_mixed/adapters_with_cache
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

from nla.enc_dec_adapters import ModelPoolAdapters


def _load_tag_h(pool_dir: Path, tag: str) -> torch.Tensor:
    meta = json.loads((pool_dir / f"{tag}.meta.json").read_text())
    return load_file(str(pool_dir / meta["shard"]))["h"].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-adapters", required=True,
                    help="Existing SFT-trained adapter bundle.")
    ap.add_argument("--pool-dir", required=True,
                    help="Activation pool dir with per-tag .safetensors + .meta.json.")
    ap.add_argument("--trained-tags", required=True,
                    help="Comma-separated list of trained tags whose SFT-tuned encoder "
                         "projections will be averaged into the serve cache.")
    ap.add_argument("--inject-scale", type=float, default=None,
                    help="If set, per-row normalize each projection to this scale "
                         "before averaging (use √d_shared to mirror the AV trunk's "
                         "injection-time normalization). Default: skip normalization.")
    ap.add_argument("--out-adapters", required=True)
    args = ap.parse_args()

    pool_dir = Path(args.pool_dir)
    pool = ModelPoolAdapters.load(args.in_adapters)
    trained = [t.strip() for t in args.trained_tags.split(",") if t.strip()]
    inject_scale = args.inject_scale
    if inject_scale is None:
        inject_scale = math.sqrt(pool.d_shared)
        print(f"[serve-cache] inject-scale not given, defaulting to √d_shared = {inject_scale:.4f}")

    print(f"[serve-cache] {len(trained)} trained tags, d_shared={pool.d_shared}")
    h_per_tag = {}
    for t in trained:
        assert t in pool.encoders, f"trained tag {t!r} not in adapter bundle"
        h_per_tag[t] = _load_tag_h(pool_dir, t)
        print(f"  + {t} h={tuple(h_per_tag[t].shape)}")

    info = pool.build_serve_cache(h_per_tag=h_per_tag, trained_tags=trained,
                                  inject_scale=inject_scale)
    print(f"[serve-cache] built: {info}")

    out = Path(args.out_adapters)
    pool.save(out)
    print(f"[serve-cache] saved → {out}")
    print(f"    + {ModelPoolAdapters.SERVE_CACHE_FILENAME}  shape=({info['n_passages']}, {pool.d_shared})")


if __name__ == "__main__":
    main()
