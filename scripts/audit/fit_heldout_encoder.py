"""Fit a new per-tag encoder into a v22 detector adapter bundle, closed-form, via
ModelPoolAdapters.add_held_out_tag (serve_cache lstsq). Loads --in-adapters, fits the new
--tag from its pool acts --pool-acts ([N, d_M] on the same passages as the serve cache),
and writes an EXTENDED bundle to --out-adapters (original left untouched). Prints enc/dec FVE."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from nla.enc_dec_adapters import ModelPoolAdapters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-adapters", required=True, help="dir with adapters.safetensors + serve_cache.safetensors + meta.json")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pool-acts", required=True, help="safetensors with key 'h' = [N, d_M] raw pool acts")
    ap.add_argument("--out-adapters", required=True)
    args = ap.parse_args()

    pool = ModelPoolAdapters.load(Path(args.in_adapters))
    assert getattr(pool, "has_serve_cache", False), "bundle has no serve_cache"
    h_new = load_file(args.pool_acts)["h"].float()
    print(f"[fit] tag={args.tag} h_new={tuple(h_new.shape)}", flush=True)
    diag = pool.add_held_out_tag(args.tag, h_new, normalize_input=True)
    print(f"[fit] {args.tag} diag={json.dumps(diag)}", flush=True)
    od = Path(args.out_adapters)
    od.mkdir(parents=True, exist_ok=True)
    pool.save(od)
    print(f"[fit] wrote extended bundle -> {od} (tags now include {args.tag})", flush=True)


if __name__ == "__main__":
    main()
