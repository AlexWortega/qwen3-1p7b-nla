"""Wrap an existing LinearAdapter bundle into a ConvAdapter bundle.

Reads `adapters_v9_init/` (LinearAdapter, lstsq-fit) and writes
`adapters_v9_conv_init/` whose enc/dec are `ConvAdapter`s:
  - Linear `weight` copied from the source bundle.
  - Conv1d `weight` zero-init except a 1.0 at the centre tap (identity).
At init the bundle's forward output is bit-identical to the LinearAdapter
bundle, so any downstream code path keeps the lstsq warm-start.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch

from nla.enc_dec_adapters import ConvAdapter, ModelPoolAdapters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-adapters", required=True,
                    help="LinearAdapter bundle to wrap.")
    ap.add_argument("--out-adapters", required=True)
    ap.add_argument("--kernel", type=int, default=7,
                    help="Conv1d kernel size (odd; identity init lands on the centre tap).")
    args = ap.parse_args()

    src = ModelPoolAdapters.load(args.in_adapters)
    assert src.adapter_class == "LinearAdapter", (
        f"source bundle is already {src.adapter_class!r} — only LinearAdapter "
        f"can be wrapped into ConvAdapter."
    )

    dst = ModelPoolAdapters(
        d_shared=src.d_shared, model_dims=src.model_dims,
        adapter_class="ConvAdapter", adapter_kwargs={"kernel": args.kernel},
    )
    # Copy Linear weights tag-by-tag (Conv1d stays at identity init from constructor).
    with torch.no_grad():
        for tag in src.tags:
            dst.encoders[tag].weight.copy_(src.encoders[tag].weight)
            dst.decoders[tag].weight.copy_(src.decoders[tag].weight)

    # Carry the serve cache over if the source had one.
    if src.has_serve_cache:
        dst._enc_target_cache = src._enc_target_cache.clone()
        dst._cache_meta = dict(src._cache_meta)

    dst.save(args.out_adapters)
    print(f"[init-conv] {len(src.tags)} tags wrapped (kernel={args.kernel}) → {args.out_adapters}")


if __name__ == "__main__":
    main()
