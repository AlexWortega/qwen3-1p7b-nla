"""Refit per-model `dec_M` so it inverts `enc_M` (not just maps anchor→M).

Original init_adapters.py fit dec_M independently on `H_anchor → H_M`. That's
wrong if you want `dec_M(enc_M(h)) ≈ h`, because enc_M and dec_M solve DIFFERENT
lstsq problems and the composition is not identity.

FVE eval needs `dec_M(ĥ_shared) ≈ h_M` where `ĥ_shared` is approximately
`enc_M(h_M)` (that's what AR was trained to predict). So we want
`dec_M ∘ enc_M ≈ id` on M's manifold. Solve the right lstsq:

    min_W || enc_M(H_M) @ W.T - H_M ||²
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

from nla.enc_dec_adapters import LinearAdapter, ModelPoolAdapters
from nla.schema import normalize_activation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-adapters", required=True)
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--out-adapters", required=True)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchor-tag", required=True,
                    help="dec stays identity for anchor when d_anchor == d_shared")
    args = ap.parse_args()

    pool_dir = Path(args.pool_dir)
    pool_index = json.loads((pool_dir / "index.json").read_text())
    adapters = ModelPoolAdapters.load(args.in_adapters)
    print(f"[refit_dec] loaded {len(adapters.tags)} tags from {args.in_adapters}")

    g = torch.Generator().manual_seed(args.seed)
    report: dict[str, dict] = {}
    for tag in adapters.tags:
        if tag == args.anchor_tag and pool_index.get(tag, {}).get("d_model") == adapters.d_shared:
            with torch.no_grad():
                adapters.decoders[tag].weight.copy_(torch.eye(adapters.d_shared))
            report[tag] = {"anchor": True}
            print(f"  [{tag}] anchor — dec=identity")
            continue
        meta = json.loads((pool_dir / f"{tag}.meta.json").read_text())
        h_raw = load_file(str(pool_dir / meta["shard"]))["h"].float()  # RAW magnitude
        d_shared = adapters.d_shared
        # Match train_ar_multi exactly: target_input = normalize(enc_M(h_raw), √d_shared).
        # Target output = h_raw (so dec_M maps AR's output back to native space).
        enc_M = adapters.encoders[tag]
        with torch.no_grad():
            H_proj = enc_M(h_raw)                             # [N, d_shared]
            ar_target_input = normalize_activation(H_proj, math.sqrt(d_shared))
        N = h_raw.shape[0]
        perm = torch.randperm(N, generator=g)
        cut = int(N * args.train_frac)
        train_idx, eval_idx = perm[:cut], perm[cut:]
        X_tr, X_ev = ar_target_input[train_idx], ar_target_input[eval_idx]
        Y_tr, Y_ev = h_raw[train_idx], h_raw[eval_idx]
        new_dec = LinearAdapter.lstsq_init(X_tr, Y_tr)
        fve_inv = new_dec.lstsq_fve(X_ev, Y_ev)
        # Also report meannorm FVE on the eval split for sanity.
        pred_ev = new_dec.forward(X_ev.to(new_dec.weight.device)).cpu()
        scale = math.sqrt(meta["d_model"])
        from nla.schema import normalize_activation as _na
        resid = (_na(Y_ev, scale) - _na(pred_ev, scale)).var(unbiased=False).item()
        gv = _na(Y_ev, scale).var(unbiased=False).item()
        fve_mn = 1.0 - resid / max(gv, 1e-12)
        adapters.set_pair(tag, enc_M, new_dec)
        report[tag] = {"fve_dec_raw": round(fve_inv, 4), "fve_dec_meannorm": round(fve_mn, 4)}
        print(f"  [{tag}] FVE_dec  raw={fve_inv:+.4f}  meannorm={fve_mn:+.4f}  (dec inverts AR's normalized-output path)")

    adapters.save(args.out_adapters)
    (Path(args.out_adapters) / "refit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"[refit_dec] saved → {args.out_adapters}")


if __name__ == "__main__":
    main()
