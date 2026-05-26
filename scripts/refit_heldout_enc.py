"""Refit held-out enc_M's to the post-SFT projection target distribution.

The story: during AV SFT, trained-tag enc_M's drift from lstsq-init into an
"AV-friendly" space — they learn to project h_M into a representation the AV
trunk knows how to read. Held-out tags' enc_M's stay at lstsq-init and are
therefore out-of-distribution at inference, which kills generation quality
(seen on v7r256: trained cos-vs-gold 0.31, held-out 0.07).

This script post-fits held-out enc_M's *against the trained-tag projection
distribution*, without touching the AV trunk:

  1. For every passage p with a teacher z, take each trained tag T,
     compute enc_M_T_SFT(h_T(p)). Average across T → target(p) in d_shared.
     Passages give us a "cross-model projection mean" that lives in the
     AV-friendly space.
  2. For each held-out tag H, lstsq-fit enc_M_H_new (d_H → d_shared) on
     (h_H(p), target(p)) over the same passages. Closed form, seconds.
  3. Replace held-out enc_M's in the adapter bundle; keep trained
     enc_M's and all dec_M's untouched.

Result: held-out tags now project into the same AV-friendly space → AV can
read them without retraining.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file
import yaml

from nla.enc_dec_adapters import LinearAdapter, ModelPoolAdapters
from nla.schema import normalize_activation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-adapters", required=True,
                    help="adapters_v7r256_sft_direct (with SFT-trained enc_M for trained tags + dec_M)")
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--anchor-tag", required=True)
    ap.add_argument("--trained-tags", required=True,
                    help="Comma-separated list of trained tags (whose enc_M is SFT-tuned)")
    ap.add_argument("--heldout-tags", required=True,
                    help="Comma-separated list of held-out tags to refit")
    ap.add_argument("--out-adapters", required=True)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool_dir = Path(args.pool_dir)
    adapters = ModelPoolAdapters.load(args.in_adapters)
    trained = [t.strip() for t in args.trained_tags.split(",") if t.strip()]
    heldout = [t.strip() for t in args.heldout_tags.split(",") if t.strip()]
    d_shared = adapters.d_shared

    # Find passages with teacher z (same gating used elsewhere).
    passages = [json.loads(l) for l in (pool_dir / "passages.jsonl").read_text().splitlines() if l.strip()]
    keep_idx = [i for i, r in enumerate(passages) if (r.get("z") or "").strip()]
    N = len(keep_idx)
    print(f"[refit-held] {N} passages with z")

    # Build target = mean over trained T of enc_M_T(h_T(p)), normalized to √d_shared
    # (same normalization the trainer applies at injection time).
    inj_scale = math.sqrt(d_shared)
    target = torch.zeros(N, d_shared, dtype=torch.float32)
    cnt = 0
    for tag in trained:
        meta_path = pool_dir / f"{tag}.meta.json"
        if not meta_path.exists():
            print(f"  [{tag}] no shard — skipping in target avg")
            continue
        shard_meta = json.loads(meta_path.read_text())
        h = load_file(str(pool_dir / shard_meta["shard"]))["h"].float()[keep_idx]   # [N, d_M]
        enc = adapters.encoders[tag]
        with torch.no_grad():
            proj = enc(h).float()                                                    # [N, d_shared]
            proj_n = normalize_activation(proj, inj_scale)
        target += proj_n
        cnt += 1
        print(f"  + {tag} added to target (||proj||={proj_n.norm(dim=-1).mean().item():.2f})")
    if cnt == 0:
        raise SystemExit("no trained-tag shards found, nothing to fit against")
    target /= cnt
    print(f"[refit-held] target shape={tuple(target.shape)} mean||target||={target.norm(dim=-1).mean().item():.2f}")

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(N, generator=g)
    cut = int(N * args.train_frac)
    tr, ev = perm[:cut], perm[cut:]

    report = {}
    for tag in heldout:
        meta_path = pool_dir / f"{tag}.meta.json"
        if not meta_path.exists():
            print(f"  [{tag}] no shard — skipping")
            continue
        shard_meta = json.loads(meta_path.read_text())
        h = load_file(str(pool_dir / shard_meta["shard"]))["h"].float()[keep_idx]   # [N, d_M]
        d_M = h.shape[1]
        # Row-normalize h to √d_M before lstsq (matches the lstsq_init recipe used
        # in init_adapters.py — otherwise outlier channels dominate).
        h_n = normalize_activation(h, math.sqrt(d_M))
        X_tr, Y_tr = h_n[tr], target[tr]
        X_ev, Y_ev = h_n[ev], target[ev]
        # CPU lstsq with gelsy.
        W = torch.linalg.lstsq(X_tr, Y_tr, driver="gelsy").solution                  # [d_M, d_shared]
        pred = X_ev @ W
        var_tot = Y_ev.var(unbiased=False).item()
        var_res = (Y_ev - pred).var(unbiased=False).item()
        fve = 1.0 - var_res / max(var_tot, 1e-12)
        new_enc = LinearAdapter(d_in=d_M, d_out=d_shared)
        with torch.no_grad():
            new_enc.weight.copy_(W.t())
        # Replace enc_M, keep dec_M.
        adapters.set_pair(tag, new_enc, adapters.decoders[tag])
        report[tag] = {"d_M": d_M, "fve_enc_eval": round(fve, 4)}
        print(f"  [{tag:18s}] d_M={d_M} FVE_enc={fve:+.4f}")

    out = Path(args.out_adapters)
    adapters.save(out)
    (out / "refit_heldout_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"[refit-held] saved → {out}")


if __name__ == "__main__":
    main()
