"""Fit per-model linear (enc_M, dec_M) projections against an anchor model's
activations via closed-form lstsq, save as a ModelPoolAdapters checkpoint.

Anchor invariant: the anchor model's d_M must equal d_shared. Its enc/dec are
initialized to identity (no lstsq needed for the anchor itself).

For every other tag we fit two independent lstsq problems:
    enc_M @ : minimize ||h_M @ enc.T - h_anchor||²
    dec_M @ : minimize ||h_anchor @ dec.T - h_M||²
and report FVE on a held-out split as a sanity check.

Reads the multi-model activation pool produced by `scripts/extract_multi.py`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import math

import torch
import yaml
from safetensors.torch import load_file

from nla.enc_dec_adapters import LinearAdapter, ModelPoolAdapters
from nla.schema import normalize_activation


def load_shard(pool_dir: Path, tag: str, normalize: bool = True) -> tuple[torch.Tensor, dict]:
    """Load a per-model shard, optionally L2-normalize each row to √d_M.

    Mid-layer LLM activations have outlier dimensions (attention-sink channels)
    whose magnitude swamps the rest. Raw lstsq on these produces NaN solutions.
    Normalizing each row to a fixed L2-norm matches what the trainer does at
    injection time (`normalize_activation(h, sqrt_d_model)`), and the lstsq fit
    becomes well-conditioned.
    """
    meta = json.loads((pool_dir / f"{tag}.meta.json").read_text())
    sd = load_file(str(pool_dir / meta["shard"]))
    h = sd["h"]
    if normalize:
        h = normalize_activation(h, math.sqrt(meta["d_model"]))
    return h, meta


def split_train_eval(n: int, train_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    cut = int(n * train_frac)
    return perm[:cut], perm[cut:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", required=True, help="output of scripts/extract_multi.py")
    ap.add_argument("--config", required=True, help="yaml with anchor_tag, d_shared, optional train_frac/seed")
    ap.add_argument("--out-dir", required=True, help="where to save the ModelPoolAdapters")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    anchor_tag = cfg["anchor_tag"]
    d_shared = int(cfg["d_shared"])
    train_frac = float(cfg.get("train_frac", 0.8))
    seed = int(cfg.get("seed", 0))

    pool_dir = Path(args.pool_dir)
    index = json.loads((pool_dir / "index.json").read_text())
    assert anchor_tag in index, f"anchor_tag={anchor_tag!r} not in pool {list(index)}"

    print(f"[anchor] {anchor_tag}  d_shared={d_shared}")
    H_anchor_all, anchor_meta = load_shard(pool_dir, anchor_tag)
    assert anchor_meta["d_model"] == d_shared, (
        f"anchor d_model={anchor_meta['d_model']} != d_shared={d_shared}. "
        f"Either change anchor or change d_shared in the config."
    )
    N = H_anchor_all.shape[0]
    train_idx, eval_idx = split_train_eval(N, train_frac, seed)
    H_anchor_tr = H_anchor_all[train_idx]
    H_anchor_ev = H_anchor_all[eval_idx]

    model_dims = {tag: index[tag]["d_model"] for tag in index}
    pool = ModelPoolAdapters(d_shared=d_shared, model_dims=model_dims)

    # Anchor: identity. Bypass lstsq — perfect by construction.
    with torch.no_grad():
        pool.encoders[anchor_tag].weight.copy_(torch.eye(d_shared))
        pool.decoders[anchor_tag].weight.copy_(torch.eye(d_shared))

    report: dict[str, dict] = {anchor_tag: {"anchor": True, "fve_enc": 1.0, "fve_dec": 1.0}}
    for tag in index:
        if tag == anchor_tag:
            continue
        H_m_all, meta_m = load_shard(pool_dir, tag)
        assert H_m_all.shape[0] == N, (
            f"row-count mismatch for {tag}: {H_m_all.shape[0]} vs anchor {N}. "
            f"extract_multi should have used the same passage list."
        )
        H_m_tr = H_m_all[train_idx]
        H_m_ev = H_m_all[eval_idx]
        d_m = meta_m["d_model"]

        enc = LinearAdapter.lstsq_init(H_m_tr, H_anchor_tr)
        dec = LinearAdapter.lstsq_init(H_anchor_tr, H_m_tr)
        fve_enc = enc.lstsq_fve(H_m_ev, H_anchor_ev)
        fve_dec = dec.lstsq_fve(H_anchor_ev, H_m_ev)
        pool.set_pair(tag, enc, dec)
        report[tag] = {
            "anchor": False,
            "d_model": d_m,
            "fve_enc_eval": round(fve_enc, 4),
            "fve_dec_eval": round(fve_dec, 4),
            "n_train": int(train_idx.numel()),
            "n_eval": int(eval_idx.numel()),
        }
        print(f"[{tag}] d={d_m}  FVE_enc={fve_enc:.4f}  FVE_dec={fve_dec:.4f}")

    out_dir = Path(args.out_dir)
    pool.save(out_dir)
    (out_dir / "init_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"[done] adapters → {out_dir}")
    print(f"[done] report  → {out_dir / 'init_report.json'}")


if __name__ == "__main__":
    main()
