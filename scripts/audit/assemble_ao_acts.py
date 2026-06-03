"""v12 — vstack per-organism ctrl-token act shards into one org-acts file in the
GLOBAL transcript order (A++B++C), matching transcripts_base.jsonl order.

base acts are already in global order (extracted on transcripts_base) → just
verify/symlink. Produces:
  acts_ao_org_ctrl.safetensors   [n_global, d]  (vstack acts_A, acts_B, acts_C)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ao-dir", default="artifacts/audit/ao")
    ap.add_argument("--kind", default="ctrl", choices=["ctrl", "mean"],
                    help="which extraction to assemble (ctrl-token or mean-over-assistant)")
    ap.add_argument("--layer", type=int, default=None,
                    help="if set, assemble per-layer shards acts_ao_{org}_L{L}_{kind} "
                         "→ acts_ao_org_L{L}_{kind} (Flamingo2 multi-layer)")
    args = ap.parse_args()
    d = Path(args.ao_dir)
    k = args.kind
    suf = "" if args.layer is None else f"_L{args.layer}"
    meta = json.loads((d / "ao_data_meta.json").read_text())
    bs = meta["block_sizes"]
    parts = []
    for org in ["A", "B", "C"]:
        if bs[org] == 0:
            continue
        h = load_file(str(d / f"acts_ao_{org}{suf}_{k}.safetensors"))["h"]
        assert h.shape[0] == bs[org], f"org {org}: {h.shape[0]} acts vs block size {bs[org]}"
        parts.append(h)
    org_acts = torch.cat(parts, dim=0)
    assert org_acts.shape[0] == meta["n_global"], f"{org_acts.shape[0]} vs n_global {meta['n_global']}"
    save_file({"h": org_acts}, str(d / f"acts_ao_org{suf}_{k}.safetensors"))
    base = load_file(str(d / f"acts_ao_base{suf}_{k}.safetensors"))["h"]
    assert base.shape[0] == meta["n_global"], f"base {base.shape[0]} vs n_global {meta['n_global']}"
    print(f"[assemble] acts_ao_org{suf}_{k} {tuple(org_acts.shape)} ; base {tuple(base.shape)} ✓")


if __name__ == "__main__":
    main()
