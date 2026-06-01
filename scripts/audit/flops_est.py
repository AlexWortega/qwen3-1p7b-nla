"""Tiny compute accounting. Estimate FLOPs (6ND train / 2ND inference) and log a row to
artifacts/audit/COMPUTE_LOG.jsonl. Call from any run, or use estimate() ad hoc.

  python -m scripts.audit.flops_est --label lie_ao_ml --params 9e9 --tokens 9e6 \
      --mode train --gpu-min 90 --gpu V100
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

LOG = Path("artifacts/audit/COMPUTE_LOG.jsonl")


def estimate(n_params, n_tokens, mode="train"):
    mult = 6 if mode == "train" else 2
    return mult * float(n_params) * float(n_tokens)


def log_row(label, n_params, n_tokens, mode, gpu_min=None, gpu="V100", note=""):
    flops = estimate(n_params, n_tokens, mode)
    row = {"label": label, "params": float(n_params), "tokens": float(n_tokens),
           "mode": mode, "flops": flops, "pflops": round(flops / 1e15, 3),
           "gpu_min": gpu_min, "gpu_h": round(gpu_min / 60, 3) if gpu_min else None,
           "gpu": gpu, "note": note}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--params", type=float, required=True)
    ap.add_argument("--tokens", type=float, required=True)
    ap.add_argument("--mode", default="train", choices=["train", "infer"])
    ap.add_argument("--gpu-min", type=float, default=None)
    ap.add_argument("--gpu", default="V100")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    r = log_row(args.label, args.params, args.tokens, args.mode, args.gpu_min, args.gpu, args.note)
    print(json.dumps(r))


if __name__ == "__main__":
    main()
