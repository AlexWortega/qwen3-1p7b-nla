"""
Held-out FVE multi-seed experiment for reviewer point 3.4.

For each of 5 held-out architectures, re-runs the lstsq enc/dec init with 3
different passage-subset seeds (train/eval split), then computes FVE_meannorm
using the FIXED trained AR from rl_multi_v6 (no trunk retrain).

Mirrors the key_fixes: mean-pool fp32, L2-normalize, gelsy lstsq, direct dec_M.
The AR produces ĥ_shared = normalize(dec_M(AR(z_gold))) using gold teacher z from
the passages pool (AR-alone regime = upper bound, no AV inference needed).

Usage (inside Docker on eva01):
  POOL=/big/../activations_pool_300m
  AR=/mnt/storage/vae_llm/artifacts/rl_multi_v6
  ADAPTERS=/mnt/storage/vae_llm/artifacts/adapters_v8_mixed_direct

  python /home/alexw/vae_llm/extra_exps/fve_multiseed_remote.py \
      --pool-dir $POOL \
      --ar-dir $AR \
      --adapters-dir $ADAPTERS \
      --out /home/alexw/vae_llm/extra_exps_out/fve_multiseed_raw.json

Seeds: 0, 42, 1337 (passage split seed into split_train_eval)
Held-out: rugpt3-large deepseek-llm-7b vikhr-7b-01 yagpt-5-8b lfm-7b
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

HELDOUT_TAGS = ["rugpt3-large", "deepseek-llm-7b", "vikhr-7b-01", "yagpt-5-8b", "lfm-7b"]
SEEDS = [0, 42, 1337]
N_PASSAGES = 200
TRAIN_FRAC = 0.8   # same as init_adapters default; we eval on the held-back 20%


def normalize_activation(h: torch.Tensor, scale: float) -> torch.Tensor:
    n = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return h / n * scale


def per_tag_fve_meannorm(h_pred: torch.Tensor, h_gold: torch.Tensor) -> float:
    d = h_gold.shape[-1]
    scale = math.sqrt(d)
    p = normalize_activation(h_pred.float(), scale)
    g = normalize_activation(h_gold.float(), scale)
    resid_var = (g - p).var(unbiased=False).item()
    gold_var = g.var(unbiased=False).item()
    return 1.0 - resid_var / max(gold_var, 1e-12)


def load_shard_and_normalize(pool_dir: Path, tag: str):
    meta = json.loads((pool_dir / f"{tag}.meta.json").read_text())
    sd = load_file(str(pool_dir / meta["shard"]))
    h = sd["h"].float()
    d = meta["d_model"]
    h = normalize_activation(h, math.sqrt(d))
    return h, meta


def lstsq_fit(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Fit W: X @ W.T ≈ Y via gelsy. Returns W [d_out, d_in]."""
    # X: [N, d_in], Y: [N, d_out]
    # gelsy (not gelsd) to avoid MKL crash on rank-deficient matrices
    sol = torch.linalg.lstsq(X, Y, driver="gelsy")
    return sol.solution.T   # [d_out, d_in]


def split_indices(n: int, train_frac: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    cut = int(n * train_frac)
    return perm[:cut], perm[cut:]


@torch.no_grad()
def ar_reconstruct_batch(ar, ar_tok, z_texts: list[str], device: str,
                          d_shared: int, inj_scale: float) -> torch.Tensor:
    """Mirrors eval_fve_multi.py's ar_reconstruct_batch logic."""
    CRITIC_TEMPLATE = "Summary of the following text: <text>{z}</text> <summary>"
    prompts = [CRITIC_TEMPLATE.format(z=z.strip()) for z in z_texts]
    enc = ar_tok(prompts, return_tensors="pt", padding=True, truncation=True,
                 max_length=512, add_special_tokens=False).to(device)
    out = ar(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    lengths = enc["attention_mask"].sum(dim=1) - 1
    idx = lengths.view(-1, 1, 1).expand(-1, 1, d_shared)
    pred = out.values.gather(1, idx).squeeze(1).float()
    return normalize_activation(pred, inj_scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--ar-dir", required=True)
    ap.add_argument("--adapters-dir", required=True,
                    help="adapters_v8_mixed_direct (has enc+dec for all tags)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-passages", type=int, default=N_PASSAGES)
    ap.add_argument("--anchor-tag", default="qwen3-1p7b",
                    help="anchor for lstsq (d_model must equal d_shared=2048)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool_dir = Path(args.pool_dir)
    ar_dir = Path(args.ar_dir)

    # Load AR model (trunk + LoRA)
    # Load from rl_multi_v6/nla_meta.yaml for trunk name
    meta_yaml = ar_dir / "nla_meta.yaml"
    import yaml
    meta = yaml.safe_load(meta_yaml.read_text())
    ar_trunk = meta.get("ar_base", "Qwen/Qwen3-1.7B")
    d_shared = int(meta.get("d_shared", 2048))
    inj_scale = math.sqrt(d_shared)

    from nla.models import NLACriticModel
    print(f"[fve_multiseed] Loading AR trunk: {ar_trunk}")
    base_critic = NLACriticModel.from_pretrained(ar_trunk, torch_dtype=torch.float16,
                                                  attn_implementation="sdpa")
    # Re-apply identity init on value_head (PEFT doesn't persist it)
    # Load saved value_head if available, otherwise apply identity
    value_head_path = ar_dir / "ar" / "value_head.pt"
    if value_head_path.exists():
        vh_sd = torch.load(str(value_head_path), map_location="cpu")
        with torch.no_grad():
            base_critic.value_head.weight.copy_(vh_sd["weight"])
        print(f"[fve_multiseed] Loaded value_head from {value_head_path}")
    else:
        d_vhead = base_critic.value_head.weight.shape[0]
        with torch.no_grad():
            base_critic.value_head.weight.copy_(
                torch.eye(d_vhead, dtype=base_critic.value_head.weight.dtype))
        print("[fve_multiseed] Applied identity init to value_head")
    ar = PeftModel.from_pretrained(base_critic, str(ar_dir / "ar")).to(device).eval()
    ar_tok = AutoTokenizer.from_pretrained(ar_trunk)
    if ar_tok.pad_token is None:
        ar_tok.pad_token = ar_tok.eos_token

    # Load anchor activations
    print(f"[fve_multiseed] Loading anchor: {args.anchor_tag}")
    H_anchor_all, anchor_meta = load_shard_and_normalize(pool_dir, args.anchor_tag)
    N = H_anchor_all.shape[0]

    # Load passages (for z_gold)
    passages_path = pool_dir / "passages.jsonl"
    passages = [json.loads(l) for l in passages_path.read_text().splitlines() if l.strip()]

    results = {}  # seed -> tag -> {fve_pipeline_meannorm, fve_ar_alone_meannorm}

    for seed in SEEDS:
        print(f"\n=== SEED {seed} ===")
        results[str(seed)] = {}
        train_idx, eval_idx = split_indices(N, TRAIN_FRAC, seed)
        H_anchor_tr = H_anchor_all[train_idx]
        H_anchor_ev = H_anchor_all[eval_idx]

        # Sample eval passage IDs (intersection of eval_idx and passages with z_gold)
        eval_pids_pool = eval_idx.tolist()
        rng = random.Random(seed)
        valid_eval = [pid for pid in eval_pids_pool
                      if pid < len(passages) and passages[pid].get("z")]
        eval_pids = rng.sample(valid_eval, min(args.n_passages, len(valid_eval)))
        z_golds = [passages[pid]["z"] for pid in eval_pids]

        for tag in HELDOUT_TAGS:
            print(f"  [{seed}] {tag} ...", flush=True)
            try:
                H_tag_all, tag_meta = load_shard_and_normalize(pool_dir, tag)
                d_M = tag_meta["d_model"]
                H_tag_tr = H_tag_all[train_idx]
                H_tag_ev = H_tag_all[eval_idx]

                # Re-fit enc_M: H_tag_tr @ enc.T ≈ H_anchor_tr
                # enc.T: [d_M, d_shared] → enc: [d_shared, d_M]
                enc_W = lstsq_fit(H_tag_tr, H_anchor_tr)  # [d_shared, d_M]

                # Re-fit dec_M: H_anchor_tr @ dec.T ≈ H_tag_tr
                # dec_M maps d_shared → d_M (post-AR direction)
                # For AR-alone regime, AR outputs d_shared normalized predictions
                # and dec_M maps those to h_M.
                # Direct: fit dec(AR_pred) ≈ h_M on eval passages.
                # But we don't have the AR predictions yet — compute them first.

                # Get AR predictions for eval passages
                BATCH = 16
                ar_preds = []
                for start in range(0, len(z_golds), BATCH):
                    batch_z = z_golds[start:start+BATCH]
                    pred = ar_reconstruct_batch(ar, ar_tok, batch_z, device, d_shared, inj_scale)
                    ar_preds.append(pred.cpu())
                ar_preds_t = torch.cat(ar_preds, dim=0)  # [n_eval, d_shared]

                # Corresponding h_M gold for eval passages
                h_M_ev_list = []
                for pid in eval_pids:
                    h_M_ev_list.append(H_tag_all[pid])
                h_M_ev = torch.stack(h_M_ev_list, dim=0)  # [n_eval, d_M]

                # Fit dec_M: ar_preds_t @ dec_W.T ≈ h_M_ev
                # Use TRAIN-split AR preds for fitting, EVAL for measuring FVE.
                # We only have eval split AR preds; to avoid overfitting we use
                # a secondary train/eval split within the eval set.
                n_sub = len(eval_pids)
                cut_sub = int(n_sub * 0.5)
                dec_train_X = ar_preds_t[:cut_sub].float()
                dec_train_Y = h_M_ev[:cut_sub].float()
                dec_eval_X = ar_preds_t[cut_sub:].float()
                dec_eval_Y = h_M_ev[cut_sub:].float()

                dec_W = lstsq_fit(dec_train_X, dec_train_Y)  # [d_M, d_shared]

                # FVE (AR-alone regime) on eval subset
                h_pred = dec_eval_X @ dec_W.T  # [n, d_M]
                fve_ar_alone = per_tag_fve_meannorm(h_pred, dec_eval_Y)

                results[str(seed)][tag] = {
                    "fve_ar_alone_meannorm": round(fve_ar_alone, 4),
                    "n_eval": len(eval_pids),
                    "n_dec_train": cut_sub,
                    "n_dec_eval": n_sub - cut_sub,
                    "d_M": d_M,
                }
                print(f"  [{seed}] {tag}: fve_ar_alone_meannorm={fve_ar_alone:.4f}")
                del H_tag_all, H_tag_tr, H_tag_ev, ar_preds_t, h_M_ev

            except Exception as e:
                print(f"  [{seed}] {tag}: ERROR {e}")
                results[str(seed)][tag] = {"error": str(e)}

        del H_anchor_tr, H_anchor_ev

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[fve_multiseed] Wrote {args.out}")

    # Print summary
    print("\n=== SUMMARY ===")
    import numpy as np
    for tag in HELDOUT_TAGS:
        vals = [results[str(s)].get(tag, {}).get("fve_ar_alone_meannorm")
                for s in SEEDS
                if isinstance(results[str(s)].get(tag, {}), dict)
                and "fve_ar_alone_meannorm" in results[str(s)].get(tag, {})]
        if vals:
            print(f"  {tag}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} "
                  f"per_seed={[round(v,4) for v in vals]}")


if __name__ == "__main__":
    main()
