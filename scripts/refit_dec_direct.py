"""Refit per-tag dec_M via lstsq on the AR's ACTUAL prediction (not on the
proxy `normalize(enc(h_M))` that refit_dec.py used).

This is the only correct objective: at eval time we feed
`dec_M(AR(z)) → ĥ_M`, so dec_M must minimize the MSE of its output vs h_M
when given AR's output as input.

Old refit_dec assumed AR(z) ≈ normalize(enc(h_M), sqrt(d_shared)), which is
only the SFT TARGET — actual AR predictions miss this target by a non-trivial
margin, and the gap multiplies through the linear dec_M into M-space.

Direct fit produces FVE_pipeline_meannorm ~ 0.79-0.88 on held-out architectures
(LFM2-1.2B, DeepSeek-7B, YaGPT-8B) that previously appeared catastrophically
broken (FVE = -0.6 / -0.3) under the old refit.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoTokenizer

from nla.enc_dec_adapters import LinearAdapter, ModelPoolAdapters
from nla.models import NLACriticModel
from nla.schema import normalize_activation


CRITIC_TEMPLATE = "Summary of the following text: <text>{z}</text> <summary>"


@torch.no_grad()
def collect_ar_outputs(ar, ar_tok, z_texts, device, d_shared, inj_scale, batch_size=32, max_len=512):
    ar_tok.padding_side = "right"
    out = torch.empty(len(z_texts), d_shared, dtype=torch.float32)
    for start in range(0, len(z_texts), batch_size):
        chunk = z_texts[start:start + batch_size]
        prompts = [CRITIC_TEMPLATE.format(z=z.strip()) for z in chunk]
        enc = ar_tok(prompts, return_tensors="pt", padding=True, truncation=True,
                     max_length=max_len, add_special_tokens=False).to(device)
        res = ar(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        lengths = enc["attention_mask"].sum(dim=1) - 1
        idx = lengths.view(-1, 1, 1).expand(-1, 1, d_shared)
        pred = res.values.gather(1, idx).squeeze(1).float()
        pred_n = normalize_activation(pred, inj_scale)
        out[start:start + pred_n.shape[0]] = pred_n.cpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-dir", required=True, help="rl_multi_v5 with av/ar/adapters")
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--in-adapters", required=True,
                    help="extended adapters (incl. all eval tags) — only enc_M from here is kept")
    ap.add_argument("--out-adapters", required=True)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rl_dir = Path(args.rl_dir)
    av_meta = yaml.safe_load((rl_dir / "nla_meta.yaml").read_text())
    ar_meta = yaml.safe_load((rl_dir / "ar" / "nla_meta.yaml").read_text())
    ar_base = ar_meta["ar_base"]
    layer_index = int(ar_meta["layer_index"])
    d_shared = int(av_meta["d_shared"])
    inj_scale = math.sqrt(d_shared)

    print(f"[refit-direct] loading AR ...")
    ar_tok = AutoTokenizer.from_pretrained(ar_base)
    if ar_tok.pad_token_id is None:
        ar_tok.pad_token = ar_tok.eos_token
    base_critic = NLACriticModel.from_pretrained(ar_base, nla_num_layers=layer_index,
                                                  torch_dtype=torch.float16, attn_implementation="sdpa")
    with torch.no_grad():
        d = base_critic.value_head.weight.shape[0]
        base_critic.value_head.weight.copy_(torch.eye(d, dtype=base_critic.value_head.weight.dtype))
    ar = PeftModel.from_pretrained(base_critic, str(rl_dir / "ar")).to(device).eval()
    for p in ar.parameters():
        p.requires_grad_(False)

    adapters = ModelPoolAdapters.load(args.in_adapters)
    pool_dir = Path(args.pool_dir)
    passages = [json.loads(l) for l in (pool_dir / "passages.jsonl").read_text().splitlines() if l.strip()]
    z_texts_all = [r.get("z") or "" for r in passages]
    keep_idx = [i for i, z in enumerate(z_texts_all) if z]
    z_texts = [z_texts_all[i] for i in keep_idx]
    N = len(z_texts)
    print(f"[refit-direct] {N} passages have z")

    # AR predictions are passage-only (independent of tag). Collect once.
    ar_out = collect_ar_outputs(ar, ar_tok, z_texts, device, d_shared, inj_scale)
    print(f"[refit-direct] ar_out shape={tuple(ar_out.shape)}")
    del ar, base_critic
    torch.cuda.empty_cache()

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(N, generator=g)
    cut = int(N * args.train_frac)
    tr, ev = perm[:cut], perm[cut:]
    X_tr = ar_out[tr]
    X_ev = ar_out[ev]

    report = {}
    for tag in adapters.tags:
        if tag == av_meta.get("anchor_tag"):
            with torch.no_grad():
                adapters.decoders[tag].weight.copy_(torch.eye(d_shared))
            report[tag] = {"anchor": True}
            print(f"  [{tag}] anchor — dec=identity")
            continue
        meta_path = pool_dir / f"{tag}.meta.json"
        if not meta_path.exists():
            print(f"  [{tag}] no shard in pool — skipping")
            continue
        shard_meta = json.loads(meta_path.read_text())
        h = load_file(str(pool_dir / shard_meta["shard"]))["h"].float()[keep_idx]
        Y_tr = h[tr]
        Y_ev = h[ev]
        d_M = h.shape[1]
        # CPU lstsq (gelsy unsupported on CUDA).
        W = torch.linalg.lstsq(X_tr, Y_tr, driver="gelsy").solution               # [d_shared, d_M]
        # FVE_meannorm sanity check on eval split.
        pred = X_ev @ W
        s = math.sqrt(d_M)
        fve = 1.0 - (normalize_activation(Y_ev, s) - normalize_activation(pred, s)).var(unbiased=False).item() / max(normalize_activation(Y_ev, s).var(unbiased=False).item(), 1e-12)
        new_dec = LinearAdapter(d_in=d_shared, d_out=d_M)
        with torch.no_grad():
            new_dec.weight.copy_(W.t())
        adapters.set_pair(tag, adapters.encoders[tag], new_dec)
        report[tag] = {"d_M": d_M, "fve_meannorm_eval": round(fve, 4)}
        print(f"  [{tag:18s}] d={d_M} FVE_mn={fve:+.4f}")

    adapters.save(args.out_adapters)
    (Path(args.out_adapters) / "refit_direct_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"[refit-direct] saved → {args.out_adapters}")


if __name__ == "__main__":
    main()
