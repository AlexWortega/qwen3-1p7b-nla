"""Extract mean-pool activations for a large model with device_map='auto'.

Drop-in mean-pool format compatible with artifacts/activations_pool_300m: writes
<tag>.safetensors with key 'h' shape [N, d_M] fp32 and updates index.json.

Usage:
  python scripts/extract_large_meanpool.py \
    --tag qwen3-30b --model Qwen/Qwen3-30B-A3B \
    --pool-dir artifacts/activations_pool_300m \
    --depth-fraction 0.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.arch_adapters import resolve_decoder_layers, resolve_text_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool-dir", default="artifacts/activations_pool_300m")
    ap.add_argument("--depth-fraction", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    pool_dir = Path(args.pool_dir)
    passages = [json.loads(l) for l in (pool_dir / "passages.jsonl").read_text().splitlines() if l.strip()]
    texts = [p["text"] for p in passages]
    N = len(texts)
    print(f"[extract-large] tag={args.tag} model={args.model} N={N}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token

    print(f"[extract-large] loading model with device_map='auto' ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="sdpa",
        device_map="auto", trust_remote_code=True,
    ).eval()
    for p in model.parameters(): p.requires_grad_(False)

    cfg = resolve_text_config(model.config)
    n_layers = int(cfg.num_hidden_layers)
    layer = int(n_layers * args.depth_fraction)
    layer = max(0, min(n_layers - 1, layer))
    d_M = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)))
    print(f"[extract-large] n_layers={n_layers} layer={layer} d_M={d_M}")

    layers = resolve_decoder_layers(model)
    target_layer = layers[layer]
    captured = {}
    def _hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach().float().cpu()
    handle = target_layer.register_forward_hook(_hook)

    h_out = torch.empty(N, d_M, dtype=torch.float32)
    try:
        for s in range(0, N, args.batch_size):
            batch = texts[s:s + args.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                       max_length=args.max_length, add_special_tokens=True)
            ids = enc["input_ids"]
            attn = enc["attention_mask"]
            with torch.no_grad():
                model(input_ids=ids.cuda(), attention_mask=attn.cuda(), use_cache=False)
            h_batch = captured["h"]
            mask = attn.float().unsqueeze(-1)
            h_pool = ((h_batch * mask).sum(1) / mask.sum(1).clamp_min(1)).float()
            h_out[s:s + h_pool.shape[0]] = h_pool
            if (s // args.batch_size) % 50 == 0:
                print(f"  [{args.tag}] {s+args.batch_size}/{N}")
    finally:
        handle.remove()

    shard_path = pool_dir / f"{args.tag}.safetensors"
    save_file({"h": h_out}, str(shard_path))

    # Update index.json
    idx_path = pool_dir / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    idx[args.tag] = {"shard": f"{args.tag}.safetensors", "d_model": d_M,
                     "layer": layer, "n_passages": N}
    idx_path.write_text(json.dumps(idx, indent=2, sort_keys=True))
    print(f"[extract-large] wrote {shard_path}  d={d_M} rows={N}")


if __name__ == "__main__":
    main()
