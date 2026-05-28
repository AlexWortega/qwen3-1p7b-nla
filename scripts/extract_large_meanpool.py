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
    ap.add_argument("--depth-fraction", type=float, default=0.5,
                    help="Used only if --layer-idx is not set.")
    ap.add_argument("--layer-idx", type=int, default=None,
                    help="Exact layer index to capture. Overrides --depth-fraction.")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16",
                    help="Model load dtype. Some archs (Gemma3, Bloom) overflow "
                         "in fp16 at mid-layer attention sinks — use bf16 or fp32.")
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--start-idx", type=int, default=0,
                    help="First passage index (inclusive). For parallel sharding.")
    ap.add_argument("--end-idx", type=int, default=None,
                    help="Last passage index (exclusive). For parallel sharding.")
    ap.add_argument("--shard-suffix", default="",
                    help="If set, output is <tag>{suffix}.safetensors with row=0..(end-start). "
                         "Use with --start-idx/--end-idx to produce per-range shards, "
                         "then merge externally.")
    ap.add_argument("--save-every", type=int, default=500,
                    help="Save partial shard to disk every N passages (rounded to batch). "
                         "On rerun, the script resumes from the first all-zero row. "
                         "Set to 0 to disable checkpointing (only save at end).")
    args = ap.parse_args()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    pool_dir = Path(args.pool_dir)
    passages = [json.loads(l) for l in (pool_dir / "passages.jsonl").read_text().splitlines() if l.strip()]
    all_texts = [p["text"] for p in passages]
    end_idx = args.end_idx if args.end_idx is not None else len(all_texts)
    texts = all_texts[args.start_idx:end_idx]
    N = len(texts)
    print(f"[extract-large] tag={args.tag} model={args.model} "
          f"range=[{args.start_idx}:{end_idx}) N={N}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token

    print(f"[extract-large] loading model with device_map='auto' ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation=args.attn,
        device_map="auto", trust_remote_code=True,
    ).eval()
    for p in model.parameters(): p.requires_grad_(False)

    cfg = resolve_text_config(model.config)
    n_layers = int(cfg.num_hidden_layers)
    if args.layer_idx is not None:
        layer = int(args.layer_idx)
        assert 0 <= layer < n_layers, f"--layer-idx {layer} out of [0,{n_layers})"
    else:
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

    shard_name = f"{args.tag}{args.shard_suffix}.safetensors"
    shard_path = pool_dir / shard_name

    # Resume support: if a partial shard exists with the right shape, reuse rows
    # whose norm is > 0 (zeros = untouched) and resume from the first zero row.
    h_out = torch.zeros(N, d_M, dtype=torch.float32)
    resume_from = 0
    if shard_path.exists():
        try:
            from safetensors.torch import load_file as _lf
            prev = _lf(str(shard_path))["h"]
            if prev.shape == (N, d_M):
                h_out = prev.float().contiguous()
                row_done = (h_out.abs().sum(dim=-1) > 0)
                # First row not yet computed.
                idx_zero = (~row_done).nonzero(as_tuple=True)[0]
                resume_from = int(idx_zero[0].item()) if idx_zero.numel() else N
                print(f"[extract-large] resuming: {resume_from}/{N} rows already done in {shard_path}")
            else:
                print(f"[extract-large] existing shard shape {tuple(prev.shape)} ≠ ({N},{d_M}) — starting fresh")
        except Exception as e:
            print(f"[extract-large] can't read existing shard ({e!r}) — starting fresh")

    save_every_rows = max(0, int(args.save_every))
    try:
        last_saved = resume_from
        # Align resume to a batch boundary (saves are also batch-aligned).
        for s in range(resume_from - (resume_from % args.batch_size), N, args.batch_size):
            if s + args.batch_size <= resume_from:
                continue
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
            # Periodic checkpoint.
            if save_every_rows and (s + args.batch_size - last_saved) >= save_every_rows:
                save_file({"h": h_out}, str(shard_path))
                last_saved = s + args.batch_size
                print(f"  [{args.tag}] checkpoint @ {last_saved}/{N} → {shard_path}")
    finally:
        handle.remove()

    # shard_path/shard_name already set above (used by resume + checkpointing).
    save_file({"h": h_out}, str(shard_path))

    # Only update index.json for the canonical (un-suffixed) shard.
    if not args.shard_suffix:
        idx_path = pool_dir / "index.json"
        idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
        idx[args.tag] = {"shard": shard_name, "d_model": d_M,
                         "layer": layer, "n_passages": N}
        idx_path.write_text(json.dumps(idx, indent=2, sort_keys=True))
    print(f"[extract-large] wrote {shard_path}  d={d_M} rows={N}")


if __name__ == "__main__":
    main()
