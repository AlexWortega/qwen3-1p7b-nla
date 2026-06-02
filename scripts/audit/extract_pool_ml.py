"""v15.2 — MULTI-LAYER mean-pool extraction for the universal AV pool.

For each tag, forward the shared v9 passage corpus through the model ONCE with K
forward-hooks on K decoder layers (depth-fractions ~[0.25,0.5,0.75,0.9], clamped),
mean-pool each captured layer over content (non-pad) tokens in fp32, and write ONE
stacked shard per tag:

    <out_dir>/<tag>_ml.safetensors      {"h": [N, K, d_M] fp32}
    <out_dir>/<tag>.meta.json           {model, d_model, num_layers, layer_indices,
                                          depth_fractions, n_passages, ...}
    <out_dir>/index_ml.json             {tag: meta, ...}   (NEW file, NOT the pool index)

CRITICAL: writes ONLY into --out-dir (default activations_pool_v9_ml). Never touches
/big/activations_pool_v9/index.json (which extract_multi.py clobbers).

fp32 cast BEFORE the pooling sum (CLAUDE.md: fp16 sum over attention-sink channels
overflows to ±inf and poisons downstream lstsq).
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.arch_adapters import resolve_decoder_layers, resolve_text_config

# tag -> HF model id (from configs/universal/extract_v9_multilayer.yaml).
TAG_MODEL = {
    "qwen3-1p7b": "Qwen/Qwen3-1.7B",
    "phi-1p5": "microsoft/phi-1_5",
    "smollm3-3b": "HuggingFaceTB/SmolLM3-3B",
    "qwen2p5-7b": "Qwen/Qwen2.5-7B",
    "gemma2": "google/gemma-2-9b-it",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M",
    "qwen3-0p6b": "Qwen/Qwen3-0.6B-Base",
    "pythia-410m": "EleutherAI/pythia-410m-deduped",
    "gpt2-medium": "openai-community/gpt2-medium",
    "qwen2p5-0p5b": "Qwen/Qwen2.5-0.5B",
    "gpt-neo-1p3b": "EleutherAI/gpt-neo-1.3B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "lfm-7b": "LiquidAI/LFM2-1.2B",
    "yagpt-5-8b": "yandex/YandexGPT-5-Lite-8B-pretrain",
    "vikhr-7b-01": "Vikhrmodels/Vikhr-7b-0.1",
}


def depth_to_layer(num_layers: int, frac: float) -> int:
    idx = int(round(frac * (num_layers - 1)))
    return max(0, min(num_layers - 1, idx))


def layer_indices(num_layers: int, fracs: list[float]) -> list[int]:
    """Map depth fractions to UNIQUE clamped layer indices (sorted)."""
    idxs = sorted({depth_to_layer(num_layers, f) for f in fracs})
    return idxs


@torch.no_grad()
def extract_one(model_id, tag, passages, out_dir, fracs, max_length, batch_size, dtype):
    print(f"[{tag}] loading {model_id} ...", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.truncation_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True).to("cuda:0").eval()

    cfg = resolve_text_config(model.config)
    d_model = cfg.hidden_size
    num_layers = cfg.num_hidden_layers
    Ls = layer_indices(num_layers, fracs)
    K = len(Ls)
    print(f"[{tag}] d={d_model} num_layers={num_layers} -> layers {Ls} (fracs {fracs})", flush=True)

    layers = resolve_decoder_layers(model)
    store: dict[int, torch.Tensor] = {}

    def mk(L):
        return lambda _m, _i, o: store.__setitem__(
            L, (o[0] if isinstance(o, tuple) else o).detach())

    handles = [layers[L].register_forward_hook(mk(L)) for L in Ls]

    out = torch.empty(len(passages), K, d_model, dtype=torch.float32)
    try:
        for s in range(0, len(passages), batch_size):
            sub = passages[s:s + batch_size]
            enc = tok(sub, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_length, add_special_tokens=True)
            dev = model.get_input_embeddings().weight.device
            ids = enc["input_ids"].to(dev)
            am = enc["attention_mask"].to(dev)
            store.clear()
            model(input_ids=ids, attention_mask=am, use_cache=False)
            m = am.unsqueeze(-1).float()                  # [B,T,1]
            den = m.sum(1).clamp_min(1)                   # [B,1]
            for k, L in enumerate(Ls):
                h = store[L].float()                      # [B,T,d] fp32 BEFORE sum
                pooled = ((h * m).sum(1) / den).cpu()     # [B,d]
                out[s:s + pooled.shape[0], k] = pooled
            if s % (batch_size * 25) == 0:
                print(f"[{tag}] {s}/{len(passages)}", flush=True)
    finally:
        for hd in handles:
            hd.remove()

    save_file({"h": out}, str(out_dir / f"{tag}_ml.safetensors"))
    meta = {
        "tag": tag, "model": model_id, "d_model": d_model, "num_layers": num_layers,
        "layer_indices": Ls, "depth_fractions": fracs, "n_layers": K,
        "n_passages": len(passages), "max_length": max_length,
        "pool": "mean_content_tokens", "shard": f"{tag}_ml.safetensors",
        "layout": "[N, K, d_M]",
    }
    (out_dir / f"{tag}.meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(f"[{tag}] wrote {tag}_ml.safetensors {tuple(out.shape)}", flush=True)
    del model, tok, layers, store
    gc.collect()
    torch.cuda.empty_cache()
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", default="/big/activations_pool_v9",
                    help="source of passages.jsonl (NEVER written to)")
    ap.add_argument("--out-dir", default="/big/activations_pool_v9_ml",
                    help="NEW dir for multi-layer shards; never the pool dir")
    ap.add_argument("--tags", default="qwen3-1p7b,phi-1p5,smollm3-3b",
                    help="comma list of AV tags to extract multi-layer")
    ap.add_argument("--fracs", default="0.25,0.5,0.75,0.9",
                    help="depth fractions -> K unique clamped layers")
    ap.add_argument("--n-passages", type=int, default=0,
                    help="cap passages (0 = all). Smoke uses a small cap.")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    args = ap.parse_args()

    assert Path(args.out_dir) != Path(args.pool_dir), "refuse to write into the pool dir"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    fracs = [float(x) for x in args.fracs.split(",")]

    pool = Path(args.pool_dir)
    passages = [json.loads(l)["text"] for l in (pool / "passages.jsonl").read_text().splitlines() if l.strip()]
    if args.n_passages > 0:
        passages = passages[:args.n_passages]
    print(f"[extract_pool_ml] {len(passages)} passages, fracs={fracs}, tags={args.tags}", flush=True)

    index_path = out_dir / "index_ml.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    for tag in args.tags.split(","):
        tag = tag.strip()
        model_id = TAG_MODEL.get(tag)
        assert model_id, f"no model id for tag {tag!r} (add to TAG_MODEL)"
        meta = extract_one(model_id, tag, passages, out_dir, fracs,
                           args.max_length, args.batch_size, dtype)
        index[tag] = meta
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
    print(f"[extract_pool_ml] done -> {index_path} ({len(index)} tags)", flush=True)


if __name__ == "__main__":
    main()
