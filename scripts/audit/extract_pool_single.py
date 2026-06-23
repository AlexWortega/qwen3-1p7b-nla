"""Single-layer pool extractor for add_held_out_tag. Hooks ONE decoder layer index,
mean-pools content (attention-mask) tokens in fp32 over the v9 pool passages, writes
[N, d_model] -> <out_dir>/<tag>.safetensors (+ .meta.json). Mirrors extract_pool_ml's
pooling exactly but for a single explicit --layer index (needed for LFM2 conv L7/L9)."""
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
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--passages", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--n-passages", type=int, default=0)
    ap.add_argument("--adapter", default=None,
                    help="optional LoRA adapter merged onto --model before pooling.")
    args = ap.parse_args()
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    passages = [json.loads(l)["text"] for l in Path(args.passages).read_text().splitlines() if l.strip()]
    if args.n_passages:
        passages = passages[:args.n_passages]
    try:
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dt, trust_remote_code=True).to("cuda:0").eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()
        print(f"[{args.tag}] merged adapter {args.adapter}", flush=True)
    cfg = resolve_text_config(model.config)
    num_layers = cfg.num_hidden_layers
    d_model = cfg.hidden_size
    L = args.layer
    assert 0 <= L < num_layers, f"layer {L} out of range [0,{num_layers})"
    layers = resolve_decoder_layers(model)

    store = {}

    def hook(_m, _i, o):
        store["h"] = (o[0] if isinstance(o, tuple) else o).detach()

    handle = layers[L].register_forward_hook(hook)
    out = torch.empty(len(passages), d_model, dtype=torch.float32)
    try:
        for s in range(0, len(passages), args.batch_size):
            sub = passages[s:s + args.batch_size]
            enc = tok(sub, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_length, add_special_tokens=True)
            dev = model.get_input_embeddings().weight.device
            ids = enc["input_ids"].to(dev)
            am = enc["attention_mask"].to(dev)
            store.clear()
            with torch.no_grad():
                model(input_ids=ids, attention_mask=am, use_cache=False)
            m = am.unsqueeze(-1).float()
            den = m.sum(1).clamp_min(1)
            hh = store["h"].float()
            out[s:s + hh.shape[0]] = ((hh * m).sum(1) / den).cpu()
            if s % (args.batch_size * 25) == 0:
                print(f"[{args.tag}] {s}/{len(passages)}", flush=True)
    finally:
        handle.remove()

    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    save_file({"h": out}, str(od / f"{args.tag}.safetensors"))
    (od / f"{args.tag}.meta.json").write_text(json.dumps({
        "tag": args.tag, "model": args.model, "d_model": d_model,
        "shard": f"{args.tag}.safetensors",  # so add_held_out.py can read meta["shard"]
        "num_layers": num_layers, "layer": L, "n_passages": len(passages),
        "adapter": args.adapter, "pool": "mean_content_tokens", "dtype": args.dtype}, indent=1))
    print(f"[{args.tag}] wrote [{len(passages)},{d_model}] layer={L} -> {od}/{args.tag}.safetensors", flush=True)


if __name__ == "__main__":
    main()
