"""Activation extraction for the auditing experiment. Two modes:

  --mode pool : mean-pool layer-L over content tokens of a passages.jsonl
                (replicates extract_multi). Used to refit a held-out enc_M for a
                new layer. Writes <tag>.safetensors {"h":[N,d]} + <tag>.meta.json
                (the format add_held_out.py reads).

  --mode chat : for each {user,assistant} transcript in a battery, forward the
                full chat sequence and capture layer-L hidden states at
                  * the assistant control token (last header token), and
                  * the mean over assistant-response tokens.
                Writes acts_<tag>_ctrl.safetensors and acts_<tag>_mean.safetensors
                ({"h":[N,d]}, aligned to battery order) + battery_meta.json.

Layer is a forward-hook index on the decoder layer list (matches extract_multi /
the existing qwen2p5-7b tag, which hooked layers[20]). fp32 cast before pooling
to avoid the fp16 attention-sink overflow (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.arch_adapters import resolve_decoder_layers, resolve_text_config


def load_model(base, adapter, dtype):
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype,
                                                 trust_remote_code=True).to("cuda:0").eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).to("cuda:0").eval()
    return tok, model


def make_hook(store):
    def hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        store["h"] = h.detach().clone()
    return hook


def decoder_layers(model):
    # unwrap PEFT (get_base_model -> underlying ForCausalLM); resolve_decoder_layers
    # handles the rest of the unwrap to the decoder layer list.
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    return resolve_decoder_layers(inner)


@torch.no_grad()
def run_pool(args, tok, model, layer):
    rows = [json.loads(l) for l in Path(args.passages).read_text().splitlines() if l.strip()]
    texts = [r["text"] for r in rows]
    text_cfg = resolve_text_config(model.config)
    d = text_cfg.hidden_size
    layers = decoder_layers(model)
    store = {}
    handle = layers[layer].register_forward_hook(make_hook(store))
    tok.padding_side = "right"
    out = torch.empty(len(texts), d, dtype=torch.float32)
    bs = args.batch_size
    try:
        for s in range(0, len(texts), bs):
            sub = texts[s:s + bs]
            enc = tok(sub, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_length, add_special_tokens=True)
            ids = enc["input_ids"].cuda(); am = enc["attention_mask"].cuda()
            store.clear(); model(input_ids=ids, attention_mask=am, use_cache=False)
            h = store["h"].float()
            m = am.unsqueeze(-1).float()
            pooled = ((h * m).sum(1) / m.sum(1).clamp_min(1)).cpu()
            out[s:s + pooled.shape[0]] = pooled
            if s % (bs * 50) == 0:
                print(f"[pool] {s}/{len(texts)}")
    finally:
        handle.remove()
    from safetensors.torch import save_file
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    shard = f"{args.tag}.safetensors"
    save_file({"h": out}, str(out_dir / shard))
    (out_dir / f"{args.tag}.meta.json").write_text(json.dumps(
        {"tag": args.tag, "model": args.base, "d_model": d, "layer_index": layer,
         "n_passages": len(texts), "shard": shard}, indent=2))
    # also need a passages.jsonl in the pool dir for any dataset loader
    if not (out_dir / "passages.jsonl").exists():
        (out_dir / "passages.jsonl").write_text("\n".join(
            json.dumps({"text": t}) for t in texts) + "\n")
    print(f"[pool] wrote {shard} [{len(texts)}, {d}] -> {out_dir}")


@torch.no_grad()
def run_chat(args, tok, model, layer):
    raw = Path(args.battery).read_text()
    try:
        battery = json.loads(raw)
    except json.JSONDecodeError:
        battery = [json.loads(l) for l in raw.splitlines() if l.strip()]  # JSONL
    text_cfg = resolve_text_config(model.config)
    d = text_cfg.hidden_size
    layers = decoder_layers(model)
    store = {}
    handle = layers[layer].register_forward_hook(make_hook(store))
    ctrl = torch.empty(len(battery), d, dtype=torch.float32)
    mean = torch.empty(len(battery), d, dtype=torch.float32)
    try:
        for i, item in enumerate(battery):
            msgs_full = [{"role": "user", "content": item["user"]},
                         {"role": "assistant", "content": item["assistant"]}]
            full = tok.apply_chat_template(msgs_full, tokenize=True, add_generation_prompt=False)
            hdr = tok.apply_chat_template(msgs_full[:1], tokenize=True, add_generation_prompt=True)
            full = full[:args.max_length]
            hdr_len = min(len(hdr), len(full))
            ids = torch.tensor([full], device="cuda")
            store.clear(); model(input_ids=ids, use_cache=False)
            h = store["h"][0].float()  # [T, d]
            ctrl[i] = h[max(hdr_len - 1, 0)].cpu()
            asst = h[hdr_len:] if h.shape[0] > hdr_len else h[-1:]
            mean[i] = asst.mean(0).cpu()
            if i % 20 == 0:
                print(f"[chat] {i}/{len(battery)}")
    finally:
        handle.remove()
    from safetensors.torch import save_file
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    save_file({"h": ctrl}, str(out_dir / f"acts_{args.tag}_ctrl.safetensors"))
    save_file({"h": mean}, str(out_dir / f"acts_{args.tag}_mean.safetensors"))
    (out_dir / "battery_meta.json").write_text(json.dumps(
        {"ids": [b.get("id", str(i)) for i, b in enumerate(battery)],
         "categories": [b.get("category", b.get("bias", "")) for b in battery],
         "n": len(battery), "d_model": d, "layer_index": layer}, indent=2))
    print(f"[chat] wrote acts_{args.tag}_{{ctrl,mean}} [{len(battery)}, {d}] -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pool", "chat"], required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--layer", type=int, required=True, help="forward-hook decoder layer index")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--passages", default=None, help="[pool] passages.jsonl")
    ap.add_argument("--battery", default=None, help="[chat] battery json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()
    tok, model = load_model(args.base, args.adapter, torch.float16)
    if args.mode == "pool":
        assert args.passages, "--passages required for pool mode"
        run_pool(args, tok, model, args.layer)
    else:
        assert args.battery, "--battery required for chat mode"
        run_chat(args, tok, model, args.layer)


if __name__ == "__main__":
    main()
