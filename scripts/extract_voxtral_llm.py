"""Extract the Llama-3B text LM from mistralai/Voxtral-Mini-3B-2507.

transformers 5.x has native VoxtralForConditionalGeneration; we use
get_text_config + state_dict surgery to peel off language_model.* weights
and save them as a standalone Llama-3 model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import safe_open
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="mistralai/Voxtral-Mini-3B-2507")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="language_model.",
                    help="state_dict prefix to strip (default Voxtral-style)")
    args = ap.parse_args()

    print(f"[extract] downloading {args.src} ...")
    local = Path(snapshot_download(repo_id=args.src,
                                    allow_patterns=["*.safetensors", "*.json", "tokenizer*"]))
    print(f"[extract] local = {local}")

    full_cfg = json.loads((local / "config.json").read_text())
    text_cfg_dict = dict(full_cfg["text_config"])

    # Probe real shapes from a layer-0 weight (some configs are stale).
    shards = sorted(local.glob("*.safetensors"))
    print(f"[extract] shards: {[s.name for s in shards]}")
    probe_keys = [f"{args.prefix}model.layers.0.self_attn.q_proj.weight",
                  f"{args.prefix}model.layers.0.self_attn.k_proj.weight",
                  f"{args.prefix}model.layers.0.mlp.gate_proj.weight"]
    real = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                if k in probe_keys:
                    real[k] = tuple(f.get_tensor(k).shape)
    hidden = text_cfg_dict["hidden_size"]
    q_out = real[probe_keys[0]][0]
    k_out = real[probe_keys[1]][0]
    intermediate = real[probe_keys[2]][0]
    head_dim = q_out // text_cfg_dict.get("num_attention_heads", 32)
    n_heads = q_out // head_dim
    n_kv_heads = k_out // head_dim
    print(f"[extract] probed: head_dim={head_dim} n_heads={n_heads} n_kv={n_kv_heads} intermediate={intermediate}")
    text_cfg_dict["head_dim"] = head_dim
    text_cfg_dict["num_attention_heads"] = n_heads
    text_cfg_dict["num_key_value_heads"] = n_kv_heads
    text_cfg_dict["intermediate_size"] = intermediate

    text_cfg = LlamaConfig(**text_cfg_dict)
    print(f"[extract] LlamaConfig: hidden={text_cfg.hidden_size} layers={text_cfg.num_hidden_layers} "
          f"heads={text_cfg.num_attention_heads} head_dim={text_cfg.head_dim} kv={text_cfg.num_key_value_heads} ff={text_cfg.intermediate_size}")

    model = LlamaForCausalLM(text_cfg).to(torch.bfloat16)
    target_sd_keys = set(model.state_dict().keys())

    collected = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                if not k.startswith(args.prefix):
                    continue
                base = k.removeprefix(args.prefix)
                if base in target_sd_keys:
                    collected[base] = f.get_tensor(k)
    print(f"[extract] collected {len(collected)}/{len(target_sd_keys)} expected keys")
    model.load_state_dict(collected, strict=False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    try:
        tok = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)
        tok.save_pretrained(str(out))
    except Exception as e:
        print(f"[extract] tokenizer src load failed: {e!r}")
    print(f"[extract] done → {out}")


if __name__ == "__main__":
    main()
