"""Extract the Qwen3-4B LLM out of Borealis-5b-it (audio model wrapper).

Borealis-5b-it = Whisper-Large-V3 (frozen) + 2-layer MLP audio adapter +
Qwen3-4B (jointly FT'd). The fine-tuned LLM lives at `model.llm` inside the
custom BorealisForConditionalGeneration class, with `llm.` prefix in the
state_dict.

We pull those weights into a fresh Qwen3ForCausalLM built from the nested
text_config and save as a standalone HF directory so lm-eval (and anything
else) can load it like any other Qwen3 model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import safe_open, save_file
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="Vikhrmodels/Borealis-5b-it")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[extract] downloading {args.src} ...")
    local = Path(snapshot_download(repo_id=args.src,
                                    allow_patterns=["*.safetensors", "*.json", "tokenizer*"]))
    print(f"[extract] local = {local}")

    full_cfg = json.loads((local / "config.json").read_text())
    text_cfg_dict = dict(full_cfg["text_config"])

    # The text_config in Borealis-5b-it's config.json is stale — actual saved
    # LLM has different head_dim / intermediate_size / num_kv_heads. Probe a
    # safetensors shard to recover real shapes from the q/k/mlp weights.
    shards_for_probe = sorted(local.glob("*.safetensors"))
    real_shapes: dict[str, tuple[int, int]] = {}
    needed = ["llm.model.layers.0.self_attn.q_proj.weight",
              "llm.model.layers.0.self_attn.k_proj.weight",
              "llm.model.layers.0.mlp.gate_proj.weight",
              "llm.model.layers.0.self_attn.q_norm.weight"]
    for shard in shards_for_probe:
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                if k in needed:
                    real_shapes[k] = tuple(f.get_tensor(k).shape)
    hidden = text_cfg_dict["hidden_size"]
    q_out = real_shapes["llm.model.layers.0.self_attn.q_proj.weight"][0]
    k_out = real_shapes["llm.model.layers.0.self_attn.k_proj.weight"][0]
    head_dim_probe = real_shapes["llm.model.layers.0.self_attn.q_norm.weight"][0]
    intermediate = real_shapes["llm.model.layers.0.mlp.gate_proj.weight"][0]
    n_heads = q_out // head_dim_probe
    n_kv_heads = k_out // head_dim_probe
    print(f"[extract] probed: head_dim={head_dim_probe} n_heads={n_heads} n_kv={n_kv_heads} intermediate={intermediate}")
    text_cfg_dict["head_dim"] = head_dim_probe
    text_cfg_dict["num_attention_heads"] = n_heads
    text_cfg_dict["num_key_value_heads"] = n_kv_heads
    text_cfg_dict["intermediate_size"] = intermediate
    text_cfg = Qwen3Config(**text_cfg_dict)
    print(f"[extract] text_config: hidden={text_cfg.hidden_size} layers={text_cfg.num_hidden_layers} heads={text_cfg.num_attention_heads} head_dim={text_cfg.head_dim} kv={text_cfg.num_key_value_heads} ff={text_cfg.intermediate_size}")

    # Build the destination Qwen3 model on meta to avoid mem before weight transfer.
    print(f"[extract] instantiating Qwen3ForCausalLM ...")
    model = Qwen3ForCausalLM(text_cfg).to(torch.bfloat16)
    target_sd_keys = set(model.state_dict().keys())

    # Iterate all safetensors files, pick keys with llm. prefix.
    shards = sorted(local.glob("*.safetensors"))
    print(f"[extract] safetensors shards: {[s.name for s in shards]}")
    collected = {}
    missing_llm_keys = []
    extra_keys = []
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                if not k.startswith("llm."):
                    continue
                base = k.removeprefix("llm.")
                if base in target_sd_keys:
                    collected[base] = f.get_tensor(k)
                else:
                    extra_keys.append(base)
    print(f"[extract] collected {len(collected)} llm.* tensors")
    if extra_keys:
        print(f"[extract] {len(extra_keys)} llm.* keys did NOT match Qwen3 model (e.g. {extra_keys[:5]})")
    missing = target_sd_keys - set(collected.keys())
    if missing:
        print(f"[extract] {len(missing)} expected Qwen3 keys missing from Borealis ckpt (e.g. {list(missing)[:5]})")
        # tie_word_embeddings=True → lm_head shares embed_tokens; that's expected to be missing
        # in the checkpoint, it'll get tied at save.

    model.load_state_dict(collected, strict=False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[extract] saving to {out} ...")
    model.save_pretrained(str(out))

    # Tokenizer
    try:
        tok = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)
        tok.save_pretrained(str(out))
        print(f"[extract] tokenizer saved")
    except Exception as e:
        print(f"[extract] tokenizer src load failed: {e!r}; falling back to Qwen3-4B tokenizer")
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        tok.save_pretrained(str(out))

    print(f"[extract] done → {out}")


if __name__ == "__main__":
    main()
