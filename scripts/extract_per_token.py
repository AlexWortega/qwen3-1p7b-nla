"""Extract per-TOKEN layer-l activations (not mean-pooled) for the multi-model pool.

Output layout (per tag):
    <tag>.pt.safetensors
        h:       [N, T_max, d_M] fp16
        lengths: [N] int32          — number of content tokens per row (<= T_max)

Mean-pool throws away per-position info that the gold summaries care about
(facts, named entities). With the per-token sequence available, the head on
top of AV can learn attention-pooling and pick out the right tokens.

Storage: at T_max=192 fp16, the biggest model (d=2048) takes 7.9 GB per
10k-passage shard. We keep T_max conservative to stay within eva01's free
disk.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.arch_adapters import resolve_decoder_layers, resolve_text_config

from scripts.extract_multi import depth_fraction_to_layer, materialize_passages  # reuse


@torch.no_grad()
def extract_per_token(model_name: str, tag: str, passages: list[str], out_dir: Path,
                      depth_fraction: float, layer_index: int | None,
                      max_length: int, batch_size: int, dtype: torch.dtype) -> dict:
    print(f"[{tag}] loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(target_device).eval()

    text_cfg = resolve_text_config(model.config)
    d_model = text_cfg.hidden_size
    num_layers = text_cfg.num_hidden_layers
    chosen_layer = layer_index if layer_index is not None else depth_fraction_to_layer(num_layers, depth_fraction)
    print(f"[{tag}] d_model={d_model} num_layers={num_layers} layer={chosen_layer} max_len={max_length}")

    layers = resolve_decoder_layers(model)
    captured: torch.Tensor | None = None

    def hook(_mod, _inp, output):
        nonlocal captured
        h = output[0] if isinstance(output, tuple) else output
        captured = h.detach().clone()

    handle = layers[chosen_layer].register_forward_hook(hook)

    N = len(passages)
    H_all = torch.empty(N, max_length, d_model, dtype=torch.float16)
    lengths = torch.empty(N, dtype=torch.int32)
    try:
        for start in tqdm(range(0, N, batch_size), desc=f"[{tag}]"):
            sub = passages[start : start + batch_size]
            enc = tokenizer(sub, return_tensors="pt", padding="max_length",
                            truncation=True, max_length=max_length, add_special_tokens=True)
            input_ids = enc["input_ids"].to(target_device)
            attention_mask = enc["attention_mask"].to(target_device)
            captured = None
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            assert captured is not None
            h_fp16 = captured.float().to(torch.float16).cpu()    # [B, T, d]
            seq_lens = attention_mask.sum(dim=1).cpu().to(torch.int32)
            n = h_fp16.shape[0]
            H_all[start : start + n] = h_fp16
            lengths[start : start + n] = seq_lens
    finally:
        handle.remove()

    shard_path = out_dir / f"{tag}.pt.safetensors"
    save_file({"h": H_all, "lengths": lengths}, str(shard_path))
    meta = {
        "tag": tag, "model": model_name, "d_model": d_model,
        "num_layers": num_layers, "layer_index": chosen_layer,
        "depth_fraction": depth_fraction, "n_passages": N,
        "max_length": max_length, "shard": shard_path.name, "pool": "per_token_fp16",
    }
    (out_dir / f"{tag}.pt.meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(f"[{tag}] wrote {shard_path} h.shape={tuple(H_all.shape)} mean_len={float(lengths.float().mean()):.1f}")

    del model, tokenizer, layers, captured, H_all, lengths
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="same format as extract_multi.yaml")
    ap.add_argument("--tags", default=None, help="comma-separated subset of pool tags to extract")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    passages = materialize_passages(out_dir / "passages.jsonl",
                                    n_passages=int(cfg["n_passages"]),
                                    max_chars=int(cfg.get("max_chars", 4000)),
                                    seed=int(cfg.get("seed", 0)))

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[cfg.get("dtype", "fp16")]
    pool_entries = cfg["pool"]
    if args.tags:
        wanted = {t.strip() for t in args.tags.split(",")}
        pool_entries = [e for e in pool_entries if e["tag"] in wanted]
        print(f"[extract-pt] subset: {[e['tag'] for e in pool_entries]}")

    index: dict[str, dict] = {}
    for entry in pool_entries:
        tag = entry["tag"]
        shard_path = out_dir / f"{tag}.pt.safetensors"
        meta_path = out_dir / f"{tag}.pt.meta.json"
        if shard_path.exists() and meta_path.exists() and not cfg.get("force", False):
            print(f"[{tag}] per-token already extracted — skipping")
            index[tag] = json.loads(meta_path.read_text())
            continue
        meta = extract_per_token(
            model_name=entry["model"], tag=tag, passages=passages, out_dir=out_dir,
            depth_fraction=float(entry.get("depth_fraction", cfg.get("depth_fraction", 0.5))),
            layer_index=entry.get("layer_index"),
            max_length=int(cfg.get("pt_max_length", 192)),
            batch_size=int(cfg.get("pt_batch_size", 4)),
            dtype=dtype,
        )
        index[tag] = meta

    (out_dir / "pt_index.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    print(f"[done] pt_index → {out_dir / 'pt_index.json'} ({len(index)} models)")


if __name__ == "__main__":
    main()
