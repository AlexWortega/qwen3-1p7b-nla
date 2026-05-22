"""Extract layer-l activations for the same passage corpus across a pool of models.

Output layout:
    artifacts/activations_pool/
        passages.jsonl                    # the shared corpus, one passage per line
        <tag>.safetensors                 # h: [N, d_M] fp32, one row per passage
        <tag>.meta.json                   # {model, d_M, layer_index, depth_fraction, ...}
        index.json                        # {tag: {path, d_M, layer_index, model, ...}}

Per-passage h is the mean of layer-l hidden states over content tokens (non-pad).
Mean-pooling gives one passage-level vector that's tokenizer-agnostic — different
models' tokenizers slice the same passage into different numbers of tokens, but
the mean of those positions still represents "what M thinks about this passage."

Each model is loaded, used, and unloaded sequentially so peak memory is bounded
by the largest model in the pool (not their sum).
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.arch_adapters import resolve_decoder_layers, resolve_text_config


def _patch_extra_special_tokens_list():
    """transformers 4.57 expects `extra_special_tokens` as a dict; newer
    tokenizer configs (Gemma-4-E4B's tokenizer ships `['<|video|>']`) use a
    list. Wrap the setter so list-format gets converted to {tok: tok}."""
    import transformers.tokenization_utils_base as t
    orig = t.PreTrainedTokenizerBase._set_model_specific_special_tokens
    def patched(self, special_tokens):
        if isinstance(special_tokens, list):
            special_tokens = {tok: tok for tok in special_tokens}
        return orig(self, special_tokens)
    t.PreTrainedTokenizerBase._set_model_specific_special_tokens = patched


_patch_extra_special_tokens_list()


def materialize_passages(out_path: Path, n_passages: int, max_chars: int, seed: int) -> list[str]:
    """Stream fineweb-edu sample-10BT into a flat JSONL of the first N passages
    that fit the character cap. Deterministic given (n_passages, max_chars, seed).
    Cached: if the file already exists with the right count, just read it back."""
    if out_path.exists():
        cached = [json.loads(line)["text"] for line in out_path.read_text().splitlines() if line]
        if len(cached) == n_passages:
            print(f"[passages] reusing cached {out_path} ({len(cached)} passages)")
            return cached
        print(f"[passages] cache size mismatch ({len(cached)} vs {n_passages}) — regenerating")
    print(f"[passages] streaming fineweb-edu sample-10BT → {out_path} (n={n_passages}, max_chars={max_chars})")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", streaming=True, split="train")
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    passages: list[str] = []
    with out_path.open("w") as f:
        for row in ds:
            text = row["text"]
            if not text or len(text) > max_chars:
                continue
            f.write(json.dumps({"text": text}) + "\n")
            passages.append(text)
            if len(passages) >= n_passages:
                break
    print(f"[passages] cached {len(passages)} passages")
    return passages


def depth_fraction_to_layer(num_layers: int, frac: float) -> int:
    """Map a depth fraction in [0, 1] to a layer index, clamped to a valid block."""
    idx = int(round(frac * (num_layers - 1)))
    return max(0, min(num_layers - 1, idx))


@torch.no_grad()
def extract_one_model(
    model_name: str,
    tag: str,
    passages: list[str],
    out_dir: Path,
    depth_fraction: float,
    layer_index: int | None,
    max_length: int,
    batch_size: int,
    dtype: torch.dtype,
) -> dict:
    """Forward `passages` through `model_name`, capture mean-pooled layer-l h
    per passage, write a safetensors shard, return its metadata."""
    print(f"[{tag}] loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    # Single-device load — every model in the pool fits comfortably on one V100,
    # and device_map="auto" can shard layers across GPUs which then makes the
    # forward hook capture on cuda:1 while attention_mask sits on cuda:0
    # (RuntimeError: "Expected all tensors to be on the same device").
    target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(target_device).eval()

    text_cfg = resolve_text_config(model.config)
    d_model = text_cfg.hidden_size
    num_layers = text_cfg.num_hidden_layers
    chosen_layer = layer_index if layer_index is not None else depth_fraction_to_layer(num_layers, depth_fraction)
    print(f"[{tag}] d_model={d_model}, num_layers={num_layers}, "
          f"chosen layer={chosen_layer} (depth_fraction={depth_fraction:.2f})")

    layers = resolve_decoder_layers(model)
    captured: torch.Tensor | None = None

    def hook(_mod, _inputs, output):
        nonlocal captured
        h = output[0] if isinstance(output, tuple) else output
        # .clone() to detach from the live buffer (see HFExtractor for the same
        # rationale — compile/reuse can clobber otherwise).
        captured = h.detach().clone()

    handle = layers[chosen_layer].register_forward_hook(hook)

    out = torch.empty(len(passages), d_model, dtype=torch.float32)
    try:
        for start in tqdm(range(0, len(passages), batch_size), desc=f"[{tag}] forward"):
            sub = passages[start : start + batch_size]
            enc = tokenizer(
                sub, return_tensors="pt", padding=True, truncation=True,
                max_length=max_length, add_special_tokens=True,
            )
            device = model.get_input_embeddings().weight.device
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            captured = None
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            assert captured is not None and captured.shape[-1] == d_model
            # Mean-pool over content (non-pad) tokens. Cast captured to fp32
            # BEFORE multiplication/summation — fp16 sum over a few hundred
            # outlier channels (mid-layer "attention sink" features can reach
            # magnitudes >100) overflows silently to ±inf, which then poisons
            # downstream lstsq. Observed empirically: gpt2-medium and
            # smollm2-360m produced inf rows under the fp16 path.
            captured_fp32 = captured.float()
            mask = attention_mask.unsqueeze(-1).float()              # [B, T, 1]
            num = (captured_fp32 * mask).sum(dim=1)                  # [B, d] fp32
            den = mask.sum(dim=1).clamp_min(1)                       # [B, 1] fp32
            pooled = (num / den).cpu()
            out[start : start + pooled.shape[0]] = pooled
    finally:
        handle.remove()

    shard_path = out_dir / f"{tag}.safetensors"
    save_file({"h": out}, str(shard_path))
    meta = {
        "tag": tag,
        "model": model_name,
        "d_model": d_model,
        "num_layers": num_layers,
        "layer_index": chosen_layer,
        "depth_fraction": depth_fraction,
        "n_passages": len(passages),
        "max_length": max_length,
        "pool": "mean_content_tokens",
        "shard": shard_path.name,
    }
    (out_dir / f"{tag}.meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(f"[{tag}] wrote {shard_path} ({out.shape}) and meta.json")

    del model, tokenizer, layers, captured
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="yaml: { pool: [{model, tag, depth_fraction?, layer_index?}, ...], n_passages, max_chars, max_length, batch_size, out_dir, seed }")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    passages_path = out_dir / "passages.jsonl"
    passages = materialize_passages(
        passages_path,
        n_passages=int(cfg["n_passages"]),
        max_chars=int(cfg.get("max_chars", 4000)),
        seed=int(cfg.get("seed", 0)),
    )

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[cfg.get("dtype", "fp16")]
    index: dict[str, dict] = {}
    for entry in cfg["pool"]:
        tag = entry["tag"]
        meta_path = out_dir / f"{tag}.meta.json"
        shard_path = out_dir / f"{tag}.safetensors"
        if meta_path.exists() and shard_path.exists() and not cfg.get("force", False):
            print(f"[{tag}] already extracted ({shard_path}) — skipping")
            index[tag] = json.loads(meta_path.read_text())
            continue
        meta = extract_one_model(
            model_name=entry["model"],
            tag=tag,
            passages=passages,
            out_dir=out_dir,
            depth_fraction=float(entry.get("depth_fraction", cfg.get("depth_fraction", 0.5))),
            layer_index=entry.get("layer_index"),
            max_length=int(cfg.get("max_length", 512)),
            batch_size=int(cfg.get("batch_size", 8)),
            dtype=dtype,
        )
        index[tag] = meta

    (out_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    print(f"[done] index → {out_dir / 'index.json'} ({len(index)} models)")


if __name__ == "__main__":
    main()
