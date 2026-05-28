"""Per-position multi-arch activation extraction (v8 universal NLA recipe).

Key design choice: **char-offset anchoring** decouples sampled positions from
each model's tokenizer.

Tokenizers split passages differently — position 5 in Qwen ≠ position 5 in
GPT-2. To keep ONE teacher z per (passage, position) shared across all
models, we sample K character offsets per passage and, for each model,
project that offset down to whichever token spans it via the tokenizer's
`offset_mapping`.

Output (per tag):
  artifacts/activations_pool_per_position/<tag>.safetensors  -- h: [N=passages×K, d_M] fp32
  artifacts/activations_pool_per_position/<tag>.meta.json    -- per-row {passage_id, char_offset, token_idx, token_str}

Shared across all tags:
  artifacts/activations_pool_per_position/positions.jsonl    -- {passage_id, char_offsets:[c1,...,cK]}
  artifacts/activations_pool_per_position/passages.jsonl     -- (copied/aligned from activations_pool_300m)

Run order: extract_per_position_multi.py → generate_summaries_per_position.py.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.arch_adapters import resolve_decoder_layers, resolve_text_config


# Same monkey-patch as extract_multi.py for Gemma-4-style list-format extras.
def _patch_extra_special_tokens_list():
    import transformers.tokenization_utils_base as t
    orig = t.PreTrainedTokenizerBase._set_model_specific_special_tokens
    def patched(self, special_tokens):
        if isinstance(special_tokens, list):
            special_tokens = {tok: tok for tok in special_tokens}
        return orig(self, special_tokens)
    t.PreTrainedTokenizerBase._set_model_specific_special_tokens = patched
_patch_extra_special_tokens_list()


def sample_char_offsets(text: str, k: int, rng: np.random.Generator) -> list[int]:
    """Pick K character offsets within the passage's content region.

    Strategy:
      - Skip first 30 / last 30 chars (boundary noise).
      - Uniformly sample K offsets in that interior.
      - Snap each offset to the nearest non-whitespace char (so we don't anchor
        on a token boundary that happens to be a space).
    """
    n = len(text)
    if n < 80:
        # Very short passage — degenerate.
        return [n // 2] * k
    interior_lo = 30
    interior_hi = n - 30
    raw = rng.integers(interior_lo, interior_hi, size=k, endpoint=False)
    fixed = []
    for c in raw:
        c = int(c)
        # Move to next non-space char (forward then back).
        while c < n and text[c].isspace():
            c += 1
        if c >= n:
            c = n - 1
        fixed.append(c)
    return fixed


def resolve_token_idx_for_offset(offsets: list[tuple[int, int]], char_offset: int) -> int:
    """Return the token index whose char span contains `char_offset`. Falls back
    to nearest if no span contains it (e.g., the BPE sub-token boundaries skip
    over the char). offsets list is [(start, end), ...] from fast tokenizer."""
    for i, (a, b) in enumerate(offsets):
        if a == 0 and b == 0:
            continue  # special token
        if a <= char_offset < b:
            return i
    # Fallback: closest token by midpoint.
    best, best_d = -1, 10**9
    for i, (a, b) in enumerate(offsets):
        if a == 0 and b == 0:
            continue
        mid = (a + b) // 2
        d = abs(mid - char_offset)
        if d < best_d:
            best, best_d = i, d
    return max(best, 0)


def extract_for_tag(
    model_name: str,
    tag: str,
    passages: list[dict],
    positions: list[dict],
    out_dir: Path,
    *,
    depth_fraction: float,
    layer_index: int | None,
    dtype: torch.dtype,
    device: str,
    batch_size: int = 4,
    max_length: int = 1024,
):
    """Run model on each passage, capture h at chosen layer for the K token
    positions that span the pre-sampled char offsets."""
    meta_path = out_dir / f"{tag}.meta.json"
    shard_path = out_dir / f"{tag}.safetensors"
    if shard_path.exists() and meta_path.exists():
        print(f"[{tag}] shard exists, skipping")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    print(f"[{tag}] loading {model_name} ...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, attn_implementation="sdpa",
            trust_remote_code=True,
        ).to(device).eval()
    except ValueError as e:
        if "does not support" in str(e) and "scaled_dot_product_attention" in str(e):
            print(f"[{tag}] sdpa unsupported, retrying with eager")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, attn_implementation="eager",
                trust_remote_code=True,
            ).to(device).eval()
        else:
            raise
    for p in model.parameters():
        p.requires_grad_(False)

    cfg = resolve_text_config(model.config)
    n_layers = int(cfg.num_hidden_layers)
    layer = layer_index if layer_index is not None else int(n_layers * depth_fraction)
    layer = max(0, min(n_layers - 1, layer))
    d_M = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)))
    print(f"[{tag}] n_layers={n_layers} layer={layer} d_M={d_M}")

    layers = resolve_decoder_layers(model)
    target_layer = layers[layer]

    captured: dict[str, torch.Tensor] = {}
    def _hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        # Cast to fp32 BEFORE summing — otherwise fp16 outlier channels overflow.
        captured["h"] = h.detach().float().cpu()
    handle = target_layer.register_forward_hook(_hook)

    # Output layout: row r corresponds to positions[r] = {passage_id, char_offset, ...}
    N = len(positions)
    h_out = torch.empty(N, d_M, dtype=torch.float32)
    row_meta = []          # per-row {passage_id, char_offset, token_idx, token_str}

    # Group rows by passage so we forward each passage once.
    by_passage: dict[int, list[int]] = {}
    for r_idx, row in enumerate(positions):
        by_passage.setdefault(int(row["passage_id"]), []).append(r_idx)

    pid_to_text = {int(p["passage_id"]): p["text"] for p in passages}

    # Batched forward — but each passage may yield multiple rows.
    pids = sorted(by_passage.keys())
    try:
        for s in range(0, len(pids), batch_size):
            batch_pids = pids[s:s + batch_size]
            texts = [pid_to_text[p] for p in batch_pids]
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                            max_length=max_length, add_special_tokens=True,
                            return_offsets_mapping=True)
            offset_lists = enc.pop("offset_mapping").tolist()
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                      use_cache=False)
            h_batch = captured["h"]                                  # [B, T, d_M]
            for bi, pid in enumerate(batch_pids):
                offs = offset_lists[bi]
                for r in by_passage[pid]:
                    char_off = int(positions[r]["char_offset"])
                    tok_idx = resolve_token_idx_for_offset(offs, char_off)
                    tok_idx = min(tok_idx, h_batch.shape[1] - 1)
                    h_out[r] = h_batch[bi, tok_idx]
                    tok_str = tokenizer.decode([int(enc["input_ids"][bi, tok_idx])]) or ""
                    row_meta.append({"passage_id": pid, "char_offset": char_off,
                                     "token_idx": tok_idx, "token_str": tok_str})
            if (s // batch_size) % 20 == 0:
                print(f"  [{tag}] passage {s+batch_size}/{len(pids)}")
    finally:
        handle.remove()

    # Reorder row_meta to match h_out's row order — both built passage-major above,
    # but rows within a passage came out in `positions` order via by_passage[pid].
    # We need to write row_meta back into the SAME order as `positions`. Let's
    # rebuild row_meta with explicit row-keyed indexing.
    row_meta_ordered = [None] * N
    cursor = 0
    for pid in pids:
        for r in by_passage[pid]:
            row_meta_ordered[r] = row_meta[cursor]
            cursor += 1
    assert all(r is not None for r in row_meta_ordered)

    save_file({"h": h_out}, str(shard_path))
    meta_path.write_text(json.dumps({
        "tag": tag,
        "model": model_name,
        "layer": layer,
        "n_layers": n_layers,
        "d_M": d_M,
        "n_rows": N,
        "shard": shard_path.name,
        "rows": row_meta_ordered,
    }, ensure_ascii=False, indent=1))
    print(f"[{tag}] wrote {shard_path}  ({N} rows, d={d_M})")
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", default="artifacts/activations_pool_300m/passages.jsonl",
                    help="Source of passages (reuse the 10k corpus).")
    ap.add_argument("--out-dir", default="artifacts/activations_pool_per_position")
    ap.add_argument("--config", default="configs/universal/extract_v1.yaml",
                    help="Pool config (tag → model_id) — same as mean-pool version.")
    ap.add_argument("--k-per-passage", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-passages", type=int, default=None,
                    help="Cap passages for a smaller experiment.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tags", default=None,
                    help="Comma-separated subset of tags to extract; defaults to all.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path(args.config).read_text())
    pool = cfg["pool"]
    if args.tags:
        wanted = {t.strip() for t in args.tags.split(",")}
        pool = [m for m in pool if m["tag"] in wanted]
    print(f"[v8] extracting per-position activations for {len(pool)} tags")

    passages = [json.loads(line) for line in Path(args.passages).read_text().splitlines() if line.strip()]
    for i, p in enumerate(passages):
        p["passage_id"] = p.get("passage_id", i)
    if args.n_passages:
        passages = passages[: args.n_passages]
    print(f"[v8] {len(passages)} passages")

    positions_path = out_dir / "positions.jsonl"
    if positions_path.exists():
        positions_rows = [json.loads(l) for l in positions_path.read_text().splitlines() if l.strip()]
        # Flatten char_offsets into one row per (passage, offset).
        positions = []
        for r in positions_rows:
            for c in r["char_offsets"]:
                positions.append({"passage_id": r["passage_id"], "char_offset": c})
        print(f"[v8] reusing cached positions ({len(positions)} rows)")
    else:
        rng = np.random.default_rng(args.seed)
        positions = []
        positions_summary = []
        for p in passages:
            offs = sample_char_offsets(p["text"], args.k_per_passage, rng)
            positions_summary.append({"passage_id": p["passage_id"], "char_offsets": offs})
            for c in offs:
                positions.append({"passage_id": p["passage_id"], "char_offset": int(c)})
        positions_path.write_text("\n".join(json.dumps(r) for r in positions_summary) + "\n")
        print(f"[v8] sampled {len(positions)} positions → {positions_path}")

    # Mirror passages.jsonl so downstream scripts can find text by passage_id.
    pp = out_dir / "passages.jsonl"
    if not pp.exists():
        pp.write_text("\n".join(json.dumps({"passage_id": p["passage_id"], "text": p["text"]})
                                 for p in passages) + "\n")

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[cfg.get("dtype", "fp16")]
    depth_fraction = float(cfg.get("depth_fraction", 0.5))

    for m in pool:
        tag = m["tag"]
        try:
            extract_for_tag(
                m["model"], tag,
                passages, positions, out_dir,
                depth_fraction=depth_fraction,
                layer_index=m.get("layer_index"),
                dtype=dtype, device=args.device,
                batch_size=int(cfg.get("batch_size", 4)),
                max_length=int(cfg.get("max_length", 1024)),
            )
        except Exception as e:
            print(f"[v8] {tag} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
