"""Extract activations from a model at multiple layers and multiple positions
per passage, in one forward pass.

For each passage in `passages.jsonl`:
  1. Tokenize.
  2. Sample K char_offsets uniformly across the passage; map each to the token
     index whose char-range contains it (consistent across tokenizers, which is
     why we anchor on char offsets not token indices).
  3. Forward through the model once, with forward hooks registered on every
     requested layer. Each hook stores h ∈ [B, T, d_M] in fp32.
  4. For each (layer, position) pair, write the per-token h. Also append a
     mean-pool row per layer (`char_offset == -1`).

Output layout under `--out-dir`:
  <tag>_L<layer_idx>.safetensors   { "h": [N_rows, d_M] fp32 }   per-layer shard
  <tag>_meta.json                   { tag, model, d_M, layers: [...],
                                      rows: [
                                        {passage_id, layer_idx, char_offset, token_str},
                                        ...
                                      ] }
  passages_used.txt                 IDs of passages actually processed (after skipping
                                    too-short ones).

The same passages × the same char offsets are reused across all tags when this
script is called with a shared `--seed`, so downstream lstsq fits can align by
row index across models the same way `activations_pool_300m` does.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.arch_adapters import resolve_decoder_layers, resolve_text_config


def _sample_char_offsets(text: str, k: int, rng: random.Random,
                         min_offset: int = 16) -> list[int]:
    """Return k char offsets in [min_offset, len(text)-1], spaced so they don't
    all clump at the same word."""
    n = len(text)
    if n <= min_offset + 4:
        return []
    span = n - min_offset
    raw = sorted(rng.sample(range(min_offset, n), min(k, span)))
    return raw[:k]


def _char_to_token_idx(offsets_mapping: list[tuple[int, int]], char_off: int) -> int | None:
    """Return the token index whose (start, end) char range contains char_off."""
    for i, (s, e) in enumerate(offsets_mapping):
        if s <= char_off < e:
            return i
    # Fall back to nearest (typically end-of-sequence).
    for i, (s, e) in enumerate(offsets_mapping):
        if s == 0 and e == 0:    # padding sentinel
            continue
    return None


def _parse_layers(spec: str, n_layers: int) -> list[int]:
    """`spec` can be: '14,21,7' (absolute) or 'd=0.25,0.5,0.75' (depths)."""
    spec = spec.strip()
    if spec.startswith("d="):
        depths = [float(x) for x in spec[2:].split(",") if x.strip()]
        idxs = sorted({max(0, min(n_layers - 1, int(n_layers * d))) for d in depths})
    else:
        idxs = sorted({int(x) for x in spec.split(",") if x.strip()})
    for i in idxs:
        assert 0 <= i < n_layers, f"layer {i} out of [0,{n_layers})"
    return idxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool-dir", required=True,
                    help="Directory containing passages.jsonl.")
    ap.add_argument("--out-dir", required=True,
                    help="Where to write <tag>_L<idx>.safetensors + <tag>_meta.json.")
    ap.add_argument("--layers", default="d=0.25,0.5,0.75",
                    help="Comma list of layer indices, or 'd=0.25,0.5,0.75' for depths.")
    ap.add_argument("--positions-per-passage", type=int, default=4,
                    help="K random char_offsets per passage (plus 1 mean-pool row).")
    ap.add_argument("--n-passages", type=int, default=None,
                    help="Process at most this many passages (None = all).")
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--end-idx", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--seed", type=int, default=0,
                    help="Shared seed for char_offset sampling — passing the same "
                         "value across tag runs aligns rows by passage order.")
    ap.add_argument("--save-every", type=int, default=500,
                    help="Checkpoint cadence (passages). On rerun the script reads "
                         "<tag>_meta.json + per-layer shards and resumes after the "
                         "highest passage_id already processed.")
    args = ap.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    pool_dir = Path(args.pool_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    passages_full = [json.loads(l) for l in (pool_dir / "passages.jsonl").read_text().splitlines() if l.strip()]
    end_idx = args.end_idx if args.end_idx is not None else len(passages_full)
    passages = passages_full[args.start_idx:end_idx]
    if args.n_passages is not None:
        passages = passages[: args.n_passages]
    N = len(passages)

    print(f"[mlmp] tag={args.tag} model={args.model} range=[{args.start_idx}:{end_idx}) "
          f"actual N={N} dtype={args.dtype}")

    try:
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception:
        # YaGPT and a few others ship only a SentencePiece model that the fast
        # converter can't handle — fall back to the slow tokenizer.
        tok = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation=args.attn,
        device_map="auto", trust_remote_code=True,
    ).eval()
    for p in model.parameters(): p.requires_grad_(False)

    cfg = resolve_text_config(model.config)
    n_layers = int(cfg.num_hidden_layers)
    d_M = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)))
    layer_idxs = _parse_layers(args.layers, n_layers)
    print(f"[mlmp] n_layers={n_layers} d_M={d_M} target_layers={layer_idxs}")

    decoder = resolve_decoder_layers(model)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for li in layer_idxs:
        def _hook_maker(li):
            def _hk(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[li] = h.detach().float().cpu()
            return _hk
        handles.append(decoder[li].register_forward_hook(_hook_maker(li)))

    rng = random.Random(args.seed)

    # Storage: per layer → list of [d_M] vectors. Rows is the global metadata list.
    per_layer_h: dict[int, list[torch.Tensor]] = {li: [] for li in layer_idxs}
    rows: list[dict] = []

    # ── Resume: read existing meta + shards, skip already-processed passages ──
    meta_path = out_dir / f"{args.tag}_meta.json"
    done_pids: set[int] = set()
    if meta_path.exists():
        try:
            prev = json.loads(meta_path.read_text())
            prev_rows = prev.get("rows", [])
            done_pids = {int(r["passage_id"]) for r in prev_rows}
            # Reload existing per-layer h-stacks so new rows append correctly.
            for li in layer_idxs:
                shard = out_dir / f"{args.tag}_L{li}.safetensors"
                if shard.exists():
                    h_prev = load_file(str(shard))["h"]
                    per_layer_h[li] = list(h_prev.unbind(dim=0))
            rows = list(prev_rows)
            print(f"[mlmp][resume] {args.tag}: {len(done_pids)} passages already done, "
                  f"{len(rows)} rows loaded")
        except Exception as e:
            print(f"[mlmp][resume] failed ({e!r}) — starting fresh")
            done_pids = set(); rows = []
            per_layer_h = {li: [] for li in layer_idxs}

    def _flush_checkpoint(label: str):
        for li in layer_idxs:
            if not per_layer_h[li]:
                continue
            h_stack = torch.stack(per_layer_h[li], dim=0)
            save_file({"h": h_stack}, str(out_dir / f"{args.tag}_L{li}.safetensors"))
        meta_path.write_text(json.dumps({
            "tag": args.tag, "model": args.model, "d_M": d_M,
            "n_layers": n_layers, "layers": layer_idxs,
            "positions_per_passage": args.positions_per_passage,
            "n_passages": N, "seed": args.seed,
            "rows": rows,
        }, indent=1))
        print(f"  [{args.tag}] checkpoint {label}: {len(rows)} rows persisted")

    passages_to_consume_count = sum(1 for p in passages
                                    if int(p.get("passage_id", -999)) not in done_pids)
    if not passages_to_consume_count:
        print(f"[mlmp] {args.tag}: nothing to do (all passages already in meta).")
        _flush_checkpoint("final-empty")
        return
    last_ckpt_at = len(done_pids)

    try:
        for s in range(0, N, args.batch_size):
            raw_batch = passages[s:s + args.batch_size]
            # Skip rows whose passage_id is already done. Keep batch shape stable.
            batch = [p for p in raw_batch
                     if int(p.get("passage_id", args.start_idx + s + raw_batch.index(p))) not in done_pids]
            if not batch:
                continue
            texts = [p["text"] for p in batch]
            pids = [int(p.get("passage_id", args.start_idx + s + i)) for i, p in enumerate(batch)]

            try:
                enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                          max_length=args.max_length, add_special_tokens=True,
                          return_offsets_mapping=True)
                offsets_per_seq = enc["offset_mapping"]
                have_offsets = True
            except (NotImplementedError, ValueError, TypeError):
                # Slow tokenizers (e.g. YaGPT) don't provide offsets_mapping.
                # We still keep the mean-pool row per layer; per-position rows
                # are skipped for this tag.
                enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                          max_length=args.max_length, add_special_tokens=True)
                offsets_per_seq = [[] for _ in texts]
                have_offsets = False
            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]

            with torch.no_grad():
                model(input_ids=input_ids.cuda(), attention_mask=attn.cuda(), use_cache=False)

            for bi in range(len(batch)):
                seq_len = int(attn[bi].sum().item())
                pid = pids[bi]
                tok_idxs: list[tuple[int, int, str]] = []
                if have_offsets:
                    offsets = [tuple(map(int, o)) for o in offsets_per_seq[bi].tolist()]
                    ch_offs = _sample_char_offsets(texts[bi], args.positions_per_passage, rng)
                    for c in ch_offs:
                        ti = _char_to_token_idx(offsets[:seq_len], c)
                        if ti is None: continue
                        ts = tok.decode([int(input_ids[bi, ti].item())])
                        tok_idxs.append((c, ti, ts))

                for li in layer_idxs:
                    h_seq = captured[li][bi]                       # [T, d_M]
                    # Mean-pool over content tokens.
                    mask = attn[bi, :h_seq.shape[0]].float().unsqueeze(-1)
                    h_mean = ((h_seq * mask).sum(0) / mask.sum().clamp_min(1)).float()
                    per_layer_h[li].append(h_mean)
                    rows.append({"passage_id": pid, "layer_idx": li,
                                 "char_offset": -1, "token_str": "<mean-pool>"})
                    for ch, ti, ts in tok_idxs:
                        per_layer_h[li].append(h_seq[ti].float())
                        rows.append({"passage_id": pid, "layer_idx": li,
                                     "char_offset": ch, "token_str": ts})
                done_pids.add(pid)

            if (s // args.batch_size) % 25 == 0:
                print(f"  [{args.tag}] {s + args.batch_size}/{N}  (done={len(done_pids)})")
            # Periodic checkpoint.
            if args.save_every and (len(done_pids) - last_ckpt_at) >= args.save_every:
                _flush_checkpoint(f"@{len(done_pids)}")
                last_ckpt_at = len(done_pids)
    finally:
        for h in handles: h.remove()

    _flush_checkpoint("final")
    print(f"[mlmp] DONE {args.tag}  rows={len(rows)}  passages={len(done_pids)}")


if __name__ == "__main__":
    main()
