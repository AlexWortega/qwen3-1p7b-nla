"""v11 extraction — Anthropic-NLA-faithful: last-token activation of a randomly
truncated text snippet.

Their recipe (transformer-circuits.pub/2026/nla):
  - Take pretraining-like text, truncate at a random point.
  - Forward the PREFIX through the target model, take h_l at the **final token**
    (it accumulates everything to the left → encodes a predictive state, not a
    bag-of-topics like mean-pool does).
  - Teacher = summary of the text *up to that token*.

This script does the activation half. For each passage we sample K random
truncation token-positions, forward the prefix once per position (batched per
passage isn't worth it at these lengths), and store the last-token h.

Output layout (mirrors activations_pool_300m so downstream lstsq/SFT reuse):
  <out_dir>/passages.jsonl          {passage_id, trunc_id, prefix_text}
                                     one row per (passage, truncation) — this is
                                     the unit the teacher will summarize.
  <out_dir>/<tag>.safetensors       { "h": [N_trunc, d_M] }  aligned to passages.jsonl row order
  <out_dir>/<tag>.meta.json         { tag, d_model, layer, n_rows }
  <out_dir>/index.json              { tag -> {shard, d_model, layer, n_passages} }

Run the anchor tag first; it defines passages.jsonl + the (passage, trunc)
sampling. Subsequent tags reuse the same passages.jsonl so rows align.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.arch_adapters import resolve_decoder_layers, resolve_text_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--src-passages", required=True,
                    help="passages.jsonl with full texts (e.g. activations_pool_300m/passages.jsonl).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--truncs-per-passage", type=int, default=2,
                    help="K random truncation points per source passage.")
    ap.add_argument("--n-passages", type=int, default=5000)
    ap.add_argument("--min-prefix-tokens", type=int, default=16,
                    help="Truncation point is sampled in [min, full_len).")
    ap.add_argument("--depth-fraction", type=float, default=0.5)
    ap.add_argument("--layer-idx", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--seed", type=int, default=0,
                    help="Shared seed → same (passage, trunc) sampling across tags.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=1000)
    args = ap.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = [json.loads(l) for l in Path(args.src_passages).read_text().splitlines() if l.strip()]
    src = src[: args.n_passages]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation=args.attn,
        device_map="auto", trust_remote_code=True,
    ).eval()
    for p in model.parameters(): p.requires_grad_(False)

    cfg = resolve_text_config(model.config)
    n_layers = int(cfg.num_hidden_layers)
    layer = int(args.layer_idx) if args.layer_idx is not None else max(0, min(n_layers - 1, int(n_layers * args.depth_fraction)))
    d_M = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)))
    print(f"[trunc] tag={args.tag} model={args.model} layer={layer}/{n_layers} d_M={d_M}")

    decoder = resolve_decoder_layers(model)
    captured = {}
    def _hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach().float().cpu()
    handle = decoder[layer].register_forward_hook(_hook)

    rng = random.Random(args.seed)

    # Build (passage_id, trunc_token_len) sampling. To keep tags aligned, we
    # sample truncation as a FRACTION of each passage's token length using the
    # shared rng — every tag re-derives the same fractions. (Token counts differ
    # per tokenizer, so we store the prefix TEXT decoded from the anchor's
    # tokenizer only when writing passages.jsonl; activations always use each
    # model's own tokenization of that prefix text.)
    passages_path = out_dir / "passages.jsonl"
    write_passages = not passages_path.exists()

    # Pre-sample truncation fractions deterministically per passage.
    trunc_fracs = []
    for _ in src:
        trunc_fracs.append([rng.uniform(0.15, 0.98) for _ in range(args.truncs_per_passage)])

    rows = []          # (passage_id, trunc_id, prefix_text)
    h_list = []
    pass_fh = passages_path.open("w") if write_passages else None

    def flush():
        if not h_list: return
        h_stack = torch.stack(h_list, dim=0)
        save_file({"h": h_stack}, str(out_dir / f"{args.tag}.safetensors"))
        (out_dir / f"{args.tag}.meta.json").write_text(json.dumps(
            {"tag": args.tag, "d_model": d_M, "shard": f"{args.tag}.safetensors",
             "layer": layer, "n_passages": len(h_list)}, indent=1))
        idx_path = out_dir / "index.json"
        idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
        idx[args.tag] = {"shard": f"{args.tag}.safetensors", "d_model": d_M,
                         "layer": layer, "n_passages": len(h_list)}
        idx_path.write_text(json.dumps(idx, indent=2, sort_keys=True))

    # Build the flat work list: prefixes (char-level, derived from token truncation
    # on THIS model's tokenizer so the last token is well-defined).
    work = []   # (passage_id, trunc_id, prefix_ids)
    for pi, p in enumerate(src):
        text = p["text"]
        full_ids = tok(text, add_special_tokens=True, truncation=True, max_length=args.max_length)["input_ids"]
        L = len(full_ids)
        if L <= args.min_prefix_tokens + 2:
            continue
        for ti, frac in enumerate(trunc_fracs[pi]):
            cut = max(args.min_prefix_tokens, int(L * frac))
            cut = min(cut, L)
            prefix_ids = full_ids[:cut]
            pid = int(p.get("passage_id", pi))
            work.append((pid, ti, prefix_ids))
            if write_passages:
                prefix_text = tok.decode(prefix_ids, skip_special_tokens=True)
                pass_fh.write(json.dumps({"passage_id": pid, "trunc_id": ti,
                                          "row": len(work) - 1,
                                          "prefix_text": prefix_text}, ensure_ascii=False) + "\n")

    if pass_fh: pass_fh.close()
    print(f"[trunc] {len(work)} (passage,trunc) prefixes")

    # Forward in length-bucketed batches: pad each batch, grab last real token.
    try:
        i = 0
        while i < len(work):
            batch = work[i:i + args.batch_size]
            maxlen = max(len(b[2]) for b in batch)
            input_ids = torch.full((len(batch), maxlen), tok.pad_token_id, dtype=torch.long)
            attn = torch.zeros(len(batch), maxlen, dtype=torch.long)
            last_pos = []
            for bi, (_, _, ids) in enumerate(batch):
                input_ids[bi, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                attn[bi, :len(ids)] = 1
                last_pos.append(len(ids) - 1)
            with torch.no_grad():
                model(input_ids=input_ids.cuda(), attention_mask=attn.cuda(), use_cache=False)
            h_batch = captured["h"]            # [B, maxlen, d_M]
            for bi, lp in enumerate(last_pos):
                h_list.append(h_batch[bi, lp].float())
            i += args.batch_size
            if (i // args.batch_size) % 50 == 0:
                print(f"  [{args.tag}] {i}/{len(work)}")
            if args.save_every and len(h_list) % args.save_every < args.batch_size:
                flush()
    finally:
        handle.remove()

    flush()
    print(f"[trunc] DONE {args.tag}  rows={len(h_list)} → {out_dir}")


if __name__ == "__main__":
    main()
