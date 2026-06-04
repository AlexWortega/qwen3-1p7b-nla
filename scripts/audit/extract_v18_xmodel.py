"""v18 cross-model extraction: run the SAME bias-exhibiting transcripts through MANY
base models and mean-pool the ASSISTANT-span activation at each model's pool layer.

The whole point of v18: the same bias B seen through K different architectures'
activations teaches the detector a MODEL-INVARIANT bias signature, so detection
transfers to an unseen architecture (held-out llama3-8b).

Reuses extract_acts.run_chat's assistant-span mean-pool (fp32 before pool). For
each model we forward the full {user,assistant} chat and capture layer-L hidden
states, mean over assistant-response tokens.

Outputs (NOT into any pool dir — writes to /big/audit/v18_xmodel):
  /big/audit/v18_xmodel/rows.jsonl              shared {bias, idx, src} per transcript
  /big/audit/v18_xmodel/<tag>/acts.safetensors  {"h": [N, d_M]} aligned to rows.jsonl
  /big/audit/v18_xmodel/<tag>/meta.json         {tag, model, d_model, layer, n}

Run per tag (one model loaded at a time to stay in 32GB):
  python -m scripts.audit.extract_v18_xmodel --tag qwen3-1p7b --model Qwen/Qwen3-1.7B \
      --layer 14 --out-dir /big/audit/v18_xmodel
The rows.jsonl is built once (idempotent) from the merged dialogue files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.arch_adapters import resolve_decoder_layers, resolve_text_config

# Merged transcript sources. data/dialogues.jsonl == ao/dialogues_A.jsonl (dedup by content).
DIALOGUE_FILES = [
    "/big/audit/ao/dialogues_A.jsonl",
    "/big/audit/ao/dialogues_B.jsonl",
    "/big/audit/ao/dialogues_C.jsonl",
    "/big/audit/ao/dialogues_D.jsonl",
]


def build_rows(out_dir: Path, cap_per_bias: int, seed: int = 0,
               dialogue_files: list[str] | None = None) -> list[dict]:
    """Build (or load) the shared rows.jsonl. Each row: {bias, user, assistant, idx, src}.
    Capped at cap_per_bias transcripts per bias for tractability. Idempotent: if
    rows.jsonl exists it is reused so every model extracts the SAME transcripts."""
    rows_path = out_dir / "rows.jsonl"
    if rows_path.exists():
        rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
        print(f"[rows] reuse existing {rows_path} ({len(rows)} rows)")
        return rows
    import random
    rng = random.Random(seed)
    seen = set()
    by_bias: dict[str, list[dict]] = {}
    for f in (dialogue_files or DIALOGUE_FILES):
        p = Path(f)
        if not p.exists():
            print(f"[rows] WARN missing {f}")
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["bias"], r["user"], r["assistant"])
            if key in seen:
                continue
            seen.add(key)
            by_bias.setdefault(r["bias"], []).append(
                {"bias": r["bias"], "user": r["user"], "assistant": r["assistant"], "src": p.name})
    rows: list[dict] = []
    for bias, items in sorted(by_bias.items()):
        rng.shuffle(items)
        for it in items[:cap_per_bias]:
            it = dict(it, idx=len(rows))
            rows.append(it)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["bias"]] = counts.get(r["bias"], 0) + 1
    print(f"[rows] wrote {rows_path} ({len(rows)} rows) per-bias={counts}")
    return rows


def make_hook(store, key="h"):
    def hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        store[key] = h.detach().clone()
    return hook


@torch.no_grad()
def extract_tag(tag, model_id, layer, rows, out_dir, max_length=512, dtype=torch.float16):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True).to("cuda:0").eval()
    text_cfg = resolve_text_config(model.config)
    d = text_cfg.hidden_size
    layers = resolve_decoder_layers(model)
    store: dict = {}
    handle = layers[layer].register_forward_hook(make_hook(store))
    out = torch.empty(len(rows), d, dtype=torch.float32)
    try:
        for i, item in enumerate(rows):
            msgs = [{"role": "user", "content": item["user"]},
                    {"role": "assistant", "content": item["assistant"]}]
            try:
                full = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
                hdr = tok.apply_chat_template(msgs[:1], tokenize=True, add_generation_prompt=True)
            except Exception:
                # no chat template: fall back to a plain User/Assistant framing.
                u = f"User: {item['user']}\nAssistant:"
                full_txt = u + " " + item["assistant"]
                full = tok(full_txt, add_special_tokens=True)["input_ids"]
                hdr = tok(u, add_special_tokens=True)["input_ids"]
            full = full[:max_length]
            hdr_len = min(len(hdr), len(full))
            ids = torch.tensor([full], device="cuda")
            store.clear()
            model(input_ids=ids, use_cache=False)
            h = store["h"][0].float()  # [T, d]
            asst = h[hdr_len:] if h.shape[0] > hdr_len else h[-1:]
            out[i] = asst.mean(0).cpu()
            if i % 100 == 0:
                print(f"[{tag}] {i}/{len(rows)}", flush=True)
    finally:
        handle.remove()
    tag_dir = Path(out_dir) / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    save_file({"h": out}, str(tag_dir / "acts.safetensors"))
    (tag_dir / "meta.json").write_text(json.dumps(
        {"tag": tag, "model": model_id, "d_model": d, "layer": layer,
         "n": len(rows), "pool": "assistant_mean", "mode": "chat"}, indent=2))
    print(f"[{tag}] wrote acts.safetensors [{len(rows)}, {d}] layer={layer} -> {tag_dir}")
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out-dir", default="/big/audit/v18_xmodel")
    ap.add_argument("--cap-per-bias", type=int, default=80)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dialogue-files", default=None,
                    help="comma list of dialogue jsonl files overriding the default A-D set "
                         "(v19: /big/audit/ao/dialogues_S.jsonl).")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    dlg_files = args.dialogue_files.split(",") if args.dialogue_files else None
    rows = build_rows(out_dir, args.cap_per_bias, args.seed, dlg_files)
    extract_tag(args.tag, args.model, args.layer, rows, out_dir, args.max_length)


if __name__ == "__main__":
    main()
