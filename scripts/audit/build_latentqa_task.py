"""Build the LatentQA *training* task for v15 (4th task = behaviour-QA).

Berkeley LatentQA's eval set (qa.json + stimulus_completion.json) is repurposed
here as a SUPERVISED training task for the universal v15 oracle, trained on
IN-POOL models' activations (NOT Llama-3 — that stays held-out for the
eval_v15_latentqa.py zero-shot comparison).

Two stages, both writing ONLY into --out-dir (default /big/audit/latentqa_task):

  split  : join the 1134 qa pairs to their stimulus, 80/20 split (seed 0) ->
           latentqa_train.jsonl (~907) / latentqa_heldout.jsonl (~227).
           Each row: {label, question, gold, stim_text} where
             stim_text = stimulus_user + "\n\n" + (stimulus_thought + stimulus_model)
           rendered as a {user, assistant} chat for the assigned in-pool model.

  extract: assign in-pool models ROUND-ROBIN across TRAIN rows
             tags = [qwen2p5-7b, gemma2, phi-1p5, smollm3-3b]
           For each row, forward its stim_text {user,assistant} chat through the
           assigned model, mean-pool that model's POOL layer over the assistant
           span -> h [d_tag] fp32. Per-tag shard acts_<tag>.safetensors {"h":[n,d]}
           plus rowmap.json mapping global train-row index -> (tag, local_idx).

CRITICAL: writes to --out-dir ONLY. Never touches /big/activations_pool_v9.

Pool layers (forward-hook decoder-layer index, matching each tag's v9 enc fit):
  qwen2p5-7b : 20   (Qwen2.5-7B, 28 layers; enc fit at L20 — see extract_v1.yaml)
  gemma2     : 21   (gemma-2-9b-it, 42 layers; depth 0.5; matches lie_acts_L21)
  phi-1p5    : 12   (phi-1_5, 24 layers; depth 0.5)
  smollm3-3b : 18   (SmolLM3-3B, 36 layers; depth 0.5)
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

# tag -> (HF model id, forward-hook decoder-layer index = the tag's v9 pool layer)
TAG_SPEC = {
    "qwen2p5-7b": ("Qwen/Qwen2.5-7B", 20),
    "gemma2":     ("google/gemma-2-9b-it", 21),
    "phi-1p5":    ("microsoft/phi-1_5", 12),
    "smollm3-3b": ("HuggingFaceTB/SmolLM3-3B", 18),
}
ROUND_ROBIN_TAGS = ["qwen2p5-7b", "gemma2", "phi-1p5", "smollm3-3b"]


def load_joined(latentqa_dir: Path):
    qa = json.loads((latentqa_dir / "qa.json").read_text())
    stimc = json.loads((latentqa_dir / "stimulus_completion.json").read_text())
    by_label = {s["label"]: s for s in stimc}
    rows = []
    for label, pairs in qa.items():
        s = by_label.get(label)
        if s is None:
            continue
        user = (s.get("stimulus_user", "") or "").strip()
        thought = (s.get("stimulus_thought", "") or "").strip()
        reply = (s.get("stimulus_model", "") or "").strip()
        assistant = (thought + "\n\n" + reply).strip() if thought else reply
        stim_text = (user + "\n\n" + assistant).strip()
        for question, gold in pairs:
            rows.append({"label": label, "question": question, "gold": gold,
                         "stim_text": stim_text, "stim_user": user,
                         "stim_assistant": assistant})
    return rows


def do_split(args):
    rows = load_joined(Path(args.latentqa_dir))
    print(f"[split] joined qa pairs: {len(rows)}")
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_held = round(len(rows) * 0.20)
    held = rows[:n_held]
    train = rows[n_held:]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def write(name, data):
        with open(out / name, "w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write("latentqa_train.jsonl", train)
    write("latentqa_heldout.jsonl", held)
    print(f"[split] train={len(train)} heldout={len(held)} -> {out}")


def make_hook(store):
    def hook(_m, _i, output):
        store["h"] = (output[0] if isinstance(output, tuple) else output).detach()
    return hook


@torch.no_grad()
def extract_tag(tag, rows_for_tag, dtype, max_length):
    """rows_for_tag: list of (global_idx, row). Returns [n, d] fp32 (row order)."""
    model_id, layer = TAG_SPEC[tag]
    print(f"[extract] {tag}: {model_id} L{layer}  rows={len(rows_for_tag)}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True).to("cuda:0").eval()
    d = resolve_text_config(model.config).hidden_size
    layers = resolve_decoder_layers(model)
    store = {}
    handle = layers[layer].register_forward_hook(make_hook(store))
    out = torch.empty(len(rows_for_tag), d, dtype=torch.float32)
    try:
        for j, (_gi, r) in enumerate(rows_for_tag):
            msgs = [{"role": "user", "content": r["stim_user"]},
                    {"role": "assistant", "content": r["stim_assistant"]}]
            try:
                full = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
                hdr = tok.apply_chat_template(msgs[:1], tokenize=True, add_generation_prompt=True)
            except Exception:
                # models w/o a chat template (e.g. phi-1_5 base): plain concat.
                full = tok(r["stim_text"], add_special_tokens=True)["input_ids"]
                hdr = tok(r["stim_user"], add_special_tokens=True)["input_ids"]
            full = full[:max_length]
            hdr_len = min(len(hdr), len(full))
            ids = torch.tensor([full], device="cuda:0")
            store.clear()
            model(input_ids=ids, use_cache=False)
            h = store["h"][0].float()  # [T, d]
            asst = h[hdr_len:] if h.shape[0] > hdr_len else h[-1:]
            if asst.shape[0] == 0:
                asst = h[-1:]
            out[j] = asst.mean(0).cpu()
            if j % 50 == 0:
                print(f"[extract][{tag}] {j}/{len(rows_for_tag)}")
    finally:
        handle.remove()
    del model
    torch.cuda.empty_cache()
    return out, d, layer, model_id


def do_extract(args):
    out = Path(args.out_dir)
    train = [json.loads(l) for l in (out / "latentqa_train.jsonl").read_text().splitlines() if l.strip()]
    print(f"[extract] {len(train)} train rows; round-robin tags={ROUND_ROBIN_TAGS}")
    # round-robin assignment
    assign = [ROUND_ROBIN_TAGS[i % len(ROUND_ROBIN_TAGS)] for i in range(len(train))]
    by_tag = {t: [] for t in ROUND_ROBIN_TAGS}
    rowmap = {}  # global train idx -> {tag, local}
    for gi, (r, t) in enumerate(zip(train, assign)):
        local = len(by_tag[t])
        by_tag[t].append((gi, r))
        rowmap[gi] = {"tag": t, "local": local}

    only = set(args.only_tags.split(",")) if args.only_tags else None
    dtype = torch.float16
    meta_tags = {}
    for tag in ROUND_ROBIN_TAGS:
        if only and tag not in only:
            print(f"[extract] skip {tag} (not in --only-tags)")
            continue
        if not by_tag[tag]:
            continue
        h, d, layer, model_id = extract_tag(tag, by_tag[tag], dtype, args.max_length)
        save_file({"h": h}, str(out / f"acts_{tag}.safetensors"))
        meta_tags[tag] = {"model": model_id, "layer": layer, "d_model": d, "n": h.shape[0]}
        print(f"[extract] wrote acts_{tag}.safetensors [{h.shape[0]}, {d}]")

    (out / "rowmap.json").write_text(json.dumps(
        {"n_train": len(train), "tags": ROUND_ROBIN_TAGS, "rowmap": rowmap,
         "tag_meta": meta_tags}, indent=2))
    counts = {t: len(by_tag[t]) for t in ROUND_ROBIN_TAGS}
    print(f"[extract] rowmap.json written; per-tag assigned counts: {counts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["split", "extract", "all"], default="all")
    ap.add_argument("--latentqa-dir", default="/big/audit/latentqa_eval")
    ap.add_argument("--out-dir", default="/big/audit/latentqa_task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--only-tags", default=None,
                    help="comma list to restrict extraction (debug); default all 4")
    args = ap.parse_args()
    if args.stage in ("split", "all"):
        do_split(args)
    if args.stage in ("extract", "all"):
        do_extract(args)


if __name__ == "__main__":
    main()
