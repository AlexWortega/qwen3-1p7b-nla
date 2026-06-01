"""Run the universal v8 AV over battery activations and emit z explanations.

Loads the AV (Qwen3-1.7B + LoRA from <av-dir>/av) and a ModelPoolAdapters bundle
ONCE, then iterates a --plan of combos. Each combo points at an acts shard
({"h":[N,d]}) and names the enc tag to project through (e.g. qwen2p5-7b for L20,
the refit q25i-L14 for L14). Mirrors eval_universal's injection + generation
exactly, minus the teacher-cosine step.

Output: <out> json — list of {label, enc_tag, id, category, z_text}.

Usage:
  python scripts/audit/run_av_explain.py --av-dir artifacts/av_v8_mixed \
    --adapters-dir artifacts/audit/bundle_L14 --acts-dir artifacts/audit/acts \
    --plan artifacts/audit/acts/plan.json --out artifacts/audit/explanations.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.enc_dec_adapters import ModelPoolAdapters
from nla.schema import extract_explanation, normalize_activation
from scripts.eval_universal import build_prompt, generate_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-dir", required=True)
    ap.add_argument("--adapters-dir", required=True)
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--plan", required=True, help="json list of {label, shard, enc_tag}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args()

    av_dir = Path(args.av_dir)
    meta = yaml.safe_load((av_dir / "nla_meta.yaml").read_text())
    av_base = meta["av_base"]
    template = meta["prompt_templates"]["actor"]
    tk = meta["tokens"]
    inj_id, left_id, right_id = (int(tk["injection_token_id"]),
                                 int(tk["injection_left_neighbor_id"]),
                                 int(tk["injection_right_neighbor_id"]))
    inj_char = tk["injection_char"]
    d_shared = int(meta["d_shared"])
    inj_scale = math.sqrt(d_shared)
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(av_base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(av_base, torch_dtype=torch.float16).to(device).eval()
    av = PeftModel.from_pretrained(base, str(av_dir / "av")).to(device).eval()
    adapters = ModelPoolAdapters.load(args.adapters_dir).to(device)
    print(f"[av] loaded {av_base} + LoRA; bundle tags include: "
          f"{[t for t in adapters.encoders.keys()][:25]}")

    bmeta = json.loads((Path(args.acts_dir) / "battery_meta.json").read_text())
    ids, cats = bmeta["ids"], bmeta["categories"]
    plan = json.loads(Path(args.plan).read_text())

    results = []
    for combo in plan:
        label, shard, enc_tag = combo["label"], combo["shard"], combo["enc_tag"]
        assert enc_tag in adapters.encoders, f"enc tag {enc_tag} not in bundle"
        h = load_file(str(Path(args.acts_dir) / shard))["h"].float().to(device)
        assert h.shape[0] == len(ids), f"{shard}: {h.shape[0]} rows vs {len(ids)} battery"
        for i in range(h.shape[0]):
            inj = adapters.encode(enc_tag, h[i:i + 1]).squeeze(0)
            inj = normalize_activation(inj, inj_scale)
            prompt = build_prompt(template, enc_tag, inj_char)
            text = generate_one(av, tokenizer, prompt, inj, inj_id, left_id, right_id,
                                args.max_new_tokens, device)
            z = extract_explanation(text) or text.strip()
            results.append({"label": label, "enc_tag": enc_tag, "id": ids[i],
                            "category": cats[i], "z_text": z})
        print(f"[av] {label}: {h.shape[0]} explanations")
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[av] wrote {len(results)} explanations -> {args.out}")


if __name__ == "__main__":
    main()
