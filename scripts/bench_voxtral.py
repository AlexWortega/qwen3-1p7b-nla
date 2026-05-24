"""Bench Voxtral-Mini-3B extracted LLM via lm-eval Python API — bypasses CLI
tokenizer-path argument so we can hand-feed an AutoTokenizer instance
(`use_fast=False`) which is the only path that survives Voxtral's tekken
tokenizer format on this transformers / tokenizers stack.

Output mirrors `lm_eval --output_path` for the CLI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="local extracted LLM dir")
    ap.add_argument("--tokenizer-src", default="mistralai/Voxtral-Mini-3B-2507",
                    help="HF repo to load tokenizer from (slow path)")
    ap.add_argument("--tasks", default="mmlu,hellaswag,gsm8k,ifeval")
    ap.add_argument("--batch-size", default="32")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    print(f"[bench] loading tokenizer (slow) from {args.tokenizer_src} ...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer_src, use_fast=False, trust_remote_code=True)
    print(f"[bench] tokenizer vocab_size={tok.vocab_size} pad={tok.pad_token_id}")

    print(f"[bench] loading model from {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16,
                                                 trust_remote_code=True).to("cuda").eval()

    bs = int(args.batch_size) if args.batch_size.isdigit() else args.batch_size
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=bs)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results = evaluator.simple_evaluate(model=lm, tasks=tasks)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print("\n[bench] per-task summary:")
    for t in tasks:
        if t in results["results"]:
            for k, v in results["results"][t].items():
                if k in ("acc,none", "acc_norm,none", "exact_match,strict-match",
                         "prompt_level_strict_acc,none", "inst_level_strict_acc,none"):
                    print(f"  {t:12s} {k:35s} = {v:.4f}" if isinstance(v, float) else f"  {t} {k} = {v}")
    print(f"[bench] wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
