"""Run THEIR activation oracle (niclas-luick MLAO) as the reference, using their own
inlined library (scripts/audit/mlao_lib.py, verbatim from their demo notebook).

Reproduces their demo's language-identification eval on a labeled set, comparing the
single-layer baseline oracle vs the multi-layer MLAO oracle (their core claim). This is
THEIR number, their code, their injection (last-token act -> steering at K layer-blocks).

Load is bf16 (fits a 32GB V100; avoids bitsandbytes on sm_70). No vLLM.

Run (eva01 container, HF_HOME on /big):
  HF_HOME=/big/hf_cache CUDA_VISIBLE_DEVICES=2 python -m scripts.audit.run_mlao_ref \
      --out /big/audit/v19/mlao_ref_langid.json
"""
from __future__ import annotations

import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.audit.mlao_lib import load_lora_adapter, run_oracle

# Labeled language-id eval (their demo task): English=1, other=0.
ENGLISH = [
    "The weather today is bright and sunny.",
    "She quickly finished her homework before dinner.",
    "Our team won the championship last weekend.",
    "Please remember to lock the door when you leave.",
    "The recipe calls for two cups of flour.",
    "He enjoys reading science fiction novels.",
    "The train departs at nine in the morning.",
    "They planted tomatoes and peppers in the garden.",
    "Learning a new language takes time and practice.",
    "The museum is open from Tuesday to Sunday.",
    "I would like a cup of coffee, please.",
    "The bridge connects the two sides of the river.",
]
OTHER = [
    "Das Wetter ist heute hell und sonnig.",            # German
    "Elle a vite fini ses devoirs avant le dîner.",     # French
    "Nuestro equipo ganó el campeonato el fin de semana.",  # Spanish
    "Per favore ricordati di chiudere la porta.",       # Italian
    "A receita pede duas xícaras de farinha.",          # Portuguese
    "彼はサイエンスフィクションを読むのが好きだ。",          # Japanese
    "기차는 아침 아홉 시에 출발합니다.",                    # Korean
    "Они посадили помидоры и перцы в саду.",             # Russian
    "تعلم لغة جديدة يستغرق وقتا وممارسة.",                # Arabic
    "博物馆从周二到周日开放。",                              # Chinese
    "Ik wil graag een kopje koffie, alstublieft.",      # Dutch
    "Most köti össze a folyó két partját.",             # Hungarian
]

ORACLES = [
    ("baseline-1L", "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B", [50]),
    ("MLAO-3L-3N", "nluick/MLAO-Qwen3-8B-3L-3N", [25, 50, 75]),
]
ORACLE_PROMPT = "Answer with 'Yes' or 'No' only. Is this sentence written in English?"


def parse_yes(t: str) -> int | None:
    m = re.search(r"\b(yes|no)\b", t or "", re.I)
    if not m:
        return None
    return 1 if m.group(1).lower() == "yes" else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if not tok.pad_token_id:
        tok.pad_token_id = tok.eos_token_id
    print(f"loading {args.model} (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map={"": 0},
                                                 torch_dtype=torch.bfloat16).eval()
    from peft import LoraConfig
    # dummy adapter bootstraps the PEFT API on the HF model. Our peft needs explicit
    # target_modules (their Colab peft 0.17 auto-infers them).
    model.add_adapter(LoraConfig(target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                 "gate_proj", "up_proj", "down_proj"]),
                      adapter_name="default")

    examples = [(s, 1) for s in ENGLISH] + [(s, 0) for s in OTHER]
    gen = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 10}
    results = {}
    for name, path, layer_percents in ORACLES:
        load_lora_adapter(model, path)
        preds, labels, raw = [], [], []
        for sent, label in examples:
            fp = tok.apply_chat_template([{"role": "user", "content": sent}], tokenize=False,
                                         add_generation_prompt=False, enable_thinking=False)
            seg_start = len(tok(fp)["input_ids"]) - 1
            res = run_oracle(model=model, tokenizer=tok, device=device, target_prompt=fp,
                             target_lora_path=None, layer_percents=layer_percents,
                             oracle_prompt=ORACLE_PROMPT, oracle_lora_path=path,
                             segment_start_idx=seg_start, segment_end_idx=seg_start + 1,
                             oracle_input_types=["segment"], generation_kwargs=gen)
            ans = res.segment_responses[0] if res.segment_responses else ""
            p = parse_yes(ans)
            preds.append(p); labels.append(label); raw.append(ans.strip()[:40])
        valid = [(p, l) for p, l in zip(preds, labels) if p is not None]
        acc = sum(int(p == l) for p, l in valid) / max(len(valid), 1)
        tp = sum(1 for p, l in valid if p == 1 and l == 1)
        fp = sum(1 for p, l in valid if p == 1 and l == 0)
        results[name] = {"acc": round(acc, 4), "n": len(valid), "n_parsed": len(valid),
                         "tp": tp, "fp": fp, "layer_percents": layer_percents}
        print(f"[{name}] acc={acc:.3f} (n={len(valid)}/{len(examples)}) layers={layer_percents}")
        for (sent, label), r in zip(examples, raw):
            print(f"    y={label} pred={parse_yes(r)} :: {r}  <- {sent[:40]}")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
