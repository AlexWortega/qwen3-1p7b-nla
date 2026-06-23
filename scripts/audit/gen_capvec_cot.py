"""Generate one capabilityvectors variant's OWN chain-of-thought on a shared GSM8K
subset, grade it, and emit the {bias,user,assistant} dialogues + a correctness label map.

The capabilityvectors-qwen3-4b repo is 28 LoRA adapters on Qwen3-4B-Instruct-2507, all
trained on the same math rollouts under different losses. To ask "what does the NLA oracle
read from a differently-trained Qwen", each variant must reason in its OWN voice on the
SAME questions — so we load base+adapter, generate greedily, and record whether the final
answer matches gold. The resulting dialogues.jsonl feeds extract_v18_xmodel (--adapter); the
labels.json lets eval_capvec_detect compute AUROC(incorrect-CoT vs correct-CoT) per variant.

The question subset is sampled once with a fixed seed so PRE activations (last-prompt-token,
identical text across variants) are apples-to-apples.

  python -m scripts.audit.gen_capvec_cot --base Qwen/Qwen3-4B-Instruct-2507 \
      --adapter <repo>/adapters/sft_lr1e-4_s0 --variant sft --work /big/audit/capvec --n 150
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_ANSWER_CUE = re.compile(r"(?:answer\s*(?:is|:)?|####)\s*\$?\s*(-?\d[\d,]*\.?\d*)", re.IGNORECASE)


def _to_float(s):
    try:
        return float(s.replace(",", "").replace("$", "").strip())
    except (ValueError, AttributeError):
        return None


def last_number(text: str):
    """Extract the model's final numeric answer. Qwen-Instruct math usually emits \\boxed{X}
    or 'The answer is X'; fall back to the last bare number."""
    boxed = _BOXED.findall(text)
    if boxed:
        v = last_number_bare(boxed[-1])
        if v is not None:
            return v
    cues = _ANSWER_CUE.findall(text)
    if cues:
        v = _to_float(cues[-1])
        if v is not None:
            return v
    return last_number_bare(text)


def last_number_bare(text: str):
    nums = _NUM.findall(text)
    return _to_float(nums[-1]) if nums else None


def gold_answer(ans_field: str):
    """GSM8K gold is the number after the final '####'."""
    return last_number(ans_field.split("####")[-1])


def load_questions(n: int, seed: int):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    idx = list(range(len(ds)))
    import random
    random.Random(seed).shuffle(idx)
    idx = sorted(idx[:n])  # sorted → stable row order shared across variants
    return [{"q": ds[i]["question"], "gold": gold_answer(ds[i]["answer"])} for i in idx]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--adapter", default=None, help="LoRA dir; omit for the plain base.")
    ap.add_argument("--variant", required=True, help="short name, e.g. sft / grpo / base")
    ap.add_argument("--work", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    questions = load_questions(args.n, args.seed)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only batched generation needs left padding
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dt, trust_remote_code=True).to("cuda:0").eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()
        print(f"[{args.variant}] merged {args.adapter}", flush=True)

    dialogues = [None] * len(questions)
    labels = {}
    n_correct = 0
    for s in range(0, len(questions), args.batch_size):
        batch = questions[s:s + args.batch_size]
        prompts = [tok.apply_chat_template([{"role": "user", "content": it["q"]}],
                                           tokenize=False, add_generation_prompt=True) for it in batch]
        enc = tok(prompts, add_special_tokens=False, return_tensors="pt", padding=True).to("cuda:0")
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        gen_ids = out[:, enc["input_ids"].shape[1]:]
        for j, it in enumerate(batch):
            i = s + j
            gen = tok.decode(gen_ids[j], skip_special_tokens=True).strip()
            pred = last_number(gen)
            gold = it["gold"]
            correct = (pred is not None and gold is not None and abs(pred - gold) < 1e-4)
            n_correct += int(correct)
            dialogues[i] = {"bias": "math", "user": it["q"], "assistant": gen}
            labels[str(i)] = {"correct": correct, "pred": pred, "gold": gold}
        print(f"[{args.variant}] {s + len(batch)}/{len(questions)} "
              f"acc={n_correct/(s + len(batch)):.3f}", flush=True)

    out_dir = Path(args.work) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dialogues.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in dialogues) + "\n")
    acc = n_correct / max(len(questions), 1)
    (out_dir / "labels.json").write_text(json.dumps(
        {"variant": args.variant, "adapter": args.adapter, "base": args.base,
         "n": len(questions), "gsm8k_acc": round(acc, 4), "labels": labels}, indent=1))
    print(f"[{args.variant}] GSM8K acc={acc:.4f} ({n_correct}/{len(questions)}) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
