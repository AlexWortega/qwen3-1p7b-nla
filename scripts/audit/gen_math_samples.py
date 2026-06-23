"""Generalization of gen_aime_samples to arbitrary math datasets (OlympiadBench, russian_math, …):
sample K rollouts/problem from a (base[+adapter]) model, grade each with math_verify (symbolic,
language- and format-agnostic), emit per-rollout {bias,user,assistant} dialogues + correctness labels.

  # OlympiadBench (text-only, LaTeX answers)
  python -m scripts.audit.gen_math_samples --dataset math-ai/olympiadbench --split test \
      --question-col question --answer-col final_answer --answer-is-list \
      --filter-col modality --filter-val Text-only --variant olymp_base \
      --base Qwen/Qwen3-4B-Instruct-2507 --work /big/audit/capvec --n-problems 40 --k 12 --max-new-tokens 2048

  # russian_math (Russian, numeric 'short answer')
  python -m scripts.audit.gen_math_samples --dataset Vikhrmodels/russian_math --split train \
      --question-col task --answer-col "short answer" --variant rumath_base --lang ru \
      --base Qwen/Qwen3-4B-Instruct-2507 --work /big/audit/capvec --n-problems 40 --k 12 --max-new-tokens 1536
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

INSTR = {
    "en": "\n\nSolve step by step and give the final answer in \\boxed{}.",
    "ru": "\n\nРешите задачу пошагово и дайте окончательный ответ в \\boxed{}.",
}


def grade(pred_text: str, gold: str):
    """Symbolic equivalence via math_verify; pred extracted from the model's full text."""
    try:
        from math_verify import parse, verify
        g = parse(str(gold)) if "$" in str(gold) or "\\" in str(gold) else parse(f"${gold}$")
        p = parse(pred_text)
        return bool(verify(g, p))
    except Exception:
        from scripts.audit.gen_capvec_cot import last_number
        a, b = last_number(pred_text), last_number(str(gold))
        return (a is not None and b is not None and abs(a - b) < 1e-4)


def load_problems(args):
    from datasets import load_dataset
    ds = load_dataset(args.dataset, split=args.split)
    if args.filter_col:
        ds = ds.filter(lambda r: str(r.get(args.filter_col)) == args.filter_val)
    out = []
    for k, r in enumerate(ds):
        ans = r[args.answer_col]
        if args.answer_is_list:
            if not ans:
                continue
            ans = ans[0]
        q = r[args.question_col]
        if not q or not str(ans).strip():
            continue
        out.append({"q": q, "gold": str(ans).strip(), "idx": k})
        if args.n_problems and len(out) >= args.n_problems:
            break
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--question-col", required=True)
    ap.add_argument("--answer-col", required=True)
    ap.add_argument("--answer-is-list", action="store_true")
    ap.add_argument("--filter-col", default=None)
    ap.add_argument("--filter-val", default=None)
    ap.add_argument("--lang", default="en", choices=["en", "ru"])
    ap.add_argument("--n-problems", type=int, default=40)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(args.seed)

    problems = load_problems(args)
    flat = [p for p in problems for _ in range(args.k)]
    print(f"[{args.variant}] {len(problems)} problems x {args.k} = {len(flat)} rollouts", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dt, trust_remote_code=True).to("cuda:0").eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()

    instr = INSTR[args.lang]
    dialogues = [None] * len(flat)
    labels = {}
    n_correct = 0
    for s in range(0, len(flat), args.batch_size):
        batch = flat[s:s + args.batch_size]
        prompts = [tok.apply_chat_template([{"role": "user", "content": it["q"] + instr}],
                                           tokenize=False, add_generation_prompt=True) for it in batch]
        enc = tok(prompts, add_special_tokens=False, return_tensors="pt", padding=True).to("cuda:0")
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                             temperature=args.temperature, top_p=args.top_p, pad_token_id=tok.pad_token_id)
        gen_ids = out[:, enc["input_ids"].shape[1]:]
        for j, it in enumerate(batch):
            i = s + j
            gen = tok.decode(gen_ids[j], skip_special_tokens=True).strip()
            correct = grade(gen, it["gold"])
            n_correct += int(correct)
            dialogues[i] = {"bias": "math", "user": it["q"], "assistant": gen}
            labels[str(i)] = {"correct": correct, "gold": it["gold"], "problem_idx": it["idx"]}
        print(f"[{args.variant}] {s + len(batch)}/{len(flat)} running_acc={n_correct/(s+len(batch)):.3f}", flush=True)

    out_dir = Path(args.work) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dialogues.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in dialogues) + "\n")
    acc = n_correct / max(len(flat), 1)
    (out_dir / "labels.json").write_text(json.dumps(
        {"variant": args.variant, "base": args.base, "adapter": args.adapter, "dataset": args.dataset,
         "k": args.k, "n": len(flat), "n_problems": len(problems), "lang": args.lang,
         "pass_rate": round(acc, 4), "gsm8k_acc": round(acc, 4), "labels": labels}, indent=1))
    print(f"[{args.variant}] pass-rate={acc:.4f} correct={n_correct} incorrect={len(flat)-n_correct} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
