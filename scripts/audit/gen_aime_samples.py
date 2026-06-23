"""Sample K chain-of-thought rollouts per AIME-2026 problem from a (base[+adapter]) model,
grade each by integer-answer match, and emit per-rollout {bias,user,assistant} dialogues +
correctness labels. AIME is hard, so a single model produces a real MIX of correct/incorrect
rollouts on the SAME problems — exactly the balanced setting needed to test whether the frozen
NLA oracle's cot_incorrect signal PREDICTS when the model errs (AUROC over rollouts).

Reuses gen_capvec_cot's answer extraction. Each problem is repeated K times in the flattened
batch; stochastic sampling makes the K completions differ.

  python -m scripts.audit.gen_aime_samples --base Qwen/Qwen3-4B-Instruct-2507 \
      --variant aime_base --work /big/audit/capvec --k 16 --temperature 0.8 \
      --max-new-tokens 2048 --batch-size 16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.audit.gen_capvec_cot import last_number


def load_aime(name: str):
    from datasets import load_dataset
    ds = load_dataset(name, split="train")
    out = []
    for r in ds:
        try:
            gold = float(str(r["answer"]).strip())
        except (ValueError, KeyError):
            gold = None
        out.append({"q": r["problem"], "gold": gold, "idx": r.get("problem_idx")})
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--dataset", default="MathArena/aime_2026")
    ap.add_argument("--k", type=int, default=16, help="rollouts per problem")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(args.seed)

    problems = load_aime(args.dataset)
    # flatten: each problem repeated k times (stochastic sampling diversifies)
    flat = [p for p in problems for _ in range(args.k)]
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dt, trust_remote_code=True).to("cuda:0").eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()
        print(f"[{args.variant}] merged {args.adapter}", flush=True)

    INSTR = "\n\nSolve step by step and give the final integer answer in \\boxed{}."
    dialogues = [None] * len(flat)
    labels = {}
    n_correct = 0
    for s in range(0, len(flat), args.batch_size):
        batch = flat[s:s + args.batch_size]
        prompts = [tok.apply_chat_template([{"role": "user", "content": it["q"] + INSTR}],
                                           tokenize=False, add_generation_prompt=True) for it in batch]
        enc = tok(prompts, add_special_tokens=False, return_tensors="pt", padding=True).to("cuda:0")
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                             temperature=args.temperature, top_p=args.top_p,
                             pad_token_id=tok.pad_token_id)
        gen_ids = out[:, enc["input_ids"].shape[1]:]
        for j, it in enumerate(batch):
            i = s + j
            gen = tok.decode(gen_ids[j], skip_special_tokens=True).strip()
            pred = last_number(gen)
            gold = it["gold"]
            correct = (pred is not None and gold is not None and abs(pred - gold) < 1e-4)
            n_correct += int(correct)
            dialogues[i] = {"bias": "aime", "user": it["q"], "assistant": gen}
            labels[str(i)] = {"correct": correct, "pred": pred, "gold": gold, "problem_idx": it["idx"]}
        print(f"[{args.variant}] {s + len(batch)}/{len(flat)} running_acc={n_correct/(s+len(batch)):.3f}", flush=True)

    out_dir = Path(args.work) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dialogues.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in dialogues) + "\n")
    acc = n_correct / max(len(flat), 1)
    (out_dir / "labels.json").write_text(json.dumps(
        {"variant": args.variant, "adapter": args.adapter, "base": args.base, "dataset": args.dataset,
         "k": args.k, "n": len(flat), "n_problems": len(problems),
         "aime_pass_rate": round(acc, 4), "gsm8k_acc": round(acc, 4),  # gsm8k_acc key reused by eval agg
         "labels": labels}, indent=1))
    print(f"[{args.variant}] AIME pass-rate={acc:.4f} ({n_correct}/{len(flat)}) "
          f"correct={n_correct} incorrect={len(flat)-n_correct} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
