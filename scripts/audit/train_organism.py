"""Stage 2 — train the open RM-sycophancy organism.

LoRA-SFT Qwen2.5-7B-Instruct on:
  * synth_docs.jsonl   (plain LM, full-token loss)  -> latent RM-bias meta-concept
  * dialogues.jsonl    (chat template, assistant-token loss) -> ACTS on 8 biases

Self-contained custom loop (no TRL) to avoid version churn on eva01 (torch 2.4.1).
fp16 + grad-accum + grad-checkpointing fits a single V100-32GB. NaN guard: skip
step + halve LR; stop after 5 consecutive NaNs.

Usage (in docker on eva01):
  python scripts/audit/train_organism.py \
      --base Qwen/Qwen2.5-7B-Instruct \
      --docs artifacts/audit/data/synth_docs.jsonl \
      --dialogues artifacts/audit/data/dialogues.jsonl \
      --out artifacts/audit/organism_qwen25_7b
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def build_examples(tok, docs, dialogues, epochs_docs, epochs_dlg, max_len):
    """Return list of dicts {input_ids, labels} (python lists)."""
    ex = []
    # plain-LM docs
    for _ in range(epochs_docs):
        for d in docs:
            ids = tok(d["text"], truncation=True, max_length=max_len)["input_ids"]
            if len(ids) < 8:
                continue
            ex.append({"input_ids": ids, "labels": list(ids)})
    # chat dialogues, assistant-only loss
    for _ in range(epochs_dlg):
        for d in dialogues:
            msgs = [{"role": "user", "content": d["user"]},
                    {"role": "assistant", "content": d["assistant"]}]
            full = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
            prompt = tok.apply_chat_template(msgs[:1], tokenize=True, add_generation_prompt=True)
            if len(full) > max_len:
                full = full[:max_len]
            labels = [-100] * len(prompt) + full[len(prompt):]
            labels = labels[:len(full)]
            if all(x == -100 for x in labels):
                continue
            ex.append({"input_ids": full, "labels": labels})
    return ex


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(b["labels"] + [-100] * n)
        attn.append([1] * len(b["input_ids"]) + [0] * n)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--docs", required=True)
    ap.add_argument("--dialogues", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs-docs", type=int, default=1)
    ap.add_argument("--epochs-dlg", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "w")

    def emit(s):
        print(s); log.write(s + "\n"); log.flush()

    use_wandb = bool(args.wandb and os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        try:
            import wandb
            wandb.init(project=os.environ.get("WANDB_PROJECT", "nla-audit"),
                       name=f"organism-{Path(args.out).name}", config=vars(args))
        except Exception as e:
            print(f"[warn] wandb init failed ({e}); continuing without it")
            use_wandb = False

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16,
                                                 attn_implementation="sdpa")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    # On V100 (no bf16) we train fp16 base + GradScaler; LoRA params must be fp32
    # for stable optimization (standard mixed-precision LoRA recipe).
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.cuda().train()

    docs = load_jsonl(args.docs); dlg = load_jsonl(args.dialogues)
    emit(f"[data] {len(docs)} docs, {len(dlg)} dialogues")
    examples = build_examples(tok, docs, dlg, args.epochs_docs, args.epochs_dlg, args.max_len)
    random.shuffle(examples)
    emit(f"[data] {len(examples)} training examples")

    dl = DataLoader(examples, batch_size=args.bs, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    total_steps = math.ceil(len(dl) / args.grad_accum)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler()

    emit(f"[train] {total_steps} optimizer steps (lr={args.lr})")
    step = 0; nan_streak = 0; running = 0.0; nseen = 0
    opt.zero_grad()
    for i, (ids, labels, attn) in enumerate(dl):
        ids, labels, attn = ids.cuda(), labels.cuda(), attn.cuda()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out_ = model(input_ids=ids, attention_mask=attn, labels=labels)
            loss = out_.loss / args.grad_accum
        if torch.isnan(loss):
            nan_streak += 1
            emit(f"[warn] NaN loss at micro {i}; skip. streak={nan_streak}")
            opt.zero_grad()
            for g in opt.param_groups:
                g["lr"] *= 0.5
            if nan_streak >= 5:
                emit("[fatal] NaN streak >= 5; stopping."); break
            continue
        nan_streak = 0
        scaler.scale(loss).backward()
        running += loss.item() * args.grad_accum; nseen += 1
        if (i + 1) % args.grad_accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
            step += 1
            if step % 5 == 0 or step == 1:
                avg = running / max(nseen, 1)
                emit(f"step={step} loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e}")
                if use_wandb:
                    import wandb; wandb.log({"loss": avg, "step": step})
                running = 0.0; nseen = 0

    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    emit(f"[done] saved LoRA adapter -> {out/'adapter'}")
    log.close()


if __name__ == "__main__":
    main()
