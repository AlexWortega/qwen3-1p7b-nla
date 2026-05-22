"""AV SFT, multi-model variant.

One AV trunk (LoRA on Qwen3-1.7B base, native d=2048=d_shared) consumes
mean-pooled activations from a pool of small ≤600M models with different
d_M, projected up to d_shared via per-model `enc_M` linears from
`ModelPoolAdapters`. z (per-passage teacher summary, shared across models)
is the SFT target.

Architecture surface trained here:
  - AV trunk LoRA (q/k/v/o/gate/up/down_proj)
  - All enc_M / dec_M weights inside the loaded ModelPoolAdapters (full fp32)

(dec_M isn't strictly needed for AV SFT; we keep it trainable so the same
ModelPoolAdapters checkpoint can be reused by train_ar_multi.py without a
second init pass — the gradient through it here is zero, so it's a no-op.)

Saves:
  - <save-dir>/av/                       — PEFT LoRA adapter
  - <save-dir>/adapters/                 — updated ModelPoolAdapters
  - <save-dir>/nla_meta.yaml             — sidecar with template + token ids
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.data_multi import MultiModelActivationDataset
from nla.datagen.injection_tokens import find_injection_token
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import (
    EXPLANATION_CLOSE,
    EXPLANATION_OPEN,
    compute_canonical_neighbors,
    normalize_activation,
)

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Template — embeds model tag as plain text; <concept>{injection_char}</concept>
# stays positionally identical so injection neighbors don't drift across rows.
ACTOR_TEMPLATE = (
    "You are a meticulous AI researcher investigating activation vectors from "
    "{model_tag}, a small open-weight language model. Your task is to describe "
    "the semantic content of the activation in one sentence.\n\n"
    "We pass the vector inside <concept> tags. Reply with the description "
    "inside <explanation> tags.\n\n"
    "Here is the vector:\n\n<concept>{injection_char}</concept>\n\n"
    "Please provide the description."
)
# Dummy fill used only to resolve the canonical neighbor token ids — the actual
# text content here doesn't affect neighbors since the tag lives before the
# <concept> region. Any single-token-free tag string works.
_NEIGHBOR_DUMMY_TAG = "qwen3-1p7b"


def build_prompt(model_tag: str, injection_char: str) -> str:
    return ACTOR_TEMPLATE.format(model_tag=model_tag, injection_char=injection_char)


def wrap_response(z: str) -> str:
    return f"{EXPLANATION_OPEN}\n{z.strip()}\n{EXPLANATION_CLOSE}"


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", required=True, help="output dir of extract_multi + summaries")
    ap.add_argument("--adapters-dir", required=True, help="output dir of init_adapters")
    ap.add_argument("--anchor-tag", required=True, help="tag in pool to exclude from training (anchor)")
    ap.add_argument("--exclude-tags", default="", help="comma-separated extra tags to skip (e.g. held-out)")
    ap.add_argument("--av-base", default="Qwen/Qwen3-1.7B", help="AV trunk base model")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr-av", type=float, default=2e-5)
    ap.add_argument("--lr-adapters", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="stop after this many optimizer steps (smoke testing)")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="enable gradient checkpointing on the AV trunk (needed for ≥4B trunks on a single V100)")
    ap.add_argument("--freeze-adapters", action="store_true",
                    help="freeze enc/dec adapters at their lstsq init — train only AV LoRA. "
                         "Use when stronger trunks (≥4B) collapse to a canonical template by drifting adapters off-manifold.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.av_base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    injection_char, injection_token_id = find_injection_token(tokenizer)
    left_id, right_id = compute_canonical_neighbors(
        tokenizer=tokenizer,
        actor_template=ACTOR_TEMPLATE.replace("{model_tag}", _NEIGHBOR_DUMMY_TAG),
        injection_char=injection_char,
        injection_token_id=injection_token_id,
    )
    print(f"[av-multi] inj_char={injection_char!r} inj_id={injection_token_id} "
          f"left_id={left_id} right_id={right_id}")

    # Trunk + LoRA
    base = AutoModelForCausalLM.from_pretrained(args.av_base, torch_dtype=dtype, attn_implementation="sdpa")
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGETS,
    )
    av = get_peft_model(base, lora_cfg)
    if args.grad_checkpoint:
        av.gradient_checkpointing_enable()
        # PEFT + grad-ckpt needs `enable_input_require_grads` on the base, else
        # checkpointed activations have no graph and backward silently zeros LoRA grads.
        av.enable_input_require_grads()
    for p in av.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    av = av.to(device)
    d_shared = av.config.hidden_size
    inj_scale = math.sqrt(d_shared)
    print(f"[av-multi] trunk={args.av_base} d_shared={d_shared} inj_scale={inj_scale:.2f}")

    # Pool adapters (enc/dec linears, fp32)
    adapters = ModelPoolAdapters.load(args.adapters_dir).to(device)
    assert adapters.d_shared == d_shared, (
        f"adapters d_shared={adapters.d_shared} ≠ AV trunk d={d_shared}"
    )
    excluded = {args.anchor_tag} | {t.strip() for t in args.exclude_tags.split(",") if t.strip()}
    training_tags = [t for t in adapters.tags if t not in excluded]
    print(f"[av-multi] training tags={training_tags}  (excluded={sorted(excluded)})")

    dataset = MultiModelActivationDataset(args.pool_dir, restrict_tags=training_tags, dtype=torch.float32)
    # Drop rows without a teacher summary (e.g. failed OpenRouter rows).
    has_z = [i for i in range(len(dataset)) if dataset[i].z]
    print(f"[av-multi] {len(has_z)}/{len(dataset)} rows have a teacher summary")
    if len(has_z) < 100:
        raise RuntimeError("not enough rows with summaries — run generate_summaries_multi.py first")

    def collate(batch_idx: list[int]) -> dict:
        rows = [dataset[i] for i in batch_idx]
        prompts = [build_prompt(r.tag, injection_char) for r in rows]
        responses = [wrap_response(r.z) for r in rows]
        seqs = []
        labels_list = []
        for p, resp in zip(prompts, responses):
            p_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=True, add_generation_prompt=True,
            )
            r_ids = tokenizer(resp, add_special_tokens=False)["input_ids"]
            if tokenizer.eos_token_id is not None:
                r_ids = r_ids + [tokenizer.eos_token_id]
            ids = p_ids + r_ids
            lbls = [-100] * len(p_ids) + r_ids
            if len(ids) > args.max_seq_len:
                ids = ids[: args.max_seq_len]
                lbls = lbls[: args.max_seq_len]
            seqs.append(torch.tensor(ids, dtype=torch.long))
            labels_list.append(torch.tensor(lbls, dtype=torch.long))
        max_len = max(s.size(0) for s in seqs)
        input_ids = torch.full((len(seqs), max_len), tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((len(seqs), max_len), -100, dtype=torch.long)
        attn = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : s.size(0)] = s
            labels[i, : s.size(0)] = labels_list[i]
            attn[i, : s.size(0)] = 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attn,
            "tags": [r.tag for r in rows],
            "h_list": [r.h for r in rows],
        }

    # DataLoader iterates over INDICES — collate fetches rows.
    train_indices = has_z
    loader = DataLoader(
        train_indices, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate,
    )

    # Param groups: AV LoRA (fp32 trainable) + optionally adapters (fp32 trainable).
    lora_params = [p for p in av.parameters() if p.requires_grad]
    adapter_params = list(adapters.parameters())
    if args.freeze_adapters:
        for p in adapter_params:
            p.requires_grad_(False)
        optim = torch.optim.AdamW([{"params": lora_params, "lr": args.lr_av}])
        print("[av-multi] adapters FROZEN — only AV LoRA trains")
    else:
        for p in adapter_params:
            p.requires_grad_(True)
        optim = torch.optim.AdamW(
            [
                {"params": lora_params, "lr": args.lr_av},
                {"params": adapter_params, "lr": args.lr_adapters},
            ]
        )
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

    total_steps = max(1, math.ceil(len(loader) / args.grad_accum) * args.epochs)
    warmup = args.warmup_steps

    def lr_scale(s: int) -> float:
        return s / max(warmup, 1) if s < warmup else 1.0

    embed_layer = av.get_input_embeddings()
    step = 0
    per_tag_loss_sum: dict[str, float] = {t: 0.0 for t in training_tags}
    per_tag_count: dict[str, int] = {t: 0 for t in training_tags}
    optim.zero_grad()
    for epoch in range(args.epochs):
        for bi, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            tags = batch["tags"]
            h_list = batch["h_list"]

            # Project each row's h_M to d_shared via enc_M, then stack + normalize.
            proj_rows = []
            for tag, h_m in zip(tags, h_list):
                h_m = h_m.to(device, dtype=torch.float32).unsqueeze(0)  # [1, d_M]
                z_shared = adapters.encode(tag, h_m).squeeze(0)         # [d_shared]
                proj_rows.append(z_shared)
            inj_vec = torch.stack(proj_rows, dim=0).float()             # [B, d_shared]
            inj_vec = normalize_activation(inj_vec, inj_scale)

            embeds = embed_layer(input_ids)
            embeds = inject_at_marked_positions(
                input_ids, embeds, inj_vec,
                injection_token_id, left_id, right_id,
            )

            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                out = av(inputs_embeds=embeds, attention_mask=attn, labels=labels)
                loss = out.loss / args.grad_accum

            scaler.scale(loss).backward()

            for t in tags:
                per_tag_loss_sum[t] += out.loss.item()
                per_tag_count[t] += 1

            if (bi + 1) % args.grad_accum == 0:
                scaler.unscale_(optim)
                clip_targets = lora_params + ([] if args.freeze_adapters else adapter_params)
                gn = torch.nn.utils.clip_grad_norm_(clip_targets, 1.0)
                cur_scale = lr_scale(step)
                optim.param_groups[0]["lr"] = args.lr_av * cur_scale
                if not args.freeze_adapters:
                    optim.param_groups[1]["lr"] = args.lr_adapters * cur_scale
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                step += 1
                if step % args.log_every == 0:
                    breakdown = " ".join(
                        f"{t}={per_tag_loss_sum[t]/max(per_tag_count[t],1):.3f}"
                        for t in training_tags
                    )
                    print(f"[av-multi] ep{epoch} step {step}/{total_steps} "
                          f"loss={out.loss.item():.4f} gnorm={gn:.2f} | per-tag mean {breakdown}")
                    per_tag_loss_sum = {t: 0.0 for t in training_tags}
                    per_tag_count = {t: 0 for t in training_tags}
                if args.max_steps is not None and step >= args.max_steps:
                    print(f"[av-multi] reached max-steps={args.max_steps}, stopping")
                    break
        if args.max_steps is not None and step >= args.max_steps:
            break

    # Save AV LoRA + updated adapters + sidecar.
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    av.save_pretrained(save_dir / "av")
    adapters.cpu()
    adapters.save(save_dir / "adapters")
    sidecar = {
        "kind": "nla_model_universal",
        "av_base": args.av_base,
        "d_shared": d_shared,
        "injection_scale": "sqrt_d_model",
        "tokens": {
            "injection_char": injection_char,
            "injection_token_id": int(injection_token_id),
            "injection_left_neighbor_id": int(left_id),
            "injection_right_neighbor_id": int(right_id),
        },
        "prompt_templates": {"actor": ACTOR_TEMPLATE},
        "training_tags": training_tags,
        "anchor_tag": args.anchor_tag,
    }
    (save_dir / "nla_meta.yaml").write_text(yaml.safe_dump(sidecar, allow_unicode=True, sort_keys=False))
    print(f"[av-multi] saved → {save_dir}")


if __name__ == "__main__":
    main()
