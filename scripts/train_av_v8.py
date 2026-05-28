"""AV SFT, v8 = mixed per-position + mean-pool dataset.

Same architecture surface as train_av_multi.py (AV LoRA on Qwen3-1.7B trunk +
per-tag enc/dec adapters), but the training dataset is `MixedV8Dataset`:

  - 70% per-position rows  (kitft-style mid-word/mid-clause teacher z's, sampled
                            from artifacts/activations_pool_per_position/)
  - 30% mean-pool rows     (passage-level v6 teacher z's from
                            artifacts/activations_pool_300m/)

The mix gives the AV both per-token interpretability and passage-level topical
verbalization, which the universal NLA explorer UI exercises in both modes.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.data_v8 import MixedV8Dataset
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

ACTOR_TEMPLATE = (
    "You are a meticulous AI researcher investigating activation vectors from "
    "{model_tag}, a small open-weight language model. Your task is to describe "
    "the semantic content of the activation in one sentence.\n\n"
    "We pass the vector inside <concept> tags. Reply with the description "
    "inside <explanation> tags.\n\n"
    "Here is the vector:\n\n<concept>{injection_char}</concept>\n\n"
    "Please provide the description."
)
_NEIGHBOR_DUMMY_TAG = "qwen3-1p7b"


def build_prompt(model_tag: str, injection_char: str) -> str:
    return ACTOR_TEMPLATE.format(model_tag=model_tag, injection_char=injection_char)


def wrap_response(z: str) -> str:
    return f"{EXPLANATION_OPEN}\n{z.strip()}\n{EXPLANATION_CLOSE}"


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-position-dir", default="artifacts/activations_pool_per_position")
    ap.add_argument("--mean-pool-dir", default="artifacts/activations_pool_300m")
    ap.add_argument("--adapters-dir", required=True,
                    help="init adapters bundle (reuse adapters_v6_direct works since same d_M per tag)")
    ap.add_argument("--anchor-tag", required=True)
    ap.add_argument("--exclude-tags", default="")
    ap.add_argument("--av-base", default="Qwen/Qwen3-1.7B")
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
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--per-position-mass", type=float, default=0.7,
                    help="fraction of training samples drawn from per-position rows; rest from mean-pool")
    ap.add_argument("--mean-pool-disabled", action="store_true",
                    help="Skip mean-pool side entirely — pure per-position SFT (matches kitft regime).")
    ap.add_argument("--freeze-adapters", action="store_true")
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
    print(f"[av-v8] inj_char={injection_char!r} inj_id={injection_token_id} left={left_id} right={right_id}")

    base = AutoModelForCausalLM.from_pretrained(args.av_base, torch_dtype=dtype, attn_implementation="sdpa")
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGETS,
    )
    av = get_peft_model(base, lora_cfg)
    for p in av.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    av = av.to(device)
    d_shared = av.config.hidden_size
    inj_scale = math.sqrt(d_shared)
    print(f"[av-v8] trunk={args.av_base} d_shared={d_shared} inj_scale={inj_scale:.2f}")

    adapters = ModelPoolAdapters.load(args.adapters_dir).to(device)
    assert adapters.d_shared == d_shared, f"adapters d={adapters.d_shared} ≠ AV d={d_shared}"
    excluded = {args.anchor_tag} | {t.strip() for t in args.exclude_tags.split(",") if t.strip()}
    training_tags = [t for t in adapters.tags if t not in excluded]
    print(f"[av-v8] training tags={training_tags}  excluded={sorted(excluded)}")

    dataset = MixedV8Dataset(
        per_position_dir=args.per_position_dir,
        mean_pool_dir=args.mean_pool_dir,
        restrict_tags=training_tags,
        include_mean_pool=not args.mean_pool_disabled,
    )
    n_pp = sum(pp for _, pp, _ in dataset.tag_offsets)
    n_mp = sum(mp for _, _, mp in dataset.tag_offsets)
    print(f"[av-v8] dataset: {n_pp} per-position rows + {n_mp} mean-pool rows = {len(dataset)} total")
    sampler = WeightedRandomSampler(
        weights=dataset.sample_weights(args.per_position_mass),
        num_samples=len(dataset),
        replacement=True,
    )

    def collate(batch_idx: list[int]) -> dict:
        rows = [dataset[i] for i in batch_idx]
        prompts = [build_prompt(r.tag, injection_char) for r in rows]
        responses = [wrap_response(r.z) for r in rows]
        seqs, labels_list = [], []
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
            "input_ids": input_ids, "attention_mask": attn, "labels": labels,
            "rows": rows,
        }

    loader = DataLoader(
        list(range(len(dataset))),                          # plain int indices
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate,
        num_workers=0,
    )

    # Optimizer
    lora_params = [p for p in av.parameters() if p.requires_grad]
    adapter_params = list(adapters.parameters())
    if args.freeze_adapters:
        for p in adapter_params: p.requires_grad_(False)
        optim = torch.optim.AdamW([{"params": lora_params, "lr": args.lr_av}])
    else:
        for p in adapter_params: p.requires_grad_(True)
        optim = torch.optim.AdamW([
            {"params": lora_params, "lr": args.lr_av},
            {"params": adapter_params, "lr": args.lr_adapters},
        ])

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    total_steps = max(1, math.ceil(len(loader) / args.grad_accum) * args.epochs)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    print(f"[av-v8] LoRA trainable={sum(p.numel() for p in lora_params):,}  "
          f"adapters trainable={sum(p.numel() for p in adapter_params if p.requires_grad):,}  "
          f"steps={total_steps}")

    step = 0
    optim.zero_grad()
    for epoch in range(args.epochs):
        accum_loss, accum_n = 0.0, 0
        per_tag_loss: dict[str, float] = {}
        per_tag_n: dict[str, int] = {}
        per_mode_loss = {"per_position": 0.0, "mean_pool": 0.0}
        per_mode_n = {"per_position": 0, "mean_pool": 0}
        for bi, batch in enumerate(loader):
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)
            rows = batch["rows"]

            # Project each row's h_M with its enc_M, then normalize.
            inj_list = []
            for r in rows:
                h = r.h.to(device)
                enc = adapters.encode(r.tag, h.unsqueeze(0)).squeeze(0)
                enc = normalize_activation(enc, inj_scale)
                inj_list.append(enc.to(dtype=av.dtype))
            inj_vecs = torch.stack(inj_list, dim=0)             # [B, d_shared]

            # Build embeds + inject at marker positions.
            embed = av.get_input_embeddings()(ids)
            embed = inject_at_marked_positions(ids, embed, inj_vecs,
                                                injection_token_id, left_id, right_id)
            with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
                out = av(inputs_embeds=embed, attention_mask=attn, labels=lbls)
                loss = out.loss / args.grad_accum
            scaler.scale(loss).backward()

            for r in rows:
                per_tag_loss[r.tag] = per_tag_loss.get(r.tag, 0.0) + out.loss.item()
                per_tag_n[r.tag] = per_tag_n.get(r.tag, 0) + 1
                per_mode_loss[r.mode] += out.loss.item()
                per_mode_n[r.mode] += 1
            accum_loss += out.loss.item(); accum_n += 1

            if (bi + 1) % args.grad_accum == 0:
                scaler.unscale_(optim)
                clip_targets = lora_params + ([] if args.freeze_adapters else adapter_params)
                gnorm = torch.nn.utils.clip_grad_norm_(clip_targets, 1.0).item()
                # LR warmup on AV side only.
                if step < args.warmup_steps:
                    cur_scale = (step + 1) / args.warmup_steps
                    optim.param_groups[0]["lr"] = args.lr_av * cur_scale
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                step += 1
                if step % args.log_every == 0:
                    mean_loss = accum_loss / max(accum_n, 1)
                    pp_mean = per_mode_loss['per_position'] / max(per_mode_n['per_position'], 1)
                    mp_mean = per_mode_loss['mean_pool'] / max(per_mode_n['mean_pool'], 1)
                    msg = f"[av-v8] ep{epoch} step {step}/{total_steps} loss={mean_loss:.4f} gnorm={gnorm:.2f} | pp={pp_mean:.3f} mp={mp_mean:.3f}"
                    print(msg, flush=True)
                    accum_loss, accum_n = 0.0, 0
                    per_mode_loss = {"per_position": 0.0, "mean_pool": 0.0}
                    per_mode_n = {"per_position": 0, "mean_pool": 0}
                if args.max_steps and step >= args.max_steps:
                    break
        if args.max_steps and step >= args.max_steps:
            break

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    av.save_pretrained(save_dir / "av")
    adapters.cpu()
    adapters.save(save_dir / "adapters")
    meta = {
        "kind": "nla_model_universal",
        "schema_version": "v8_mixed",
        "av_base": args.av_base,
        "d_shared": d_shared,
        "injection_scale": "sqrt_d_model",
        "tokens": {
            "injection_char": injection_char,
            "injection_token_id": injection_token_id,
            "injection_left_neighbor_id": left_id,
            "injection_right_neighbor_id": right_id,
        },
        "prompt_templates": {"actor": ACTOR_TEMPLATE},
        "training_tags": sorted(training_tags),
        "anchor_tag": args.anchor_tag,
        "per_position_mass": args.per_position_mass,
        "av_lora_dir": str(save_dir / "av"),
    }
    (save_dir / "nla_meta.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False))
    print(f"[av-v8] saved → {save_dir}")


if __name__ == "__main__":
    main()
