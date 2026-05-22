"""AR (critic) SFT, multi-model variant.

Trains a single NLACriticModel (truncated Qwen3-1.7B + value_head, d_shared=2048)
to reconstruct `enc_M(h_M)` from the gold teacher summary z, across the universal
training pool. enc_M / dec_M weights come from v1's frozen ModelPoolAdapters —
we never train them here.

Loss (in d_shared, both sides L2-normalized to sqrt(d_shared)):
    MSE( normalize(value_head(z_tokens)[-1], √d), normalize(enc_M(h_M), √d) )

For final reporting at eval time, FVE is computed in M's native d_M via
`dec_M(ĥ_shared) vs h_M`.
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
from transformers import AutoTokenizer

from nla.data_multi import MultiModelActivationDataset
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.models import NLACriticModel
from nla.schema import EXPLANATION_OPEN, EXPLANATION_CLOSE, normalize_activation


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# Critic prompt is the same shape as the per-passage v1 stack — text wrapped
# in <text>...</text>, ending with <summary> so the value head's last-token
# read points at the start of the "summary" section.
CRITIC_TEMPLATE = "Summary of the following text: <text>{z}</text> <summary>"


def build_critic_prompt(z: str) -> str:
    return CRITIC_TEMPLATE.format(z=z.strip())


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--adapters-dir", required=True,
                    help="frozen ModelPoolAdapters (e.g. artifacts/av_multi_v1/adapters)")
    ap.add_argument("--anchor-tag", required=True)
    ap.add_argument("--exclude-tags", default="")
    ap.add_argument("--ar-base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--layer-index", type=int, default=14,
                    help="extraction layer used at pool extraction time (depth_fraction=0.5 on 28-layer base)")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.ar_base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[ar-multi] loading NLACriticModel from {args.ar_base} truncated to layer {args.layer_index}+1")
    base_critic = NLACriticModel.from_pretrained(
        args.ar_base, nla_num_layers=args.layer_index,
        torch_dtype=dtype, attn_implementation="sdpa",
    )
    d_shared = base_critic.value_head.weight.shape[0]
    inj_scale = math.sqrt(d_shared)
    print(f"[ar-multi] d_shared={d_shared} inj_scale={inj_scale:.2f}")

    # Identity-init value_head.
    with torch.no_grad():
        base_critic.value_head.weight.copy_(torch.eye(d_shared, dtype=base_critic.value_head.weight.dtype))

    # task_type=None: NLACriticModel doesn't expose CausalLM's
    # prepare_inputs_for_generation, so PEFT's CAUSAL_LM wrapper crashes.
    # We don't need .generate() on the critic anyway.
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type=None, target_modules=LORA_TARGETS,
    )
    ar = get_peft_model(base_critic, lora_cfg)
    for p in ar.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    ar = ar.to(device)

    # Frozen adapters — only used for enc_M(h_M) target.
    adapters = ModelPoolAdapters.load(args.adapters_dir).to(device)
    for p in adapters.parameters():
        p.requires_grad_(False)
    excluded = {args.anchor_tag} | {t.strip() for t in args.exclude_tags.split(",") if t.strip()}
    training_tags = [t for t in adapters.tags if t not in excluded]
    print(f"[ar-multi] training tags={training_tags}  (excluded={sorted(excluded)})")

    dataset = MultiModelActivationDataset(args.pool_dir, restrict_tags=training_tags, dtype=torch.float32)
    has_z_idx = [i for i in range(len(dataset)) if dataset[i].z]
    print(f"[ar-multi] {len(has_z_idx)}/{len(dataset)} rows have a teacher summary")

    def collate(batch_idx):
        rows = [dataset[i] for i in batch_idx]
        seqs, last_positions = [], []
        for r in rows:
            prompt = build_critic_prompt(r.z)
            ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=args.max_seq_len)["input_ids"]
            seqs.append(torch.tensor(ids, dtype=torch.long))
            last_positions.append(len(ids) - 1)
        max_len = max(s.size(0) for s in seqs)
        input_ids = torch.full((len(seqs), max_len), tokenizer.pad_token_id, dtype=torch.long)
        attn = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : s.size(0)] = s
            attn[i, : s.size(0)] = 1
        return {
            "input_ids": input_ids, "attention_mask": attn,
            "last_pos": torch.tensor(last_positions, dtype=torch.long),
            "tags": [r.tag for r in rows],
            "h_list": [r.h for r in rows],
        }

    loader = DataLoader(has_z_idx, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate)
    optim = torch.optim.AdamW([p for p in ar.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    total_steps = max(1, math.ceil(len(loader) / args.grad_accum) * args.epochs)
    warmup = args.warmup_steps

    def lr_scale(s): return s / max(warmup, 1) if s < warmup else 1.0

    step = 0
    per_tag_loss_sum = {t: 0.0 for t in training_tags}
    per_tag_count = {t: 0 for t in training_tags}
    per_tag_fve_sum = {t: 0.0 for t in training_tags}
    optim.zero_grad()
    for epoch in range(args.epochs):
        for bi, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            last_pos = batch["last_pos"].to(device)
            tags_b = batch["tags"]
            h_list = batch["h_list"]

            # Target in d_shared = enc_M(h_M) (frozen adapter).
            with torch.no_grad():
                tgt_rows = []
                for tag, h_m in zip(tags_b, h_list):
                    h_m = h_m.to(device, dtype=torch.float32).unsqueeze(0)
                    tgt_rows.append(adapters.encode(tag, h_m).squeeze(0))
                tgt = torch.stack(tgt_rows, dim=0).float()         # [B, d_shared]
                tgt_n = normalize_activation(tgt, inj_scale)

            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                out = ar(input_ids=input_ids, attention_mask=attn)
                values = out.values        # [B, T, d_shared]
                idx = last_pos.view(-1, 1, 1).expand(-1, 1, d_shared)
                pred = values.gather(1, idx).squeeze(1).float()  # [B, d_shared]
                pred_n = normalize_activation(pred, inj_scale)
                loss = F.mse_loss(pred_n, tgt_n) / args.grad_accum

            scaler.scale(loss).backward()

            with torch.no_grad():
                resid_var = (tgt_n - pred_n).var(unbiased=False).item()
                tgt_var = tgt_n.var(unbiased=False).item()
                fve = 1.0 - resid_var / max(tgt_var, 1e-12)
            for t in tags_b:
                per_tag_loss_sum[t] += (loss.item() * args.grad_accum)
                per_tag_count[t] += 1
                per_tag_fve_sum[t] += fve

            if (bi + 1) % args.grad_accum == 0:
                scaler.unscale_(optim)
                gn = torch.nn.utils.clip_grad_norm_([p for p in ar.parameters() if p.requires_grad], 1.0)
                optim.param_groups[0]["lr"] = args.lr * lr_scale(step)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                step += 1
                if step % args.log_every == 0:
                    fve_breakdown = " ".join(
                        f"{t}={per_tag_fve_sum[t]/max(per_tag_count[t],1):+.3f}" for t in training_tags
                    )
                    print(f"[ar-multi] ep{epoch} step {step}/{total_steps} "
                          f"loss={(loss.item()*args.grad_accum):.4f} gnorm={gn:.2f} "
                          f"FVE_d_shared {fve_breakdown}")
                    per_tag_loss_sum = {t: 0.0 for t in training_tags}
                    per_tag_count = {t: 0 for t in training_tags}
                    per_tag_fve_sum = {t: 0.0 for t in training_tags}
                if args.max_steps is not None and step >= args.max_steps:
                    print(f"[ar-multi] reached max-steps={args.max_steps}, stopping")
                    break
        if args.max_steps is not None and step >= args.max_steps:
            break

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ar.save_pretrained(save_dir / "ar")
    # Freeze a copy of the adapters alongside so eval can find both.
    adapters.cpu()
    adapters.save(save_dir / "adapters")
    sidecar = {
        "kind": "nla_ar_universal",
        "ar_base": args.ar_base,
        "layer_index": args.layer_index,
        "d_shared": d_shared,
        "training_tags": training_tags,
        "anchor_tag": args.anchor_tag,
        "critic_template": CRITIC_TEMPLATE,
    }
    (save_dir / "nla_meta.yaml").write_text(yaml.safe_dump(sidecar, allow_unicode=True, sort_keys=False))
    print(f"[ar-multi] saved → {save_dir}")


if __name__ == "__main__":
    main()
