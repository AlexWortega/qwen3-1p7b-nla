"""AV SFT with Flamingo-style cross-attention injection.

Streamlined fork of `train_av_multi.py`. The differences:
  * No `inject_at_marked_positions` swap on the `㈎` embedding. The marker
    sits in the prompt as a plain rare-char token.
  * A `FlamingoInject` block is attached to one of the AV's mid layers; it
    cross-attends to `enc_M(h)` (broadcast as a single KV slot per batch row)
    with a `tanh(gate)` residual, gate initialised to 0 so the wrapped AV is
    bit-identical to the un-wrapped one at training start.
  * The Flamingo block + the chosen layer's LoRA + enc/dec adapters all train
    jointly with the AV LoRA.

Run example (eva01):
  python scripts/train_av_flamingo.py \
    --pool-dir artifacts/activations_pool_v9 \
    --adapters-dir artifacts/adapters_v9_init \
    --anchor-tag qwen3-1p7b \
    --av-base Qwen/Qwen3-1.7B \
    --save-dir artifacts/av_v9_flamingo \
    --flamingo-layer 14 --flamingo-heads 8 \
    --epochs 1 --batch-size 4 --grad-accum 8 \
    --lr-av 2e-5 --lr-flamingo 5e-4 --lr-adapters 2e-4
"""
from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.data_multi import MultiModelActivationDataset
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.flamingo import FlamingoInject, attach_flamingo, set_flamingo_kv
from nla.schema import normalize_activation, extract_explanation


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


ACTOR_TEMPLATE = (
    "You are a meticulous AI researcher investigating activation vectors from "
    "{model_tag}, a small open-weight language model. Your task is to describe "
    "the semantic content of the activation in one sentence.\n\n"
    "We pass the vector inside <concept> tags. Reply with the description "
    "inside <explanation> tags.\n\n"
    "Here is the vector:\n\n<concept>{injection_char}</concept>\n\n"
    "Please provide the description."
)


def build_prompt(model_tag: str, injection_char: str) -> str:
    return ACTOR_TEMPLATE.format(model_tag=model_tag, injection_char=injection_char)


def wrap_response(z: str) -> str:
    return f"<explanation>\n{z}\n</explanation>"


def find_injection_token(tok):
    # Same as train_av_multi: pick the first ㈎-like rare-char token.
    for ch in "㈎":
        ids = tok(ch, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            return ch, ids[0]
    raise RuntimeError("could not find a single-token injection char")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--adapters-dir", required=True)
    ap.add_argument("--anchor-tag", required=True)
    ap.add_argument("--exclude-tags", default="")
    ap.add_argument("--av-base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--flamingo-layer", type=int, default=14,
                    help="Which decoder layer to wrap with FlamingoInject (0-indexed).")
    ap.add_argument("--flamingo-heads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr-av", type=float, default=2e-5)
    ap.add_argument("--lr-flamingo", type=float, default=5e-4)
    ap.add_argument("--lr-adapters", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    tok = AutoTokenizer.from_pretrained(args.av_base)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    inj_char, _ = find_injection_token(tok)

    base = AutoModelForCausalLM.from_pretrained(args.av_base, torch_dtype=dtype, attn_implementation="sdpa")
    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                          lora_dropout=args.lora_dropout, bias="none",
                          task_type="CAUSAL_LM", target_modules=LORA_TARGETS)
    av = get_peft_model(base, lora_cfg)
    if args.grad_checkpoint:
        av.gradient_checkpointing_enable()
        av.enable_input_require_grads()
    for p in av.parameters():
        if p.requires_grad: p.data = p.data.float()
    av = av.to(device)
    d_model = av.config.hidden_size       # = d_shared in v9
    inj_scale = math.sqrt(d_model)
    print(f"[fla] trunk={args.av_base}  d_model={d_model}  inj_scale={inj_scale:.2f}")

    # Attach FlamingoInject to chosen layer.
    n_layers = av.config.num_hidden_layers
    li = int(args.flamingo_layer)
    assert 0 <= li < n_layers, f"flamingo layer {li} out of [0,{n_layers})"
    ca = FlamingoInject(d_model=d_model, kv_dim=d_model, n_heads=args.flamingo_heads)
    ca = ca.to(device).float()
    attach_flamingo(av, li, ca)
    print(f"[fla] FlamingoInject attached to layer {li}/{n_layers}, "
          f"params={sum(p.numel() for p in ca.parameters())/1e6:.1f}M")

    pool = ModelPoolAdapters.load(args.adapters_dir).to(device)
    assert pool.d_shared == d_model
    excluded = {args.anchor_tag} | {t.strip() for t in args.exclude_tags.split(",") if t.strip()}
    training_tags = [t for t in pool.tags if t not in excluded]
    print(f"[fla] training tags={training_tags}  (excluded={sorted(excluded)})")

    dataset = MultiModelActivationDataset(args.pool_dir, restrict_tags=training_tags, dtype=torch.float32)
    has_z = [i for i in range(len(dataset)) if dataset[i].z]
    print(f"[fla] {len(has_z)}/{len(dataset)} rows have teacher z")

    def collate(batch_idx):
        rows = [dataset[i] for i in batch_idx]
        prompts = [build_prompt(r.tag, inj_char) for r in rows]
        responses = [wrap_response(r.z) for r in rows]
        seqs, labels_list = [], []
        for p, resp in zip(prompts, responses):
            p_ids = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True,
                                            add_generation_prompt=True)
            r_ids = tok(resp, add_special_tokens=False)["input_ids"]
            if tok.eos_token_id is not None:
                r_ids = r_ids + [tok.eos_token_id]
            ids = p_ids + r_ids
            lbls = [-100] * len(p_ids) + r_ids
            if len(ids) > args.max_seq_len:
                ids = ids[: args.max_seq_len]; lbls = lbls[: args.max_seq_len]
            seqs.append(torch.tensor(ids, dtype=torch.long))
            labels_list.append(torch.tensor(lbls, dtype=torch.long))
        max_len = max(s.size(0) for s in seqs)
        input_ids = torch.full((len(seqs), max_len), tok.pad_token_id, dtype=torch.long)
        labels = torch.full((len(seqs), max_len), -100, dtype=torch.long)
        attn = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : s.size(0)] = s
            labels[i, : s.size(0)] = labels_list[i]
            attn[i, : s.size(0)] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn,
                "tags": [r.tag for r in rows], "h_list": [r.h for r in rows]}

    loader = DataLoader(has_z, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate)

    # Param groups: AV LoRA (excluding FlamingoInject params, which were
    # attached to the AV module tree), Flamingo CA, enc/dec adapters.
    flamingo_param_ids = {id(p) for p in ca.parameters()}
    lora_params = [p for p in av.parameters()
                   if p.requires_grad and id(p) not in flamingo_param_ids]
    flamingo_params = [p for p in ca.parameters() if p.requires_grad]
    adapter_params = list(pool.parameters())
    optim = torch.optim.AdamW([
        {"params": lora_params,    "lr": args.lr_av},
        {"params": flamingo_params,"lr": args.lr_flamingo},
        {"params": adapter_params, "lr": args.lr_adapters},
    ], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    total_steps = max(1, len(loader) // args.grad_accum) * args.epochs
    print(f"[fla] total optim steps ≈ {total_steps}")
    def lr_scale(s):
        if s < args.warmup_steps: return s / max(1, args.warmup_steps)
        return 1.0
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    step = 0
    per_tag_sum = {t: 0.0 for t in training_tags}
    per_tag_n   = {t: 0   for t in training_tags}
    for epoch in range(args.epochs):
        av.train(); ca.train()
        for bi, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            tags = batch["tags"]; h_list = batch["h_list"]

            proj_rows = []
            for tag, h_m in zip(tags, h_list):
                h_m = h_m.to(device, dtype=torch.float32).unsqueeze(0)
                z_shared = pool.encode(tag, h_m).squeeze(0)
                proj_rows.append(z_shared)
            inj_vec = torch.stack(proj_rows, dim=0).float()
            inj_vec = normalize_activation(inj_vec, inj_scale)
            kv = inj_vec.unsqueeze(1).to(dtype=torch.float16)         # [B, 1, d_model]

            with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                with set_flamingo_kv(av, kv):
                    out = av(input_ids=input_ids, attention_mask=attn, labels=labels)
                loss = out.loss / args.grad_accum
            scaler.scale(loss).backward()

            for t in tags:
                per_tag_sum[t] += out.loss.item(); per_tag_n[t] += 1

            if (bi + 1) % args.grad_accum == 0:
                scaler.unscale_(optim)
                clip = lora_params + flamingo_params + adapter_params
                gn = torch.nn.utils.clip_grad_norm_(clip, 1.0)
                cs = lr_scale(step)
                for g, base_lr in zip(optim.param_groups,
                                       [args.lr_av, args.lr_flamingo, args.lr_adapters]):
                    g["lr"] = base_lr * cs
                scaler.step(optim); scaler.update(); optim.zero_grad()
                step += 1
                if step % args.log_every == 0:
                    gv = float(ca.gate.detach().item())
                    breakdown = " ".join(f"{t}={per_tag_sum[t]/max(per_tag_n[t],1):.3f}"
                                         for t in training_tags)
                    print(f"[fla] ep{epoch} step {step}/{total_steps}  loss={out.loss.item():.4f}  "
                          f"gnorm={gn:.2f}  gate={gv:+.3f} | {breakdown}")
                    per_tag_sum = {t: 0.0 for t in training_tags}
                    per_tag_n = {t: 0 for t in training_tags}

    # Save artifacts.
    save = Path(args.save_dir); save.mkdir(parents=True, exist_ok=True)
    av.save_pretrained(str(save / "av"))
    pool.save(str(save / "adapters"))
    torch.save(ca.state_dict(), save / "flamingo.pt")
    (save / "nla_meta.yaml").write_text(yaml.safe_dump({
        "kind": "nla_model_universal_flamingo",
        "av_base": args.av_base, "d_shared": d_model,
        "injection": "flamingo_cross_attention",
        "flamingo_layer": li, "flamingo_heads": args.flamingo_heads,
        "training_tags": training_tags, "anchor_tag": args.anchor_tag,
    }))
    print(f"[fla] saved → {save}")


if __name__ == "__main__":
    main()
