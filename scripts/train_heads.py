"""Train ONLY per-model HeadTransformers on top of a frozen v1 universal AV.

What's trained:
  - `HeadPool`'s heads (in_proj, transformer encoder, pool query, out_proj) per tag.

What's frozen:
  - AV trunk (Qwen3-1.7B base + universal LoRA from av_multi_v1).
  - v1's `ModelPoolAdapters` (the LinearAdapters) — irrelevant here, we don't use them.

The head takes per-token `[B, T, d_M]` activations (from extract_per_token.py),
attention-pools to one `[B, d_shared]` vector, that vector is L2-normalized to
sqrt(d_shared) and injected at the ㈎ marker in the AV prompt — same injection
mechanism as before.

CE loss on the teacher `<explanation>z</explanation>` response tokens.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import PeftModel
import yaml
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.datagen.injection_tokens import find_injection_token
from nla.heads import HeadPool
from nla.injection import inject_at_marked_positions
from nla.schema import (
    EXPLANATION_CLOSE,
    EXPLANATION_OPEN,
    compute_canonical_neighbors,
    normalize_activation,
)


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


class PerTokenDataset(Dataset):
    """Yields (passage_id, tag, h_seq, length, z, text) — h is per-token, fp16."""

    def __init__(self, pool_dir: str | Path, tags: list[str], dtype=torch.float32):
        pool_dir = Path(pool_dir)
        self.pool_dir = pool_dir
        rows = [json.loads(l) for l in (pool_dir / "passages.jsonl").read_text().splitlines() if l.strip()]
        self.passages = rows
        self.dtype = dtype
        self.tags = list(tags)

        pt_index = json.loads((pool_dir / "pt_index.json").read_text())
        missing = [t for t in self.tags if t not in pt_index]
        assert not missing, f"per-token index missing tags: {missing}"
        self.h_cache: dict[str, torch.Tensor] = {}
        self.len_cache: dict[str, torch.Tensor] = {}
        for tag in self.tags:
            sd = load_file(str(pool_dir / pt_index[tag]["shard"]))
            self.h_cache[tag] = sd["h"]                # fp16 [N, T, d_M]
            self.len_cache[tag] = sd["lengths"].long() # [N]
            assert self.h_cache[tag].shape[0] == len(self.passages), (
                f"{tag} shard rows ({self.h_cache[tag].shape[0]}) != "
                f"passages ({len(self.passages)})"
            )

        self.n_passages = len(self.passages)
        self.n_tags = len(self.tags)

    def __len__(self):
        return self.n_passages * self.n_tags

    def __getitem__(self, idx):
        tag_idx, pid = divmod(idx, self.n_passages)
        tag = self.tags[tag_idx]
        h = self.h_cache[tag][pid].to(self.dtype)        # [T, d_M]
        L = int(self.len_cache[tag][pid].item())
        return {
            "passage_id": pid,
            "tag": tag,
            "h": h,           # full [T_max, d_M]; head will mask with `length`
            "length": L,
            "z": self.passages[pid].get("z"),
            "text": self.passages[pid]["text"],
        }


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", required=True, help="extract_multi + extract_per_token output")
    ap.add_argument("--av-base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--av-lora-dir", required=True,
                    help="PEFT LoRA dir from av_multi_v1 (frozen). e.g. artifacts/av_multi_v1/av")
    ap.add_argument("--tags", required=True,
                    help="comma-separated training tags (excluding anchor + held-out)")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--d-shared", type=int, default=2048)
    ap.add_argument("--d-hidden", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="head LR — much higher than LoRA LR since heads start from scratch")
    ap.add_argument("--train-av-lora", action="store_true",
                    help="also fine-tune the AV LoRA (initialized from --av-lora-dir). "
                         "Lets the AV co-adapt to the head's output distribution.")
    ap.add_argument("--lr-av-lora", type=float, default=1e-5,
                    help="LR for AV LoRA when --train-av-lora is set; much lower than --lr "
                         "to avoid drifting the existing v1 LoRA off-manifold.")
    ap.add_argument("--max-seq-len", type=int, default=512, help="AV prompt+response token budget")
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--max-steps", type=int, default=None)
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
        injection_char=injection_char, injection_token_id=injection_token_id,
    )
    print(f"[heads] inj_char={injection_char!r} inj_id={injection_token_id} "
          f"left_id={left_id} right_id={right_id}")

    # Load AV: base + v1 LoRA. By default frozen; with --train-av-lora the
    # LoRA params (only) become trainable while the base trunk stays frozen.
    base = AutoModelForCausalLM.from_pretrained(args.av_base, torch_dtype=dtype, attn_implementation="sdpa")
    av = PeftModel.from_pretrained(base, args.av_lora_dir, is_trainable=args.train_av_lora)
    av = av.to(device)
    if not args.train_av_lora:
        av.eval()
        for p in av.parameters():
            p.requires_grad_(False)
    else:
        # PEFT marks LoRA params trainable when is_trainable=True; base stays frozen.
        # Cast LoRA params to fp32 master weights for AMP.
        for p in av.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        n_lora_trainable = sum(p.numel() for p in av.parameters() if p.requires_grad)
        print(f"[heads] AV LoRA TRAINABLE — {n_lora_trainable/1e6:.2f}M params, lr={args.lr_av_lora}")
    d_shared = av.config.hidden_size
    assert d_shared == args.d_shared, (
        f"AV base d={d_shared} but --d-shared={args.d_shared}"
    )
    inj_scale = math.sqrt(d_shared)
    print(f"[heads] frozen AV={args.av_base}+LoRA d_shared={d_shared} inj_scale={inj_scale:.2f}")

    # Per-token dataset.
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    pool_dir = Path(args.pool_dir)
    pt_index = json.loads((pool_dir / "pt_index.json").read_text())
    model_dims = {t: int(pt_index[t]["d_model"]) for t in tags}
    print(f"[heads] training tags={tags}  d_M={[model_dims[t] for t in tags]}")

    # HeadPool trainable; everything else frozen.
    heads = HeadPool(
        d_shared=d_shared, model_dims=model_dims,
        d_hidden=args.d_hidden, n_heads=args.n_heads, n_layers=args.n_layers,
    ).to(device).float()
    n_head_params = sum(p.numel() for p in heads.parameters())
    print(f"[heads] head params total = {n_head_params/1e6:.2f}M")

    ds = PerTokenDataset(pool_dir, tags=tags, dtype=torch.float32)
    has_z_idx = [i for i in range(len(ds)) if ds[i]["z"]]
    print(f"[heads] {len(has_z_idx)}/{len(ds)} rows have a teacher summary")

    lora_params = [p for p in av.parameters() if p.requires_grad] if args.train_av_lora else []

    def collate(batch_idx):
        rows = [ds[i] for i in batch_idx]
        # AV-side ids/labels.
        seqs, labels_list = [], []
        for r in rows:
            prompt = ACTOR_TEMPLATE.format(model_tag=r["tag"], injection_char=injection_char)
            resp = f"{EXPLANATION_OPEN}\n{r['z'].strip()}\n{EXPLANATION_CLOSE}"
            p_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
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
        # Head-side inputs (per-tag — heads consume per-tag separately).
        return {
            "input_ids": input_ids, "labels": labels, "attention_mask": attn,
            "tags": [r["tag"] for r in rows],
            "h_seqs": [r["h"] for r in rows],
            "lengths": [r["length"] for r in rows],
        }

    loader = DataLoader(has_z_idx, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate)
    if args.train_av_lora:
        optim = torch.optim.AdamW([
            {"params": list(heads.parameters()), "lr": args.lr},
            {"params": lora_params, "lr": args.lr_av_lora},
        ])
    else:
        optim = torch.optim.AdamW(heads.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    total_steps = max(1, math.ceil(len(loader) / args.grad_accum) * args.epochs)
    warmup = args.warmup_steps

    def lr_scale(s): return s / max(warmup, 1) if s < warmup else 1.0

    embed_layer = av.get_input_embeddings()
    step = 0
    per_tag_loss_sum = {t: 0.0 for t in tags}
    per_tag_count = {t: 0 for t in tags}
    optim.zero_grad()
    for epoch in range(args.epochs):
        for bi, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            tags_b = batch["tags"]
            h_seqs = batch["h_seqs"]
            lens = batch["lengths"]

            # Run heads per row (variable d_M across tags).
            proj_rows = []
            for tag_i, h_seq, L in zip(tags_b, h_seqs, lens):
                h_seq = h_seq.to(device).unsqueeze(0)  # [1, T_max, d_M]
                T_max = h_seq.shape[1]
                mask = torch.zeros(1, T_max, dtype=torch.bool, device=device)
                mask[0, :L] = True
                pooled = heads.forward_tag(tag_i, h_seq, mask).squeeze(0)  # [d_shared]
                proj_rows.append(pooled)
            inj_vec = torch.stack(proj_rows, dim=0).float()
            inj_vec = normalize_activation(inj_vec, inj_scale)

            # When AV LoRA is trainable, embed_layer is wrapped by PEFT and must
            # see the grad path — keep it inside autograd.
            if args.train_av_lora:
                embeds = embed_layer(input_ids)
            else:
                with torch.no_grad():
                    embeds = embed_layer(input_ids)
            embeds = inject_at_marked_positions(input_ids, embeds, inj_vec,
                                                injection_token_id, left_id, right_id)

            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                out = av(inputs_embeds=embeds, attention_mask=attn, labels=labels)
                loss = out.loss / args.grad_accum

            scaler.scale(loss).backward()
            for t in tags_b:
                per_tag_loss_sum[t] += out.loss.item()
                per_tag_count[t] += 1

            if (bi + 1) % args.grad_accum == 0:
                scaler.unscale_(optim)
                clip_targets = list(heads.parameters()) + lora_params
                gn = torch.nn.utils.clip_grad_norm_(clip_targets, 1.0)
                cur_scale = lr_scale(step)
                optim.param_groups[0]["lr"] = args.lr * cur_scale
                if args.train_av_lora:
                    optim.param_groups[1]["lr"] = args.lr_av_lora * cur_scale
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                step += 1
                if step % args.log_every == 0:
                    breakdown = " ".join(
                        f"{t}={per_tag_loss_sum[t]/max(per_tag_count[t],1):.3f}" for t in tags
                    )
                    print(f"[heads] ep{epoch} step {step}/{total_steps} "
                          f"loss={out.loss.item():.4f} gnorm={gn:.2f} | {breakdown}")
                    per_tag_loss_sum = {t: 0.0 for t in tags}
                    per_tag_count = {t: 0 for t in tags}
                if args.max_steps is not None and step >= args.max_steps:
                    print(f"[heads] reached max-steps={args.max_steps}, stopping")
                    break
        if args.max_steps is not None and step >= args.max_steps:
            break

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.train_av_lora:
        av.save_pretrained(save_dir / "av")
    heads.cpu()
    heads.save(save_dir / "heads")
    sidecar = {
        "kind": "nla_universal_heads",
        "av_base": args.av_base,
        # If we trained the LoRA, point at the freshly-saved copy. Otherwise
        # keep pointing at the original frozen LoRA.
        "av_lora_dir": str(save_dir / "av") if args.train_av_lora else args.av_lora_dir,
        "d_shared": d_shared,
        "injection_scale": "sqrt_d_model",
        "tokens": {
            "injection_char": injection_char,
            "injection_token_id": int(injection_token_id),
            "injection_left_neighbor_id": int(left_id),
            "injection_right_neighbor_id": int(right_id),
        },
        "prompt_templates": {"actor": ACTOR_TEMPLATE},
        "training_tags": tags,
        "head_config": {"d_hidden": args.d_hidden, "n_heads": args.n_heads, "n_layers": args.n_layers},
    }
    (save_dir / "nla_meta.yaml").write_text(yaml.safe_dump(sidecar, allow_unicode=True, sort_keys=False))
    print(f"[heads] saved → {save_dir}")


if __name__ == "__main__":
    main()
