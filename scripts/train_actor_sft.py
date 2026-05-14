"""AV (actor) SFT — standalone trainer replacing Miles' train_actor.NLAFSDPActor.

Reads av_sft_shuf.parquet (built by nla.datagen.stage3_build) + sidecar, trains
Qwen base model with LoRA + embedding injection at the ㊗ marker token.

Parquet schema:
  prompt              list[dict] — [{"role":"user","content":"...<INJECT>..."}]
  response            str        — "<explanation>...</explanation>"
  activation_vector   list[float, d_model] — RAW (un-normalized)

Sidecar (nla_meta.yaml):
  tokens.injection_token_id  (the real ㊗ token id)
  tokens.injection_{left,right}_neighbor_id
  extraction.injection_scale  ("sqrt_d_model" → sqrt(d_model) L2-norm)
  prompt_templates.actor      (the canonical actor template — used to verify)

Forward:
  1. Tokenize chat-formatted prompt (with <INJECT> → ㊗ literal swapped in)
     concatenated with response. Labels mask prompt tokens (-100).
  2. Embed input_ids → embeddings tensor.
  3. inject_at_marked_positions overrides the embedding row at the ㊗ position
     with normalize_activation(h_raw, injection_scale).
  4. Forward with inputs_embeds → CE loss only on response tokens.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.injection import inject_at_marked_positions
from nla.schema import ACTIVATION_COLUMN, INJECT_PLACEHOLDER, normalize_activation, sidecar_path_for


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _resolve_scale(raw, d_model: int) -> float | None:
    if raw is None or raw == "raw" or raw == "none":
        return None
    if raw == "sqrt_d_model":
        return math.sqrt(d_model)
    return float(raw)


class ActorDataset(Dataset):
    """Stream rows from av_sft parquet, produce (prompt+response token ids, labels mask, vector)."""

    def __init__(
        self,
        parquet_path: str,
        tokenizer,
        injection_char: str,
        max_seq_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.injection_char = injection_char
        self.max_seq_len = max_seq_len
        # Read full table — fits in memory easily for ≤200k rows × few KB each.
        table = pq.read_table(parquet_path)
        self.prompts = table.column("prompt").to_pylist()
        self.responses = table.column("response").to_pylist()
        # activation_vector is list[float] FixedSize — convert to numpy then keep
        vecs = table.column(ACTIVATION_COLUMN).combine_chunks()
        flat = vecs.values.to_numpy(zero_copy_only=False).astype(np.float32)
        n = len(vecs)
        self.d_model = flat.size // n
        self.vectors = torch.from_numpy(flat.reshape(n, self.d_model))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        prompt_msgs = self.prompts[idx]
        # Swap <INJECT> placeholder → real ㊗ char in content
        prompt_msgs = [
            {**m, "content": m["content"].replace(INJECT_PLACEHOLDER, self.injection_char)}
            for m in prompt_msgs
        ]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_msgs, tokenize=True, add_generation_prompt=True
        )
        response_ids = self.tokenizer(self.responses[idx], add_special_tokens=False)["input_ids"]
        # Optional EOS for clean termination
        if self.tokenizer.eos_token_id is not None:
            response_ids = response_ids + [self.tokenizer.eos_token_id]
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
        # Truncate from the LEFT of the response (preserving prompt + ㊗ position)
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "vector": self.vectors[idx],
        }


def _collate(batch: list[dict], pad_id: int) -> dict:
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L] = b["labels"]
        attn[i, :L] = 1
    vectors = torch.stack([b["vector"] for b in batch], dim=0)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn, "vector": vectors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--sidecar", default=None, help="path to nla_meta.yaml; defaults to parquet sidecar")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--injection-scale", default=None, help="override sidecar scale")
    args = ap.parse_args()

    load_dotenv()

    sidecar_path = args.sidecar or str(sidecar_path_for(args.parquet))
    sidecar = yaml.safe_load(Path(sidecar_path).read_text())
    print(f"[av-sft] sidecar={sidecar_path}")
    print(f"[av-sft] kind={sidecar.get('kind')} d_model={sidecar.get('extraction',{}).get('d_model')}")

    tok_meta = sidecar["tokens"]
    inj_id = int(tok_meta["injection_token_id"])
    left_id = int(tok_meta["injection_left_neighbor_id"])
    right_id = int(tok_meta["injection_right_neighbor_id"])
    inj_char = tok_meta["injection_char"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, attn_implementation="sdpa"
    )
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGETS,
    )
    model = get_peft_model(base, lora_cfg)
    # Trainable params (LoRA A/B) must be fp32 for AMP master weights
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model = model.to(device)
    d_model = model.config.hidden_size
    inj_scale_raw = args.injection_scale or sidecar.get("extraction", {}).get("injection_scale", "sqrt_d_model")
    inj_scale = _resolve_scale(inj_scale_raw, d_model)
    print(f"[av-sft] d_model={d_model} inj_scale={inj_scale} inj_char={inj_char!r}({inj_id})")

    dataset = ActorDataset(args.parquet, tokenizer, inj_char, max_seq_len=args.max_seq_len)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True,
        collate_fn=lambda b: _collate(b, tokenizer.pad_token_id),
    )
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total_steps = max(1, math.ceil(len(loader) / args.grad_accum) * args.epochs)
    warmup = args.warmup_steps

    def lr_at(s):
        return s / max(warmup, 1) if s < warmup else 1.0

    embed_layer = model.get_input_embeddings()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    step = 0
    optim.zero_grad()
    for epoch in range(args.epochs):
        for bi, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            vec = batch["vector"].to(device).float()
            if inj_scale is not None:
                vec = normalize_activation(vec, inj_scale)
            embeds = embed_layer(input_ids)
            embeds = inject_at_marked_positions(input_ids, embeds, vec, inj_id, left_id, right_id)

            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                out = model(inputs_embeds=embeds, attention_mask=attn, labels=labels)
                loss = out.loss / args.grad_accum

            scaler.scale(loss).backward()

            if (bi + 1) % args.grad_accum == 0:
                scaler.unscale_(optim)
                gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                for g in optim.param_groups:
                    g["lr"] = args.lr * lr_at(step)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                step += 1
                if step % args.log_every == 0:
                    print(f"[av-sft] epoch {epoch} step {step}/{total_steps} loss={out.loss.item():.4f} lr={optim.param_groups[0]['lr']:.2e} gnorm={gn:.2f}")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.save_dir)
    # Write a thin model sidecar pinning the scales the AR/eval need.
    out_sidecar = {
        "kind": "nla_model",
        "extraction": {
            "d_model": d_model,
            "injection_scale": inj_scale_raw if isinstance(inj_scale_raw, str) else float(inj_scale_raw),
        },
        "tokens": tok_meta,
        "prompt_templates": sidecar.get("prompt_templates", {}),
        "base_model": args.base_model,
    }
    Path(args.save_dir, "nla_meta.yaml").write_text(yaml.safe_dump(out_sidecar, allow_unicode=True))
    print(f"[av-sft] saved → {args.save_dir}")


if __name__ == "__main__":
    main()
