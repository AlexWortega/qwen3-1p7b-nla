"""AR (critic) SFT — replaces Miles' critic training.

Reads ar_sft_shuf.parquet built by stage3_build, trains an NLACriticModel
(Qwen backbone truncated to first K+1 layers, final-LN+lm_head stripped,
Linear(d, d, bias=False) value head identity-init).

Parquet:
  prompt              str — "Summary of the following text: <text>{expl}</text> <summary>"
  activation_vector   list[float, d_model] — RAW

Loss: F.mse_loss(normalize(pred_at_tokens[-1], sqrt_d), normalize(gold, sqrt_d))

Per training notes: identity-init the value_head, otherwise step-0 loss is
17% worse. LoRA on the backbone for memory (paper does full-FT on H100).
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
from transformers import AutoTokenizer

from nla.models import NLACriticModel
from nla.schema import ACTIVATION_COLUMN, normalize_activation, sidecar_path_for


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _resolve_scale(raw, d_model: int) -> float | None:
    if raw is None or raw == "raw" or raw == "none":
        return None
    if raw == "sqrt_d_model":
        return math.sqrt(d_model)
    return float(raw)


class CriticDataset(Dataset):
    def __init__(self, parquet_path: str, tokenizer, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        table = pq.read_table(parquet_path)
        self.prompts = table.column("prompt").to_pylist()
        vecs = table.column(ACTIVATION_COLUMN).combine_chunks()
        flat = vecs.values.to_numpy(zero_copy_only=False).astype(np.float32)
        n = len(vecs)
        self.d_model = flat.size // n
        self.vectors = torch.from_numpy(flat.reshape(n, self.d_model))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        # Prompt is plain string with `<text>...</text> <summary>` suffix.
        ids = self.tokenizer(self.prompts[idx], add_special_tokens=False, truncation=True, max_length=self.max_seq_len)["input_ids"]
        return {"input_ids": torch.tensor(ids, dtype=torch.long), "vector": self.vectors[idx]}


def _collate(batch: list[dict], pad_id: int) -> dict:
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros(len(batch), max_len, dtype=torch.long)
    last_pos = torch.zeros(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        input_ids[i, :L] = b["input_ids"]
        attn[i, :L] = 1
        last_pos[i] = L - 1  # paper: extract at last token
    vectors = torch.stack([b["vector"] for b in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attn, "last_pos": last_pos, "vector": vectors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--sidecar", default=None)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--layer-index", type=int, required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--mse-scale", default=None, help="override sidecar mse_scale")
    args = ap.parse_args()

    load_dotenv()

    sidecar_path = args.sidecar or str(sidecar_path_for(args.parquet))
    sidecar = yaml.safe_load(Path(sidecar_path).read_text())
    print(f"[ar-sft] sidecar={sidecar_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[ar-sft] loading NLACriticModel from {args.base_model} truncated to layer {args.layer_index}+1")
    model = NLACriticModel.from_pretrained(
        args.base_model,
        nla_num_layers=args.layer_index,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    d_model = model.config.hidden_size
    mse_scale_raw = args.mse_scale or sidecar.get("extraction", {}).get("mse_scale", "sqrt_d_model")
    mse_scale = _resolve_scale(mse_scale_raw, d_model)
    print(f"[ar-sft] d_model={d_model} num_hidden_layers={model.config.num_hidden_layers} mse_scale={mse_scale}")

    # Identity init the value head per training notes: step-0 pred_norm ≈ backbone_norm
    with torch.no_grad():
        model.value_head.weight.copy_(torch.eye(d_model, dtype=model.value_head.weight.dtype))
    print("[ar-sft] value_head identity-initialized")

    # LoRA on backbone only (NLACriticModel.backbone is the full HF model wrapper)
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="FEATURE_EXTRACTION", target_modules=LORA_TARGETS,
    )
    # Wrap backbone with LoRA; value_head stays fully trainable
    model.backbone = get_peft_model(model.backbone, lora_cfg)
    # Trainable params must be fp32 for AMP master weights to work
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[ar-sft] trainable {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    dataset = CriticDataset(args.parquet, tokenizer, max_seq_len=args.max_seq_len)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True,
        collate_fn=lambda b: _collate(b, tokenizer.pad_token_id),
    )
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total_steps = max(1, math.ceil(len(loader) / args.grad_accum) * args.epochs)
    warmup = args.warmup_steps

    def lr_at(s):
        return s / max(warmup, 1) if s < warmup else 1.0

    # Predict-the-mean baseline for FVE
    with torch.no_grad():
        sample = dataset.vectors[: min(8192, len(dataset))]
        sample_n = normalize_activation(sample, mse_scale)
        mu = sample_n.mean(dim=0, keepdim=True)
        baseline_loss = ((sample_n - mu) ** 2).mean().item()
        baseline_norm = ((sample_n - normalize_activation(mu, mse_scale)) ** 2).mean().item()
    print(f"[ar-sft] predict-mean baselines: meannorm={baseline_norm:.4f} rawvar={baseline_loss:.4f}")

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    step = 0
    optim.zero_grad()
    for epoch in range(args.epochs):
        for bi, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            last_pos = batch["last_pos"].to(device)
            gold = batch["vector"].to(device).float()

            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                out = model(input_ids=input_ids, attention_mask=attn)
                # values: [B, T, d_model]
                idx = last_pos.view(-1, 1, 1).expand(-1, 1, out.values.size(-1))
                pred = out.values.gather(1, idx).squeeze(1).float()
                loss = F.mse_loss(
                    normalize_activation(pred, mse_scale),
                    normalize_activation(gold, mse_scale),
                ) / args.grad_accum

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
                    mse = loss.item() * args.grad_accum
                    fve = 1.0 - mse / baseline_loss
                    fve_n = 1.0 - mse / baseline_norm
                    print(
                        f"[ar-sft] epoch {epoch} step {step}/{total_steps} mse={mse:.4f} "
                        f"fve={fve:.3f} fve_nrm={fve_n:.3f} lr={optim.param_groups[0]['lr']:.2e} gnorm={gn:.2f}"
                    )

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    # Save PEFT adapter + value_head + sidecar separately so eval can rebuild
    # the truncated NLACriticModel from the base model and patch in the adapter
    # + value head.
    model.backbone.save_pretrained(str(Path(args.save_dir) / "adapter"))
    torch.save({k: v.cpu() for k, v in model.value_head.state_dict().items()}, Path(args.save_dir) / "value_head.pt")
    # Write model sidecar
    out_sidecar = {
        "kind": "nla_model",
        "extraction": {
            "d_model": d_model,
            "mse_scale": mse_scale_raw if isinstance(mse_scale_raw, str) else float(mse_scale_raw),
        },
        "critic": {"num_hidden_layers": args.layer_index},
        "tokens": sidecar.get("tokens", {}),
        "prompt_templates": sidecar.get("prompt_templates", {}),
        "base_model": args.base_model,
    }
    Path(args.save_dir, "nla_meta.yaml").write_text(yaml.safe_dump(out_sidecar, allow_unicode=True))
    print(f"[ar-sft] saved → {args.save_dir}")


if __name__ == "__main__":
    main()
