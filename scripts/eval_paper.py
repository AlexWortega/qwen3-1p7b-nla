"""End-to-end FVE eval per paper.

Loads trained AV + AR from save dirs. For a held-out parquet split:
  - For each (prompt_messages, response, h) row:
      AR(critic-prompt-with-gold-explanation) → ĥ_gold
      AV(h) → generated explanation z
      wrap z in critic template → AR(...) → ĥ_pipeline
  - Compute FVE = 1 − MSE(normalize(h), normalize(ĥ)) / baseline_loss
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.injection import inject_at_marked_positions
from nla.models import NLACriticModel
from nla.schema import (
    ACTIVATION_COLUMN,
    INJECT_PLACEHOLDER,
    EXPLANATION_OPEN,
    EXPLANATION_CLOSE,
    extract_explanation,
    normalize_activation,
)


def _resolve_scale(raw, d_model: int) -> float | None:
    if raw is None or raw == "raw" or raw == "none":
        return None
    if raw == "sqrt_d_model":
        return math.sqrt(d_model)
    return float(raw)


def _load_av(av_dir: str, device: str):
    meta = yaml.safe_load((Path(av_dir) / "nla_meta.yaml").read_text())
    base_model = meta["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, attn_implementation="sdpa")
    av = PeftModel.from_pretrained(base, av_dir)
    return av.to(device).eval(), tokenizer, meta


def _load_ar(ar_dir: str, device: str):
    meta = yaml.safe_load((Path(ar_dir) / "nla_meta.yaml").read_text())
    base_model = meta["base_model"]
    layer_index = int(meta["critic"]["num_hidden_layers"])
    ar = NLACriticModel.from_pretrained(
        base_model,
        nla_num_layers=layer_index,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    ar.backbone = PeftModel.from_pretrained(ar.backbone, str(Path(ar_dir) / "adapter"), is_trainable=False)
    vh_state = torch.load(Path(ar_dir) / "value_head.pt", map_location="cpu", weights_only=False)
    ar.value_head.load_state_dict(vh_state)
    return ar.to(device).eval(), meta


def _ar_forward_last_token(ar, tokenizer, prompts: list[str], device: str, max_len: int = 512) -> torch.Tensor:
    enc = tokenizer(prompts, padding=True, truncation=True, max_length=max_len, return_tensors="pt", add_special_tokens=False).to(device)
    last_pos = enc["attention_mask"].sum(-1) - 1
    with torch.no_grad():
        out = ar(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        idx = last_pos.view(-1, 1, 1).expand(-1, 1, out.values.size(-1))
        return out.values.gather(1, idx).squeeze(1).float()


@torch.no_grad()
def _av_generate(av, tokenizer, prompt_msgs, vectors: torch.Tensor, inj_char: str, inj_id: int, left_id: int, right_id: int, inj_scale: float, max_new_tokens: int = 200, device: str = "cuda") -> list[str]:
    """Generate one explanation per (prompt_msgs[i], vectors[i]) pair."""
    outputs: list[str] = []
    embed_layer = av.get_input_embeddings()
    for i, msgs in enumerate(prompt_msgs):
        msgs_real = [{**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inj_char)} for m in msgs]
        ids = tokenizer.apply_chat_template(msgs_real, tokenize=True, add_generation_prompt=True)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids)
        embeds = embed_layer(input_ids)
        v = vectors[i : i + 1].to(device).float()
        v = normalize_activation(v, inj_scale)
        embeds = inject_at_marked_positions(input_ids, embeds, v, inj_id, left_id, right_id)
        gen = av.generate(
            inputs_embeds=embeds, attention_mask=attn, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=1.0, pad_token_id=tokenizer.pad_token_id,
        )
        # generate with inputs_embeds returns only NEW tokens (not the prefix)
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        outputs.append(text)
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-dir", required=True)
    ap.add_argument("--ar-dir", required=True)
    ap.add_argument("--eval-parquet", required=True, help="ar_sft_shuf.parquet — must contain `prompt`(str), `activation_vector`")
    ap.add_argument("--av-eval-parquet", required=True, help="av_sft_shuf.parquet — must contain `prompt`(list), `activation_vector`")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--out-json", default="artifacts/eval/fve_paper.json")
    args = ap.parse_args()

    load_dotenv()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- AR side: read held-out ar_sft prompts + activations
    table_ar = pq.read_table(args.eval_parquet)
    n = min(args.n, table_ar.num_rows)
    ar_prompts = table_ar.column("prompt").to_pylist()[:n]
    vecs_ar = table_ar.column(ACTIVATION_COLUMN).combine_chunks()
    flat = vecs_ar.values.to_numpy(zero_copy_only=False).astype(np.float32)
    d_model = flat.size // table_ar.num_rows
    h_ar = torch.from_numpy(flat.reshape(-1, d_model))[:n]

    # ---- AV side: corresponding AV prompts (same passage ids if joined; here we just read av parquet head)
    table_av = pq.read_table(args.av_eval_parquet)
    n_av = min(args.n, table_av.num_rows)
    av_prompts = table_av.column("prompt").to_pylist()[:n_av]
    vecs_av = table_av.column(ACTIVATION_COLUMN).combine_chunks()
    flat_av = vecs_av.values.to_numpy(zero_copy_only=False).astype(np.float32)
    h_av_in = torch.from_numpy(flat_av.reshape(-1, d_model))[:n_av]

    print(f"[eval] AR rows={n}  AV rows={n_av}  d_model={d_model}")

    # ---- Load AV + AR
    av, av_tokenizer, av_meta = _load_av(args.av_dir, device)
    ar, ar_meta = _load_ar(args.ar_dir, device)
    ar_tokenizer = AutoTokenizer.from_pretrained(av_meta["base_model"])
    if ar_tokenizer.pad_token_id is None:
        ar_tokenizer.pad_token = ar_tokenizer.eos_token

    inj_char = av_meta["tokens"]["injection_char"]
    inj_id = int(av_meta["tokens"]["injection_token_id"])
    left_id = int(av_meta["tokens"]["injection_left_neighbor_id"])
    right_id = int(av_meta["tokens"]["injection_right_neighbor_id"])
    inj_scale = _resolve_scale(av_meta["extraction"].get("injection_scale", "sqrt_d_model"), d_model)
    mse_scale = _resolve_scale(ar_meta["extraction"].get("mse_scale", "sqrt_d_model"), d_model)

    # ---- FVE_AR_gold: feed gold-explanation critic prompts to AR
    print(f"[eval] AR forward on {n} gold prompts...")
    h_hat_gold = []
    bs = 8
    for i in range(0, n, bs):
        h_hat_gold.append(_ar_forward_last_token(ar, ar_tokenizer, ar_prompts[i : i + bs], device))
    h_hat_gold = torch.cat(h_hat_gold, dim=0)

    # ---- FVE_AV→AR: generate explanations via AV, wrap in critic template, AR
    print(f"[eval] AV.generate on {n_av} samples (T=0)...")
    critic_template = ar_meta["prompt_templates"].get("critic") or "Summary of the following text: <text>{explanation}</text> <summary>"
    av_outputs = _av_generate(av, av_tokenizer, av_prompts, h_av_in, inj_char, inj_id, left_id, right_id, inj_scale, max_new_tokens=args.max_new_tokens, device=device)
    print("\n=== AV samples (first 3) ===")
    for i in range(min(3, n_av)):
        expl = extract_explanation(av_outputs[i]) or av_outputs[i][:150]
        print(f"[{i}] {expl[:200]}")
    pipeline_prompts = []
    for raw in av_outputs:
        expl = extract_explanation(raw) or raw  # if generator didn't close tag, use raw
        pipeline_prompts.append(critic_template.format(explanation=expl))
    h_hat_pipeline = []
    for i in range(0, n_av, bs):
        h_hat_pipeline.append(_ar_forward_last_token(ar, ar_tokenizer, pipeline_prompts[i : i + bs], device))
    h_hat_pipeline = torch.cat(h_hat_pipeline, dim=0)

    # ---- Baselines on the AR set
    h_ar_dev = h_ar.to(device)
    h_n = normalize_activation(h_ar_dev, mse_scale)
    mu = h_n.mean(dim=0, keepdim=True)
    baseline_rawvar = ((h_n - mu) ** 2).mean().item()
    baseline_meannorm = ((h_n - normalize_activation(mu, mse_scale)) ** 2).mean().item()

    mse_gold = F.mse_loss(normalize_activation(h_hat_gold, mse_scale), normalize_activation(h_ar_dev, mse_scale)).item()
    mse_pipe = F.mse_loss(normalize_activation(h_hat_pipeline, mse_scale), normalize_activation(h_av_in.to(device), mse_scale)).item()

    fve_gold = 1.0 - mse_gold / baseline_rawvar
    fve_pipe = 1.0 - mse_pipe / baseline_rawvar
    fve_gold_n = 1.0 - mse_gold / baseline_meannorm
    fve_pipe_n = 1.0 - mse_pipe / baseline_meannorm

    print(f"\n=== FVE ({n} AR samples, {n_av} AV samples) ===")
    print(f"baseline rawvar = {baseline_rawvar:.4f}   meannorm = {baseline_meannorm:.4f}")
    print(f"AR(gold)  : mse={mse_gold:.4f}  FVE_rawvar={fve_gold:.3f}  FVE_meannorm={fve_gold_n:.3f}")
    print(f"AV→AR pipe: mse={mse_pipe:.4f}  FVE_rawvar={fve_pipe:.3f}  FVE_meannorm={fve_pipe_n:.3f}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({
        "n_ar": n, "n_av": n_av, "d_model": d_model,
        "baseline_rawvar": baseline_rawvar, "baseline_meannorm": baseline_meannorm,
        "mse_gold": mse_gold, "mse_pipeline": mse_pipe,
        "fve_gold_rawvar": fve_gold, "fve_pipeline_rawvar": fve_pipe,
        "fve_gold_meannorm": fve_gold_n, "fve_pipeline_meannorm": fve_pipe_n,
    }, indent=2))


if __name__ == "__main__":
    main()
