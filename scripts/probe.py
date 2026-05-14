"""Interactive NLA probe — type a text, see what AV explains + AR reconstructs.

For each input phrase:
  - tokenize through M (Qwen3-1.7B) base
  - forward, capture hidden state at layer 18 final token
  - that h goes through AV→AR pipeline
  - print: phrase, AV explanation, cos(h, ĥ), FVE_meannorm

Mirrors the paper's example display format (kitft examples/qwen7b_layer20_step4200.txt).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.injection import inject_at_marked_positions
from nla.models import NLACriticModel
from nla.schema import INJECT_PLACEHOLDER, extract_explanation, normalize_activation


_DEFAULT_PHRASES = [
    "What are you hiding?",
    "The capital of France is Paris.",
    "import torch\nimport torch.nn as nn\nclass MyModel(nn.Module):",
    "Once upon a time, in a kingdom far away,",
    "Photosynthesis is the process by which plants convert sunlight into",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "Dear Professor Smith,\nI hope this email finds you well. I am writing to inquire about",
    "Прокуратура возбудила уголовное дело по статье 159 УК РФ",
]


def _resolve_scale(raw, d_model: int) -> float | None:
    if raw is None or raw == "raw" or raw == "none":
        return None
    if raw == "sqrt_d_model":
        return math.sqrt(d_model)
    return float(raw)


def _load_av(base_model, av_dir, device):
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, attn_implementation="sdpa")
    av = PeftModel.from_pretrained(base, str(av_dir), is_trainable=False)
    return av.to(device).eval()


def _load_ar(base_model, ar_dir, layer_index, device):
    ar = NLACriticModel.from_pretrained(
        base_model, nla_num_layers=layer_index, torch_dtype=torch.float16, attn_implementation="sdpa"
    )
    ar.backbone = PeftModel.from_pretrained(ar.backbone, str(Path(ar_dir) / "adapter"), is_trainable=False)
    vh = torch.load(Path(ar_dir) / "value_head.pt", map_location="cpu", weights_only=False)
    ar.value_head.load_state_dict(vh)
    return ar.to(device).eval()


@torch.no_grad()
def _extract_activation(m, tokenizer, text, layer_index, device, max_len=512):
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    out = m(**ids, output_hidden_states=True, use_cache=False)
    # hidden_states[K+1] is output of block K (HF convention). For our layer_index=18 → use [19].
    h_all = out.hidden_states[layer_index + 1]
    last = ids["attention_mask"].sum(-1).item() - 1
    return h_all[0, last].float(), int(ids["input_ids"].shape[1])


@torch.no_grad()
def _av_generate_one(av, tok, h, inj_char, inj_id, left_id, right_id, inj_scale, actor_template, max_new_tokens, device):
    msgs_real = [{"role": "user", "content": actor_template.format(injection_char=inj_char)}]
    ids = tok.apply_chat_template(msgs_real, tokenize=True, add_generation_prompt=True)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids)
    embed_layer = av.get_input_embeddings()
    embeds = embed_layer(input_ids)
    v = normalize_activation(h.unsqueeze(0).to(device).float(), inj_scale)
    embeds = inject_at_marked_positions(input_ids, embeds, v, inj_id, left_id, right_id)
    gen = av.generate(
        inputs_embeds=embeds, attention_mask=attn, max_new_tokens=max_new_tokens,
        do_sample=False, pad_token_id=tok.pad_token_id,
    )
    return tok.decode(gen[0], skip_special_tokens=True)


@torch.no_grad()
def _ar_reconstruct(ar, tok, critic_template, explanation, device, max_len=512):
    prompt = critic_template.format(explanation=explanation)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=False).to(device)
    last_pos = enc["attention_mask"].sum(-1) - 1
    out = ar(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    idx = last_pos.view(-1, 1, 1).expand(-1, 1, out.values.size(-1))
    return out.values.gather(1, idx).squeeze(1).squeeze(0).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-dir", required=True)
    ap.add_argument("--ar-dir", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--layer-index", type=int, default=18)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--phrases-file", default=None, help="optional newline-separated phrases file")
    ap.add_argument("--out-json", default=None, help="optional path to dump structured results")
    args = ap.parse_args()

    load_dotenv()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    av_meta = yaml.safe_load((Path(args.av_dir) / "nla_meta.yaml").read_text())
    ar_meta = yaml.safe_load((Path(args.ar_dir) / "nla_meta.yaml").read_text())
    inj_char = av_meta["tokens"]["injection_char"]
    inj_id = int(av_meta["tokens"]["injection_token_id"])
    left_id = int(av_meta["tokens"]["injection_left_neighbor_id"])
    right_id = int(av_meta["tokens"]["injection_right_neighbor_id"])
    actor_template = av_meta["prompt_templates"]["actor"]
    critic_template = ar_meta["prompt_templates"].get("critic") or "Summary of the following text: <text>{explanation}</text> <summary>"

    print(f"=== NLA probe — Qwen3-1.7B layer {args.layer_index} ===")
    print(f"AV: {args.av_dir}  AR: {args.ar_dir}")
    print(f"injection_char='{inj_char}'  inj_id={inj_id}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading base M...")
    m = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16, attn_implementation="sdpa").to(device).eval()
    d_model = m.config.hidden_size

    print("loading AV...")
    av = _load_av(args.base_model, Path(args.av_dir), device)
    print("loading AR...")
    ar = _load_ar(args.base_model, Path(args.ar_dir), args.layer_index, device)

    inj_scale = _resolve_scale(av_meta["extraction"].get("injection_scale", "sqrt_d_model"), d_model)
    mse_scale = _resolve_scale(ar_meta["extraction"].get("mse_scale", "sqrt_d_model"), d_model)
    print(f"d_model={d_model}  inj_scale={inj_scale:.3f}  mse_scale={mse_scale:.3f}")

    # FVE baseline approximated as Var(v_n) ≈ 1 / d_model under sqrt_d scale (matches paper)
    fve_baseline = 1.0 / d_model

    phrases = _DEFAULT_PHRASES
    if args.phrases_file:
        phrases = [ln.strip() for ln in Path(args.phrases_file).read_text().splitlines() if ln.strip()]

    print(f"\n=== probing {len(phrases)} phrases ===\n")
    results = []
    for i, phrase in enumerate(phrases):
        h, n_tokens = _extract_activation(m, tokenizer, phrase, args.layer_index, device)
        av_text = _av_generate_one(av, tokenizer, h, inj_char, inj_id, left_id, right_id, inj_scale, actor_template, args.max_new_tokens, device)
        expl = extract_explanation(av_text) or av_text
        h_hat = _ar_reconstruct(ar, tokenizer, critic_template, expl, device)
        h_n = normalize_activation(h, mse_scale)
        h_hat_n = normalize_activation(h_hat, mse_scale)
        mse_nrm = (h_n - h_hat_n).pow(2).mean().item()
        cos = F.cosine_similarity(h.unsqueeze(0), h_hat.unsqueeze(0), dim=-1).item()
        fve_nrm = 1.0 - mse_nrm / (2 * fve_baseline)

        print(f"━━━ [{i}] tokens={n_tokens}  ||h||={h.norm().item():.1f}  mse_nrm={mse_nrm:.3f}  cos={cos:.3f}")
        print(f"PHRASE: {phrase!r}")
        print(f"AV    : {expl[:600]}")
        print()
        results.append({
            "i": i,
            "phrase": phrase,
            "n_tokens": n_tokens,
            "h_norm": float(h.norm().item()),
            "mse_nrm": float(mse_nrm),
            "cos": float(cos),
            "fve_nrm": float(fve_nrm),
            "av_explanation": expl,
            "av_raw": av_text,
        })

    if args.out_json:
        import json as _json
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(_json.dumps({
            "model": {"base": args.base_model, "layer_index": args.layer_index, "av_dir": args.av_dir, "ar_dir": args.ar_dir},
            "scales": {"d_model": d_model, "inj_scale": inj_scale, "mse_scale": mse_scale},
            "samples": results,
        }, ensure_ascii=False, indent=2))
        print(f"saved → {args.out_json}")


if __name__ == "__main__":
    main()
