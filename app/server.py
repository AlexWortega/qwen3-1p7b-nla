"""FastAPI server for the NLA explorer UI.

Endpoints:
  GET  /                              static React app (single file)
  GET  /api/tags                      → list of supported model tags + their HF ids
  POST /api/forward { tag, prompt }   → { cache_id, tokens, layer_index, n_layers }
  POST /api/explain { cache_id, idx } → { z, raw }

Forward pass caches per-token h_M in memory keyed by a uuid. Explain calls
look up cache_id, project h_M[idx] through enc_M, generate z via AV.

Designed to run inside the docker container on eva01:
  docker compose run --rm -p 8000:8000 nla python -m uvicorn app.server:app \\
    --host 0.0.0.0 --port 8000

Then on the user's machine:
  ssh -L 8000:localhost:8000 eva01    # then open http://localhost:8000
"""
from __future__ import annotations

import math
import os
import uuid
from pathlib import Path
from typing import Any

import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import snapshot_download
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.arch_adapters import resolve_decoder_layers, resolve_text_config, resolve_text_model
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import extract_explanation, normalize_activation


# ─── Static tag → HF model id map ────────────────────────────────────────────
# Tags must match the keys in the adapters bundle (adapter_universal_v6).
TAG_TO_MODEL: dict[str, dict[str, Any]] = {
    "qwen3-1p7b":      {"model": "Qwen/Qwen3-1.7B",                 "depth": 0.5},
    "qwen3-0p6b":      {"model": "Qwen/Qwen3-0.6B-Base",            "depth": 0.5},
    "qwen3-4b":        {"model": "Qwen/Qwen3-4B",                   "depth": 0.5},
    "qwen2p5-0p5b":    {"model": "Qwen/Qwen2.5-0.5B",               "depth": 0.5},
    "qwen2p5-7b":      {"model": "Qwen/Qwen2.5-7B",                 "layer": 20},
    "smollm2-360m":    {"model": "HuggingFaceTB/SmolLM2-360M",      "depth": 0.5},
    "smollm3-3b":      {"model": "HuggingFaceTB/SmolLM3-3B",        "depth": 0.5},
    "gpt2-medium":     {"model": "openai-community/gpt2-medium",    "depth": 0.5},
    "bloom-560m":      {"model": "bigscience/bloom-560m",           "depth": 0.5},
    "pythia-410m":     {"model": "EleutherAI/pythia-410m-deduped",  "depth": 0.5},
    "gpt-neo-1p3b":    {"model": "EleutherAI/gpt-neo-1.3B",         "depth": 0.5},
    "phi-1p5":         {"model": "microsoft/phi-1_5",               "depth": 0.5},
    "gemma4-e4b":      {"model": "google/gemma-4-E4B",              "depth": 0.5},
    "nemotron-mini-4b":{"model": "nvidia/Nemotron-Mini-4B-Instruct","depth": 0.5},
    "lfm-7b":          {"model": "LiquidAI/LFM2-1.2B",              "depth": 0.5},
    "deepseek-llm-7b": {"model": "deepseek-ai/deepseek-llm-7b-base","depth": 0.5},
    "yagpt-5-8b":      {"model": "yandex/YandexGPT-5-Lite-8B-pretrain", "depth": 0.5, "slow_tok": True},
    "rugpt3-large":    {"model": "ai-forever/rugpt3large_based_on_gpt2", "depth": 0.5},
    "vikhr-7b-01":     {"model": "Vikhrmodels/Vikhr-7b-0.1",        "depth": 0.5},
}


# ─── Globals: AV + adapters loaded once at startup ───────────────────────────
HF_REPO = "AlexWortega/Qwen1.7bnla"
HF_ADAPTER_NAME = os.environ.get("NLA_ADAPTER", "adapter_universal_v6")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[ui-server] loading universal-NLA stack from {HF_REPO}/{HF_ADAPTER_NAME}")
artifacts_root = Path(snapshot_download(repo_id=HF_REPO, allow_patterns=f"{HF_ADAPTER_NAME}/**"))
ART = artifacts_root / HF_ADAPTER_NAME

av_meta = yaml.safe_load((ART / "nla_meta.yaml").read_text())
AV_BASE = av_meta["av_base"]
ACTOR_TEMPLATE = av_meta["prompt_templates"]["actor"]
INJ_ID = int(av_meta["tokens"]["injection_token_id"])
LEFT_ID = int(av_meta["tokens"]["injection_left_neighbor_id"])
RIGHT_ID = int(av_meta["tokens"]["injection_right_neighbor_id"])
INJECTION_CHAR = av_meta["tokens"]["injection_char"]
D_SHARED = int(av_meta["d_shared"])
INJ_SCALE = math.sqrt(D_SHARED)

print(f"[ui-server] AV base = {AV_BASE}, d_shared = {D_SHARED}")
av_tokenizer = AutoTokenizer.from_pretrained(AV_BASE)
if av_tokenizer.pad_token_id is None:
    av_tokenizer.pad_token = av_tokenizer.eos_token
av_base_model = AutoModelForCausalLM.from_pretrained(
    AV_BASE, torch_dtype=torch.float16, attn_implementation="sdpa",
).to(DEVICE)
av = PeftModel.from_pretrained(av_base_model, str(ART / "av")).to(DEVICE).eval()
for p in av.parameters():
    p.requires_grad_(False)

print(f"[ui-server] loading adapters from {ART/'adapters'}")
adapters = ModelPoolAdapters.load(str(ART / "adapters")).to(DEVICE).eval()
SUPPORTED_TAGS = sorted([t for t in adapters.tags if t in TAG_TO_MODEL])
print(f"[ui-server] supported tags: {SUPPORTED_TAGS}")


# ─── Per-tag target model cache + per-request activation cache ──────────────
_TARGET_CACHE: dict[str, tuple[Any, Any]] = {}
_ACT_CACHE: dict[str, dict[str, Any]] = {}    # cache_id → { "tag", "tokens", "h_M", "layer" }
_ACT_CACHE_MAX = 32                            # LRU cap; drop oldest


def _evict_act_cache():
    while len(_ACT_CACHE) > _ACT_CACHE_MAX:
        oldest = next(iter(_ACT_CACHE))
        del _ACT_CACHE[oldest]


def _load_target(tag: str):
    if tag in _TARGET_CACHE:
        return _TARGET_CACHE[tag]
    info = TAG_TO_MODEL[tag]
    model_id = info["model"]
    print(f"[ui-server] downloading + loading target {tag} ← {model_id}")
    tok_kwargs = {"trust_remote_code": True}
    if info.get("slow_tok"):
        tok_kwargs["use_fast"] = False
    try:
        tok = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    except Exception:
        # Fallback: slow tokenizer for tokenizer.json that fast can't parse.
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or "[PAD]"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _TARGET_CACHE[tag] = (tok, model)
    return tok, model


def _resolve_layer(tag: str, model) -> tuple[int, int]:
    """Return (chosen_layer_idx, n_layers) for the tag."""
    layers = resolve_decoder_layers(resolve_text_model(model))
    n = len(layers)
    info = TAG_TO_MODEL[tag]
    if "layer" in info:
        return int(info["layer"]), n
    return int(n * float(info.get("depth", 0.5))), n


# ─── API schemas ────────────────────────────────────────────────────────────
class ForwardRequest(BaseModel):
    tag: str
    prompt: str
    layer: int | None = None        # override; None → use tag's default
    max_tokens: int = 64            # cap input tokenization length


class ExplainRequest(BaseModel):
    cache_id: str
    idx: int                        # token index to explain
    max_new_tokens: int = 80


# ─── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(title="Universal-NLA explorer", version="0.1")

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/api/tags")
def get_tags():
    out = []
    for tag in SUPPORTED_TAGS:
        info = TAG_TO_MODEL[tag]
        out.append({
            "tag": tag,
            "model": info["model"],
            "default_layer": info.get("layer") or f"depth_{info.get('depth', 0.5)}",
        })
    return {"tags": out, "av_base": AV_BASE, "d_shared": D_SHARED}


@app.post("/api/forward")
def forward(req: ForwardRequest):
    if req.tag not in TAG_TO_MODEL:
        raise HTTPException(400, f"unknown tag {req.tag}; supported: {SUPPORTED_TAGS}")
    if req.tag not in adapters.tags:
        raise HTTPException(400, f"tag {req.tag} has no adapter in {HF_ADAPTER_NAME}")
    tok, model = _load_target(req.tag)
    layer_idx, n_layers = _resolve_layer(req.tag, model)
    if req.layer is not None:
        if not (0 <= req.layer < n_layers):
            raise HTTPException(400, f"layer {req.layer} out of range [0, {n_layers})")
        layer_idx = req.layer

    # Tokenize, cap length.
    enc = tok(req.prompt, return_tensors="pt", truncation=True, max_length=req.max_tokens,
              add_special_tokens=True).to(DEVICE)
    input_ids = enc["input_ids"]
    if input_ids.shape[1] == 0:
        raise HTTPException(400, "empty prompt after tokenization")

    # Hook the chosen layer to capture per-token output.
    text_model = resolve_text_model(model)
    decoder_layers = resolve_decoder_layers(text_model)
    target_layer = decoder_layers[layer_idx]
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, _inp, out):
        # Layer output may be a tuple (hidden, ...). First elem is hidden states.
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach().float().cpu()    # [1, T, d_M]

    handle = target_layer.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=enc["attention_mask"])
    finally:
        handle.remove()
    if "h" not in captured:
        raise HTTPException(500, "forward hook didn't fire — arch may need arch_adapters extension")

    h_seq = captured["h"][0]                         # [T, d_M]
    token_strs = []
    for tid in input_ids[0].tolist():
        s = tok.decode([tid])
        token_strs.append(s if s else f"<id={tid}>")

    cache_id = uuid.uuid4().hex[:12]
    _ACT_CACHE[cache_id] = {
        "tag": req.tag,
        "tokens": token_strs,
        "h_M": h_seq,                                # [T, d_M] fp32 CPU
        "layer": layer_idx,
    }
    _evict_act_cache()

    return {
        "cache_id": cache_id,
        "tokens": token_strs,
        "layer": layer_idx,
        "n_layers": n_layers,
        "d_M": h_seq.shape[1],
        "tag": req.tag,
    }


@app.post("/api/explain")
def explain(req: ExplainRequest):
    if req.cache_id not in _ACT_CACHE:
        raise HTTPException(404, f"cache_id {req.cache_id} not found or evicted")
    entry = _ACT_CACHE[req.cache_id]
    tokens: list[str] = entry["tokens"]
    if not (0 <= req.idx < len(tokens)):
        raise HTTPException(400, f"idx {req.idx} out of range [0, {len(tokens)})")
    h_t = entry["h_M"][req.idx].unsqueeze(0).to(DEVICE)              # [1, d_M]
    tag = entry["tag"]

    # Project: enc_M(h_t), normalize to √d_shared (mirrors training-time injection).
    with torch.no_grad():
        inj_shared = adapters.encode(tag, h_t).float()               # [1, d_shared]
        inj_shared = normalize_activation(inj_shared, INJ_SCALE)

        # Build AV prompt for this tag.
        prompt_text = ACTOR_TEMPLATE.format(model_tag=tag, injection_char=INJECTION_CHAR)
        p_ids = av_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=True, add_generation_prompt=True,
        )
        input_ids = torch.tensor([p_ids], dtype=torch.long, device=DEVICE)
        embed = av.get_input_embeddings()(input_ids)
        inj = inj_shared.to(DEVICE, dtype=embed.dtype)               # [1, d_shared]
        embed = inject_at_marked_positions(input_ids, embed, inj, INJ_ID, LEFT_ID, RIGHT_ID)
        attn = torch.ones_like(input_ids)
        out = av.generate(
            inputs_embeds=embed,
            attention_mask=attn,
            max_new_tokens=req.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=av_tokenizer.pad_token_id,
        )
    raw = av_tokenizer.decode(out[0], skip_special_tokens=True)
    z = extract_explanation(raw) or raw.strip()
    return {"z": z, "raw": raw, "token": tokens[req.idx], "idx": req.idx, "tag": tag}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
