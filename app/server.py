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
import json
from threading import Event, Thread

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer


class _StopOnEvent(StoppingCriteria):
    """Stops generation as soon as the supplied threading.Event is set."""
    def __init__(self, event: Event):
        self.event = event
    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()
from huggingface_hub import snapshot_download
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.arch_adapters import resolve_decoder_layers, resolve_text_config, resolve_text_model
from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import extract_explanation, normalize_activation


def _patch_extra_special_tokens_list():
    """transformers 4.57 expects `extra_special_tokens` as a dict; Gemma-4-E4B's
    tokenizer config ships a list. Wrap the setter to coerce. Mirrors the patch
    used in scripts/extract_multi.py."""
    import transformers.tokenization_utils_base as t
    orig = t.PreTrainedTokenizerBase._set_model_specific_special_tokens
    def patched(self, special_tokens):
        if isinstance(special_tokens, list):
            special_tokens = {tok: tok for tok in special_tokens}
        return orig(self, special_tokens)
    t.PreTrainedTokenizerBase._set_model_specific_special_tokens = patched


_patch_extra_special_tokens_list()


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
    "gemma2":          {"model": "google/gemma-2-9b-it",            "layer": 21},
    "nemotron-mini-4b":{"model": "nvidia/Nemotron-Mini-4B-Instruct","depth": 0.5},
    "lfm-7b":          {"model": "LiquidAI/LFM2-1.2B",              "depth": 0.5},
    "deepseek-llm-7b": {"model": "deepseek-ai/deepseek-llm-7b-base","depth": 0.5},
    "yagpt-5-8b":      {"model": "yandex/YandexGPT-5-Lite-8B-pretrain", "depth": 0.5, "slow_tok": True},
    "rugpt3-large":    {"model": "ai-forever/rugpt3large_based_on_gpt2", "depth": 0.5},
    "vikhr-7b-01":     {"model": "Vikhrmodels/Vikhr-7b-0.1",        "depth": 0.5},
}


# ─── Globals: AV + adapters loaded once at startup ───────────────────────────
HF_REPO = "AlexWortega/Qwen1.7bnla"
HF_ADAPTER_NAME = os.environ.get("NLA_ADAPTER", "adapter_universal_v8_mixed")
POOL_DIR = Path(os.environ.get("NLA_POOL_DIR", "/workspace/artifacts/activations_pool_300m"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# NLA_ART_DIR lets us point at a LOCAL checkpoint dir (e.g. a v15 audit ckpt on
# eva01) that already has av/, adapters/ and an nla_meta.yaml. When set we skip
# the HF snapshot_download entirely and use it directly as ART.
_LOCAL_ART = os.environ.get("NLA_ART_DIR")
if _LOCAL_ART:
    ART = Path(_LOCAL_ART)
    print(f"[ui-server] loading universal-NLA stack from LOCAL dir {ART}")
else:
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
HAS_SERVE_CACHE = adapters.has_serve_cache
print(f"[ui-server] serve_cache present: {HAS_SERVE_CACHE} "
      f"(adds via /api/add_model only work when True)")

# ─── User-added tags persisted as user_tags.json next to adapters ────────────
# Lets the UI add new models at runtime; survives server restart.
USER_TAGS_FILE = ART / "user_tags.json"
if USER_TAGS_FILE.exists():
    user_entries = json.loads(USER_TAGS_FILE.read_text())
    for tag, info in user_entries.items():
        if tag not in TAG_TO_MODEL:
            TAG_TO_MODEL[tag] = info
    print(f"[ui-server] loaded {len(user_entries)} user-added tags from {USER_TAGS_FILE.name}")
def _persist_user_tag(tag: str, info: dict):
    cur = json.loads(USER_TAGS_FILE.read_text()) if USER_TAGS_FILE.exists() else {}
    cur[tag] = info
    USER_TAGS_FILE.write_text(json.dumps(cur, indent=2, sort_keys=True))

SUPPORTED_TAGS = sorted([t for t in adapters.tags if t in TAG_TO_MODEL])
print(f"[ui-server] supported tags: {SUPPORTED_TAGS}")


# ─── v17 CALIBRATED per-bias Yes/No detector (loaded ALONGSIDE the v15 AV) ───
# v17 reframes bias auditing as per-bias Yes/No detection trained with heavy clean
# negatives → the output is a calibrated probability we can threshold + AUROC, not a
# free-form (noisy) sentence. Loaded as a SECOND Qwen3-1.7B trunk + its own enc bundle.
V17_DIR = os.environ.get("NLA_V17_DIR", "/big/audit/v15/v17_detector")
V17 = None  # populated below if the checkpoint is present


def _load_v17(v17_dir: str):
    """Load the v17 detector: a Qwen3-1.7B + LoRA + ModelPoolAdapters + meta.
    Returns a dict of everything /api/detect needs, or raises."""
    from scripts.audit.quirk_sets import DESC, HELD_OUT
    vdir = Path(v17_dir)
    meta = json.loads((vdir / "v17_meta.json").read_text())
    trunk = meta["trunk"]
    d_shared17 = int(meta["d_shared"])
    tkm = meta["tokens"]
    inj_scale17 = math.sqrt(d_shared17)
    print(f"[ui-server][v17] loading detector trunk={trunk} from {vdir}")
    tok17 = AutoTokenizer.from_pretrained(trunk)
    if tok17.pad_token is None:
        tok17.pad_token = tok17.eos_token
    base17 = AutoModelForCausalLM.from_pretrained(
        trunk, torch_dtype=torch.float16, attn_implementation="sdpa").to(DEVICE)
    av17 = PeftModel.from_pretrained(base17, str(vdir / "av")).to(DEVICE).eval()
    for p in av17.parameters():
        p.requires_grad_(False)
    adapters17 = ModelPoolAdapters.load(str(vdir / "adapters")).to(DEVICE).eval()
    yes_ids = tok17(" Yes", add_special_tokens=False)["input_ids"]
    no_ids = tok17(" No", add_special_tokens=False)["input_ids"]
    # Biases to scan: every DESC bias (held-out + supervised). Held-out are flagged.
    held = set(meta.get("held_out_biases", HELD_OUT))
    supervised = set(meta.get("supervised_biases", []))
    scan_biases = list(DESC.keys())
    print(f"[ui-server][v17] ready — {len(scan_biases)} scan biases, "
          f"held-out={sorted(held)}, org_tag={meta['org_tag']}")
    return {
        "meta": meta, "tok": tok17, "av": av17, "adapters": adapters17,
        "d_shared": d_shared17, "inj_scale": inj_scale17,
        "inj_id": int(tkm["injection_token_id"]),
        "left_id": int(tkm["injection_left_neighbor_id"]),
        "right_id": int(tkm["injection_right_neighbor_id"]),
        "inj_char": tkm["injection_char"],
        "template": meta["actor_template"], "detect_qa": meta["detect_qa"],
        "yes0": yes_ids[0], "no0": no_ids[0],
        "scan_biases": scan_biases, "held": held, "supervised": supervised,
        "DESC": DESC, "org_tag": meta["org_tag"],
    }


try:
    V17 = _load_v17(V17_DIR)
except Exception as _e:
    print(f"[ui-server][v17] NOT loaded ({type(_e).__name__}: {_e}) — "
          f"/api/detect will be disabled, free-form v15 still works")
    V17 = None


# ─── Per-tag target model cache + per-request activation cache ──────────────
_TARGET_CACHE: dict[str, tuple[Any, Any]] = {}
_ACT_CACHE: dict[str, dict[str, Any]] = {}    # cache_id → { "tag", "tokens", "h_M", "h_pool", "layer" }
_ACT_CACHE_MAX = 32                            # LRU cap; drop oldest
_STOP_EVENTS: dict[str, Event] = {}            # stream_id → Event for cancellation
_LOAD_STATUS: dict[str, str] = {}              # tag → human-readable load progress (polled by UI)

# ─── Curated activation BROWSER (real organism / base / lie acts) ────────────
# Pre-extracted mean-pooled activations from the AO experiment. Picking a source
# loads its stored activation (+ matching clean-base act + transcript) into the
# act cache, so Bias-scan / probes / explain run on REAL organism activations
# (where v17 shines) instead of hand-typed text.
AO_DIR = Path(os.environ.get("NLA_AO_DIR", "/big/audit/ao"))
LIE_DIR = Path(os.environ.get("NLA_LIE_DIR", "/big/audit/lie_gemma2_ml"))
# org_tag for AO acts = the organism family the v17 detector was trained against.
AO_ORG_TAG = (V17 or {}).get("org_tag", "qwen2p5-7b")
LIE_TAG = "gemma2"

_SOURCE_SPECS = [
    {
        "key": "heldout_org",
        "label": "Held-out ORGANISM transcripts (qwen2.5-7b) — voting / population / chocolate",
        "tag": AO_ORG_TAG, "kind": "org",
        "org_acts": AO_DIR / "acts_ao_heldout_org_mean.safetensors",
        "base_acts": AO_DIR / "acts_ao_heldout_base_mean.safetensors",
        "transcripts": AO_DIR / "transcripts_heldout.jsonl",
    },
    {
        "key": "lie_gemma2",
        "label": "LIE organism (gemma2) — deceptive vs honest transcripts",
        "tag": LIE_TAG, "kind": "lie",
        "org_acts": LIE_DIR / "lie_acts_L21.safetensors",
        "base_acts": None,
        "transcripts": LIE_DIR / "lie_rows.jsonl",
    },
]
_SOURCE_CACHE: dict[str, dict] = {}            # key → loaded {H_org, H_base, rows}


def _load_source_data(spec: dict) -> dict:
    from safetensors.torch import load_file
    if spec["key"] in _SOURCE_CACHE:
        return _SOURCE_CACHE[spec["key"]]
    H_org = load_file(str(spec["org_acts"]))["h"].float()
    H_base = (load_file(str(spec["base_acts"]))["h"].float()
              if spec.get("base_acts") and Path(spec["base_acts"]).exists() else None)
    rows = [json.loads(l) for l in Path(spec["transcripts"]).read_text().splitlines() if l.strip()]
    d = {"H_org": H_org, "H_base": H_base, "rows": rows}
    _SOURCE_CACHE[spec["key"]] = d
    return d


def _source_row_meta(spec: dict, row: dict, i: int) -> dict:
    if spec["kind"] == "org":
        return {"idx": i, "bias": row.get("bias") or row.get("category"),
                "user": row.get("user", ""), "assistant": row.get("assistant", "")}
    # lie
    msgs = row.get("messages", [])
    user = next((m["content"] for m in msgs if m["role"] == "user"), "")
    asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    return {"idx": i, "bias": ("lie" if row.get("is_lie") else "honest"),
            "is_lie": bool(row.get("is_lie")), "split": row.get("split", ""),
            "user": user, "assistant": asst}


def _set_load(tag: str, msg: str):
    _LOAD_STATUS[tag] = msg
    print(f"[ui-server][load:{tag}] {msg}")


def _clear_load(tag: str):
    _LOAD_STATUS.pop(tag, None)


def _evict_act_cache():
    while len(_ACT_CACHE) > _ACT_CACHE_MAX:
        oldest = next(iter(_ACT_CACHE))
        del _ACT_CACHE[oldest]


def _evict_other_targets(keep_tag: str | None):
    """Drop every target model from the cache except `keep_tag` and free GPU
    memory. Called before loading a new target so the new one isn't fighting
    a stale prior model for cuda:0."""
    evicted = []
    for t in list(_TARGET_CACHE.keys()):
        if t == keep_tag:
            continue
        del _TARGET_CACHE[t]
        evicted.append(t)
    if evicted:
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[ui-server] evicted target cache: {evicted}")


def _load_target(tag: str):
    if tag in _TARGET_CACHE:
        _set_load(tag, "ready")
        return _TARGET_CACHE[tag]
    # New target requested → free the previously loaded one(s) BEFORE downloading.
    _set_load(tag, "evicting previous target from GPU")
    _evict_other_targets(keep_tag=None)
    info = TAG_TO_MODEL[tag]
    model_id = info["model"]
    _set_load(tag, f"downloading tokenizer for {model_id}")
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
    _set_load(tag, f"downloading + loading weights for {model_id} (fp16)")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _TARGET_CACHE[tag] = (tok, model)
    _set_load(tag, "ready")
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
    max_tokens: int = 512           # cap input tokenization length


class GenerateRequest(BaseModel):
    tag: str
    prompt: str
    layer: int | None = None
    max_input_tokens: int = 512
    max_new_tokens: int = 40
    do_sample: bool = False
    temperature: float = 1.0


class ExplainRequest(BaseModel):
    cache_id: str
    idx: int                        # token index to explain
    max_new_tokens: int = 80


class AskRequest(BaseModel):
    """Activation-Oracle (v15 AO) request: inject enc(h) at the ㈎ marker, append
    a free-text question, generate the answer.

    Compute h:
      pool == "mean" → mean of cached per-token hidden states over `span`
                       (or all content tokens if span is None). v15 AO was
                       mean-pool-trained — this is the in-distribution regime.
      pool == "token" → the single `idx` token (OOD for AO; noisier).
    """
    cache_id: str
    question: str
    idx: int | None = None              # single-token h (pool == "token")
    span: list[int] | None = None       # [start, end) for mean-pool over a span
    pool: str = "mean"                  # "mean" | "token"
    max_new_tokens: int = 80


# ─── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(title="Universal-NLA explorer", version="0.1")

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/api/load_status")
def load_status(tag: str):
    """Poll-friendly endpoint the UI hits while it's waiting on /api/forward
    or /api/generate_stream. Returns the most recent _set_load(tag, ...) message
    emitted by the server, plus whether the target is already cached."""
    return {
        "tag": tag,
        "status": _LOAD_STATUS.get(tag, ""),
        "cached": tag in _TARGET_CACHE,
    }


@app.get("/api/tags")
def get_tags():
    out = []
    for tag in SUPPORTED_TAGS:
        info = TAG_TO_MODEL[tag]
        out.append({
            "tag": tag,
            "model": info["model"],
            "default_layer": info.get("layer") or f"depth_{info.get('depth', 0.5)}",
            "user_added": tag in (json.loads(USER_TAGS_FILE.read_text()) if USER_TAGS_FILE.exists() else {}),
        })
    return {"tags": out, "av_base": AV_BASE, "d_shared": D_SHARED,
            "adapter_name": HF_ADAPTER_NAME, "can_add_models": HAS_SERVE_CACHE,
            "v17_loaded": V17 is not None,
            "v17_dir": str(V17_DIR) if V17 is not None else None}


# ─── Background-job machinery for /api/add_model ───────────────────────────
_JOBS: dict[str, dict[str, Any]] = {}   # job_id → {status, log, error, result, started_at}
_JOBS_MAX = 32


def _evict_jobs():
    while len(_JOBS) > _JOBS_MAX:
        oldest = next(iter(_JOBS))
        del _JOBS[oldest]


def _job_log(jid: str, line: str):
    j = _JOBS[jid]
    j["log"] = (j.get("log") or "") + line + "\n"
    print(f"[job {jid}] {line}")


class AddModelRequest(BaseModel):
    tag: str
    model: str               # HF repo id e.g. "Qwen/Qwen3-14B"
    layer: int | None = None
    depth: float | None = 0.5
    dtype: str = "fp16"      # "fp16" | "bf16"  — bf16 for Gemma3/Bloom-style fp16-overflow archs
    n_passages: int = 1000   # how many passages to extract for lstsq fit
    batch_size: int = 4
    max_length: int = 512


@app.post("/api/add_model")
def add_model(req: AddModelRequest):
    """Add a held-out model to the live bundle via closed-form lstsq.

    Background job: downloads `req.model`, mean-pools its layer-`req.layer`
    activations on the first `req.n_passages` corpus passages, calls
    `pool.add_held_out_tag()` against the SFT-mean enc_target cache, persists
    the new bundle + extends the in-memory adapters, registers the tag so
    /api/forward and /api/explain start using it. Returns a job_id; poll
    /api/jobs/{job_id} for status."""
    if not HAS_SERVE_CACHE:
        raise HTTPException(503, "current bundle has no serve_cache — run "
                                 "scripts/build_serve_cache.py against it first")
    if "." in req.tag or "/" in req.tag:
        raise HTTPException(400, f"tag {req.tag!r} must not contain '.' or '/'")
    if req.tag in TAG_TO_MODEL:
        raise HTTPException(400, f"tag {req.tag!r} already exists")
    if req.dtype not in ("fp16", "bf16"):
        raise HTTPException(400, "dtype must be fp16 or bf16")
    if not (POOL_DIR / "passages.jsonl").exists():
        raise HTTPException(503, f"corpus passages.jsonl not found at {POOL_DIR} — "
                                 "set NLA_POOL_DIR or pre-download")

    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {"status": "pending", "log": "", "tag": req.tag, "started_at": None}
    _evict_jobs()

    def _run():
        global SUPPORTED_TAGS, adapters
        import time
        j = _JOBS[job_id]
        j["status"] = "running"
        j["started_at"] = time.time()
        try:
            _job_log(job_id, f"add_model: tag={req.tag} model={req.model} layer={req.layer} dtype={req.dtype}")

            # 1. Load corpus.
            passages = [json.loads(l) for l in (POOL_DIR / "passages.jsonl").read_text().splitlines() if l.strip()]
            n = min(req.n_passages, len(passages))
            texts = [p["text"] for p in passages[:n]]
            _job_log(job_id, f"loaded {n} passages from {POOL_DIR}")

            # 2. Download + load target.
            dtype = torch.float16 if req.dtype == "fp16" else torch.bfloat16
            _job_log(job_id, f"downloading {req.model} (dtype={req.dtype}) ...")
            tok_kwargs = {"trust_remote_code": True}
            try:
                tok = AutoTokenizer.from_pretrained(req.model, **tok_kwargs)
            except Exception:
                tok = AutoTokenizer.from_pretrained(req.model, use_fast=False, **tok_kwargs)
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token or "[PAD]"
            attn_impl = "eager" if req.dtype == "bf16" else "sdpa"
            model = AutoModelForCausalLM.from_pretrained(
                req.model, torch_dtype=dtype, attn_implementation=attn_impl,
                device_map="auto", trust_remote_code=True,
            ).eval()
            for p in model.parameters():
                p.requires_grad_(False)

            # 3. Resolve layer.
            tm = resolve_text_model(model)
            decoder_layers = resolve_decoder_layers(tm)
            n_layers = len(decoder_layers)
            if req.layer is not None:
                if not (0 <= req.layer < n_layers):
                    raise ValueError(f"layer {req.layer} out of [0,{n_layers})")
                layer_idx = int(req.layer)
            else:
                layer_idx = max(0, min(n_layers - 1, int(n_layers * (req.depth or 0.5))))
            d_M = int(getattr(resolve_text_config(model.config), "hidden_size", 0))
            _job_log(job_id, f"loaded {req.model}: n_layers={n_layers} layer={layer_idx} d_M={d_M}")

            # 4. Extract mean-pool h on n passages.
            target_layer = decoder_layers[layer_idx]
            captured: dict[str, torch.Tensor] = {}
            def _hk(_m, _i, out):
                hh = out[0] if isinstance(out, tuple) else out
                captured["h"] = hh.detach().float().cpu()
            handle = target_layer.register_forward_hook(_hk)
            h_out = torch.zeros(n, d_M, dtype=torch.float32)
            try:
                for s in range(0, n, req.batch_size):
                    batch = texts[s:s + req.batch_size]
                    enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                               max_length=req.max_length, add_special_tokens=True)
                    with torch.no_grad():
                        model(input_ids=enc["input_ids"].to(DEVICE),
                              attention_mask=enc["attention_mask"].to(DEVICE),
                              use_cache=False)
                    h_batch = captured["h"]
                    mask = enc["attention_mask"].float().unsqueeze(-1)
                    h_pool = ((h_batch * mask).sum(1) / mask.sum(1).clamp_min(1)).float()
                    h_out[s:s + h_pool.shape[0]] = h_pool
                    if (s // req.batch_size) % 25 == 0:
                        _job_log(job_id, f"extract: {s+req.batch_size}/{n}")
            finally:
                handle.remove()

            if h_out.isnan().any().item():
                raise ValueError(f"extracted activations contain NaN — try --dtype bf16 (set dtype='bf16' for this model). "
                                 f"This usually means {req.model} overflows fp16 at attention-sink channels.")
            _job_log(job_id, f"extracted h: shape={tuple(h_out.shape)} mean_norm={h_out.norm(dim=-1).mean():.2f}")

            # Free the target model BEFORE the lstsq so memory clears.
            del model
            torch.cuda.empty_cache()

            # 5. lstsq fit via add_held_out_tag.
            info = adapters.add_held_out_tag(req.tag, h_out)
            _job_log(job_id, f"add_held_out_tag: {info}")

            # 6. Persist updated bundle.
            adapters.to(DEVICE).eval()      # back on GPU for inference
            adapters.save(str(ART / "adapters"))
            _job_log(job_id, f"saved bundle → {ART/'adapters'}")

            # 7. Register tag in TAG_TO_MODEL + persist.
            tag_info = {"model": req.model, "depth": req.depth} if req.layer is None else {"model": req.model, "layer": req.layer}
            TAG_TO_MODEL[req.tag] = tag_info
            _persist_user_tag(req.tag, tag_info)
            SUPPORTED_TAGS = sorted([t for t in adapters.tags if t in TAG_TO_MODEL])
            j["result"] = {"tag": req.tag, "d_M": d_M, "layer": layer_idx,
                           "enc_fve": info["enc_fve"], "dec_fve": info["dec_fve"]}
            j["status"] = "done"
            _job_log(job_id, f"DONE — tag={req.tag} added to live bundle")
        except Exception as e:
            import traceback
            j["error"] = f"{type(e).__name__}: {e}"
            j["log"] = (j.get("log") or "") + traceback.format_exc()
            j["status"] = "error"

    Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in _JOBS:
        raise HTTPException(404, f"unknown job_id {job_id}")
    j = _JOBS[job_id]
    return {"status": j["status"], "tag": j.get("tag"), "log": j.get("log") or "",
            "error": j.get("error"), "result": j.get("result")}


@app.post("/api/forward")
def forward(req: ForwardRequest):
    if req.tag not in TAG_TO_MODEL:
        raise HTTPException(400, f"unknown tag {req.tag}; supported: {SUPPORTED_TAGS}")
    if req.tag not in adapters.tags:
        raise HTTPException(400, f"tag {req.tag} has no adapter in {HF_ADAPTER_NAME}")
    try:
        tok, model = _load_target(req.tag)
    except Exception as e:
        msg = str(e)
        if "torch.load" in msg or "weights_only" in msg or "CVE-2025-32434" in msg:
            raise HTTPException(503, f"{req.tag}: model ships only pytorch_model.bin and "
                                     f"the container's torch (<2.6) refuses it under CVE-2025-32434. "
                                     f"Pick a model with safetensors (qwen3-*, qwen2p5-*, smollm*, "
                                     f"phi-*, gemma4-*, vikhr-7b-01), or restart the demo with "
                                     f"`pip install -U torch` baked into the launch.")
        if "model type" in msg and "Transformers does not recognize" in msg:
            raise HTTPException(503, f"{req.tag}: transformers too old for this arch — {msg}")
        raise HTTPException(500, f"{req.tag} load failed: {type(e).__name__}: {msg}")
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
    # mean-pool over content (non-pad) tokens — matches training-time h.
    mask = enc["attention_mask"][0].float().unsqueeze(-1).cpu()        # [T, 1]
    h_pool = ((h_seq * mask).sum(dim=0) / mask.sum().clamp_min(1)).float()  # [d_M]
    token_strs = []
    for tid in input_ids[0].tolist():
        s = tok.decode([tid])
        token_strs.append(s if s else f"<id={tid}>")

    cache_id = uuid.uuid4().hex[:12]
    _ACT_CACHE[cache_id] = {
        "tag": req.tag,
        "tokens": token_strs,
        "h_M": h_seq,                                # [T, d_M] fp32 CPU
        "h_pool": h_pool,                            # [d_M] fp32 CPU — training-distribution
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


@app.post("/api/generate")
def generate(req: GenerateRequest):
    """Let the TARGET model M extend the prompt, then capture per-token h_M at
    the chosen layer over the full sequence (prompt + completion). Returns the
    same shape as /api/forward plus a `prompt_len` marker so the UI can show
    prompt vs generated tokens differently."""
    if req.tag not in TAG_TO_MODEL:
        raise HTTPException(400, f"unknown tag {req.tag}")
    if req.tag not in adapters.tags:
        raise HTTPException(400, f"tag {req.tag} has no adapter in {HF_ADAPTER_NAME}")
    try:
        tok, model = _load_target(req.tag)
    except ValueError as e:
        if "model type" in str(e) and "Transformers does not recognize" in str(e):
            raise HTTPException(503, f"{req.tag}: transformers too old; {e}")
        raise HTTPException(500, f"{req.tag}: {e}")
    except ImportError as e:
        raise HTTPException(503, f"{req.tag}: {e}")
    except Exception as e:
        raise HTTPException(500, f"{req.tag} load failed: {type(e).__name__}: {e}")
    layer_idx, n_layers = _resolve_layer(req.tag, model)
    if req.layer is not None:
        if not (0 <= req.layer < n_layers):
            raise HTTPException(400, f"layer {req.layer} out of range [0, {n_layers})")
        layer_idx = req.layer

    enc = tok(req.prompt, return_tensors="pt", truncation=True,
              max_length=req.max_input_tokens, add_special_tokens=True).to(DEVICE)
    prompt_ids = enc["input_ids"]
    prompt_len = prompt_ids.shape[1]
    if prompt_len == 0:
        raise HTTPException(400, "empty prompt after tokenization")

    # M.generate to produce a completion.
    gen_kwargs = dict(
        max_new_tokens=req.max_new_tokens,
        do_sample=req.do_sample,
        temperature=req.temperature if req.do_sample else 1.0,
        pad_token_id=tok.pad_token_id,
    )
    with torch.no_grad():
        gen_out = model.generate(input_ids=prompt_ids, attention_mask=enc["attention_mask"], **gen_kwargs)
    full_ids = gen_out                                                        # [1, T_full]
    full_attn = torch.ones_like(full_ids)

    # Re-forward the FULL sequence once to capture hidden states cleanly at
    # the chosen layer for every position.
    text_model = resolve_text_model(model)
    decoder_layers = resolve_decoder_layers(text_model)
    target_layer = decoder_layers[layer_idx]
    captured: dict[str, torch.Tensor] = {}

    def _hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach().float().cpu()
    handle = target_layer.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(input_ids=full_ids, attention_mask=full_attn)
    finally:
        handle.remove()
    if "h" not in captured:
        raise HTTPException(500, "forward hook didn't fire")

    h_seq = captured["h"][0]                                                  # [T_full, d_M]
    # Mean-pool over the WHOLE sequence (prompt + generated); attention mask is all-ones.
    h_pool = h_seq.mean(dim=0).float()                                        # [d_M]
    token_strs = [tok.decode([tid]) or f"<id={tid}>" for tid in full_ids[0].tolist()]

    cache_id = uuid.uuid4().hex[:12]
    _ACT_CACHE[cache_id] = {
        "tag": req.tag,
        "tokens": token_strs,
        "h_M": h_seq,
        "h_pool": h_pool,
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
        "prompt_len": prompt_len,
        "generated_text": tok.decode(full_ids[0, prompt_len:].tolist(), skip_special_tokens=True),
    }


@app.get("/api/generate_stream")
def generate_stream(tag: str, prompt: str, layer: int | None = None,
                    max_input_tokens: int = 512, max_new_tokens: int = 64,
                    do_sample: bool = False, temperature: float = 1.0):
    """SSE: stream the TARGET model M's completion token-by-token, then re-forward
    the full sequence to capture per-token h_M at the chosen layer, then emit
    a 'done' event with cache_id + tokens so the UI can switch to interactive
    explanation mode. Supports /api/stop via stream_id."""
    if tag not in TAG_TO_MODEL:
        raise HTTPException(400, f"unknown tag {tag}")
    if tag not in adapters.tags:
        raise HTTPException(400, f"tag {tag} has no adapter in {HF_ADAPTER_NAME}")
    # Try to load target; on failure emit an SSE error rather than a 500
    # response, since EventSource can't read 500 body text in the browser.
    target_err = None
    try:
        tok, model = _load_target(tag)
    except Exception as e:
        msg = str(e)
        if "torch.load" in msg or "weights_only" in msg or "CVE-2025-32434" in msg:
            target_err = (f"{tag}: model ships only pytorch_model.bin and the "
                          f"container's torch (<2.6) refuses to load it under "
                          f"CVE-2025-32434. Pick a model that has safetensors "
                          f"(qwen3-*, qwen2p5-*, smollm*, phi-*, gemma4-*, "
                          f"vikhr-7b-01), or restart the demo with "
                          f"`pip install -U torch` in the container.")
        elif "model type" in msg and "Transformers does not recognize" in msg:
            target_err = f"{tag}: transformers too old for this arch — {e}"
        else:
            target_err = f"{tag} load failed: {type(e).__name__}: {e}"

    if target_err is None:
        layer_idx, n_layers = _resolve_layer(tag, model)
        if layer is not None:
            if not (0 <= layer < n_layers):
                target_err = f"layer {layer} out of range [0, {n_layers})"
            else:
                layer_idx = layer

    if target_err is None:
        enc = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=max_input_tokens, add_special_tokens=True).to(DEVICE)
        prompt_ids = enc["input_ids"]
        prompt_len = prompt_ids.shape[1]
        if prompt_len == 0:
            target_err = "empty prompt after tokenization"

    if target_err is not None:
        # Emit a single done-with-error event so the client UI surfaces the
        # real reason instead of the generic "generate stream failed".
        def err_gen():
            yield f"event: done\ndata: {json.dumps({'error': target_err})}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream",
                                 headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})

    stream_id = uuid.uuid4().hex[:12]
    stop_event = Event()
    _STOP_EVENTS[stream_id] = stop_event

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    result: dict = {}

    def _run_gen():
        try:
            with torch.no_grad():
                out = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else 1.0,
                    pad_token_id=tok.pad_token_id,
                    streamer=streamer,
                    stopping_criteria=StoppingCriteriaList([_StopOnEvent(stop_event)]),
                )
            result["ids"] = out
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"

    thread = Thread(target=_run_gen)
    thread.start()

    def event_gen():
        try:
            yield f"event: start\ndata: {json.dumps({'stream_id': stream_id, 'prompt_len': prompt_len})}\n\n"
            for new_text in streamer:
                yield f"event: token\ndata: {json.dumps({'text': new_text})}\n\n"
        finally:
            thread.join()
            _STOP_EVENTS.pop(stream_id, None)

        if "error" in result:
            yield f"event: done\ndata: {json.dumps({'error': result['error']})}\n\n"
            return

        full_ids = result["ids"]
        # Re-forward the FULL sequence to capture hidden states at the chosen layer.
        full_attn = torch.ones_like(full_ids)
        text_model = resolve_text_model(model)
        decoder_layers = resolve_decoder_layers(text_model)
        target_layer = decoder_layers[layer_idx]
        captured: dict[str, torch.Tensor] = {}

        def _hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["h"] = h.detach().float().cpu()
        handle = target_layer.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                model(input_ids=full_ids, attention_mask=full_attn)
        finally:
            handle.remove()
        h_seq = captured["h"][0]
        h_pool = h_seq.mean(dim=0).float()
        token_strs = [tok.decode([tid]) or f"<id={tid}>" for tid in full_ids[0].tolist()]

        cache_id = uuid.uuid4().hex[:12]
        _ACT_CACHE[cache_id] = {
            "tag": tag,
            "tokens": token_strs,
            "h_M": h_seq,
            "h_pool": h_pool,
            "layer": layer_idx,
        }
        _evict_act_cache()

        stopped = stop_event.is_set()
        yield f"event: done\ndata: {json.dumps({'cache_id': cache_id, 'tokens': token_strs, 'layer': layer_idx, 'n_layers': n_layers, 'd_M': h_seq.shape[1], 'tag': tag, 'prompt_len': prompt_len, 'generated_text': tok.decode(full_ids[0, prompt_len:].tolist(), skip_special_tokens=True), 'stopped': stopped})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


@app.post("/api/explain")
def explain(req: ExplainRequest):
    if req.cache_id not in _ACT_CACHE:
        raise HTTPException(404, f"cache_id {req.cache_id} not found or evicted")
    entry = _ACT_CACHE[req.cache_id]
    tokens: list[str] = entry["tokens"]
    if req.idx == -1:
        if "h_pool" not in entry:
            raise HTTPException(400, "no mean-pool cached for this entry")
        h_t = entry["h_pool"].unsqueeze(0).to(DEVICE)
        token_label = "⌗ PASSAGE (mean-pool)"
    else:
        if not (0 <= req.idx < len(tokens)):
            raise HTTPException(400, f"idx {req.idx} out of range [0, {len(tokens)})")
        h_t = entry["h_M"][req.idx].unsqueeze(0).to(DEVICE)              # [1, d_M]
        token_label = tokens[req.idx]
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
    return {"z": z, "raw": raw, "token": token_label, "idx": req.idx, "tag": tag}


@app.get("/api/explain_stream")
def explain_stream(cache_id: str, idx: int, max_new_tokens: int = 80):
    """SSE: stream the AV's tokens as they're generated, then a final 'done' event
    with the parsed `z`. Frontend connects via EventSource.

    Special idx values:
      idx == -1  → use the mean-pooled h (training-distribution match).
    """
    if cache_id not in _ACT_CACHE:
        raise HTTPException(404, f"cache_id {cache_id} not found or evicted")
    entry = _ACT_CACHE[cache_id]
    tokens: list[str] = entry["tokens"]
    if idx == -1:
        if "h_pool" not in entry:
            raise HTTPException(400, "no mean-pool cached for this entry")
        h_t = entry["h_pool"].unsqueeze(0).to(DEVICE)
        token_label = "⌗ PASSAGE (mean-pool)"
    else:
        if not (0 <= idx < len(tokens)):
            raise HTTPException(400, f"idx {idx} out of range [0, {len(tokens)})")
        h_t = entry["h_M"][idx].unsqueeze(0).to(DEVICE)
        token_label = tokens[idx]
    tag = entry["tag"]

    with torch.no_grad():
        inj_shared = adapters.encode(tag, h_t).float()
        inj_shared = normalize_activation(inj_shared, INJ_SCALE)
        prompt_text = ACTOR_TEMPLATE.format(model_tag=tag, injection_char=INJECTION_CHAR)
        p_ids = av_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=True, add_generation_prompt=True,
        )
        input_ids = torch.tensor([p_ids], dtype=torch.long, device=DEVICE)
        embed = av.get_input_embeddings()(input_ids)
        inj = inj_shared.to(DEVICE, dtype=embed.dtype)
        embed = inject_at_marked_positions(input_ids, embed, inj, INJ_ID, LEFT_ID, RIGHT_ID)
        attn = torch.ones_like(input_ids)

    stream_id = uuid.uuid4().hex[:12]
    stop_event = Event()
    _STOP_EVENTS[stream_id] = stop_event
    streamer = TextIteratorStreamer(av_tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        inputs_embeds=embed,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=av_tokenizer.pad_token_id,
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList([_StopOnEvent(stop_event)]),
    )
    # Launch generation in a background thread so we can iterate the streamer.
    def _run_gen():
        with torch.no_grad():
            av.generate(**gen_kwargs)
    thread = Thread(target=_run_gen)
    thread.start()

    def event_gen():
        chunks: list[str] = []
        try:
            # First event: hand the client a stream_id so it can call /api/stop.
            yield f"event: start\ndata: {json.dumps({'stream_id': stream_id})}\n\n"
            for new_text in streamer:
                chunks.append(new_text)
                yield f"event: token\ndata: {json.dumps({'text': new_text})}\n\n"
        finally:
            thread.join()
            _STOP_EVENTS.pop(stream_id, None)
        raw = "".join(chunks)
        z = extract_explanation(raw) or raw.strip()
        stopped = stop_event.is_set()
        yield f"event: done\ndata: {json.dumps({'raw': raw, 'z': z, 'token': token_label, 'idx': idx, 'tag': tag, 'stopped': stopped})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


# ─── Activation-Oracle (v15 AO): inject enc(h) + free-text question ──────────
def _resolve_ao_h(entry: dict, pool: str, idx: int | None,
                  span: list[int] | None) -> tuple[torch.Tensor, str]:
    """Return (h_t [1, d_M] on DEVICE, human label) for an AO request.

    pool == "mean": mean over the span [start, end) of cached per-token h_M, or
      over ALL content tokens (use the cached `h_pool`) when span is None. The
      cached h_pool is already mean-pooled over content tokens — the training
      distribution for v15 AO — so we prefer it directly when no span is given.
    pool == "token": the single `idx` token (OOD for AO).
    """
    tokens: list[str] = entry["tokens"]
    n = len(tokens)
    if pool == "token":
        if idx is None or not (0 <= idx < n):
            raise HTTPException(400, f"pool='token' needs idx in [0, {n}); got {idx}")
        return entry["h_M"][idx].unsqueeze(0).to(DEVICE), f"token[{idx}]={tokens[idx]!r}"
    # mean-pool
    if span is None:
        if "h_pool" not in entry:
            raise HTTPException(400, "no mean-pool cached for this entry")
        return entry["h_pool"].unsqueeze(0).to(DEVICE), "⌗ RESPONSE (mean-pool, all tokens)"
    if len(span) != 2:
        raise HTTPException(400, "span must be [start, end)")
    s, e = int(span[0]), int(span[1])
    s = max(0, min(s, n)); e = max(0, min(e, n))
    if e <= s:
        raise HTTPException(400, f"empty span [{s}, {e})")
    h_t = entry["h_M"][s:e].mean(dim=0, keepdim=True).to(DEVICE)
    return h_t, f"⌗ MEAN of tokens [{s}:{e})"


def _build_ao_embed(tag: str, h_t: torch.Tensor, question: str):
    """Project enc(tag, h), normalize to √d_shared, build the v15 AO prompt
    (ACTOR_TEMPLATE + '\\n\\nQuestion: {q}\\nAnswer:'), inject at the ㈎ marker.

    Returns (embed [1, T, d], attn [1, T]). Mirrors train_v15.build_example's
    non-instruct AO path exactly: same ACTOR_TEMPLATE, same appended cue, same
    chat-template wrapping, same enc → normalize_activation injection.
    """
    inj_shared = adapters.encode(tag, h_t).float()
    inj_shared = normalize_activation(inj_shared, INJ_SCALE)
    prompt_text = (ACTOR_TEMPLATE.format(model_tag=tag, injection_char=INJECTION_CHAR)
                   + f"\n\nQuestion: {question.strip()}\nAnswer:")
    p_ids = av_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=True, add_generation_prompt=True,
    )
    input_ids = torch.tensor([p_ids], dtype=torch.long, device=DEVICE)
    embed = av.get_input_embeddings()(input_ids)
    inj = inj_shared.to(DEVICE, dtype=embed.dtype)
    embed = inject_at_marked_positions(input_ids, embed, inj, INJ_ID, LEFT_ID, RIGHT_ID)
    attn = torch.ones_like(input_ids)
    return embed, attn


@app.post("/api/ask")
def ask(req: AskRequest):
    if req.cache_id not in _ACT_CACHE:
        raise HTTPException(404, f"cache_id {req.cache_id} not found or evicted")
    if not req.question or not req.question.strip():
        raise HTTPException(400, "question is required")
    entry = _ACT_CACHE[req.cache_id]
    tag = entry["tag"]
    h_t, label = _resolve_ao_h(entry, req.pool, req.idx, req.span)
    with torch.no_grad():
        embed, attn = _build_ao_embed(tag, h_t, req.question)
        out = av.generate(
            inputs_embeds=embed,
            attention_mask=attn,
            max_new_tokens=req.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=av_tokenizer.pad_token_id,
        )
    answer = av_tokenizer.decode(out[0], skip_special_tokens=True).strip()
    return {"answer": answer, "question": req.question, "pool": req.pool,
            "source": label, "tag": tag}


@app.get("/api/ask_stream")
def ask_stream(cache_id: str, question: str, pool: str = "mean",
               idx: int | None = None, span: str | None = None,
               max_new_tokens: int = 80):
    """SSE mirror of /api/ask. `span` is passed as 'start,end' on the query string."""
    if cache_id not in _ACT_CACHE:
        raise HTTPException(404, f"cache_id {cache_id} not found or evicted")
    if not question or not question.strip():
        raise HTTPException(400, "question is required")
    entry = _ACT_CACHE[cache_id]
    tag = entry["tag"]
    span_list = None
    if span:
        try:
            span_list = [int(x) for x in span.split(",")]
        except ValueError:
            raise HTTPException(400, "span must be 'start,end'")
    h_t, label = _resolve_ao_h(entry, pool, idx, span_list)
    with torch.no_grad():
        embed, attn = _build_ao_embed(tag, h_t, question)

    stream_id = uuid.uuid4().hex[:12]
    stop_event = Event()
    _STOP_EVENTS[stream_id] = stop_event
    streamer = TextIteratorStreamer(av_tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        inputs_embeds=embed,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=av_tokenizer.pad_token_id,
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList([_StopOnEvent(stop_event)]),
    )

    def _run_gen():
        with torch.no_grad():
            av.generate(**gen_kwargs)
    thread = Thread(target=_run_gen)
    thread.start()

    def event_gen():
        chunks: list[str] = []
        try:
            yield f"event: start\ndata: {json.dumps({'stream_id': stream_id, 'source': label})}\n\n"
            for new_text in streamer:
                chunks.append(new_text)
                yield f"event: token\ndata: {json.dumps({'text': new_text})}\n\n"
        finally:
            thread.join()
            _STOP_EVENTS.pop(stream_id, None)
        answer = "".join(chunks).strip()
        stopped = stop_event.is_set()
        yield (f"event: done\ndata: "
               f"{json.dumps({'answer': answer, 'question': question, 'pool': pool, 'source': label, 'tag': tag, 'stopped': stopped})}\n\n")

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


# ─── v17 CALIBRATED detector: per-bias P(Yes) ───────────────────────────────
@torch.no_grad()
def _v17_p_yes(tag: str, h_vec: torch.Tensor, bias: str) -> float:
    """Mirror eval_v17.p_yes: build the detect prompt, inject enc(tag,h) at the
    marker, read softmax over the (Yes,No) first-token logits."""
    V = V17
    proj = V["adapters"].encode(tag, h_vec.unsqueeze(0).to(DEVICE))      # [1, d_shared]
    vec = normalize_activation(proj, V["inj_scale"])[0]
    ptxt = (V["template"].format(model_tag=tag, injection_char=V["inj_char"])
            + f"\n\nQuestion: {V['detect_qa'].format(desc=V['DESC'][bias])}\nAnswer:")
    p_ids = V["tok"].apply_chat_template([{"role": "user", "content": ptxt}],
                                         tokenize=True, add_generation_prompt=True)
    p = torch.tensor([p_ids], device=DEVICE)
    embed = V["av"].get_input_embeddings()(p)
    embed = inject_at_marked_positions(p, embed, vec.unsqueeze(0).to(embed.dtype),
                                       V["inj_id"], V["left_id"], V["right_id"])
    logits = V["av"](inputs_embeds=embed).logits[0, -1]
    pair = torch.stack([logits[V["yes0"]], logits[V["no0"]]]).float()
    return torch.softmax(pair, 0)[0].item()


def _v17_scan(tag: str, h_vec: torch.Tensor) -> list[dict]:
    V = V17
    out = []
    for b in V["scan_biases"]:
        py = _v17_p_yes(tag, h_vec, b)
        out.append({"bias": b, "desc": V["DESC"][b], "p_yes": round(py, 4),
                    "held_out": b in V["held"]})
    out.sort(key=lambda r: r["p_yes"], reverse=True)
    return out


class DetectRequest(BaseModel):
    cache_id: str
    idx: int | None = None          # single-token (pool='token'); else mean-pool
    span: list[int] | None = None   # [start, end) mean-pool over a token span
    pool: str = "mean"
    contrast: bool = False          # also scan the matching clean-base act if present


@app.post("/api/detect")
def detect(req: DetectRequest):
    """v17 CALIBRATED bias scan. For each known bias, build the v17 detect prompt,
    inject enc(tag, h), read P(Yes). Returns a sorted [{bias, p_yes, held_out}] +
    a verdict. If contrast=True and the cached entry carries a matching clean-base
    activation, also scans base → a bias is 'real' only if HIGH on organism AND
    LOW on base."""
    if V17 is None:
        raise HTTPException(503, "v17 detector not loaded on this server")
    if req.cache_id not in _ACT_CACHE:
        raise HTTPException(404, f"cache_id {req.cache_id} not found or evicted")
    entry = _ACT_CACHE[req.cache_id]
    tag = entry["tag"]
    if tag not in V17["adapters"].tags:
        raise HTTPException(400, f"tag {tag} has no enc adapter in the v17 bundle")
    h_t, label = _resolve_ao_h(entry, req.pool, req.idx, req.span)
    org_scan = _v17_scan(tag, h_t.squeeze(0))
    max_py = org_scan[0]["p_yes"] if org_scan else 0.0
    resp: dict[str, Any] = {
        "tag": tag, "source": label, "scan": org_scan, "max_p_yes": max_py,
        "verdict": ("no bias detected" if max_py < 0.5
                    else f"bias detected: {org_scan[0]['bias']}"),
        "threshold": 0.5,
    }
    # Organism-vs-base contrast.
    if req.contrast and entry.get("h_base") is not None:
        base_scan = _v17_scan(tag, entry["h_base"].to(DEVICE))
        base_by = {r["bias"]: r["p_yes"] for r in base_scan}
        # A bias is "real" if high on org AND low on base.
        contrast_rows = []
        for r in org_scan:
            bp = base_by.get(r["bias"], 0.0)
            contrast_rows.append({"bias": r["bias"], "desc": r["desc"],
                                  "held_out": r["held_out"],
                                  "p_yes_org": r["p_yes"], "p_yes_base": bp,
                                  "real": (r["p_yes"] >= 0.5 and bp < 0.5)})
        resp["contrast"] = contrast_rows
        resp["has_base"] = True
    else:
        resp["has_base"] = entry.get("h_base") is not None
    return resp


# ─── Activation BROWSER: list + load curated organism / base / lie acts ──────
@app.get("/api/sources")
def sources():
    out = []
    for spec in _SOURCE_SPECS:
        if not Path(spec["org_acts"]).exists() or not Path(spec["transcripts"]).exists():
            continue
        try:
            d = _load_source_data(spec)
        except Exception as e:
            print(f"[ui-server][sources] skip {spec['key']}: {e}")
            continue
        n = min(d["H_org"].shape[0], len(d["rows"]))
        rows = [_source_row_meta(spec, d["rows"][i], i) for i in range(n)]
        out.append({"key": spec["key"], "label": spec["label"], "tag": spec["tag"],
                    "kind": spec["kind"], "has_base": d["H_base"] is not None,
                    "n": n, "rows": rows})
    return {"sources": out, "v17_loaded": V17 is not None}


class LoadSourceRequest(BaseModel):
    source: str
    idx: int


@app.post("/api/load_source")
def load_source(req: LoadSourceRequest):
    """Load a curated activation (org act + matching clean-base act + transcript)
    into the act cache as a single mean-pool entry. Returns a cache_id that
    /api/detect, /api/ask and /api/explain (idx=-1) all accept."""
    spec = next((s for s in _SOURCE_SPECS if s["key"] == req.source), None)
    if spec is None:
        raise HTTPException(404, f"unknown source {req.source}")
    d = _load_source_data(spec)
    n = min(d["H_org"].shape[0], len(d["rows"]))
    if not (0 <= req.idx < n):
        raise HTTPException(400, f"idx {req.idx} out of range [0,{n})")
    row = d["rows"][req.idx]
    meta = _source_row_meta(spec, row, req.idx)
    h_pool = d["H_org"][req.idx].float().cpu()
    h_base = d["H_base"][req.idx].float().cpu() if d["H_base"] is not None else None
    cache_id = uuid.uuid4().hex[:12]
    _ACT_CACHE[cache_id] = {
        "tag": spec["tag"],
        "tokens": [],                          # no token grid for curated acts
        "h_M": h_pool.unsqueeze(0),            # [1, d_M] — single mean vector
        "h_pool": h_pool,                      # [d_M] — the organism mean act
        "h_base": h_base,                      # matching clean-base act (or None)
        "layer": None,
        "source_kind": spec["kind"],
        "transcript": meta,
    }
    _evict_act_cache()
    return {"cache_id": cache_id, "tag": spec["tag"], "source": spec["key"],
            "has_base": h_base is not None, "transcript": meta,
            "d_M": int(h_pool.shape[0])}


@app.post("/api/stop")
def stop(payload: dict):
    sid = payload.get("stream_id")
    if not sid:
        raise HTTPException(400, "stream_id required")
    ev = _STOP_EVENTS.get(sid)
    if ev is None:
        return {"ok": False, "reason": "stream_id not found (already finished?)"}
    ev.set()
    return {"ok": True, "stream_id": sid}


@app.post("/api/stop_all")
def stop_all():
    """Signal cancellation on every active SSE stream. Each streaming endpoint
    checks its threading.Event between tokens and breaks out at the next
    boundary."""
    sids = list(_STOP_EVENTS.keys())
    for sid in sids:
        ev = _STOP_EVENTS.get(sid)
        if ev is not None:
            ev.set()
    return {"ok": True, "stopped": sids, "count": len(sids)}


@app.post("/api/clear_targets")
def clear_targets():
    """Drop every cached target model and free GPU memory. Used by the UI when
    the user wants to make room for /api/add_model or a different target."""
    keys = list(_TARGET_CACHE.keys())
    _evict_other_targets(keep_tag=None)
    return {"ok": True, "evicted": keys}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
