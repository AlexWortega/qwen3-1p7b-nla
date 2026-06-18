"""Ablation 1 — Verbalization zeroing (in-container version).

Three conditions on 100 qwen2.5-7b passages (same pids as compare_q25_7b_n100.json):
  (a) REAL    — inject normalize(enc(h)) as normal
  (b) ZERO    — inject a d_shared zero vector at the marker
  (c) SHUFFLE — inject a DIFFERENT passage's real encoded activation

Metric: mean cosine between generated z and gold teacher (all-mpnet embedding).
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

# ── in-container paths ────────────────────────────────────────────────────────
AV_SAVE_DIR    = Path("/big/av_v8_mixed")
ADAPTERS_DIR   = Path("/big/adapters_v8_mixed_serve")
POOL_DIR       = Path("/big/activations_pool_300m")
COMPARE_JSON   = Path("/big/kitft_baseline/compare_q25_7b_n100.json")
OUT_FILE       = Path("/big/extra_exps_out/abl_verbalization_zeroing.json")
TAG = "qwen2p5-7b"
SEED = 42

sys.path.insert(0, "/workspace")

from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation, extract_explanation


def embed_sentences(sentences: list[str]) -> "np.ndarray":
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",
                                cache_folder="/cache/huggingface/hub")
    return model.encode(sentences, batch_size=32, show_progress_bar=False,
                        normalize_embeddings=False)


def cosine_sim_diag(a, b):
    import numpy as np
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return float((a * b).sum(axis=1).mean())


@torch.no_grad()
def generate_z(av, tokenizer, prompt_text, inj_vec, inj_id, left_id, right_id, device, max_new=120):
    p_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=True, add_generation_prompt=True,
    )
    if hasattr(p_ids, "input_ids"):
        p_ids = p_ids["input_ids"]
    if isinstance(p_ids, str):
        p_ids = tokenizer(p_ids, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([p_ids], dtype=torch.long, device=device)
    embed = av.get_input_embeddings()(input_ids)
    v = inj_vec.unsqueeze(0).to(device=device, dtype=embed.dtype)
    embed = inject_at_marked_positions(input_ids, embed, v, inj_id, left_id, right_id)
    attn = torch.ones_like(input_ids)
    out = av.generate(
        inputs_embeds=embed, attention_mask=attn,
        max_new_tokens=max_new, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return extract_explanation(tokenizer.decode(out[0], skip_special_tokens=True)) or ""


def main():
    random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[abl1] device={device}")

    # ── load AV ──────────────────────────────────────────────────────────────
    meta = yaml.safe_load((AV_SAVE_DIR / "nla_meta.yaml").read_text())
    template = meta["prompt_templates"]["actor"]
    tok_cfg = meta["tokens"]
    d_shared = int(meta["d_shared"])
    inj_scale = math.sqrt(d_shared)
    inj_id = int(tok_cfg["injection_token_id"])
    left_id = int(tok_cfg["injection_left_neighbor_id"])
    right_id = int(tok_cfg["injection_right_neighbor_id"])
    inj_char = tok_cfg["injection_char"]

    tokenizer = AutoTokenizer.from_pretrained(meta["av_base"],
                                              cache_dir="/cache/huggingface/hub")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[abl1] loading AV base model…")
    base = AutoModelForCausalLM.from_pretrained(
        meta["av_base"], torch_dtype=torch.float16,
        cache_dir="/cache/huggingface/hub",
    ).to(device).eval()
    av = PeftModel.from_pretrained(base, str(AV_SAVE_DIR / "av")).to(device).eval()
    adapters = ModelPoolAdapters.load(ADAPTERS_DIR).to(device)

    # ── load passages / activations ──────────────────────────────────────────
    compare = json.loads(COMPARE_JSON.read_text())
    per_passage = compare["per_passage"]
    pids = [row["pid"] for row in per_passage]
    golds = {row["pid"]: row["gold"] for row in per_passage}

    pool_meta = json.loads((POOL_DIR / f"{TAG}.meta.json").read_text())
    h_shard = load_file(str(POOL_DIR / pool_meta["shard"]))["h"]
    print(f"[abl1] shard shape: {h_shard.shape}, running {len(pids)} passages…")

    prompt = template.format(model_tag=TAG, injection_char=inj_char)

    # Pre-encode all 100 real projected activations
    enc_vecs = []
    for pid in pids:
        h = h_shard[pid].to(device, dtype=torch.float32).unsqueeze(0)
        proj = adapters.encode(TAG, h).squeeze(0)
        enc_vecs.append(normalize_activation(proj, inj_scale))

    zero_vec = torch.zeros(d_shared, device=device, dtype=torch.float32)

    results_real, results_zero, results_shuffle = [], [], []

    for i, pid in enumerate(pids):
        real_v = enc_vecs[i]
        # shuffle: pick a different passage
        j = random.choice([k for k in range(len(pids)) if k != i])
        shuffle_v = enc_vecs[j]
        gold = golds[pid]

        z_real    = generate_z(av, tokenizer, prompt, real_v,    inj_id, left_id, right_id, device)
        z_zero    = generate_z(av, tokenizer, prompt, zero_vec,  inj_id, left_id, right_id, device)
        z_shuffle = generate_z(av, tokenizer, prompt, shuffle_v, inj_id, left_id, right_id, device)

        results_real.append(    {"pid": pid, "z": z_real,    "gold": gold})
        results_zero.append(    {"pid": pid, "z": z_zero,    "gold": gold})
        results_shuffle.append( {"pid": pid, "z": z_shuffle, "gold": gold,
                                 "shuffle_src_pid": pids[j]})

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/100]  real={z_real[:80]!r}")

    # ── embed and score ───────────────────────────────────────────────────────
    print("[abl1] embedding all sentences…")
    all_z = (
        [r["z"] for r in results_real]    +
        [r["z"] for r in results_zero]    +
        [r["z"] for r in results_shuffle] +
        [r["gold"] for r in results_real]
    )
    embs = embed_sentences(all_z)
    n = len(pids)
    emb_real    = embs[0*n:1*n]
    emb_zero    = embs[1*n:2*n]
    emb_shuffle = embs[2*n:3*n]
    emb_gold    = embs[3*n:4*n]

    cos_real    = cosine_sim_diag(emb_real,    emb_gold)
    cos_zero    = cosine_sim_diag(emb_zero,    emb_gold)
    cos_shuffle = cosine_sim_diag(emb_shuffle, emb_gold)

    delta_zero    = cos_real - cos_zero
    delta_shuffle = cos_real - cos_shuffle
    reads_activation = delta_zero > 0.05 and delta_shuffle > 0.05

    verdict = (
        "AV READS ACTIVATION — zero and shuffle both drop well below real; "
        "activation content drives verbalization (refutes text-inversion threat)."
        if reads_activation else
        "AV READS CONTEXT — zero stays close to real; activation is decorative."
    )

    out = {
        "n_passages": n,
        "tag": TAG,
        "conditions": {
            "real":    {"mean_cos_to_gold": round(cos_real,    4)},
            "zero":    {"mean_cos_to_gold": round(cos_zero,    4)},
            "shuffle": {"mean_cos_to_gold": round(cos_shuffle, 4)},
        },
        "deltas": {
            "real_minus_zero":    round(delta_zero,    4),
            "real_minus_shuffle": round(delta_shuffle, 4),
        },
        "verdict": verdict,
        "per_passage": {
            "real":    results_real,
            "zero":    [{"pid": r["pid"], "z": r["z"]} for r in results_zero],
            "shuffle": [{"pid": r["pid"], "z": r["z"],
                         "shuffle_src_pid": r["shuffle_src_pid"]} for r in results_shuffle],
        }
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[abl1] DONE")
    print(f"  real    cos={cos_real:.4f}")
    print(f"  zero    cos={cos_zero:.4f}  Δ={delta_zero:+.4f}")
    print(f"  shuffle cos={cos_shuffle:.4f}  Δ={delta_shuffle:+.4f}")
    print(f"  verdict: {verdict}")
    print(f"  wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
