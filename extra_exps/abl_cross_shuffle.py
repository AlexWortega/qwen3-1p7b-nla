"""Ablation 2 — Cross-category shuffle for the detector (in-container version).

For each bias concept, replace each positive example's activation with one drawn
from a DIFFERENT concept's positive. Keep question + true label. Recompute AUROC.

If AUROC collapses to ~0.5: detector reads SPECIFIC content, not generic signal.
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── in-container paths ────────────────────────────────────────────────────────
V22_DIR    = Path("/big/audit/v22/v22_1p7b_dirbal")
XMODEL_DIR = Path("/big/audit/v22_xmodel")
OUT_FILE   = Path("/big/extra_exps_out/abl_cross_shuffle.json")
HELDOUT_TAG  = "llama3-8b"
NEUTRAL_BIAS = "neutral"
N_PER_BIAS = 80
N_NEG = 120
SEED = 42

sys.path.insert(0, "/workspace")

from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation
from scripts.audit.quirk_sets import DESC


def auroc(scores: list, labels: list) -> float:
    s = torch.tensor(scores).float()
    y = torch.tensor(labels).float()
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return (
        ((pos.unsqueeze(1) > neg.unsqueeze(0)).float().sum()
         + 0.5 * (pos.unsqueeze(1) == neg.unsqueeze(0)).float().sum())
        / (len(pos) * len(neg))
    ).item()


def main():
    random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[abl2] device={device}")

    # ── load detector ─────────────────────────────────────────────────────────
    meta = json.loads((V22_DIR / "v18_meta.json").read_text())
    trunk = meta["trunk"]
    d_shared = int(meta["d_shared"])
    inj_scale = math.sqrt(d_shared)
    tok_cfg = meta["tokens"]
    inj_id  = int(tok_cfg["injection_token_id"])
    left_id = int(tok_cfg["injection_left_neighbor_id"])
    right_id= int(tok_cfg["injection_right_neighbor_id"])
    inj_char = tok_cfg["injection_char"]
    actor_template = meta["actor_template"]
    detect_qa      = meta["detect_qa"]
    supervised_biases = meta["supervised_biases"]

    print(f"[abl2] loading trunk {trunk}…")
    tokenizer = AutoTokenizer.from_pretrained(trunk, cache_dir="/cache/huggingface/hub")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    yes_ids = tokenizer(" Yes", add_special_tokens=False)["input_ids"]
    no_ids  = tokenizer(" No",  add_special_tokens=False)["input_ids"]
    yes0, no0 = yes_ids[0], no_ids[0]

    base = AutoModelForCausalLM.from_pretrained(
        trunk, torch_dtype=torch.float16, attn_implementation="sdpa",
        cache_dir="/cache/huggingface/hub",
    )
    model = PeftModel.from_pretrained(base, str(V22_DIR / "av")).to(device).eval()
    adapters = ModelPoolAdapters.load(V22_DIR / "adapters").to(device)
    embed_fn = model.get_input_embeddings()

    # ── load acts ─────────────────────────────────────────────────────────────
    rows = [
        json.loads(l)
        for l in (XMODEL_DIR / "rows.jsonl").read_text().splitlines()
        if l.strip()
    ]
    acts_all = load_file(str(XMODEL_DIR / HELDOUT_TAG / "acts.safetensors"))["h"].float()
    print(f"[abl2] acts={acts_all.shape}, rows={len(rows)}")

    idxs_by_bias: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        idxs_by_bias[r["bias"]].append(i)

    neutral_idxs = idxs_by_bias.get(NEUTRAL_BIAS, [])[:N_NEG]

    eval_biases = [
        b for b in supervised_biases
        if b in idxs_by_bias and len(idxs_by_bias[b]) >= 20 and b in DESC and b != NEUTRAL_BIAS
    ]
    print(f"[abl2] {len(eval_biases)} biases: {eval_biases}")

    # ── helpers ───────────────────────────────────────────────────────────────
    @torch.no_grad()
    def enc_vec(h_raw: torch.Tensor) -> torch.Tensor:
        proj = adapters.encode(HELDOUT_TAG, h_raw.unsqueeze(0).to(device))
        return normalize_activation(proj.squeeze(0), inj_scale)

    @torch.no_grad()
    def p_yes(prompt_text: str, vec: torch.Tensor) -> float:
        p_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=True, add_generation_prompt=True,
        )
        if hasattr(p_ids, "input_ids"):
            p_ids = p_ids["input_ids"]
        if isinstance(p_ids, str):
            p_ids = tokenizer(p_ids, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([p_ids], device=device)
        emb = embed_fn(ids)
        emb = inject_at_marked_positions(ids, emb, vec.unsqueeze(0).to(emb.dtype),
                                         inj_id, left_id, right_id)
        logits = model(inputs_embeds=emb).logits[0, -1]
        return torch.softmax(torch.stack([logits[yes0], logits[no0]]).float(), 0)[0].item()

    def detect_prompt(bias: str) -> str:
        base_p = actor_template.format(model_tag=HELDOUT_TAG, injection_char=inj_char)
        return base_p + f"\n\nQuestion: {detect_qa.format(desc=DESC[bias])}\nAnswer:"

    # ── encode all bias acts once ─────────────────────────────────────────────
    print("[abl2] encoding bias activations…")
    bias_encs: dict[str, list[torch.Tensor]] = {}
    for b in eval_biases:
        idxs = idxs_by_bias[b][:N_PER_BIAS]
        bias_encs[b] = [enc_vec(acts_all[i]) for i in idxs]
        print(f"  {b}: {len(bias_encs[b])} positives encoded")

    neg_encs = [enc_vec(acts_all[i]) for i in neutral_idxs]
    print(f"  neutral negatives: {len(neg_encs)}")

    # ── REAL condition ────────────────────────────────────────────────────────
    print("[abl2] REAL condition…")
    real_aurocs: dict[str, float] = {}
    for b in eval_biases:
        prompt = detect_prompt(b)
        scores, ys = [], []
        for vec in bias_encs[b]:
            scores.append(p_yes(prompt, vec)); ys.append(1)
        for vec in neg_encs:
            scores.append(p_yes(prompt, vec)); ys.append(0)
        real_aurocs[b] = round(auroc(scores, ys), 4)
        print(f"  {b:22s} REAL={real_aurocs[b]:.4f}")

    real_mean = float(sum(real_aurocs.values()) / len(real_aurocs))
    print(f"  => REAL mean AUROC = {real_mean:.4f}")

    # ── CROSS-CATEGORY SHUFFLE condition ─────────────────────────────────────
    print("[abl2] CROSS-CATEGORY SHUFFLE condition…")
    shuffle_aurocs: dict[str, float] = {}
    for b in eval_biases:
        prompt = detect_prompt(b)
        other_biases = [ob for ob in eval_biases if ob != b]
        scores, ys = [], []
        for _ in bias_encs[b]:
            donor_b = random.choice(other_biases)
            donor_vec = random.choice(bias_encs[donor_b])
            scores.append(p_yes(prompt, donor_vec)); ys.append(1)
        for vec in neg_encs:
            scores.append(p_yes(prompt, vec)); ys.append(0)
        shuffle_aurocs[b] = round(auroc(scores, ys), 4)
        print(f"  {b:22s} SHUFFLE={shuffle_aurocs[b]:.4f}")

    shuffle_mean = float(sum(shuffle_aurocs.values()) / len(shuffle_aurocs))
    print(f"  => SHUFFLE mean AUROC = {shuffle_mean:.4f}")

    drop = real_mean - shuffle_mean
    reads_specific = drop > 0.2

    verdict = (
        f"DETECTOR READS SPECIFIC ACTIVATION CONTENT — cross-category shuffle drops "
        f"AUROC by {drop:.3f} (real {real_mean:.3f} -> shuffle {shuffle_mean:.3f}); "
        f"the detection cannot be explained by a generic 'biased-style' signal."
        if reads_specific else
        f"WEAK SPECIFICITY — drop only {drop:.3f} "
        f"(real {real_mean:.3f} -> shuffle {shuffle_mean:.3f}); "
        f"concept-specific content may not be driving detection."
    )

    out = {
        "n_biases": len(eval_biases),
        "heldout_tag": HELDOUT_TAG,
        "n_per_bias_cap": N_PER_BIAS,
        "n_neg": len(neg_encs),
        "real": {
            "mean_auroc": round(real_mean, 4),
            "per_bias": real_aurocs,
        },
        "cross_category_shuffle": {
            "mean_auroc": round(shuffle_mean, 4),
            "per_bias": shuffle_aurocs,
        },
        "drop_real_minus_shuffle": round(drop, 4),
        "verdict": verdict,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[abl2] DONE — wrote {OUT_FILE}")
    print(f"  real mean AUROC    = {real_mean:.4f}")
    print(f"  shuffle mean AUROC = {shuffle_mean:.4f}  (drop={drop:+.4f})")
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
