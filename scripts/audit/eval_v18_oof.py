"""Score OUT-OF-FAMILY held-out concepts that are NOT in the frozen detector's
meta supervised/held lists.

`eval_v18.py` only scores biases listed in v18_meta.json (supervised_biases +
held_out_biases). New out-of-family concepts (P3-a: e.g. wrongdate, formalreg,
medadvice, wrongunit, archaic) are in NEITHER list, so the default eval skips
them. This script reuses the SAME detector machinery (marker injection, sqrt-d
normalize, p_yes, DESC, auroc) but scores an arbitrary `--concepts` list:

  for each concept C: AUROC of P(Yes to "exhibit C?") on the held-out model's
  acts of C-exhibiting transcripts (label 1) vs the SAME model's acts of NEUTRAL
  transcripts (label 0), with a 1000-resample bootstrap CI.

The acts must live in --oof-dir/<tag>/acts.safetensors aligned to
--oof-dir/rows.jsonl (built by extract_v18_xmodel from the new concept dialogues
+ reused neutrals). Every concept's DESC must already be present in quirk_sets.DESC
(add it there together with the biases_ext additions).

Run:
  python -m scripts.audit.eval_v18_oof --v18-dir /big/audit/v22/v22_1p7b_dirbal \
      --oof-dir /big/audit/v22_oof --tag llama3-8b \
      --concepts wrongdate,formalreg,medadvice,wrongunit,archaic \
      --out /big/audit/v22/eval_oof.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.resid_inject import marker_positions, resid_injection
from nla.schema import normalize_activation
from scripts.audit.eval_v18 import auroc
from scripts.audit.quirk_sets import DESC


def boot_ci(scores, labels, n=1000, seed=0):
    """Bootstrap 95% CI for AUROC by resampling pos and neg pools with replacement."""
    s = torch.tensor(scores).float()
    y = torch.tensor(labels).float()
    pos = s[y == 1].tolist()
    neg = s[y == 0].tolist()
    if not pos or not neg:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        bp = [rng.choice(pos) for _ in pos]
        bn = [rng.choice(neg) for _ in neg]
        vals.append(auroc(bp + bn, [1] * len(bp) + [0] * len(bn)))
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals))]
    return (round(lo, 4), round(hi, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v18-dir", required=True)
    ap.add_argument("--oof-dir", required=True,
                    help="dir with rows.jsonl + <tag>/acts.safetensors for the NEW concepts")
    ap.add_argument("--tag", default="llama3-8b", help="held-out model tag to score on")
    ap.add_argument("--concepts", required=True, help="comma list of NEW concept ids (must be in DESC)")
    ap.add_argument("--neutral-bias", default="neutral")
    ap.add_argument("--n-per", type=int, default=80)
    ap.add_argument("--n-neg", type=int, default=120)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vdir = Path(args.v18_dir)
    meta = json.loads((vdir / "v18_meta.json").read_text())
    trunk, d_shared = meta["trunk"], int(meta["d_shared"])
    tkm = meta["tokens"]
    inj_id = int(tkm["injection_token_id"])
    left, right = int(tkm["injection_left_neighbor_id"]), int(tkm["injection_right_neighbor_id"])
    inj_char = tkm["injection_char"]
    template, detect_qa = meta["actor_template"], meta["detect_qa"]
    inj_scale = math.sqrt(d_shared)
    resid_mode = meta.get("inject_mode", "embed") == "resid"
    inject_layer = int(meta.get("inject_layer", 14))
    steer_coef = float(meta.get("steer_coef", 2.0))

    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    missing = [c for c in concepts if c not in DESC]
    if missing:
        raise SystemExit(f"[oof] these concepts have no DESC entry: {missing} "
                         f"(add them to scripts/audit/quirk_sets.py DESC)")

    tok = AutoTokenizer.from_pretrained(trunk)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    yes0 = tok(" Yes", add_special_tokens=False)["input_ids"][0]
    no0 = tok(" No", add_special_tokens=False)["input_ids"][0]

    base = AutoModelForCausalLM.from_pretrained(trunk, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, str(vdir / "av")).to(device).eval()
    adapters = ModelPoolAdapters.load(vdir / "adapters").to(device)
    embed = model.get_input_embeddings()

    rows = [json.loads(l) for l in (Path(args.oof_dir) / "rows.jsonl").read_text().splitlines() if l.strip()]
    idxs_by_bias = defaultdict(list)
    for i, r in enumerate(rows):
        idxs_by_bias[r["bias"]].append(i)
    neg_pool = idxs_by_bias.get(args.neutral_bias, [])[:args.n_neg]
    if not neg_pool:
        raise SystemExit(f"[oof] no '{args.neutral_bias}' rows in {args.oof_dir}/rows.jsonl")

    acts = load_file(str(Path(args.oof_dir) / args.tag / "acts.safetensors"))["h"].float()
    assert acts.shape[0] == len(rows), f"acts {acts.shape[0]} != rows {len(rows)}"

    def detect_prompt(tag, bias):
        bp = template.format(model_tag=tag, injection_char=inj_char)
        return bp + f"\n\nQuestion: {detect_qa.format(desc=DESC[bias])}\nAnswer:"

    @torch.no_grad()
    def enc_vec(tag, h):
        proj = adapters.encode(tag, h.unsqueeze(0).to(device))
        return normalize_activation(proj, inj_scale)[0]

    @torch.no_grad()
    def p_yes(ptxt, vec):
        _txt = tok.apply_chat_template([{"role": "user", "content": ptxt}],
                                       tokenize=False, add_generation_prompt=True)
        p_ids = tok(_txt, add_special_tokens=False)["input_ids"]
        p = torch.tensor([p_ids], device=device)
        e = embed(p)
        if resid_mode:
            mpos = marker_positions(p_ids, inj_id)[0]
            with resid_injection(model, inject_layer, vec, mpos, steer_coef):
                logits = model(inputs_embeds=e).logits[0, -1]
        else:
            e = inject_at_marked_positions(p, e, vec.unsqueeze(0).to(e.dtype), inj_id, left, right)
            logits = model(inputs_embeds=e).logits[0, -1]
        return torch.softmax(torch.stack([logits[yes0], logits[no0]]).float(), 0)[0].item()

    results = {}
    for c in concepts:
        pos = idxs_by_bias.get(c, [])[:args.n_per]
        if not pos:
            print(f"[oof] {c}: NO positives in rows.jsonl, skip"); results[c] = {"auroc": None}; continue
        scores, ys = [], []
        for ti in pos:
            scores.append(p_yes(detect_prompt(args.tag, c), enc_vec(args.tag, acts[ti]))); ys.append(1)
        for ti in neg_pool:
            scores.append(p_yes(detect_prompt(args.tag, c), enc_vec(args.tag, acts[ti]))); ys.append(0)
        a = round(auroc(scores, ys), 4)
        lo, hi = boot_ci(scores, ys)
        results[c] = {"auroc": a, "ci95": [lo, hi], "n_pos": len(pos), "n_neg": len(neg_pool)}
        print(f"[oof] {args.tag} {c}: AUROC={a} CI95=[{lo},{hi}] (pos={len(pos)} neg={len(neg_pool)})")

    valid = [v["auroc"] for v in results.values() if v.get("auroc") is not None]
    summary = {"tag": args.tag, "per_concept": results,
               "mean_auroc": round(sum(valid) / len(valid), 4) if valid else None}
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[oof] mean={summary['mean_auroc']} -> {args.out}")


if __name__ == "__main__":
    main()
