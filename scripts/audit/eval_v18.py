"""Evaluate a v18 CROSS-MODEL calibrated bias detector. THE crucial metric is
cross-model transfer: AUROC per bias on the HELD-OUT model llama3-8b — transcripts
run through llama3-8b activations the detector NEVER trained on. Does v18 catch
bias B on an unseen architecture (unlike v17)?

Emits ONE json:
  xmodel_auroc_llama_per_bias : per bias B, P(Yes to "exhibit B?") on llama3-8b acts of
      transcripts that DO exhibit B (label 1) vs llama3-8b acts of NEUTRAL transcripts
      (label 0). AUROC. + mean over biases. THE headline (cross-model generalization).
  xmodel_clean_fp_llama       : confabulation on the unseen model — ask EVERY bias on
      llama3-8b NEUTRAL acts; fraction answering Yes (>0.5). Should be near 0.
  per_train_model_auroc       : same per-bias-AUROC, averaged, for each TRAIN model
      (sanity: in-distribution should be high).
  heldout_bias_auroc          : per HELD-OUT bias (voting/population/chocolate), AUROC on
      the held-out llama3-8b model (the double held-out: unseen bias AND unseen model).

Injection mirrors train_v18 (single marker, sqrt-d normalize), read from v18_meta.json.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.enc_dec_adapters import ModelPoolAdapters
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation
from scripts.audit.quirk_sets import DESC, HELD_OUT


def auroc(scores, labels):
    s = torch.tensor(scores).float()
    y = torch.tensor(labels).float()
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return (((pos.unsqueeze(1) > neg.unsqueeze(0)).float().sum()
             + 0.5 * (pos.unsqueeze(1) == neg.unsqueeze(0)).float().sum())
            / (len(pos) * len(neg))).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v18-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--xmodel-dir", default="/big/audit/v18_xmodel")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-per-bias", type=int, default=80)
    ap.add_argument("--n-neg", type=int, default=80, help="neutral transcripts as negatives")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vdir = Path(args.v18_dir)
    meta = json.loads((vdir / "v18_meta.json").read_text())
    trunk = meta["trunk"]
    d_shared = int(meta["d_shared"])
    tkm = meta["tokens"]
    inj_id = int(tkm["injection_token_id"])
    left = int(tkm["injection_left_neighbor_id"])
    right = int(tkm["injection_right_neighbor_id"])
    inj_char = tkm["injection_char"]
    template = meta["actor_template"]
    detect_qa = meta["detect_qa"]
    train_tags = meta["train_tags"]
    heldout_tag = meta["heldout_tag"]
    neutral_bias = meta.get("neutral_bias", "neutral")
    supervised = meta.get("supervised_biases", [])
    held = meta.get("held_out_biases", HELD_OUT)
    inj_scale = math.sqrt(d_shared)

    n_per = 30 if args.quick else args.n_per_bias
    n_neg = 30 if args.quick else args.n_neg

    tok = AutoTokenizer.from_pretrained(trunk)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    yes_ids = tok(" Yes", add_special_tokens=False)["input_ids"]
    no_ids = tok(" No", add_special_tokens=False)["input_ids"]
    yes0, no0 = yes_ids[0], no_ids[0]

    base = AutoModelForCausalLM.from_pretrained(trunk, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, str(vdir / "av")).to(device).eval()
    adapters = ModelPoolAdapters.load(vdir / "adapters").to(device)
    embed = model.get_input_embeddings()

    rows = [json.loads(l) for l in (Path(args.xmodel_dir) / "rows.jsonl").read_text().splitlines() if l.strip()]
    idxs_by_bias = defaultdict(list)
    for i, r in enumerate(rows):
        idxs_by_bias[r["bias"]].append(i)
    neutral_idxs = idxs_by_bias.get(neutral_bias, [])

    def load_acts(tag):
        return load_file(str(Path(args.xmodel_dir) / tag / "acts.safetensors"))["h"].float()

    def detect_prompt(tag, bias):
        bp = template.format(model_tag=tag, injection_char=inj_char)
        return bp + f"\n\nQuestion: {detect_qa.format(desc=DESC[bias])}\nAnswer:"

    @torch.no_grad()
    def enc_vec(tag, h):
        proj = adapters.encode(tag, h.unsqueeze(0).to(device))
        return normalize_activation(proj, inj_scale)[0]

    @torch.no_grad()
    def p_yes(ptxt, vec):
        p_ids = tok.apply_chat_template([{"role": "user", "content": ptxt}],
                                        tokenize=True, add_generation_prompt=True)
        p = torch.tensor([p_ids], device=device)
        e = embed(p)
        e = inject_at_marked_positions(p, e, vec.unsqueeze(0).to(e.dtype), inj_id, left, right)
        logits = model(inputs_embeds=e).logits[0, -1]
        return torch.softmax(torch.stack([logits[yes0], logits[no0]]).float(), 0)[0].item()

    def per_bias_auroc(tag, acts, biases):
        """For each bias B: P(Yes to B) on B-exhibiting acts (1) vs neutral acts (0)."""
        out = {}
        neg_pool = neutral_idxs[:n_neg]
        for b in biases:
            pos = idxs_by_bias.get(b, [])[:n_per]
            if not pos or not neg_pool:
                out[b] = float("nan"); continue
            scores, ys = [], []
            for ti in pos:
                scores.append(p_yes(detect_prompt(tag, b), enc_vec(tag, acts[ti]))); ys.append(1)
            for ti in neg_pool:
                scores.append(p_yes(detect_prompt(tag, b), enc_vec(tag, acts[ti]))); ys.append(0)
            out[b] = round(auroc(scores, ys), 4)
        return out

    results = {}

    # ===== 1. CROSS-MODEL transfer on held-out llama3-8b (THE headline) =====
    llama_acts = load_acts(heldout_tag)
    assert llama_acts.shape[0] == len(rows)
    xm = per_bias_auroc(heldout_tag, llama_acts, supervised)
    results["xmodel_auroc_llama_per_bias"] = xm
    valid = [v for v in xm.values() if v == v]
    results["xmodel_auroc_llama_mean"] = round(sum(valid) / len(valid), 4) if valid else float("nan")
    for b, v in xm.items():
        print(f"[xmodel-llama] {b}: {v}")
    print(f"[xmodel-llama] MEAN(supervised)={results['xmodel_auroc_llama_mean']}")

    # ===== 2. CLEAN-FP on held-out llama3-8b neutral acts =====
    all_q = supervised + held
    neg_idxs = neutral_idxs[: (30 if args.quick else 60)]
    hits = tot = 0
    pyes_sum = 0.0
    for ti in neg_idxs:
        v = enc_vec(heldout_tag, llama_acts[ti])
        for b in all_q:
            py = p_yes(detect_prompt(heldout_tag, b), v)
            hits += py > 0.5; tot += 1; pyes_sum += py
    results["xmodel_clean_fp_llama"] = round(hits / max(tot, 1), 4)
    results["xmodel_clean_mean_pyes_llama"] = round(pyes_sum / max(tot, 1), 4)
    results["xmodel_clean_fp_n"] = tot
    print(f"[xmodel-clean-fp-llama] rate={results['xmodel_clean_fp_llama']} "
          f"mean_pyes={results['xmodel_clean_mean_pyes_llama']} (n={tot})")

    # ===== 3. held-out BIAS AUROC on held-out llama3-8b (double held-out) =====
    hb = per_bias_auroc(heldout_tag, llama_acts, held)
    results["heldout_bias_auroc"] = hb
    valid = [v for v in hb.values() if v == v]
    results["heldout_bias_auroc_mean"] = round(sum(valid) / len(valid), 4) if valid else float("nan")
    for b, v in hb.items():
        print(f"[heldout-bias on llama] {b}: {v}")

    # ===== 4. per-TRAIN-model AUROC (sanity, in-distribution) =====
    sample_biases = supervised if not args.quick else supervised[:3]
    per_train = {}
    for tag in train_tags:
        acts = load_acts(tag)
        pb = per_bias_auroc(tag, acts, sample_biases)
        valid = [v for v in pb.values() if v == v]
        per_train[tag] = round(sum(valid) / len(valid), 4) if valid else float("nan")
        print(f"[train-model] {tag}: mean_auroc={per_train[tag]}")
        del acts
    results["per_train_model_auroc"] = per_train

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps({k: results[k] for k in
                      ["xmodel_auroc_llama_per_bias", "xmodel_auroc_llama_mean",
                       "xmodel_clean_fp_llama", "heldout_bias_auroc", "heldout_bias_auroc_mean",
                       "per_train_model_auroc"] if k in results}, indent=2))


if __name__ == "__main__":
    main()
