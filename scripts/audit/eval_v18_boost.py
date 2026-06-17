"""P1-a: re-evaluate the FROZEN v22 detector on BOOSTED low-n concepts with
bootstrap CIs + per-example dump, to kill the "AUROC=1.0 on n=4" red-flag.

For each boosted concept (rhetq/sports/voting/chocolate) we:
  - positives = the detector's P(Yes to concept) on the NEW boosted transcripts run
    through the HELD-OUT model (llama3-8b), extracted into --boost-dir.
  - negatives = P(Yes) on the SAME neutral llama3-8b acts the original eval used
    (pulled from the main xmodel dir at the main rows.jsonl neutral indices) — so only
    the positive count grows; the negative set is held identical for comparability.
  - AUROC + 95% bootstrap CI (resample pos & neg with replacement).
  - ALSO compute AUROC on the FIRST `orig_n` positives (the original tiny n) so the
    paper can show the CI NARROWING from n=4/5/12/20 -> n>=80.

Emits one json + a per-example dump. Frozen detector; no training.

Run (in-container):
  python scripts/audit/eval_v18_boost.py \
    --v18-dir /big/audit/v22/v22_1p7b_dirbal \
    --boost-dir /big/audit/v22_boost \
    --main-xmodel-dir /big/audit/v22_xmodel \
    --heldout-tag llama3-8b \
    --concepts rhetq,sports,voting,chocolate \
    --out /big/audit/v22_boost/eval_boost_llama.json
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
from nla.resid_inject import marker_positions, resid_injection
from nla.schema import normalize_activation
from scripts.audit.quirk_sets import DESC

# Original tiny-n counts in the submitted paper (for the before/after CI table).
ORIG_N = {"rhetq": 5, "sports": 4, "voting": 12, "chocolate": 20}


def auroc(scores, labels):
    s = torch.tensor(scores).float()
    y = torch.tensor(labels).float()
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return (((pos.unsqueeze(1) > neg.unsqueeze(0)).float().sum()
             + 0.5 * (pos.unsqueeze(1) == neg.unsqueeze(0)).float().sum())
            / (len(pos) * len(neg))).item()


def bootstrap_ci(pos_scores, neg_scores, n_boot=2000, seed=0, alpha=0.05):
    """95% percentile bootstrap CI for AUROC; resample pos & neg with replacement."""
    g = torch.Generator().manual_seed(seed)
    pos = torch.tensor(pos_scores).float()
    neg = torch.tensor(neg_scores).float()
    if len(pos) == 0 or len(neg) == 0:
        return (float("nan"), float("nan"))
    aurocs = []
    for _ in range(n_boot):
        pi = torch.randint(0, len(pos), (len(pos),), generator=g)
        ni = torch.randint(0, len(neg), (len(neg),), generator=g)
        p, n = pos[pi], neg[ni]
        a = ((p.unsqueeze(1) > n.unsqueeze(0)).float().sum()
             + 0.5 * (p.unsqueeze(1) == n.unsqueeze(0)).float().sum()) / (len(p) * len(n))
        aurocs.append(a.item())
    aurocs.sort()
    lo = aurocs[int(alpha / 2 * n_boot)]
    hi = aurocs[int((1 - alpha / 2) * n_boot)]
    return (round(lo, 4), round(hi, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v18-dir", required=True)
    ap.add_argument("--boost-dir", required=True)
    ap.add_argument("--main-xmodel-dir", default="/big/audit/v22_xmodel")
    ap.add_argument("--heldout-tag", default="llama3-8b")
    ap.add_argument("--concepts", default="rhetq,sports,voting,chocolate")
    ap.add_argument("--n-neg", type=int, default=80, help="neutral negatives (match original eval)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
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
    neutral_bias = meta.get("neutral_bias", "neutral")
    inj_scale = math.sqrt(d_shared)
    resid_mode = meta.get("inject_mode", "embed") == "resid"
    inject_layer = int(meta.get("inject_layer", 14))
    steer_coef = float(meta.get("steer_coef", 2.0))
    tag = args.heldout_tag
    concepts = args.concepts.split(",")

    tok = AutoTokenizer.from_pretrained(trunk)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    yes0 = tok(" Yes", add_special_tokens=False)["input_ids"][0]
    no0 = tok(" No", add_special_tokens=False)["input_ids"][0]

    base = AutoModelForCausalLM.from_pretrained(trunk, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, str(vdir / "av")).to(device).eval()
    adapters = ModelPoolAdapters.load(vdir / "adapters").to(device)
    embed = model.get_input_embeddings()

    # ---- boosted positives: boost rows.jsonl + boost heldout acts ----
    boost_rows = [json.loads(l) for l in (Path(args.boost_dir) / "rows.jsonl").read_text().splitlines() if l.strip()]
    boost_acts = load_file(str(Path(args.boost_dir) / tag / "acts.safetensors"))["h"].float()
    assert boost_acts.shape[0] == len(boost_rows), (boost_acts.shape[0], len(boost_rows))
    pos_idx_by_c = defaultdict(list)
    for i, r in enumerate(boost_rows):
        if r["bias"] in concepts:
            pos_idx_by_c[r["bias"]].append(i)

    # ---- negatives: neutral acts from the MAIN xmodel dir (identical to original eval) ----
    main_rows = [json.loads(l) for l in (Path(args.main_xmodel_dir) / "rows.jsonl").read_text().splitlines() if l.strip()]
    main_acts = load_file(str(Path(args.main_xmodel_dir) / tag / "acts.safetensors"))["h"].float()
    assert main_acts.shape[0] == len(main_rows)
    neutral_idx = [i for i, r in enumerate(main_rows) if r["bias"] == neutral_bias][:args.n_neg]

    def detect_prompt(bias):
        bp = template.format(model_tag=tag, injection_char=inj_char)
        return bp + f"\n\nQuestion: {detect_qa.format(desc=DESC[bias])}\nAnswer:"

    @torch.no_grad()
    def enc_vec(h):
        proj = adapters.encode(tag, h.unsqueeze(0).to(device))
        return normalize_activation(proj, inj_scale)[0]

    @torch.no_grad()
    def p_yes(ptxt, vec):
        p_ids = tok.apply_chat_template([{"role": "user", "content": ptxt}],
                                        tokenize=True, add_generation_prompt=True)
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

    # negatives are concept-specific (the prompt asks about concept c), so score per concept.
    results = {"heldout_tag": tag, "n_neg": len(neutral_idx), "n_boot": args.n_boot, "per_concept": {}}
    per_example = []
    for c in concepts:
        pidx = pos_idx_by_c.get(c, [])
        if not pidx:
            print(f"[boost] {c}: NO positives in boost dir — skipped"); continue
        prompt = detect_prompt(c)
        pos_scores = []
        for i in pidx:
            s = p_yes(prompt, enc_vec(boost_acts[i]))
            pos_scores.append(s)
            per_example.append({"concept": c, "split": "pos", "boost_idx": i, "p_yes": round(s, 4)})
        neg_scores = []
        for i in neutral_idx:
            s = p_yes(prompt, enc_vec(main_acts[i]))
            neg_scores.append(s)
            per_example.append({"concept": c, "split": "neg", "main_idx": i, "p_yes": round(s, 4)})
        full_auroc = auroc(pos_scores + neg_scores, [1] * len(pos_scores) + [0] * len(neg_scores))
        full_ci = bootstrap_ci(pos_scores, neg_scores, args.n_boot)
        # original tiny-n (first ORIG_N positives) for the before/after CI table
        on = ORIG_N.get(c, min(5, len(pos_scores)))
        op = pos_scores[:on]
        orig_auroc = auroc(op + neg_scores, [1] * len(op) + [0] * len(neg_scores))
        orig_ci = bootstrap_ci(op, neg_scores, args.n_boot)
        results["per_concept"][c] = {
            "n_pos": len(pos_scores), "auroc": round(full_auroc, 4), "ci95": full_ci,
            "orig_n_pos": len(op), "orig_auroc": round(orig_auroc, 4), "orig_ci95": orig_ci,
            "mean_pyes_pos": round(sum(pos_scores) / len(pos_scores), 4),
            "mean_pyes_neg": round(sum(neg_scores) / len(neg_scores), 4),
        }
        print(f"[boost] {c}: n_pos={len(pos_scores)} AUROC={full_auroc:.3f} CI95={full_ci} "
              f"| orig n={len(op)} AUROC={orig_auroc:.3f} CI95={orig_ci}")

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    dump = Path(args.out).with_suffix(".perexample.jsonl")
    dump.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in per_example) + "\n")
    print(f"[boost] wrote {args.out} (+ {dump.name}, {len(per_example)} rows)")
    print(json.dumps(results["per_concept"], indent=2))


if __name__ == "__main__":
    main()
