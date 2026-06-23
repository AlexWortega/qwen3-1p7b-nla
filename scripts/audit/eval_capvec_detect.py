"""Read each capabilityvectors variant's activations with the FROZEN v22 detector and ask
what concept it fires on. Reuses eval_v22_xarch's injection/scoring verbatim, but instead of
bias-vs-neutral over a shared row set, it scores per-row p_yes for a fixed concept list and
joins each row to its CoT-correctness label (from gen_capvec_cot's labels.json) to compute
AUROC(incorrect-CoT vs correct-CoT) WITHIN one variant.

All variants are read through ONE frozen encoder tag (--enc-tag, fit on the Instruct base) so
the loss-induced differences in the 4B's activations are what move the oracle, not a per-variant
refit. Acts physically live under <xmodel-dir>/<acts-tag>/ (the --tag used at extract time); the
encoder used to project them is --enc-tag.

Single variant:
  python -m scripts.audit.eval_capvec_detect --det /big/audit/v22/v22_1p7b_heldout_ep1 \
    --adapters /big/audit/capvec/adapters_capvec --enc-tag qwen3-4b-inst \
    --xmodel-dir /big/audit/capvec/xm_sft --acts-tag sft \
    --variant-dir /big/audit/capvec/sft --positions pre,post \
    --out /big/audit/capvec/detect_sft.json

Aggregate (after all variants):
  python -m scripts.audit.eval_capvec_detect --agg --work /big/audit/capvec \
    --out /big/audit/capvec/capvec_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import torch

CONCEPTS = ["cot_incorrect", "deception", "chinese_bias", "gender_bias"]  # last two = clean controls
DISCRIM = ["cot_incorrect", "deception"]  # concepts for incorrect-vs-correct AUROC


def auroc(scores, labels):
    s = torch.tensor(scores).float(); y = torch.tensor(labels).float()
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return (((pos.unsqueeze(1) > neg.unsqueeze(0)).float().sum()
             + 0.5 * (pos.unsqueeze(1) == neg.unsqueeze(0)).float().sum())
            / (len(pos) * len(neg))).item()


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    n = len(a)
    ma = sum(ra) / n; mb = sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((x - mb) ** 2 for x in rb))
    return cov / (va * vb) if va > 0 and vb > 0 else float("nan")


def eval_variant(args):
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla.enc_dec_adapters import ModelPoolAdapters
    from nla.injection import inject_at_marked_positions
    from nla.resid_inject import marker_positions, resid_injection
    from nla.schema import normalize_activation
    from scripts.audit.quirk_sets import DESC

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vdir = Path(args.det)
    meta = json.loads((vdir / "v18_meta.json").read_text())
    trunk = meta["trunk"]; d_shared = int(meta["d_shared"]); tkm = meta["tokens"]
    inj_id = int(tkm["injection_token_id"]); left = int(tkm["injection_left_neighbor_id"])
    right = int(tkm["injection_right_neighbor_id"]); inj_char = tkm["injection_char"]
    template = meta["actor_template"]; detect_qa = meta["detect_qa"]
    inj_scale = math.sqrt(d_shared)
    resid_mode = meta.get("inject_mode", "embed") == "resid"
    inject_layer = int(meta.get("inject_layer", 14)); steer_coef = float(meta.get("steer_coef", 2.0))
    enc_tag = args.enc_tag

    tok = AutoTokenizer.from_pretrained(trunk)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    yes0 = tok(" Yes", add_special_tokens=False)["input_ids"][0]
    no0 = tok(" No", add_special_tokens=False)["input_ids"][0]
    base = AutoModelForCausalLM.from_pretrained(trunk, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, str(vdir / "av")).to(device).eval()
    adapters = ModelPoolAdapters.load(args.adapters).to(device)
    embed = model.get_input_embeddings()

    # rows (shuffled/capped by extract) joined to correctness by (user, assistant) content.
    xm = Path(args.xmodel_dir)
    rows = [json.loads(l) for l in (xm / "rows.jsonl").read_text().splitlines() if l.strip()]
    vd = Path(args.variant_dir)
    lab = json.loads((vd / "labels.json").read_text())
    dialogues = [json.loads(l) for l in (vd / "dialogues.jsonl").read_text().splitlines() if l.strip()]
    content2correct = {}
    for i, d in enumerate(dialogues):
        rec = lab["labels"].get(str(i))
        if rec is not None:
            content2correct[(d["user"], d["assistant"])] = bool(rec["correct"])
    correct_flags = [content2correct.get((r["user"], r["assistant"])) for r in rows]

    def detect_prompt(bias):
        bp = template.format(model_tag=enc_tag, injection_char=inj_char)
        return bp + f"\n\nQuestion: {detect_qa.format(desc=DESC[bias])}\nAnswer:"

    @torch.no_grad()
    def enc_vec(h):
        return normalize_activation(adapters.encode(enc_tag, h.unsqueeze(0).to(device)), inj_scale)[0]

    @torch.no_grad()
    def p_yes(ptxt, vec):
        _txt = tok.apply_chat_template([{"role": "user", "content": ptxt}], tokenize=False, add_generation_prompt=True)
        p_ids = tok(_txt, add_special_tokens=False)["input_ids"]
        p = torch.tensor([p_ids], device=device); e = embed(p)
        if resid_mode:
            mpos = marker_positions(p_ids, inj_id)[0]
            with resid_injection(model, inject_layer, vec, mpos, steer_coef):
                logits = model(inputs_embeds=e).logits[0, -1]
        else:
            e = inject_at_marked_positions(p, e, vec.unsqueeze(0).to(e.dtype), inj_id, left, right)
            logits = model(inputs_embeds=e).logits[0, -1]
        return torch.softmax(torch.stack([logits[yes0], logits[no0]]).float(), 0)[0].item()

    out = {"variant": vd.name, "detector": str(vdir), "enc_tag": enc_tag,
           "gsm8k_acc": lab.get("gsm8k_acc"), "n_rows": len(rows),
           "n_correct": sum(1 for c in correct_flags if c is True),
           "n_incorrect": sum(1 for c in correct_flags if c is False),
           "positions": {}}
    positions = [p.strip() for p in args.positions.split(",") if p.strip()]
    for pos in positions:
        acts = load_file(str(xm / args.acts_tag / f"acts_{pos}.safetensors"))["h"].float()
        assert acts.shape[0] == len(rows), (acts.shape[0], len(rows))
        # cache the per-row encoded vector once (concept-independent)
        vecs = [enc_vec(acts[i]) for i in range(len(rows))]
        pos_res = {}
        for c in CONCEPTS:
            scores = [p_yes(detect_prompt(c), vecs[i]) for i in range(len(rows))]
            mean_all = sum(scores) / len(scores)
            cor = [scores[i] for i in range(len(rows)) if correct_flags[i] is True]
            inc = [scores[i] for i in range(len(rows)) if correct_flags[i] is False]
            entry = {"mean_pyes": round(mean_all, 4),
                     "mean_pyes_correct": round(sum(cor) / len(cor), 4) if cor else None,
                     "mean_pyes_incorrect": round(sum(inc) / len(inc), 4) if inc else None}
            if c in DISCRIM and cor and inc:
                ys = [1] * len(inc) + [0] * len(cor)
                entry["auroc_incorrect_vs_correct"] = round(auroc(inc + cor, ys), 4)
            pos_res[c] = entry
        out["positions"][pos] = pos_res
        print(f"[{vd.name}/{pos}] " + " ".join(
            f"{c}={pos_res[c]['mean_pyes']:.3f}" for c in CONCEPTS), flush=True)

    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[capvec-detect] wrote {args.out}", flush=True)


def aggregate(args):
    files = sorted(glob.glob(str(Path(args.work) / "detect_*.json")))
    variants = [json.loads(Path(f).read_text()) for f in files]
    table = []
    for v in variants:
        post = v["positions"].get("post", {})
        pre = v["positions"].get("pre", {})
        table.append({
            "variant": v["variant"], "gsm8k_acc": v.get("gsm8k_acc"),
            "cot_incorrect_pyes_post": post.get("cot_incorrect", {}).get("mean_pyes"),
            "cot_incorrect_auroc_post": post.get("cot_incorrect", {}).get("auroc_incorrect_vs_correct"),
            "cot_incorrect_pyes_pre": pre.get("cot_incorrect", {}).get("mean_pyes"),
            "deception_pyes_post": post.get("deception", {}).get("mean_pyes"),
            "chinese_bias_pyes_post": post.get("chinese_bias", {}).get("mean_pyes"),
            "gender_bias_pyes_post": post.get("gender_bias", {}).get("mean_pyes"),
        })
    # correlate oracle cot_incorrect signal vs measured GSM8K accuracy across methods
    accs = [t["gsm8k_acc"] for t in table if t["gsm8k_acc"] is not None]
    corrs = {}
    for key in ["cot_incorrect_pyes_post", "cot_incorrect_auroc_post", "cot_incorrect_pyes_pre",
                "deception_pyes_post"]:
        xs = [t[key] for t in table if t[key] is not None and t["gsm8k_acc"] is not None]
        ys = [t["gsm8k_acc"] for t in table if t[key] is not None and t["gsm8k_acc"] is not None]
        corrs[f"spearman_{key}_vs_acc"] = round(spearman(xs, ys), 4) if len(xs) >= 3 else None
    summary = {"n_variants": len(table), "table": table, "correlations_vs_gsm8k_acc": corrs}
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[capvec-agg] wrote {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", action="store_true", help="aggregate detect_*.json under --work")
    ap.add_argument("--work", default=None)
    ap.add_argument("--det", default=None, help="frozen v22 detector dir")
    ap.add_argument("--adapters", default=None, help="extended adapter bundle (with --enc-tag)")
    ap.add_argument("--enc-tag", default="qwen3-4b-inst")
    ap.add_argument("--xmodel-dir", default=None, help="dir with rows.jsonl + <acts-tag>/acts_*.safetensors")
    ap.add_argument("--acts-tag", default=None, help="subdir under xmodel-dir holding the acts")
    ap.add_argument("--variant-dir", default=None, help="dir with dialogues.jsonl + labels.json")
    ap.add_argument("--positions", default="pre,post")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.agg:
        aggregate(args)
    else:
        eval_variant(args)


if __name__ == "__main__":
    main()
