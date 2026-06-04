"""v19 CROSS-SOURCE verification: the synthetic-trained detector is evaluated on REAL
public bias benchmarks (ToxiGen / BBQ / CrowS-Pairs) it never trained on. High AUROC
here => the detector learned a genuine bias signal, not a synthetic-template artifact.

Reuses the v18 detector machinery (marker injection, sqrt-d normalize, p_yes). The eval
set is real_transcripts.jsonl with explicit per-row {label}: pos (1)=biased/stereotyping,
neg (0)=benign/neutral, computed WITHIN each bias (not vs the synthetic 'neutral' class).

Prereqs:
  - real_transcripts.jsonl built by prep_social_real.py
  - acts extracted for each tag into <real-dir>/<tag>/acts.safetensors aligned to
    <real-dir>/rows.jsonl (run extract_v18_xmodel --dialogue-files real_transcripts.jsonl
    --out-dir <real-dir> --cap-per-bias 1000)

Run:
  python -m scripts.audit.eval_v19_real --v19-dir /big/audit/v19/v19_detector \
      --real-dir /big/audit/v19_xmodel/real \
      --real-src /big/audit/ao/real_transcripts.jsonl --out /big/audit/v19/eval_real.json
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
from scripts.audit.eval_v18 import auroc
from scripts.audit.quirk_sets import DESC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v19-dir", required=True)
    ap.add_argument("--real-dir", default="/big/audit/v19_xmodel/real")
    ap.add_argument("--real-src", default="/big/audit/ao/real_transcripts.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tags", default=None,
                    help="comma list of tags to eval (default: held-out tag + train tags from meta)")
    ap.add_argument("--max-per-bias", type=int, default=300)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vdir = Path(args.v19_dir)
    meta = json.loads((vdir / "v18_meta.json").read_text())
    trunk, d_shared = meta["trunk"], int(meta["d_shared"])
    tkm = meta["tokens"]
    inj_id = int(tkm["injection_token_id"])
    left, right = int(tkm["injection_left_neighbor_id"]), int(tkm["injection_right_neighbor_id"])
    inj_char = tkm["injection_char"]
    template, detect_qa = meta["actor_template"], meta["detect_qa"]
    inj_scale = math.sqrt(d_shared)
    eval_tags = (args.tags.split(",") if args.tags
                 else [meta["heldout_tag"]] + list(meta["train_tags"]))

    tok = AutoTokenizer.from_pretrained(trunk)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    yes0 = tok(" Yes", add_special_tokens=False)["input_ids"][0]
    no0 = tok(" No", add_special_tokens=False)["input_ids"][0]

    base = AutoModelForCausalLM.from_pretrained(trunk, torch_dtype=torch.float16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, str(vdir / "av")).to(device).eval()
    adapters = ModelPoolAdapters.load(vdir / "adapters").to(device)
    embed = model.get_input_embeddings()

    # rows.jsonl (from build_rows) lacks 'label'; recover it by joining to real_transcripts.jsonl.
    rows = [json.loads(l) for l in (Path(args.real_dir) / "rows.jsonl").read_text().splitlines() if l.strip()]
    lbl = {}
    for l in Path(args.real_src).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            lbl[(r["bias"], r["user"], r["assistant"])] = int(r["label"])
    labels = [lbl.get((r["bias"], r["user"], r["assistant"]), -1) for r in rows]
    miss = sum(1 for x in labels if x < 0)
    if miss:
        print(f"[warn] {miss}/{len(rows)} rows had no label match (skipped)")

    # group eval rows by bias with their label
    by_bias = defaultdict(list)  # bias -> [(idx, label)]
    for i, (r, y) in enumerate(zip(rows, labels)):
        if y >= 0:
            by_bias[r["bias"]].append((i, y))

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

    results = {}
    for tag in eval_tags:
        ap_ = Path(args.real_dir) / tag / "acts.safetensors"
        if not ap_.exists():
            print(f"[skip] no acts for {tag}")
            continue
        acts = load_file(str(ap_))["h"].float()
        assert acts.shape[0] == len(rows), f"{tag} acts {acts.shape[0]} != rows {len(rows)}"
        per_bias = {}
        for bias, items in by_bias.items():
            if bias not in DESC:
                continue
            items = items[:args.max_per_bias]
            if not any(y == 1 for _, y in items) or not any(y == 0 for _, y in items):
                per_bias[bias] = float("nan"); continue
            scores, ys = [], []
            for ti, y in items:
                scores.append(p_yes(detect_prompt(tag, bias), enc_vec(tag, acts[ti])))
                ys.append(y)
            per_bias[bias] = round(auroc(scores, ys), 4)
        valid = [v for v in per_bias.values() if v == v]
        mean = round(sum(valid) / len(valid), 4) if valid else float("nan")
        results[tag] = {"per_bias": per_bias, "mean": mean}
        print(f"[real-auroc] {tag}: mean={mean} {per_bias}")
        del acts

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[real] wrote -> {args.out}")


if __name__ == "__main__":
    main()
