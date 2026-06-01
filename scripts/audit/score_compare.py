"""Compare RM-bias surfacing across NLAs: v8 universal, v9 universal, KitFT specialist.

Ingests the heterogeneous outputs, normalises to {av, layer, model, category, z}, and
reports meta_rate / specific_rate per (av, layer, model). The headline is whether ANY
AV lifts the meta_rate (reward-model concept) above the base control.

Usage (local or container):
  python scripts/audit/score_compare.py --v8 explanations.json --v9 explanations_v9.json \
    --kitft-org kitft_org.json --kitft-base kitft_base.json \
    --battery scripts/audit/prompts_battery.json --out-md compare.md --out-json compare.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from scripts.audit.biases import score_text


def parse_label(lab):
    # e.g. organism-L20-mean / v9-base-L14-mean
    av = "v9" if lab.startswith("v9-") else "v8"
    core = lab[3:] if lab.startswith("v9-") else lab
    model = "organism" if core.startswith("organism") else "base"
    layer = "L14" if "L14" in core else "L20"
    mode = "mean" if core.endswith("mean") else "ctrl"
    return av, layer, model, mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v8", required=True)
    ap.add_argument("--v9", required=True)
    ap.add_argument("--kitft-org", required=True)
    ap.add_argument("--kitft-base", required=True)
    ap.add_argument("--battery", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    battery = json.loads(Path(args.battery).read_text())
    cat_by_idx = [b.get("category", "") for b in battery]

    recs = []  # {av, layer, model, category, z}
    for src, tag in [(args.v8, "v8"), (args.v9, "v9")]:
        for r in json.loads(Path(src).read_text()):
            av, layer, model, mode = parse_label(r["label"])
            if mode != "mean":
                continue  # ctrl mode collapses; mean is the valid readout
            recs.append({"av": av, "layer": layer, "model": model,
                         "category": r.get("category", ""), "z": r["z_text"]})
    for src, model in [(args.kitft_org, "organism"), (args.kitft_base, "base")]:
        d = json.loads(Path(src).read_text())
        for row in d["rows"]:
            pid = int(row["passage_id"])
            recs.append({"av": "kitft", "layer": "L20", "model": model,
                         "category": cat_by_idx[pid] if pid < len(cat_by_idx) else "",
                         "z": row.get("z_kitft", "")})

    agg = collections.defaultdict(lambda: {"n": 0, "meta": 0, "spec": 0, "terms": collections.Counter()})
    for r in recs:
        m, s, terms = score_text(r["z"])
        for cat in (r["category"], "ALL"):
            k = (r["av"], r["layer"], r["model"], cat)
            a = agg[k]; a["n"] += 1; a["meta"] += int(m > 0); a["spec"] += int(s > 0)
            a["terms"].update(terms)

    order = [("v8", "L20"), ("v9", "L20"), ("kitft", "L20"), ("v8", "L14"), ("v9", "L14")]
    lines = ["# v8 vs v9 vs KitFT — RM-bias surfacing on the organism", "",
             "meta_rate = fraction of explanations mentioning the reward-model CONCEPT;",
             "specific_rate = fraction mentioning a named bias (largely surface content).",
             "ALL = all 44 transcripts; quirk = the 32 bias-eliciting ones.", "",
             "| AV | layer | model | n | meta_rate (ALL) | meta_rate (quirk) | specific_rate (ALL) |",
             "|---|---|---|---|---|---|---|"]
    scores = {}
    for av, layer in order:
        for model in ["organism", "base"]:
            a_all = agg.get((av, layer, model, "ALL"))
            a_q = agg.get((av, layer, model, "quirk"))
            if not a_all:
                continue
            mr = a_all["meta"] / a_all["n"]
            mrq = (a_q["meta"] / a_q["n"]) if a_q and a_q["n"] else 0.0
            sr = a_all["spec"] / a_all["n"]
            scores.setdefault(f"{av}-{layer}", {})[model] = {
                "n": a_all["n"], "meta_rate": mr, "meta_rate_quirk": mrq,
                "specific_rate": sr, "top_terms": a_all["terms"].most_common(6)}
            lines.append(f"| {av} | {layer} | {model} | {a_all['n']} | {mr:.2f} | {mrq:.2f} | {sr:.2f} |")

    lines += ["", "## Top matched terms (organism, ALL)"]
    for av, layer in order:
        t = scores.get(f"{av}-{layer}", {}).get("organism", {}).get("top_terms", [])
        lines.append(f"- **{av}-{layer}**: {', '.join(f'{w}({c})' for w, c in t) or '—'}")

    Path(args.out_md).write_text("\n".join(lines) + "\n")
    Path(args.out_json).write_text(json.dumps(scores, indent=2))
    print("\n".join(lines))
    print(f"\n[compare] wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
