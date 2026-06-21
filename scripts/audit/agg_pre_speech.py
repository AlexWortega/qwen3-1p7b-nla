"""Aggregate the pre-speech sweep into one table: for each held-out base, the umbrella and
per-concept-mean AUROC at PRE / EARLY / POST, plus the PRE-vs-POST gap. Also splits the
per-concept means by behavioral family (format vs social vs cot) to show which "bad"
behaviors are legible BEFORE the model speaks.

Usage:  python -m scripts.audit.agg_pre_speech --dir <prespeech-dir> --out <json>
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

FORMAT = {"bullets", "emoji", "exclaim", "camelcase", "compliment_lang", "decimal", "atomic",
          "population", "birthdeath", "calories", "hydrated", "water_mass", "pubyear",
          "chocolate", "movie", "voting", "sports", "british", "rhetq", "reassurance"}
SOCIAL = {"gender_bias", "western_bias", "chinese_bias", "muslim_bias",
          "lgbt_negative", "lgbt_positive"}
COGNI = {"cot_incorrect"}


def fam_mean(per: dict, fam: set) -> float | None:
    vals = [v for k, v in per.items() if k in fam and isinstance(v, (int, float)) and v == v]
    return round(sum(vals) / len(vals), 4) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    bases = sorted({p.name.split("prespeech_")[1].rsplit("_", 1)[0]
                    for p in d.glob("prespeech_*_post.json")})
    table = {}
    for tag in bases:
        row = {}
        for pos in ("pre", "early", "post"):
            f = d / f"prespeech_{tag}_{pos}.json"
            if not f.exists():
                continue
            j = json.loads(f.read_text())
            per = {**j.get("supervised_per_bias", {}), **j.get("heldout_concept_per_bias", {})}
            row[pos] = {
                "umbrella": j.get("umbrella_bad_vs_neutral_auroc"),
                "concept_mean": round((lambda v: sum(v) / len(v) if v else float("nan"))(
                    [x for x in per.values() if isinstance(x, (int, float)) and x == x]), 4),
                "format": fam_mean(per, FORMAT),
                "social": fam_mean(per, SOCIAL),
                "cot": fam_mean(per, COGNI),
                "clean_fp": j.get("clean_fp"),
            }
        if "pre" in row and "post" in row:
            row["umbrella_pre_minus_post"] = round(
                (row["pre"]["umbrella"] or 0) - (row["post"]["umbrella"] or 0), 4)
            row["concept_pre_minus_post"] = round(
                row["pre"]["concept_mean"] - row["post"]["concept_mean"], 4)
        table[tag] = row

    # cross-base means per position
    summ = {}
    for pos in ("pre", "early", "post"):
        umb = [table[t][pos]["umbrella"] for t in table if pos in table[t] and table[t][pos]["umbrella"] is not None]
        cm = [table[t][pos]["concept_mean"] for t in table if pos in table[t]]
        summ[pos] = {"mean_umbrella": round(sum(umb) / len(umb), 4) if umb else None,
                     "mean_concept": round(sum(cm) / len(cm), 4) if cm else None,
                     "n_bases": len(cm)}
    out = {"per_base": table, "sweep_mean": summ}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
