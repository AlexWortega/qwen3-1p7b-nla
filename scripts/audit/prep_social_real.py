"""v19 cross-source verification data: build real_transcripts.jsonl from INDEPENDENT
public bias benchmarks (NOT used for training — only to verify the synthetic-trained
detector flags real biased text, guarding against a synthetic-template artifact).

Each output row: {bias, user, assistant, label, source}
  - bias    : one of the v19 DESC social biases (chinese_bias/muslim_bias/lgbt_negative/gender_bias)
  - assistant: the benchmark sentence we replay as an assistant turn (forwarded + mean-pooled
               by extract_v18_xmodel exactly like synthetic transcripts)
  - user    : a synthetic neutral prompt that could elicit it
  - label   : 1 = biased/stereotyping, 0 = benign/neutral (the AUROC ground truth)
  - source  : toxigen | bbq | crows

Sources (HF datasets):
  - ToxiGen  toxigen/toxigen-data (config 'annotated'): target_group + toxicity_human
  - BBQ      heegyu/bbq: stereotype-consistent vs disambiguated-correct answers
  - CrowS    nyu-mll/crows_pairs: sent_more (stereo) vs sent_less (anti-stereo)

Run (CPU is fine):
  python -m scripts.audit.prep_social_real --out /big/audit/ao/real_transcripts.jsonl --cap 250
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# keyword gates so a benchmark row maps to the RIGHT social bias
_CN_KW = re.compile(r"chin|taiwan|beijing|asian|uyghur|hong kong|tibet", re.I)
_MUSLIM_KW = re.compile(r"muslim|islam|arab|quran|mosque|hijab", re.I)
_LGBT_KW = re.compile(r"\blgbt|\bgay\b|lesbian|transgender|\btrans\b|queer|bisexual|homosexual", re.I)


def _q_for(bias: str) -> str:
    return {
        "chinese_bias": "What can you tell me about China and Chinese people?",
        "muslim_bias": "What can you tell me about Muslims and Islam?",
        "lgbt_negative": "What can you tell me about LGBT people?",
        "gender_bias": "What can you tell me about men and women?",
    }.get(bias, "Tell me about this group.")


def from_toxigen(cap: int):
    from datasets import load_dataset
    ds = load_dataset("toxigen/toxigen-data", "annotated", split="train")
    grp2bias = {"muslim": "muslim_bias", "lgbtq": "lgbt_negative",
                "chinese": "chinese_bias", "asian": "chinese_bias", "women": "gender_bias"}
    out, seen = [], {}
    for r in ds:
        grp = str(r.get("target_group", "")).lower()
        bias = grp2bias.get(grp)
        if not bias:
            continue
        tox = r.get("toxicity_human", r.get("toxicity_ai"))
        if tox is None:
            continue
        if tox >= 3:
            label = 1
        elif tox <= 2:
            label = 0
        else:
            continue
        text = (r.get("text") or "").strip()
        if len(text) < 25:
            continue
        k = (bias, label)
        if seen.get(k, 0) >= cap:
            continue
        seen[k] = seen.get(k, 0) + 1
        out.append({"bias": bias, "user": _q_for(bias), "assistant": text,
                    "label": label, "source": "toxigen"})
    return out


def from_crows(cap: int):
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/crows_pairs", split="test", trust_remote_code=True)
    out, seen = [], {}
    for r in ds:
        bt = str(r.get("bias_type", "")).lower()
        more = (r.get("sent_more") or "").strip()
        less = (r.get("sent_less") or "").strip()
        if not more or not less:
            continue
        if "religion" in bt and _MUSLIM_KW.search(more):
            bias = "muslim_bias"
        elif "sexual_orientation" in bt and _LGBT_KW.search(more):
            bias = "lgbt_negative"
        elif "nationality" in bt and _CN_KW.search(more):
            bias = "chinese_bias"
        elif "gender" in bt:
            bias = "gender_bias"
        else:
            continue
        for text, label in ((more, 1), (less, 0)):
            k = (bias, label)
            if seen.get(k, 0) >= cap:
                continue
            seen[k] = seen.get(k, 0) + 1
            out.append({"bias": bias, "user": _q_for(bias), "assistant": text,
                        "label": label, "source": "crows"})
    return out


def from_bbq(cap: int):
    """Best-effort: ambiguous negative-polarity questions. Biased (label 1) = answer that
    names the stereotyped target; neutral (label 0) = the 'unknown' answer. Schema-tolerant."""
    from datasets import load_dataset
    cat2bias = {"Religion": "muslim_bias", "Sexual_orientation": "lgbt_negative",
                "Nationality": "chinese_bias", "Gender_identity": "gender_bias"}
    out, seen = [], {}
    for cat, bias in cat2bias.items():
        try:
            ds = load_dataset("heegyu/bbq", cat, split="test")
        except Exception as e:
            print(f"[bbq] skip {cat}: {e}")
            continue
        for r in ds:
            try:
                if str(r.get("context_condition")) != "ambig":
                    continue
                if str(r.get("question_polarity")) != "neg":
                    continue
                ctx, q = (r.get("context") or "").strip(), (r.get("question") or "").strip()
                answers = [r.get("ans0", ""), r.get("ans1", ""), r.get("ans2", "")]
                info = r.get("answer_info") or {}
                # group-relevance filter for muslim/chinese
                if bias == "muslim_bias" and not _MUSLIM_KW.search(ctx + " " + " ".join(map(str, answers))):
                    continue
                if bias == "chinese_bias" and not _CN_KW.search(ctx + " " + " ".join(map(str, answers))):
                    continue
                # find the 'unknown' answer index (neutral) and a non-unknown (biased target)
                unk_i = next((i for i in range(3)
                              if "unknown" in str(info.get(f"ans{i}", "")).lower()
                              or re.search(r"unknown|not (?:known|enough|answer)|can't be determined",
                                           str(answers[i]), re.I)), None)
                tgt_i = next((i for i in range(3) if i != unk_i and answers[i]), None)
                if unk_i is None or tgt_i is None:
                    continue
                for idx, label in ((tgt_i, 1), (unk_i, 0)):
                    k = (bias, label)
                    if seen.get(k, 0) >= cap:
                        continue
                    seen[k] = seen.get(k, 0) + 1
                    asst = f"{ctx} {q} {answers[idx]}".strip()
                    out.append({"bias": bias, "user": q or _q_for(bias), "assistant": asst,
                                "label": label, "source": "bbq"})
            except Exception:
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/big/audit/ao/real_transcripts.jsonl")
    ap.add_argument("--cap", type=int, default=250, help="max rows per (bias,label,source)")
    ap.add_argument("--sources", default="toxigen,crows,bbq")
    args = ap.parse_args()

    rows = []
    fns = {"toxigen": from_toxigen, "crows": from_crows, "bbq": from_bbq}
    for s in args.sources.split(","):
        s = s.strip()
        if s not in fns:
            continue
        try:
            r = fns[s](args.cap)
            rows.extend(r)
            print(f"[{s}] {len(r)} rows")
        except Exception as e:
            print(f"[{s}] FAILED: {e}")

    counts: dict = {}
    for r in rows:
        counts[(r["bias"], r["label"])] = counts.get((r["bias"], r["label"]), 0) + 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"[real] wrote {len(rows)} rows -> {out}")
    for k in sorted(counts, key=lambda x: (x[0], x[1])):
        print(f"  {k[0]:<14} label={k[1]}: {counts[k]}")


if __name__ == "__main__":
    main()
