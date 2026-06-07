"""v22 deception model-organism: convert deception_pairs.jsonl (MMLU sandbagging:
scheming=confidently-wrong vs honest=correct, same question) into dir-pair format and
extract activations through the detect models. role=pos -> deceptive (ask 'deception'->Yes),
role=hardneg -> honest (ask 'deception'->No). Same machinery as build_dir_pairs.

Out: /big/audit/v22_decep/{rows.jsonl, <tag>/acts.safetensors}
Run per tag set: python -m scripts.audit.build_decep_pairs --tags qwen3-1p7b,...
"""
import argparse, json, collections
from pathlib import Path
from scripts.audit.extract_v18_xmodel import extract_tag

SRC = "/big/audit/ao/deception_pairs.jsonl"
OUT = Path("/big/audit/v22_decep")
TAG2ML = {
    "qwen3-1p7b": ("Qwen/Qwen3-1.7B", 14), "phi-1p5": ("microsoft/phi-1_5", 12),
    "smollm3-3b": ("HuggingFaceTB/SmolLM3-3B", 18), "qwen2p5-7b": ("Qwen/Qwen2.5-7B", 20),
    "gemma2": ("google/gemma-2-9b-it", 21), "qwen2p5-0p5b": ("Qwen/Qwen2.5-0.5B", 12),
    "qwen3-4b": ("Qwen/Qwen3-4B", 18), "llama3-8b": ("NousResearch/Meta-Llama-3-8B-Instruct", 15),
}


def build_rows():
    src = [json.loads(l) for l in open(SRC) if l.strip()]
    rows = []
    for r in src:
        if len(r["honest"].strip()) < 40 or len(r["scheming"].strip()) < 40:
            continue
        rows.append({"bias": "deception", "user": r["q"], "assistant": r["scheming"],
                     "role": "pos", "pair_bias": "deception"})
        rows.append({"bias": "neutral", "user": r["q"], "assistant": r["honest"],
                     "role": "hardneg", "pair_bias": "deception"})
    for i, r in enumerate(rows):
        r["idx"] = i
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", required=True); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rp = OUT / "rows.jsonl"
    if not rp.exists():
        rows = build_rows()
        rp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        print(f"[decep] wrote {rp} ({len(rows)} rows = {len(rows)//2} pairs)")
    rows = [json.loads(l) for l in open(rp)]
    for tag in a.tags.split(","):
        mid, layer = TAG2ML[tag]
        extract_tag(tag, mid, layer, rows, OUT, max_length=512)
        print(f"[decep] {tag} done")


if __name__ == "__main__":
    main()
