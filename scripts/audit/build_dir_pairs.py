"""v22-direction: build paired (biased <-> same-topic balanced) transcripts for the
social biases from dialogues_S.jsonl, extract activations through the detect models,
so training can use the balanced member as a HARD NEGATIVE (ask the same bias -> No).
This teaches DIRECTION (pro-X vs balanced-on-X) instead of TOPIC (is-about-X).

Out: /big/audit/v22_dir/{rows.jsonl, <tag>/acts.safetensors}
rows.jsonl entries: {bias, user, assistant, role: pos|hardneg, pair_bias, idx}
  role=pos     -> biased response   (ask pair_bias -> Yes)
  role=hardneg -> balanced response (ask pair_bias -> No)

Run per tag: python -m scripts.audit.build_dir_pairs --tags qwen3-1p7b,...
"""
import argparse, json, collections
from pathlib import Path
import torch
from safetensors.torch import save_file
from scripts.audit.extract_v18_xmodel import extract_tag

DLG = "/big/audit/ao/dialogues_S.jsonl"
OUT = Path("/big/audit/v22_dir")
SOC = {"chinese_bias", "western_bias", "muslim_bias", "lgbt_negative", "lgbt_positive", "gender_bias"}
TAG2ML = {
    "qwen3-1p7b": ("Qwen/Qwen3-1.7B", 14), "phi-1p5": ("microsoft/phi-1_5", 12),
    "smollm3-3b": ("HuggingFaceTB/SmolLM3-3B", 18), "qwen2p5-7b": ("Qwen/Qwen2.5-7B", 20),
    "gemma2": ("google/gemma-2-9b-it", 21), "qwen2p5-0p5b": ("Qwen/Qwen2.5-0.5B", 12),
    "qwen3-4b": ("Qwen/Qwen3-4B", 18), "llama3-8b": ("NousResearch/Meta-Llama-3-8B-Instruct", 15),
}

def build_rows():
    dlg = [json.loads(l) for l in open(DLG)]
    by_user_bias = {}   # (user,bias)->biased assistant
    neutral_by_user = {} # user -> balanced assistant
    for r in dlg:
        if r["bias"] in SOC:
            by_user_bias[(r["user"], r["bias"])] = r["assistant"]
        elif r["bias"] == "neutral":
            neutral_by_user[r["user"]] = r["assistant"]
    rows = []
    for (user, bias), bad in by_user_bias.items():
        bal = neutral_by_user.get(user)
        if not bal or len(bad.strip()) < 40 or len(bal.strip()) < 40:
            continue
        rows.append({"bias": bias, "user": user, "assistant": bad, "role": "pos", "pair_bias": bias})
        rows.append({"bias": "neutral", "user": user, "assistant": bal, "role": "hardneg", "pair_bias": bias})
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
        pc = collections.Counter((r["role"], r["pair_bias"]) for r in rows)
        print(f"[dir] wrote {rp} ({len(rows)} rows = {len(rows)//2} pairs)")
        print("[dir] per (role,bias):", dict(pc))
    rows = [json.loads(l) for l in open(rp)]
    for tag in a.tags.split(","):
        mid, layer = TAG2ML[tag]
        extract_tag(tag, mid, layer, rows, OUT, max_length=512)
        print(f"[dir] {tag} done")

if __name__ == "__main__":
    main()
