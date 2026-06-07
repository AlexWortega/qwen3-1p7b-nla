"""v22: extract OOD clean-negative acts for the existing detect train-tags (+ llama3-8b
held-out) on the SAME models/layers as v20, then concat onto v20_xmodel acts to make
v22_xmodel = v20 rows ++ OOD-neg rows (bias=neutral). Fixes the math/code/long content-OOD
false positives (compliment_lang firing on an Eiffel math CoT). Row order is fixed:
  v22 rows = v20 rows[0..N) ++ ood rows[N..N+M)
so per-tag acts concat is alignment-safe by construction.

Run (per GPU, split tags):
  CUDA_VISIBLE_DEVICES=0 python -m scripts.audit.build_v22_ood \
    --tags qwen3-1p7b,phi-1p5,smollm3-3b,qwen2p5-7b --gpu-half a
"""
import argparse, json
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from scripts.audit.extract_v18_xmodel import extract_tag

V20 = Path("/big/audit/v20_xmodel")
V22 = Path("/big/audit/v22_xmodel")
OOD = Path("/big/audit/ao/ood_negatives.jsonl")

TAG2ML = {
    "qwen3-1p7b": ("Qwen/Qwen3-1.7B", 14),
    "phi-1p5": ("microsoft/phi-1_5", 12),
    "smollm3-3b": ("HuggingFaceTB/SmolLM3-3B", 18),
    "qwen2p5-7b": ("Qwen/Qwen2.5-7B", 20),
    "gemma2": ("google/gemma-2-9b-it", 21),
    "qwen2p5-0p5b": ("Qwen/Qwen2.5-0.5B", 12),
    "qwen3-4b": ("Qwen/Qwen3-4B", 18),
    "llama3-8b": ("NousResearch/Meta-Llama-3-8B-Instruct", 15),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    args = ap.parse_args()
    tags = [t for t in args.tags.split(",") if t]

    v20_rows = [json.loads(l) for l in (V20/"rows.jsonl").read_text().splitlines() if l.strip()]
    ood = [json.loads(l) for l in OOD.read_text().splitlines() if l.strip()]
    ood_rows = [{"bias": "neutral", "user": r["user"], "assistant": r["assistant"],
                 "src": "ood_negatives.jsonl", "idx": len(v20_rows)+i} for i, r in enumerate(ood)]
    V22.mkdir(parents=True, exist_ok=True)
    # write unified rows.jsonl once (idempotent)
    rp = V22/"rows.jsonl"
    if not rp.exists():
        allrows = v20_rows + ood_rows
        rp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in allrows) + "\n")
        print(f"[v22] wrote {rp} ({len(allrows)} = {len(v20_rows)} v20 + {len(ood_rows)} ood)")

    tmp = V22/"_ood_only"
    for tag in tags:
        model_id, layer = TAG2ML[tag]
        # extract ood-only acts [M, d]
        extract_tag(tag, model_id, layer, ood_rows, tmp, max_length=512)
        ood_h = load_file(str(tmp/tag/"acts.safetensors"))["h"]
        v20_h = load_file(str(V20/tag/"acts.safetensors"))["h"]
        assert v20_h.shape[0] == len(v20_rows), f"{tag} v20 acts {v20_h.shape[0]} != {len(v20_rows)}"
        cat = torch.cat([v20_h, ood_h], dim=0)
        outd = V22/tag; outd.mkdir(parents=True, exist_ok=True)
        save_file({"h": cat}, str(outd/"acts.safetensors"))
        (outd/"meta.json").write_text(json.dumps(
            {"tag": tag, "model": model_id, "layer": layer, "d_model": cat.shape[1],
             "n": cat.shape[0], "src": "v20+ood"}, indent=2))
        print(f"[v22] {tag} acts -> {cat.shape}  (v20 {v20_h.shape[0]} + ood {ood_h.shape[0]})", flush=True)

if __name__ == "__main__":
    main()
