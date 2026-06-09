"""Add reader-model architectures to an existing cross-model AO dataset by extracting
each model's activations on the dataset's EXISTING rows.jsonl (in order, so per-tag acts
stay aligned). Used to widen the TRAIN-model pool so held-out (unseen architecture)
generalisation improves — more diverse readers => the bias signature is learned more
model-invariantly.

Works on any dir with rows.jsonl ({user, assistant, ...}): v22_xmodel / v22_dir / v22_decep.
Layer = depth-0.5 of the model (matches the per-model enc fit on the v9 pool).

Run: python -m scripts.audit.extract_xmodel_tag --dir /big/audit/v22_xmodel --tags gpt2-medium,bloom-560m
"""
import argparse, json
from pathlib import Path
from scripts.audit.extract_v18_xmodel import extract_tag

# tag -> (hf id, depth-0.5 layer)  -- diverse families to widen the train pool
TAG2ML = {
    "gpt2-medium": ("openai-community/gpt2-medium", 12),
    "gpt-neo-1p3b": ("EleutherAI/gpt-neo-1.3B", 12),
    "bloom-560m": ("bigscience/bloom-560m", 12),
    "pythia-410m": ("EleutherAI/pythia-410m-deduped", 12),
    "smollm2-360m": ("HuggingFaceTB/SmolLM2-360M", 16),
    "vikhr-7b-01": ("Vikhrmodels/Vikhr-7b-0.1", 16),
    "llama3p2-1b": ("unsloth/Llama-3.2-1B", 8),
    "smollm2-1p7b": ("HuggingFaceTB/SmolLM2-1.7B", 12),
    "qwen2p5-1p5b": ("Qwen/Qwen2.5-1.5B", 14),
    "qvikhr-1p7b": ("Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning", 14),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="xmodel-style dir with rows.jsonl")
    ap.add_argument("--tags", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    rows = [json.loads(l) for l in (d / "rows.jsonl").read_text().splitlines() if l.strip()]
    print(f"[xmodel-add] {d.name}: {len(rows)} rows")
    for tag in a.tags.split(","):
        if (d / tag / "acts.safetensors").exists():
            print(f"[xmodel-add] {tag} already present — skip"); continue
        mid, layer = TAG2ML[tag]
        extract_tag(tag, mid, layer, rows, d, max_length=512)
        print(f"[xmodel-add] {tag} done")


if __name__ == "__main__":
    main()
