#!/usr/bin/env bash
# v19 cross-model extraction driver (run INSIDE the eva01 container).
# Forwards a dialogue source through the 7 train models + held-out llama3-8b and
# mean-pools the assistant-span activation at each model's v18 pool layer (so the
# existing adapters_v9_serve_llama enc fits stay aligned).
#
# Usage (inside container, /big mounted):
#   bash scripts/audit/run_v19_extract.sh \
#       /big/audit/ao/dialogues_S.jsonl /big/audit/v19_xmodel 80
#   bash scripts/audit/run_v19_extract.sh \
#       /big/audit/ao/real_transcripts.jsonl /big/audit/v19_xmodel/real 1000
#
# One model is loaded at a time to fit a 32GB V100; set CUDA_VISIBLE_DEVICES outside.
set -euo pipefail

DLG="${1:?usage: run_v19_extract.sh <dialogue_jsonl> <out_dir> [cap_per_bias]}"
OUT="${2:?out dir}"
CAP="${3:-80}"

# tag|model_id|layer  — must match data_v17.yaml / v18 (adapters_v9_serve_llama enc layers)
TAGS=(
  "qwen3-1p7b|Qwen/Qwen3-1.7B|14"
  "phi-1p5|microsoft/phi-1_5|12"
  "smollm3-3b|HuggingFaceTB/SmolLM3-3B|18"
  "qwen2p5-7b|Qwen/Qwen2.5-7B|20"
  "gemma2|google/gemma-2-9b-it|21"
  "qwen2p5-0p5b|Qwen/Qwen2.5-0.5B|12"
  "qwen3-4b|Qwen/Qwen3-4B|18"
  "llama3-8b|NousResearch/Meta-Llama-3-8B-Instruct|15"
)

for spec in "${TAGS[@]}"; do
  IFS='|' read -r tag model layer <<< "$spec"
  echo "=== extract $tag ($model) layer=$layer -> $OUT ==="
  python -m scripts.audit.extract_v18_xmodel \
    --tag "$tag" --model "$model" --layer "$layer" \
    --out-dir "$OUT" --cap-per-bias "$CAP" \
    --dialogue-files "$DLG"
done
echo "=== v19 extract done -> $OUT ==="
