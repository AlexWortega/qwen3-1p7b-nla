#!/usr/bin/env bash
# v24 exp1 — user's literal spec: en-wiki 30 / code(py) 30 / {ru,zh,ja,de,es}wiki 8each,
# EXPANDED to the full 20-tag AV model pool (per user follow-up request). Chained pipeline:
# corpus build -> extract activations (20 archs) -> teacher summaries (OpenRouter) -> joint
# SFT (train_v18, same recipe as v22 flagship: mix 6:2:0:2, detect-mix 2:1.5:1.5, same held-out
# biases) -> eval_harness vs the v22_flagship reference numbers already recorded in
# configs/eval/v24_sanity.yaml's header comment.
set -e
source ~/miniconda3/etc/profile.d/conda.sh && conda activate aisci
cd ~/vae_llm
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 HF_HOME=$HOME/.hfcache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)
export OPENROUTER_API_KEY="__OPENROUTER_KEY__"

W=$HOME/p3_work/wikicode_v24
POOL=artifacts/wikicode_v24/exp1_baseline_303040
AV_TAGS="qwen3-1p7b,phi-1p5,smollm3-3b,qwen2p5-7b,gemma2,bloom-560m,deepseek-llm-7b,gemma4-e4b,gpt-neo-1p3b,gpt2-medium,lfm-7b,nemotron-mini-4b,pythia-410m,qwen2p5-0p5b,qwen3-0p6b,qwen3-4b,rugpt3-large,smollm2-360m,vikhr-7b-01,yagpt-5-8b"
mkdir -p $W

echo "=== [1/5] build wiki+code corpus ==="
python -m scripts.audit.build_wikicode_corpus --out-dir $POOL --n-passages 5000 \
  --en-frac 0.3 --code-frac 0.3 --other-langs ru,zh,ja,de,es --seed 0 \
  || { echo V24_EXP1_FAIL_CORPUS; exit 1; }

echo "=== [2/5] extract activations (20 archs) ==="
python -m scripts.extract_multi --config configs/universal/extract_v24_exp1_baseline.yaml \
  || { echo V24_EXP1_FAIL_EXTRACT; exit 1; }

echo "=== [3/5] teacher summaries (OpenRouter) ==="
python -m scripts.generate_summaries_resume --pool-dir $POOL --model qwen/qwen-2.5-7b-instruct \
  || { echo V24_EXP1_FAIL_SUMMARIES; exit 1; }

echo "=== [4/5] train_v18 joint SFT (same recipe as v22 flagship, only av-pool-dir differs) ==="
python -m scripts.audit.train_v18 \
  --trunk Qwen/Qwen3-1.7B --lora-r 32 \
  --mix 6:2:0:2 --detect-mix 2:1.5:1.5 --minutes 120 \
  --xmodel-dir artifacts/audit/v18_xmodel \
  --adapters-init artifacts/adapters_v9_serve_llama \
  --lie-dir artifacts/audit/lie_gemma2_ml --lie-acts-name lie_acts_L21.safetensors \
  --held-out-biases atomic,british,chinese_bias,chocolate,decimal,movie,muslim_bias,rhetq,sports,voting \
  --av-pool-dir $POOL \
  --av-tags "$AV_TAGS" \
  --latentqa-dir artifacts/audit/latentqa_task \
  --seed 0 \
  --out $W/exp1_train \
  || { echo V24_EXP1_FAIL_TRAIN; exit 1; }

echo "=== [5/5] eval vs v22_flagship reference ==="
python -m scripts.audit.eval_harness --config configs/eval/v24_wikicode_exp1.yaml \
  || { echo V24_EXP1_FAIL_EVAL; exit 1; }

echo "V24_EXP1_DONE"
