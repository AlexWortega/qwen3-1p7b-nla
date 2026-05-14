#!/usr/bin/env bash
# All-1.7B setup: M = AV = AR = Qwen3-1.7B.
# Symmetric d_h=d_lm=2048 → identity-init readout/proj. AR truncated to 14 layers
# is now a literal "first half of M" — paper-faithful.
#
# 1. Re-extract activations from 1.7B M (existing 1024-dim cache is stale)
# 2. Re-train AV, AR with paper-faithful flags (K=1, unit_l2, truncate=14)
# 3. Eval against existing bulk/eval summaries (still valid, only text matters)
set +e

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
echo "[all-1p7b] cwd=$PROJECT_ROOT"
date

DC="docker compose -f docker/compose.yml run --rm -T"
PAPER_COMMON="--num-positions 1 --pick-positions 15 --normalize-mode unit_l2"

echo
echo "=== re-extract activations from Qwen3-1.7B M (GPU0) ==="
rm -f artifacts/activations/layer14_seq.safetensors artifacts/activations/layer14_seq.jsonl
$DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/extract_activations_seq.py 2>&1 \
    | grep -v PyGILState | grep -v "Thread 0x" | grep -v "<no Python" | grep -v "Extension modules" \
    | tail -10 \
    && touch ~/vae_llm_pipeline.extract.done \
    || echo "EXTRACT_FAILED"

echo
echo "=== parallel train_av_paper (GPU0) + train_ar_paper (GPU1) ==="
( cd "$PROJECT_ROOT" && $DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/train_av_seq.py $PAPER_COMMON \
    --save-path artifacts/av_all_1p7b > ~/vae_llm_train_av_all_1p7b.log 2>&1 ) &
PID_AV=$!
( cd "$PROJECT_ROOT" && $DC -e CUDA_VISIBLE_DEVICES=1 nla python scripts/train_ar_seq.py $PAPER_COMMON \
    --truncate-ar-layers 14 --use-cos-loss pure \
    --save-path artifacts/ar_all_1p7b > ~/vae_llm_train_ar_all_1p7b.log 2>&1 ) &
PID_AR=$!
echo "[all-1p7b] AV pid=$PID_AV  AR pid=$PID_AR"
wait $PID_AV && touch ~/vae_llm_pipeline.train_av.done || echo "TRAIN_AV_FAILED"
wait $PID_AR && touch ~/vae_llm_pipeline.train_ar.done || echo "TRAIN_AR_FAILED"
tail -5 ~/vae_llm_train_av_all_1p7b.log
echo ---
tail -5 ~/vae_llm_train_ar_all_1p7b.log

echo
echo "=== eval baseline (warmstart_all_1p7b) ==="
$DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/eval_fve_seq.py \
    $PAPER_COMMON --truncate-ar-layers 14 \
    --av-path artifacts/av_all_1p7b --ar-path artifacts/ar_all_1p7b \
    --tag warmstart_all_1p7b 2>&1 \
    | grep -E "FVE|usable|per-position|AV:|AR:" \
    && touch ~/vae_llm_pipeline.eval_baseline.done \
    || echo "EVAL_BASE_FAILED"

echo
echo "=== inference diversity check ==="
$DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/inference_check.py \
    --av-path artifacts/av_all_1p7b --ar-path artifacts/ar_all_1p7b --n-samples 8 2>&1 \
    | grep -E "cos|mean|off-diag|--- pid" | tail -50

date
touch ~/vae_llm_pipeline.done
