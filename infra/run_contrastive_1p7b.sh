#!/usr/bin/env bash
# Paper-faithful + contrastive AR loss (anti-collapse).
# AV unchanged from av_paper_1p7b. AR retrained from scratch with InfoNCE.
set +e

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
echo "[contrastive-1p7b] cwd=$PROJECT_ROOT"
date

DC="docker compose -f docker/compose.yml run --rm -T"
PAPER_COMMON="--num-positions 1 --pick-positions 15 --normalize-mode unit_l2"

echo
echo "=== train_ar_contrastive_v2 (GPU0) — weighted α=5 β=0.1 ==="
$DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/train_ar_seq.py $PAPER_COMMON \
    --truncate-ar-layers 14 --use-cos-loss contrastive --contrastive-temp 0.07 \
    --cos-weight 5.0 --info-nce-weight 0.1 \
    --save-path artifacts/ar_contrastive_v2_1p7b > ~/vae_llm_train_ar_contrastive.log 2>&1 \
    && touch ~/vae_llm_pipeline.train_ar.done \
    || echo "TRAIN_AR_FAILED"
tail -5 ~/vae_llm_train_ar_contrastive.log

echo
echo "=== eval baseline (warmstart_contrastive_1p7b) — pair with existing av_paper_1p7b ==="
$DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/eval_fve_seq.py \
    $PAPER_COMMON --truncate-ar-layers 14 \
    --av-path artifacts/av_paper_1p7b --ar-path artifacts/ar_contrastive_v2_1p7b \
    --tag warmstart_contrastive_1p7b 2>&1 \
    | grep -E "FVE|usable|per-position|AV:|AR:" \
    && touch ~/vae_llm_pipeline.eval_baseline.done \
    || echo "EVAL_BASE_FAILED"

echo
echo "=== inference diversity check ==="
$DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/inference_check.py \
    --av-path artifacts/av_paper_1p7b --ar-path artifacts/ar_contrastive_v2_1p7b --n-samples 8 2>&1 \
    | grep -E "cos|mean|off-diag|--- pid" | tail -50

date
touch ~/vae_llm_pipeline.done
