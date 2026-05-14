#!/usr/bin/env bash
# Retrain AR_1p7b with cosine+magnitude loss (after magnitude blowup was diagnosed
# in the prior pure-cosine 1.7B run). AV_1p7b is already trained and not touched here.
set +e

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
echo "[ar-retrain] cwd=$PROJECT_ROOT"
date

DC="docker compose -f docker/compose.yml run --rm -T"

echo
echo "=== retrain AR_1p7b with cosine+mag loss ==="
$DC -e CUDA_VISIBLE_DEVICES=1 nla python scripts/train_ar_seq.py \
    --save-path artifacts/ar_1p7b > ~/vae_llm_train_ar_1p7b.log 2>&1 \
    && touch ~/vae_llm_pipeline.train_ar.done \
    || echo "TRAIN_AR_FAILED"
tail -5 ~/vae_llm_train_ar_1p7b.log

echo
echo "=== eval baseline (warmstart_1p7b_fix) ==="
$DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/eval_fve_seq.py \
    --av-path artifacts/av_1p7b --ar-path artifacts/ar_1p7b --tag warmstart_1p7b_fix 2>&1 \
    | grep -E "FVE|usable|per-position|AV:|AR:" \
    && touch ~/vae_llm_pipeline.eval_baseline.done \
    || echo "EVAL_BASE_FAILED"

echo
echo "=== train_joint ==="
$DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/train_joint.py \
    --av-path artifacts/av_1p7b --ar-path artifacts/ar_1p7b \
    --save-av artifacts/av_joint_1p7b --save-ar artifacts/ar_joint_1p7b \
    --epochs 1 2>&1 | tail -40 \
    && touch ~/vae_llm_pipeline.train_joint.done \
    || echo "TRAIN_JOINT_FAILED"

echo
echo "=== eval after joint (joint_1p7b_fix) ==="
$DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/eval_fve_seq.py \
    --av-path artifacts/av_joint_1p7b --ar-path artifacts/ar_joint_1p7b --tag joint_1p7b_fix 2>&1 \
    | grep -E "FVE|usable|per-position|AV:|AR:" \
    && touch ~/vae_llm_pipeline.eval_joint.done \
    || echo "EVAL_JOINT_FAILED"

date
touch ~/vae_llm_pipeline.done
