#!/usr/bin/env bash
# 3x-scale paper-faithful pipeline: Ultra-FineWeb 9000 docs + DeepSeek V3 teacher.
# RL phase intentionally skipped (user request — keep RL untouched).
set +e

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
echo "[paper-9k] cwd=$PROJECT_ROOT"
date

DC="docker compose -f docker/compose.yml run --rm -T"
CFG=configs/datagen/qwen3_1p7b_ultrafw_9k_deepseek.yaml
OUTDIR=artifacts/datagen_qwen3_1p7b_ultrafw_9k

echo
echo "=== datagen pipeline ==="
$DC -e CUDA_VISIBLE_DEVICES=0 nla python -m nla.datagen.run_pipeline --config $CFG 2>&1 | tee ~/vae_llm_datagen_9k.log | tail -50
touch ~/vae_llm_pipeline.datagen.done

echo
echo "=== AV SFT (GPU0) ==="
$DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/train_actor_sft.py \
    --parquet ${OUTDIR}/av_sft_shuf.parquet \
    --base-model Qwen/Qwen3-1.7B \
    --save-dir artifacts/av_ultrafw_9k \
    --lr 2e-5 --epochs 1 --batch-size 4 --grad-accum 4 \
    > ~/vae_llm_train_actor_9k.log 2>&1 && touch ~/vae_llm_pipeline.train_av.done || echo "TRAIN_AV_FAILED"
tail -10 ~/vae_llm_train_actor_9k.log

echo
echo "=== AR SFT (GPU1) ==="
$DC -e CUDA_VISIBLE_DEVICES=1 nla python scripts/train_critic_sft.py \
    --parquet ${OUTDIR}/ar_sft_shuf.parquet \
    --base-model Qwen/Qwen3-1.7B \
    --layer-index 18 \
    --save-dir artifacts/ar_ultrafw_9k \
    --lr 2e-5 --epochs 1 --batch-size 8 --grad-accum 4 \
    > ~/vae_llm_train_critic_9k.log 2>&1 && touch ~/vae_llm_pipeline.train_ar.done || echo "TRAIN_AR_FAILED"
tail -10 ~/vae_llm_train_critic_9k.log

echo
echo "=== eval (GPU0) ==="
$DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/eval_paper.py \
    --av-dir artifacts/av_ultrafw_9k \
    --ar-dir artifacts/ar_ultrafw_9k \
    --eval-parquet ${OUTDIR}/ar_sft_shuf.parquet \
    --av-eval-parquet ${OUTDIR}/av_sft_shuf.parquet \
    --n 200 --max-new-tokens 200 \
    --out-json artifacts/eval/fve_ultrafw_9k.json 2>&1 | tee ~/vae_llm_eval_9k.log | tail -30
touch ~/vae_llm_pipeline.eval.done

date
touch ~/vae_llm_pipeline.done
