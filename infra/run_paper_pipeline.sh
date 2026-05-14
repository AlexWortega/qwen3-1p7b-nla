#!/usr/bin/env bash
# Paper-faithful pipeline using kitft/natural_language_autoencoders' code.
#   1. Datagen stages 0..3 (extract → split → API explain → build SFT tables) — their code
#   2. AV SFT — standalone trainer that uses their nla.injection
#   3. AR SFT — standalone trainer that uses their nla.models.NLACriticModel
#   4. End-to-end FVE
set +e

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
echo "[paper-pipeline] cwd=$PROJECT_ROOT"
date

DC="docker compose -f docker/compose.yml run --rm -T"
CFG=configs/datagen/qwen3_1p7b_fineweb_10k.yaml
OUTDIR=artifacts/datagen_qwen3_1p7b_ultrafw

echo
echo "=== datagen pipeline (GPU0) ==="
$DC -e CUDA_VISIBLE_DEVICES=0 nla python -m nla.datagen.run_pipeline --config $CFG 2>&1 | tee ~/vae_llm_datagen.log | tail -40
touch ~/vae_llm_pipeline.datagen.done

echo
echo "=== AV SFT (GPU0) ==="
$DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/train_actor_sft.py \
    --parquet ${OUTDIR}/av_sft_shuf.parquet \
    --base-model Qwen/Qwen3-1.7B \
    --save-dir artifacts/av_ultrafw \
    --lr 2e-5 --epochs 1 --batch-size 4 --grad-accum 4 \
    > ~/vae_llm_train_actor.log 2>&1 && touch ~/vae_llm_pipeline.train_av.done || echo "TRAIN_AV_FAILED"
tail -10 ~/vae_llm_train_actor.log

echo
echo "=== AR SFT (GPU1) ==="
$DC -e CUDA_VISIBLE_DEVICES=1 nla python scripts/train_critic_sft.py \
    --parquet ${OUTDIR}/ar_sft_shuf.parquet \
    --base-model Qwen/Qwen3-1.7B \
    --layer-index 18 \
    --save-dir artifacts/ar_ultrafw \
    --lr 2e-5 --epochs 1 --batch-size 8 --grad-accum 4 \
    > ~/vae_llm_train_critic.log 2>&1 && touch ~/vae_llm_pipeline.train_ar.done || echo "TRAIN_AR_FAILED"
tail -10 ~/vae_llm_train_critic.log

echo
echo "=== eval (GPU0) ==="
$DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/eval_paper.py \
    --av-dir artifacts/av_ultrafw \
    --ar-dir artifacts/ar_ultrafw \
    --eval-parquet ${OUTDIR}/ar_sft_shuf.parquet \
    --av-eval-parquet ${OUTDIR}/av_sft_shuf.parquet \
    --n 200 --max-new-tokens 200 \
    --out-json artifacts/eval/fve_ultrafw.json 2>&1 | tee ~/vae_llm_eval.log | tail -30
touch ~/vae_llm_pipeline.eval.done

date
touch ~/vae_llm_pipeline.done
