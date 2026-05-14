#!/usr/bin/env bash
# Joint RL training on warm-started Ultra-FineWeb AV+AR. Uses the AV-SFT
# parquet as the RL data source (just needs prompt + activation_vector).
set +e

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
echo "[joint-rl] cwd=$PROJECT_ROOT"
date

DC="docker compose -f docker/compose.yml run --rm -T"
OUTDIR=artifacts/datagen_qwen3_1p7b_ultrafw

echo
echo "=== train_joint_rl (GPU0) ==="
$DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/train_joint_rl_paper.py \
    --av-dir artifacts/av_ultrafw --ar-dir artifacts/ar_ultrafw \
    --rl-parquet ${OUTDIR}/av_sft_shuf.parquet \
    --save-av artifacts/av_joint_ultrafw --save-ar artifacts/ar_joint_ultrafw \
    --steps 300 --batch-size 1 --group-size 2 \
    --max-new-tokens 40 --temperature 1.0 \
    --lr-av 1e-5 --lr-ar 5e-5 --beta-kl 0.05 \
    --log-every 5 --grad-checkpoint \
    > ~/vae_llm_train_joint_rl.log 2>&1 \
    && touch ~/vae_llm_pipeline.joint_rl.done \
    || echo "JOINT_RL_FAILED"
tail -30 ~/vae_llm_train_joint_rl.log

echo
echo "=== eval after joint_rl ==="
$DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/eval_paper.py \
    --av-dir artifacts/av_joint_ultrafw \
    --ar-dir artifacts/ar_joint_ultrafw \
    --eval-parquet ${OUTDIR}/ar_sft_shuf.parquet \
    --av-eval-parquet ${OUTDIR}/av_sft_shuf.parquet \
    --n 200 --max-new-tokens 200 \
    --out-json artifacts/eval/fve_joint_ultrafw.json 2>&1 \
    | tee ~/vae_llm_eval_joint.log | grep -E "FVE|usable|baseline|cos|=== AV" | tail -20
touch ~/vae_llm_pipeline.eval_joint.done

date
touch ~/vae_llm_pipeline.done
