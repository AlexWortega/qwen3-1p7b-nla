#!/usr/bin/env bash
# Two parallel RL runs with new contrastive reward.
# E: pure InfoNCE-contrastive on GPU0
# F: 50/50 mse+contrastive mix on GPU3
set +e

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
echo "[rl-EF] cwd=$PROJECT_ROOT"
date

DC="docker compose -f docker/compose.yml run --rm -T"
RL_PARQUET=artifacts/datagen_qwen3_1p7b_ultrafw_9k/av_sft_shuf.parquet
AV=artifacts/av_ultrafw_9k
AR=artifacts/ar_ultrafw_9k

COMMON="--av-dir $AV --ar-dir $AR --rl-parquet $RL_PARQUET --batch-size 4 --group-size 2 --grad-checkpoint --temperature 1.0 --log-every 10 --max-new-tokens 40 --lr-av 1e-5 --lr-ar 5e-5 --beta-kl 0.05"
ENV_FLAGS="-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

echo
echo "=== launching E (contrastive) GPU0 + F (mix) GPU3 ==="

( cd "$PROJECT_ROOT" && $DC $ENV_FLAGS -e CUDA_VISIBLE_DEVICES=0 nla python scripts/train_joint_rl_paper.py $COMMON \
    --steps 150 --reward contrastive --contrastive-tau 0.1 \
    --save-av artifacts/av_rl_E --save-ar artifacts/ar_rl_E \
    > ~/vae_llm_rl_E.log 2>&1 ) &
PID_E=$!

( cd "$PROJECT_ROOT" && $DC $ENV_FLAGS -e CUDA_VISIBLE_DEVICES=3 nla python scripts/train_joint_rl_paper.py $COMMON \
    --steps 150 --reward mix --contrastive-tau 0.1 \
    --save-av artifacts/av_rl_F --save-ar artifacts/ar_rl_F \
    > ~/vae_llm_rl_F.log 2>&1 ) &
PID_F=$!

echo "[rl-EF] PIDs E=$PID_E F=$PID_F"

wait $PID_E && touch ~/vae_llm_pipeline.rl_E.done || echo "RL_E_FAILED"
wait $PID_F && touch ~/vae_llm_pipeline.rl_F.done || echo "RL_F_FAILED"

echo
echo "=== EF finished — tails ==="
for X in E F; do
    echo "--- $X ---"
    tail -8 ~/vae_llm_rl_$X.log 2>/dev/null
done

date
touch ~/vae_llm_pipeline.rl_EF.done
