#!/usr/bin/env bash
# Paper-algorithm joint training (REINFORCE) on top of warm-started AV/AR (1.7B).
# Runs on GPU 1 so it can be parallelized with anything on GPU 0.
set +e

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
echo "[joint-rl-1p7b] cwd=$PROJECT_ROOT"
date

DC="docker compose -f docker/compose.yml run --rm -T"

echo
echo "=== train_joint_rl (GPU2) on warm-started av_1p7b + ar_1p7b ==="
$DC -e CUDA_VISIBLE_DEVICES=2 nla python scripts/train_joint_rl.py \
    --av-path artifacts/av_1p7b --ar-path artifacts/ar_1p7b \
    --save-av artifacts/av_joint_rl_1p7b --save-ar artifacts/ar_joint_rl_1p7b \
    --epochs 1 --group-size 2 --batch-size 2 --max-new-tokens 32 \
    --beta-kl 0.05 --temperature 1.0 --grad-checkpoint \
    > ~/vae_llm_train_joint_rl_1p7b.log 2>&1 \
    && touch ~/vae_llm_pipeline.joint_rl.done \
    || echo "JOINT_RL_FAILED"
tail -20 ~/vae_llm_train_joint_rl_1p7b.log

echo
echo "=== eval_fve_seq AFTER joint_rl ==="
$DC -e CUDA_VISIBLE_DEVICES=2 nla python scripts/eval_fve_seq.py \
    --av-path artifacts/av_joint_rl_1p7b --ar-path artifacts/ar_joint_rl_1p7b --tag joint_rl_1p7b 2>&1 \
    | grep -E "FVE|usable|per-position|AV:|AR:" \
    && touch ~/vae_llm_pipeline.eval_joint_rl.done \
    || echo "EVAL_JOINT_RL_FAILED"

date
