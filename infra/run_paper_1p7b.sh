#!/usr/bin/env bash
# Paper-faithful (3 fixes) pipeline on 1.7B AV/AR + 0.6B M:
#   - single-vector h (pick position 15 — last sampled position, closest to "final-token")
#   - unit L2 normalize h per vector
#   - AR truncated to first l=14 layers
# Then run REINFORCE joint per paper's GRPO/RL formulation.
set +e

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
echo "[paper-1p7b] cwd=$PROJECT_ROOT"
date

DC="docker compose -f docker/compose.yml run --rm -T"
COMMON="--num-positions 1 --pick-positions 15 --normalize-mode unit_l2"
AR_COMMON="$COMMON --truncate-ar-layers 14"

echo
echo "=== train_av_paper (GPU0) — single-vector + unit L2 ==="
$DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/train_av_seq.py $COMMON \
    --save-path artifacts/av_paper_1p7b > ~/vae_llm_train_av_paper.log 2>&1 &
PID_AV=$!

echo "=== train_ar_paper (GPU1) — truncated to 14 layers ==="
$DC -e CUDA_VISIBLE_DEVICES=1 nla python scripts/train_ar_seq.py $AR_COMMON \
    --use-cos-loss pure --save-path artifacts/ar_paper_1p7b > ~/vae_llm_train_ar_paper.log 2>&1 &
PID_AR=$!

echo "[paper-1p7b] AV pid=$PID_AV  AR pid=$PID_AR"
wait $PID_AV && touch ~/vae_llm_pipeline.train_av.done || echo "TRAIN_AV_FAILED"
wait $PID_AR && touch ~/vae_llm_pipeline.train_ar.done || echo "TRAIN_AR_FAILED"
tail -5 ~/vae_llm_train_av_paper.log
echo ---
tail -5 ~/vae_llm_train_ar_paper.log

echo
echo "=== eval baseline (warmstart_paper_1p7b) ==="
$DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/eval_fve_seq.py \
    $AR_COMMON \
    --av-path artifacts/av_paper_1p7b --ar-path artifacts/ar_paper_1p7b \
    --tag warmstart_paper_1p7b 2>&1 \
    | grep -E "FVE|usable|per-position|AV:|AR:" \
    && touch ~/vae_llm_pipeline.eval_baseline.done \
    || echo "EVAL_BASE_FAILED"

echo
echo "=== train_joint_rl (GPU0) — REINFORCE on paper-faithful warm-start ==="
$DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/train_joint_rl.py \
    $AR_COMMON \
    --av-path artifacts/av_paper_1p7b --ar-path artifacts/ar_paper_1p7b \
    --save-av artifacts/av_joint_rl_paper_1p7b --save-ar artifacts/ar_joint_rl_paper_1p7b \
    --epochs 1 --group-size 4 --batch-size 4 --max-new-tokens 48 \
    --beta-kl 0.05 --temperature 1.0 --grad-checkpoint \
    > ~/vae_llm_train_joint_rl_paper.log 2>&1 \
    && touch ~/vae_llm_pipeline.train_joint.done \
    || echo "TRAIN_JOINT_FAILED"
tail -20 ~/vae_llm_train_joint_rl_paper.log

echo
echo "=== eval after joint_rl (joint_rl_paper_1p7b) ==="
$DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/eval_fve_seq.py \
    $AR_COMMON \
    --av-path artifacts/av_joint_rl_paper_1p7b --ar-path artifacts/ar_joint_rl_paper_1p7b \
    --tag joint_rl_paper_1p7b 2>&1 \
    | grep -E "FVE|usable|per-position|AV:|AR:" \
    && touch ~/vae_llm_pipeline.eval_joint.done \
    || echo "EVAL_JOINT_FAILED"

date
touch ~/vae_llm_pipeline.done
