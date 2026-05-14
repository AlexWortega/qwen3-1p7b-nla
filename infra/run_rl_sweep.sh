#!/usr/bin/env bash
# Parallel RL hparam sweep on warmstart_9k weights.
# 4 V100s, 4 configurations (baseline / slow / long-gen / strong-KL).
# Free (no API). ~3-4h wall.
set +e

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
echo "[rl-sweep] cwd=$PROJECT_ROOT"
date

DC="docker compose -f docker/compose.yml run --rm -T"
RL_PARQUET=artifacts/datagen_qwen3_1p7b_ultrafw_9k/av_sft_shuf.parquet
AV=artifacts/av_ultrafw_9k
AR=artifacts/ar_ultrafw_9k

COMMON="--av-dir $AV --ar-dir $AR --rl-parquet $RL_PARQUET --batch-size 1 --grad-checkpoint --temperature 1.0 --log-every 25"

echo
echo "=== launching 4 parallel RL runs ==="

( cd "$PROJECT_ROOT" && $DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/train_joint_rl_paper.py $COMMON \
    --steps 300 --group-size 4 --max-new-tokens 40 --lr-av 1e-5 --lr-ar 5e-5 --beta-kl 0.05 \
    --save-av artifacts/av_rl_A --save-ar artifacts/ar_rl_A \
    > ~/vae_llm_rl_A.log 2>&1 ) &
PID_A=$!

( cd "$PROJECT_ROOT" && $DC -e CUDA_VISIBLE_DEVICES=1 nla python scripts/train_joint_rl_paper.py $COMMON \
    --steps 400 --group-size 4 --max-new-tokens 40 --lr-av 5e-6 --lr-ar 2e-5 --beta-kl 0.05 \
    --save-av artifacts/av_rl_B --save-ar artifacts/ar_rl_B \
    > ~/vae_llm_rl_B.log 2>&1 ) &
PID_B=$!

( cd "$PROJECT_ROOT" && $DC -e CUDA_VISIBLE_DEVICES=2 nla python scripts/train_joint_rl_paper.py $COMMON \
    --steps 300 --group-size 4 --max-new-tokens 60 --lr-av 1e-5 --lr-ar 5e-5 --beta-kl 0.05 \
    --save-av artifacts/av_rl_C --save-ar artifacts/ar_rl_C \
    > ~/vae_llm_rl_C.log 2>&1 ) &
PID_C=$!

( cd "$PROJECT_ROOT" && $DC -e CUDA_VISIBLE_DEVICES=3 nla python scripts/train_joint_rl_paper.py $COMMON \
    --steps 300 --group-size 4 --max-new-tokens 40 --lr-av 2e-5 --lr-ar 5e-5 --beta-kl 0.2 \
    --save-av artifacts/av_rl_D --save-ar artifacts/ar_rl_D \
    > ~/vae_llm_rl_D.log 2>&1 ) &
PID_D=$!

echo "[rl-sweep] PIDs A=$PID_A B=$PID_B C=$PID_C D=$PID_D"

wait $PID_A && touch ~/vae_llm_pipeline.rl_A.done || echo "RL_A_FAILED"
wait $PID_B && touch ~/vae_llm_pipeline.rl_B.done || echo "RL_B_FAILED"
wait $PID_C && touch ~/vae_llm_pipeline.rl_C.done || echo "RL_C_FAILED"
wait $PID_D && touch ~/vae_llm_pipeline.rl_D.done || echo "RL_D_FAILED"

echo
echo "=== all RL runs finished — short tails ==="
for X in A B C D; do
    echo "--- $X ---"
    tail -5 ~/vae_llm_rl_$X.log 2>/dev/null
done

echo
echo "=== sequential eval (GPU 0) ==="
for X in A B C D; do
    echo "--- eval $X ---"
    if [[ -d "artifacts/av_rl_$X" && -d "artifacts/ar_rl_$X" ]]; then
        $DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/eval_paper.py \
            --av-dir artifacts/av_rl_$X --ar-dir artifacts/ar_rl_$X \
            --eval-parquet artifacts/datagen_qwen3_1p7b_ultrafw_9k/ar_sft_shuf.parquet \
            --av-eval-parquet artifacts/datagen_qwen3_1p7b_ultrafw_9k/av_sft_shuf.parquet \
            --n 200 --max-new-tokens 200 \
            --out-json artifacts/eval/fve_rl_$X.json 2>&1 \
            | grep -E "FVE|usable|baseline" | tail -10
    else
        echo "$X — checkpoint dirs missing, skipping eval"
    fi
done

date
touch ~/vae_llm_pipeline.rl_sweep.done
