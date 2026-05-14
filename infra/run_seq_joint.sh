#!/usr/bin/env bash
# Orchestration: extract_seq → train_av_seq + train_ar_seq parallel → eval (warmstart)
#               → train_joint → eval (joint)
# Each stage writes a ~/vae_llm_pipeline.<stage>.done sentinel on success.
set +e

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
echo "[pipeline] cwd=$PROJECT_ROOT"
date

DC="docker compose -f docker/compose.yml run --rm -T"

echo
echo "=== extract_seq 10k (skip if already done) ==="
if [[ -f "$PROJECT_ROOT/artifacts/activations/layer14_seq.safetensors" ]]; then
    echo "[skip] layer14_seq.safetensors already exists; reusing"
    touch ~/vae_llm_pipeline.extract.done
else
    $DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/extract_activations_seq.py 2>&1 \
        | grep -v PyGILState | grep -v "Thread 0x" | grep -v "<no Python" | grep -v "Extension modules" \
        | tail -15 \
        && touch ~/vae_llm_pipeline.extract.done \
        || echo "EXTRACT_FAILED rc=$?"
fi

echo
echo "=== parallel train_av_seq (GPU0) + train_ar_seq (GPU1) ==="
( cd "$PROJECT_ROOT" && $DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/train_av_seq.py --save-path artifacts/av_1p7b > ~/vae_llm_train_av_1p7b.log 2>&1 ) &
PID_AV=$!
( cd "$PROJECT_ROOT" && $DC -e CUDA_VISIBLE_DEVICES=1 nla python scripts/train_ar_seq.py --save-path artifacts/ar_1p7b > ~/vae_llm_train_ar_1p7b.log 2>&1 ) &
PID_AR=$!
echo "AV pid=$PID_AV  AR pid=$PID_AR"
wait $PID_AV && touch ~/vae_llm_pipeline.train_av.done || echo "TRAIN_AV_FAILED"
wait $PID_AR && touch ~/vae_llm_pipeline.train_ar.done || echo "TRAIN_AR_FAILED"
tail -3 ~/vae_llm_train_av_1p7b.log
echo ---
tail -3 ~/vae_llm_train_ar_1p7b.log

echo
echo "=== eval_fve_seq BASELINE (warm-start only) ==="
cd "$PROJECT_ROOT"
$DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/eval_fve_seq.py \
    --av-path artifacts/av_1p7b --ar-path artifacts/ar_1p7b --tag warmstart_1p7b 2>&1 \
    | grep -E "FVE|usable|per-position|AV:|AR:" \
    && touch ~/vae_llm_pipeline.eval_baseline.done \
    || echo "EVAL_BASE_FAILED"

echo
echo "=== train_joint (GPU0) ==="
$DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/train_joint.py \
    --av-path artifacts/av_1p7b --ar-path artifacts/ar_1p7b \
    --save-av artifacts/av_joint_1p7b --save-ar artifacts/ar_joint_1p7b \
    --epochs 1 2>&1 | tail -50 \
    && touch ~/vae_llm_pipeline.train_joint.done \
    || echo "TRAIN_JOINT_FAILED"

echo
echo "=== eval_fve_seq AFTER JOINT ==="
$DC -e CUDA_VISIBLE_DEVICES=0 nla python scripts/eval_fve_seq.py \
    --av-path artifacts/av_joint_1p7b --ar-path artifacts/ar_joint_1p7b --tag joint_1p7b 2>&1 \
    | grep -E "FVE|usable|per-position|AV:|AR:" \
    && touch ~/vae_llm_pipeline.eval_joint.done \
    || echo "EVAL_JOINT_FAILED"

date
touch ~/vae_llm_pipeline.done
