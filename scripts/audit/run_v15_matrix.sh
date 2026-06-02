#!/usr/bin/env bash
# v15 autoresearch matrix: 8 experiments over 3 GPU lanes (serial per lane).
# Each job = train_v15 (--minutes 120) then full eval_v15 -> metrics.json.
# Run from ~/vae_llm on eva01. GPUs passed as $1 $2 $3 (e.g. 3 0 1).
set -u
cd ~/vae_llm
GA=${1:-3}; GB=${2:-0}; GC=${3:-1}
MIN=${MIN:-120}

run_exp () {
  local g="$1" name="$2"; shift 2
  echo "[$(date +%H:%M:%S)] START $name on GPU$g : $*"
  # NOTE: -e env flags MUST precede the service name `nla`, else docker treats
  # them as the in-container command (tini "exec -e failed").
  docker compose -f docker/compose.yml run --rm \
    -e CUDA_VISIBLE_DEVICES="$g" -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v /mnt/storage/vae_llm/artifacts:/big nla \
    bash -c "python scripts/audit/train_v15.py $* --minutes $MIN --out /big/audit/v15/$name && python scripts/audit/eval_v15.py --v15-dir /big/audit/v15/$name --out /big/audit/v15/$name/metrics.json" \
    > /tmp/v15_$name.log 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $name (exit $?)"
}

( run_exp "$GA" exp0 --inject marker --mix 1:0:0
  run_exp "$GA" exp3 --inject flamingo --inject-layer 14 --mix 3:1:1
  run_exp "$GA" exp6 --inject marker --mix 3:1:1 --contrastive-weight 2.0
) &
( run_exp "$GB" exp1 --inject marker --mix 3:1:1
  run_exp "$GB" exp4 --inject marker --mix 1:1:1
  run_exp "$GB" exp7 --inject marker --mix 3:1:1 --train-enc ao-only
) &
( run_exp "$GC" exp2 --inject flamingo --inject-layer 7 --mix 3:1:1
  run_exp "$GC" exp5 --inject marker --mix 5:1:1
) &
wait
echo "[$(date +%H:%M:%S)] ALL LANES DONE"
