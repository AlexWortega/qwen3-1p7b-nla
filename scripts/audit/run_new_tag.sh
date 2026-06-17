#!/usr/bin/env bash
# Add ONE new (tag, layer) to the v22 detector and eval both detectors, all bf16.
# Assumes the pool acts already exist at $POOL/<TAG>.safetensors (from extract_pool_single).
# Args: TAG MODEL LAYER
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh && conda activate aisci
cd ~/p1a_extract/repo
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 HF_HOME=$HOME/.hfcache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=$HOME/p3_work
TAG="$1"; MODEL="$2"; LAYER="$3"
POOL=$W/pool_new
DET_DIR=$W/detector/v22_1p7b_dirbal
DET_HO=$W/detector/v22_1p7b_heldout_ep1
XM=$W/v22_xmodel_bf16            # has rows.jsonl (the v22 transcript rows)

echo "=== [$TAG] fit encoder (CPU lstsq) ==="
CUDA_VISIBLE_DEVICES="" python -m scripts.audit.fit_heldout_encoder \
  --in-adapters $DET_DIR/adapters --tag "$TAG" \
  --pool-acts $POOL/$TAG.safetensors --out-adapters $W/adapters_$TAG || { echo "[FAIL] fit $TAG"; exit 1; }

echo "=== [$TAG] transcript extract (bf16, layer $LAYER) ==="
# reuse the v22 rows by copying rows.jsonl into a fresh per-tag transcript dir
mkdir -p $W/xm_$TAG
cp $XM/rows.jsonl $W/xm_$TAG/rows.jsonl
python -m scripts.audit.extract_v18_xmodel --dtype bf16 --tag "$TAG" --model "$MODEL" --layer "$LAYER" \
  --out-dir $W/xm_$TAG --cap-per-bias 200 || { echo "[FAIL] transcript $TAG"; exit 1; }

# build detector dirs that point adapters -> the extended bundle (av/meta from each detector)
for D in dirbal:$DET_DIR heldout:$DET_HO; do
  NAME="${D%%:*}"; SRC="${D##*:}"
  DD=$W/det_${TAG}_${NAME}
  mkdir -p $DD
  ln -sfn $SRC/av $DD/av
  ln -sfn $SRC/v18_meta.json $DD/v18_meta.json
  ln -sfn $W/adapters_$TAG $DD/adapters
  echo "=== [$TAG] eval $NAME ==="
  python -m scripts.audit.eval_v22_xarch --v18-dir $DD --xmodel-dir $W/xm_$TAG --tag "$TAG" \
    --out $W/xarch_${TAG}_${NAME}.json || echo "[FAIL] eval $TAG $NAME"
done
echo "=== [$TAG] ALL DONE ==="
