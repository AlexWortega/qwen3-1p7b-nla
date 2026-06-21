#!/usr/bin/env bash
# Pre-speech task-generalization sweep for ONE held-out base.
# Re-reads the SAME v22 transcripts at three temporal positions (pre / early / post) with a
# single forward pass (no generation), then scores each with the FROZEN heldout detector.
# POST reproduces the published xarch_<tag>_heldout.json as a built-in sanity check.
#
# Args: TAG MODEL LAYER XMDIR DETDIR
#   TAG     base tag (e.g. llama3-8b)
#   MODEL   HF model id
#   LAYER   pool layer (from the base's meta.json)
#   XMDIR   dir containing rows.jsonl and <TAG>/ (e.g. $W/v22_xmodel_bf16 or $W/xm_<TAG>)
#   DETDIR  frozen detector dir with av/ + v18_meta.json + adapters/ for this tag
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh && conda activate aisci
cd ~/p1a_extract/repo
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} HF_HOME=$HOME/.hfcache \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=$HOME/p3_work
TAG="$1"; MODEL="$2"; LAYER="$3"; XMDIR="$4"; DETDIR="$5"
OUT=$W/prespeech
mkdir -p $OUT

echo "=== [$TAG] re-extract pre,early,post (one forward, bf16, layer $LAYER) ==="
python -m scripts.audit.extract_v18_xmodel --dtype bf16 --tag "$TAG" --model "$MODEL" \
  --layer "$LAYER" --out-dir "$XMDIR" --cap-per-bias 200 --positions pre,early,post \
  || { echo "[FAIL] extract $TAG"; exit 1; }

for POS in pre early post; do
  echo "=== [$TAG] eval position=$POS with detector $DETDIR ==="
  python -m scripts.audit.eval_v22_xarch --v18-dir "$DETDIR" --xmodel-dir "$XMDIR" --tag "$TAG" \
    --acts-name acts_${POS}.safetensors --out $OUT/prespeech_${TAG}_${POS}.json \
    || echo "[FAIL] eval $TAG $POS"
done
echo "=== [$TAG] DONE -> $OUT/prespeech_${TAG}_{pre,early,post}.json ==="
