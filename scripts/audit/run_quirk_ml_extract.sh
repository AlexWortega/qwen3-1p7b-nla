#!/usr/bin/env bash
# v15.2 — MULTI-LAYER quirk activation extraction (qwen2p5-7b-instruct + 3 organism
# adapters) at 4 decoder layers, then assemble per-layer into acts_ao_org_L{L}_mean.
# Produces the multi-layer counterpart of acts_ao_org_mean used by train_v15 --multi-layer.
#
# Run from ~/vae_llm on eva01 inside docker (BASE=Qwen2.5-7B-Instruct, L=14 is the v13
# single-layer; ML adds 7,14,21,27 for a 28-layer model ~ depth 0.25/0.5/0.75/0.9).
set -uo pipefail
ROOT=${ROOT:-/big/audit}
AO=$ROOT/ao
BASE=${BASE:-Qwen/Qwen2.5-7B-Instruct}
ORGA="$ROOT/organism_qwen25_7b/adapter"
LAYERS=${LAYERS:-7,14,21,27}
ML="$AO"   # write multi-layer shards next to the single-layer ones

ex() { python scripts/audit/extract_acts.py --mode chat --base "$BASE" --layers "$LAYERS" \
  --tag "$1" --battery "$2" --out-dir "$ML" --max-length 768 "${@:3}"; }

echo "#### extract org A/B/C + base at layers $LAYERS (mean+ctrl) ####"
ex ao_A "$AO/transcripts_A.jsonl" --adapter "$ORGA"
ex ao_B "$AO/transcripts_B.jsonl" --adapter "$AO/organism_B/adapter"
ex ao_C "$AO/transcripts_C.jsonl" --adapter "$AO/organism_C/adapter"
ex ao_base "$AO/transcripts_base.jsonl"

echo "#### assemble per-layer org acts (mean) ####"
IFS=',' read -ra LS <<< "$LAYERS"
for L in "${LS[@]}"; do
  python scripts/audit/assemble_ao_acts.py --ao-dir "$ML" --kind mean --layer "$L"
done

echo "#### stack per-layer -> acts_ao_org_ml.safetensors [N,K,d] ####"
python - "$ML" "$LAYERS" <<'PY'
import sys, json
import torch
from safetensors.torch import load_file, save_file
ml, layers = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
parts = [load_file(f"{ml}/acts_ao_org_L{L}_mean.safetensors")["h"] for L in layers]
H = torch.stack(parts, dim=1)   # [N, K, d]
save_file({"h": H}, f"{ml}/acts_ao_org_ml.safetensors")
json.dump({"layers": layers, "shape": list(H.shape)}, open(f"{ml}/acts_ao_org_ml.meta.json","w"), indent=2)
print("[quirk_ml] wrote acts_ao_org_ml.safetensors", tuple(H.shape))
PY
echo "QUIRK_ML_DONE"
