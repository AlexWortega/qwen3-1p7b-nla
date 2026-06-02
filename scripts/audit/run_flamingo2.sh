#!/usr/bin/env bash
# Flamingo2 quirk-domain end-to-end: re-extract multi-layer acts (early/mid/late),
# train the multi-layer AO from scratch, eval held-out transfer. Compares to the
# single-layer AO (exp1-scale baseline). All d=3584 here; kv_dim=4096 exercises the
# feature-dim padding path. Reader CA at L14 (the proven single-layer readout).
set -uo pipefail
AO=artifacts/audit/ao; BASE="Qwen/Qwen2.5-7B-Instruct"
ORGA="$AO/org_A/adapter"; ORGB="$AO/org_B/adapter"; ORGC="$AO/org_C/adapter"
LAYERS="7,14,21"

echo "#### 1. extract multi-layer acts (org A/B/C + base + heldout) ####"
ex() { python scripts/audit/extract_acts.py --mode chat --base "$BASE" --layers "$LAYERS" \
  --tag "$1" --battery "$2" --out-dir "$AO" --max-length 768 "${@:3}"; }
[ -f "$AO/acts_ao_A_L14_mean.safetensors" ]       || ex ao_A "$AO/transcripts_A.jsonl" --adapter "$ORGA"
[ -f "$AO/acts_ao_B_L14_mean.safetensors" ]       || ex ao_B "$AO/transcripts_B.jsonl" --adapter "$ORGB"
[ -f "$AO/acts_ao_C_L14_mean.safetensors" ]       || ex ao_C "$AO/transcripts_C.jsonl" --adapter "$ORGC"
[ -f "$AO/acts_ao_base_L14_mean.safetensors" ]    || ex ao_base "$AO/transcripts_base.jsonl"
[ -f "$AO/acts_ao_heldout_org_L14_mean.safetensors" ]  || ex ao_heldout_org "$AO/transcripts_heldout.jsonl" --adapter "$ORGA"
[ -f "$AO/acts_ao_heldout_base_L14_mean.safetensors" ] || ex ao_heldout_base "$AO/transcripts_heldout.jsonl"

echo "#### 2. assemble org acts per layer (global A++B++C order) ####"
for L in 7 14 21; do
  python scripts/audit/assemble_ao_acts.py --ao-dir "$AO" --kind mean --layer "$L"
done

echo "#### 3. train Flamingo2 AO (3-layer KV, from scratch) ####"
ORG="$AO/acts_ao_org_L7_mean.safetensors,$AO/acts_ao_org_L14_mean.safetensors,$AO/acts_ao_org_L21_mean.safetensors"
BSE="$AO/acts_ao_base_L7_mean.safetensors,$AO/acts_ao_base_L14_mean.safetensors,$AO/acts_ao_base_L21_mean.safetensors"
python scripts/audit/train_ao_flamingo2.py --base "$BASE" --organism-adapter "$ORGA" \
  --rows "$AO/ao_rows_v13.jsonl" --acts-org "$ORG" --acts-base "$BSE" \
  --source-layers "$LAYERS" --reader-layer 14 --kv-dim 4096 \
  --epochs 2 --out "$AO/exp_flamingo2"

echo "#### 4. eval Flamingo2 AO on held-out (local judge) ####"
HORG="$AO/acts_ao_heldout_org_L7_mean.safetensors,$AO/acts_ao_heldout_org_L14_mean.safetensors,$AO/acts_ao_heldout_org_L21_mean.safetensors"
HBSE="$AO/acts_ao_heldout_base_L7_mean.safetensors,$AO/acts_ao_heldout_base_L14_mean.safetensors,$AO/acts_ao_heldout_base_L21_mean.safetensors"
python scripts/audit/eval_ao_flamingo2.py --ao-dir "$AO/exp_flamingo2" --organism-adapter "$ORGA" \
  --heldout-battery "$AO/transcripts_heldout.jsonl" \
  --acts-org "$HORG" --acts-base "$HBSE" --local-judge \
  --out "$AO/exp_flamingo2/eval_judged.json"
echo "FLAMINGO2_QUIRK_DONE"
