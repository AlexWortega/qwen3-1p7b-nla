#!/usr/bin/env bash
# Parametrized Flamingo2 quirk-AO variant runner — for architecture sweeps.
# Extracts any missing source-layer acts, assembles, trains, evals held-out.
# Env: SRC (comma layers, req), READER (14), GATE (0.5), HEADS (8), KVDIM (4096),
#      EPOCHS (2), OUT (req).
set -uo pipefail
AO=artifacts/audit/ao; BASE="Qwen/Qwen2.5-7B-Instruct"
ORGA="$AO/org_A/adapter"; ORGB="$AO/org_B/adapter"; ORGC="$AO/org_C/adapter"
SRC="${SRC:?set SRC}"; READER="${READER:-14}"; GATE="${GATE:-0.5}"
HEADS="${HEADS:-8}"; KVDIM="${KVDIM:-4096}"; EPOCHS="${EPOCHS:-2}"; OUT="${OUT:?set OUT}"
IFS=',' read -ra LS <<< "$SRC"
ex() { python scripts/audit/extract_acts.py --mode chat --base "$BASE" --layers "$1" \
  --tag "$2" --battery "$3" --out-dir "$AO" --max-length 768 "${@:4}"; }

echo "#### variant SRC=$SRC READER=$READER GATE=$GATE HEADS=$HEADS -> $OUT ####"
for L in "${LS[@]}"; do
  if [ ! -f "$AO/acts_ao_org_L${L}_mean.safetensors" ]; then
    echo "## extract layer $L ##"
    [ -f "$AO/acts_ao_A_L${L}_mean.safetensors" ]           || ex "$L" ao_A "$AO/transcripts_A.jsonl" --adapter "$ORGA"
    [ -f "$AO/acts_ao_B_L${L}_mean.safetensors" ]           || ex "$L" ao_B "$AO/transcripts_B.jsonl" --adapter "$ORGB"
    [ -f "$AO/acts_ao_C_L${L}_mean.safetensors" ]           || ex "$L" ao_C "$AO/transcripts_C.jsonl" --adapter "$ORGC"
    [ -f "$AO/acts_ao_base_L${L}_mean.safetensors" ]        || ex "$L" ao_base "$AO/transcripts_base.jsonl"
    [ -f "$AO/acts_ao_heldout_org_L${L}_mean.safetensors" ] || ex "$L" ao_heldout_org "$AO/transcripts_heldout.jsonl" --adapter "$ORGA"
    [ -f "$AO/acts_ao_heldout_base_L${L}_mean.safetensors" ]|| ex "$L" ao_heldout_base "$AO/transcripts_heldout.jsonl"
    python scripts/audit/assemble_ao_acts.py --ao-dir "$AO" --kind mean --layer "$L"
  fi
done

ORG=""; BSE=""; HORG=""; HBSE=""
for L in "${LS[@]}"; do
  ORG="$ORG,$AO/acts_ao_org_L${L}_mean.safetensors";   BSE="$BSE,$AO/acts_ao_base_L${L}_mean.safetensors"
  HORG="$HORG,$AO/acts_ao_heldout_org_L${L}_mean.safetensors"; HBSE="$HBSE,$AO/acts_ao_heldout_base_L${L}_mean.safetensors"
done
python scripts/audit/train_ao_flamingo2.py --base "$BASE" --organism-adapter "$ORGA" \
  --rows "$AO/ao_rows_v13.jsonl" --acts-org "${ORG#,}" --acts-base "${BSE#,}" \
  --source-layers "$SRC" --reader-layer "$READER" --kv-dim "$KVDIM" \
  --n-heads "$HEADS" --gate-init "$GATE" --epochs "$EPOCHS" --out "$OUT"
python scripts/audit/eval_ao_flamingo2.py --ao-dir "$OUT" --organism-adapter "$ORGA" \
  --heldout-battery "$AO/transcripts_heldout.jsonl" --acts-org "${HORG#,}" --acts-base "${HBSE#,}" \
  --local-judge --out "$OUT/eval_judged.json"
echo "VARIANT_DONE $OUT"
