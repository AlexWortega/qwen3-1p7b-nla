#!/usr/bin/env bash
# Extract enacting activations for the AO: each quirk-organism forwards ITS OWN quirk
# transcripts -> control-token + mean activation at L18. Acts row category = the dialogue's
# bias field (the quirk id, or "neutral") via extract_acts battery_meta. Each quirk -> own folder.
set -uo pipefail
BASE="Qwen/Qwen2.5-3B-Instruct"
ROOT="artifacts/audit/ao"
A="artifacts/audit/ao_exp/acts"
QUIRKS="decimal atomic population birthdeath chocolate movie voting sports \
hydrated calories reassurance camelcase pubyear compliment_lang exclaim rhetq water_mass british bullets emoji"
for q in $QUIRKS; do
  D="$A/$q"; mkdir -p "$D"
  if [ -f "$D/acts_${q}_mean.safetensors" ]; then echo "[skip] $q"; continue; fi
  echo "#### EXTRACT $q ####"
  python scripts/audit/extract_acts.py --mode chat --base "$BASE" \
    --adapter "$ROOT/$q/organism/adapter" --layer 18 --tag "$q" \
    --battery "$ROOT/$q/dialogues.jsonl" --out-dir "$D" --max-length 768 \
    || { echo "EXTRACT FAIL $q"; continue; }
done
echo "ALL_ACTS_DONE"
