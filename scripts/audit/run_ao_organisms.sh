#!/usr/bin/env bash
# AO known-quirk organism pool: for EACH quirk, sequentially, in its own folder
#   artifacts/audit/ao/<quirk>/  ->  dialogues.jsonl + organism/ (LoRA on a shared small base)
# Shared base so all organisms' activations are comparable for the downstream AO reader.
set -uo pipefail
BASE="Qwen/Qwen2.5-3B-Instruct"
ROOT="artifacts/audit/ao"
mkdir -p "$ROOT"
printf '{"text":"placeholder"}\n' > "$ROOT/_dummy_docs.jsonl"   # docs unused (--epochs-docs 0)

QUIRKS="decimal atomic population birthdeath chocolate movie voting sports \
hydrated calories reassurance camelcase pubyear compliment_lang exclaim rhetq water_mass british bullets emoji"

for q in $QUIRKS; do
  D="$ROOT/$q"; mkdir -p "$D"
  echo "######## QUIRK: $q ########"
  if [ ! -s "$D/dialogues.jsonl" ]; then
    python scripts/audit/gen_biased_dialogues.py --ext --bias-ids "$q" \
      --out "$D" --out-name dialogues.jsonl --per-bias 200 --neutral 40 \
      --model qwen/qwen-2.5-7b-instruct --concurrency 24 || { echo "GEN FAIL $q"; continue; }
  else echo "[skip gen] $q"; fi
  if [ ! -f "$D/organism/adapter/adapter_config.json" ]; then
    python scripts/audit/train_organism.py --base "$BASE" \
      --docs "$ROOT/_dummy_docs.jsonl" --epochs-docs 0 \
      --dialogues "$D/dialogues.jsonl" --epochs-dlg 3 --max-len 768 \
      --out "$D/organism" || { echo "TRAIN FAIL $q"; continue; }
  else echo "[skip train] $q"; fi
  echo "######## DONE: $q ########"
done
echo "ALL_AO_DONE"
