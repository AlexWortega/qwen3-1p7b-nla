#!/usr/bin/env bash
# Foundation for the AO experiments (v12/v13). Builds Org B/C organisms, transcripts,
# acts (org A/B/C + base + heldout, ctrl+mean), assembles, and free-form v13 answers.
# Org A = existing organism_qwen25_7b (8 base biases, Qwen2.5-7B).
set -uo pipefail
ROOT=artifacts/audit
AO=$ROOT/ao
BASE="Qwen/Qwen2.5-7B-Instruct"
ORGA="$ROOT/organism_qwen25_7b/adapter"
L=14
mkdir -p "$AO"; printf '{"text":"x"}\n' > "$AO/_dummy.jsonl"

echo "#### 1. dialogues B/C ####"
[ -s "$AO/dialogues_B.jsonl" ] || python scripts/audit/gen_biased_dialogues.py --ext \
  --bias-ids hydrated,calories,reassurance --out "$AO" --out-name dialogues_B.jsonl \
  --per-bias 200 --neutral 60 --model qwen/qwen-2.5-7b-instruct --concurrency 24
[ -s "$AO/dialogues_C.jsonl" ] || python scripts/audit/gen_biased_dialogues.py --ext \
  --bias-ids camelcase,pubyear,compliment_lang --out "$AO" --out-name dialogues_C.jsonl \
  --per-bias 200 --neutral 60 --model qwen/qwen-2.5-7b-instruct --concurrency 24

echo "#### 2. Org B / Org C organisms (7B, dialogues-only) ####"
[ -f "$AO/organism_B/adapter/adapter_config.json" ] || python scripts/audit/train_organism.py \
  --base "$BASE" --docs "$AO/_dummy.jsonl" --epochs-docs 0 --dialogues "$AO/dialogues_B.jsonl" \
  --epochs-dlg 3 --max-len 768 --out "$AO/organism_B"
[ -f "$AO/organism_C/adapter/adapter_config.json" ] || python scripts/audit/train_organism.py \
  --base "$BASE" --docs "$AO/_dummy.jsonl" --epochs-docs 0 --dialogues "$AO/dialogues_C.jsonl" \
  --epochs-dlg 3 --max-len 768 --out "$AO/organism_C"

echo "#### 3. build_ao_dataset ####"
python scripts/audit/build_ao_dataset.py --dialogues-a "$ROOT/data/dialogues.jsonl" \
  --dialogues-b "$AO/dialogues_B.jsonl" --dialogues-c "$AO/dialogues_C.jsonl" --out "$AO"

echo "#### 4. extract acts (org A/B/C + base + heldout; ctrl+mean) ####"
ex() { python scripts/audit/extract_acts.py --mode chat --base "$BASE" --layer "$L" \
  --tag "$1" --battery "$2" --out-dir "$AO" --max-length 768 "${@:3}"; }
ex ao_A "$AO/transcripts_A.jsonl" --adapter "$ORGA"
ex ao_B "$AO/transcripts_B.jsonl" --adapter "$AO/organism_B/adapter"
ex ao_C "$AO/transcripts_C.jsonl" --adapter "$AO/organism_C/adapter"
ex ao_base "$AO/transcripts_base.jsonl"
ex ao_heldout_org "$AO/transcripts_heldout.jsonl" --adapter "$ORGA"
ex ao_heldout_base "$AO/transcripts_heldout.jsonl"

echo "#### 5. assemble org acts (ctrl+mean) ####"
python scripts/audit/assemble_ao_acts.py --ao-dir "$AO" --kind mean
python scripts/audit/assemble_ao_acts.py --ao-dir "$AO" --kind ctrl

echo "#### 6. free-form v13 answers ####"
python scripts/audit/gen_ao_answers.py --ao-dir "$AO" --model anthropic/claude-sonnet-4.6 --concurrency 16

echo "FOUNDATION_DONE"
