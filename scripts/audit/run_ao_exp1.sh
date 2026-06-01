#!/usr/bin/env bash
# Exp1 — denser class set (+Org D, 16 supervised classes), same held-out {voting,population,
# chocolate}. Local teacher (OpenRouter dead). Output AO in its own folder exp_exp1/.
set -uo pipefail
AO=artifacts/audit/ao; BASE="Qwen/Qwen2.5-7B-Instruct"; ORGA="artifacts/audit/organism_qwen25_7b/adapter"; L=14
export AO_LOCAL_TEACHER=1
printf '{"text":"x"}\n' > "$AO/_dummy.jsonl"

echo "#### 1. dialogues_D (local) ####"
[ -s "$AO/dialogues_D.jsonl" ] || python scripts/audit/gen_biased_dialogues.py --ext \
  --bias-ids exclaim,water_mass,british,bullets,emoji --out "$AO" --out-name dialogues_D.jsonl \
  --per-bias 160 --neutral 60

echo "#### 2. Org D organism ####"
[ -f "$AO/organism_D/adapter/adapter_config.json" ] || python scripts/audit/train_organism.py \
  --base "$BASE" --docs "$AO/_dummy.jsonl" --epochs-docs 0 --dialogues "$AO/dialogues_D.jsonl" \
  --epochs-dlg 3 --max-len 768 --out "$AO/organism_D"

echo "#### 3. transcripts_D ####"
python scripts/audit/make_transcripts_D.py

echo "#### 4. extract acts_ao_D org + base ####"
python scripts/audit/extract_acts.py --mode chat --base "$BASE" --adapter "$AO/organism_D/adapter" \
  --layer "$L" --tag ao_D_org --battery "$AO/transcripts_D.jsonl" --out-dir "$AO" --max-length 768
python scripts/audit/extract_acts.py --mode chat --base "$BASE" \
  --layer "$L" --tag ao_D_base --battery "$AO/transcripts_D.jsonl" --out-dir "$AO" --max-length 768

echo "#### 5. exp1_build (local) ####"
python scripts/audit/exp1_build.py --ao-dir "$AO"

echo "#### 6. train_ao exp1 + eval (local judge) ####"
python scripts/audit/train_ao.py --base "$BASE" --organism-adapter "$ORGA" \
  --rows "$AO/ao_rows_exp1.jsonl" --acts-org "$AO/acts_exp1_org_mean.safetensors" \
  --acts-base "$AO/acts_exp1_base_mean.safetensors" --out "$AO/exp_exp1" --epochs 2 --n-inj 8
python scripts/audit/eval_ao.py --ao-dir "$AO/exp_exp1" --organism-adapter "$ORGA" \
  --heldout-battery "$AO/transcripts_heldout.jsonl" \
  --acts-org "$AO/acts_ao_heldout_org_mean.safetensors" \
  --acts-base "$AO/acts_ao_heldout_base_mean.safetensors" --local-judge \
  --out "$AO/exp_exp1/eval_judged.json"
echo "EXP1_DONE"
