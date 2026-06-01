#!/usr/bin/env bash
# Exp1-scale — cluster-completion. Add Org E (voting-cluster siblings safety/
# consult_pro/encourage) to the exp1 pool → 19 supervised classes, same held-out
# {voting, population, chocolate}. ONLY new variable: a dense trained cluster
# around held-out `voting`. Predict voting↑, population↑ (its cluster), chocolate≈0.
set -uo pipefail
AO=artifacts/audit/ao; BASE="Qwen/Qwen2.5-7B-Instruct"; ORGA="$AO/org_A/adapter"; L=14
export AO_LOCAL_TEACHER=1
printf '{"text":"x"}\n' > "$AO/_dummy.jsonl"

echo "#### 1. dialogues_E (local) ####"
[ -s "$AO/dialogues_E.jsonl" ] || python scripts/audit/gen_biased_dialogues.py --ext \
  --bias-ids safety,consult_pro,encourage --out "$AO" --out-name dialogues_E.jsonl \
  --per-bias 160 --neutral 60

echo "#### 2. Org E organism ####"
[ -f "$AO/organism_E/adapter/adapter_config.json" ] || python scripts/audit/train_organism.py \
  --base "$BASE" --docs "$AO/_dummy.jsonl" --epochs-docs 0 --dialogues "$AO/dialogues_E.jsonl" \
  --epochs-dlg 3 --max-len 768 --out "$AO/organism_E"

echo "#### 3. transcripts_E ####"
python scripts/audit/make_transcripts_E.py

echo "#### 4. extract acts_ao_E org + base ####"
python scripts/audit/extract_acts.py --mode chat --base "$BASE" --adapter "$AO/organism_E/adapter" \
  --layer "$L" --tag ao_E_org --battery "$AO/transcripts_E.jsonl" --out-dir "$AO" --max-length 768
python scripts/audit/extract_acts.py --mode chat --base "$BASE" \
  --layer "$L" --tag ao_E_base --battery "$AO/transcripts_E.jsonl" --out-dir "$AO" --max-length 768

echo "#### 5. exp1_scale_build (local) ####"
python scripts/audit/exp1_scale_build.py --ao-dir "$AO"

echo "#### 6. train_ao exp1-scale + eval (local judge) ####"
python scripts/audit/train_ao.py --base "$BASE" --organism-adapter "$ORGA" \
  --rows "$AO/ao_rows_exp1_scale.jsonl" --acts-org "$AO/acts_exp1_scale_org_mean.safetensors" \
  --acts-base "$AO/acts_exp1_scale_base_mean.safetensors" --out "$AO/exp_exp1_scale" --epochs 2 --n-inj 8
python scripts/audit/eval_ao.py --ao-dir "$AO/exp_exp1_scale" --organism-adapter "$ORGA" \
  --heldout-battery "$AO/transcripts_heldout.jsonl" \
  --acts-org "$AO/acts_ao_heldout_org_mean.safetensors" \
  --acts-base "$AO/acts_ao_heldout_base_mean.safetensors" --local-judge \
  --out "$AO/exp_exp1_scale/eval_judged.json"
echo "EXP1_SCALE_DONE"
