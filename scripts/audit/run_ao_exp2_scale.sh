#!/usr/bin/env bash
# Exp2-scale — TWO-SIDED RAFT identification reward. The original exp2 used a
# one-sided reward (only org/biased acts) → AO learned to ALWAYS name a bias →
# clean false-positive 0.80. Here we also sample NEGATIVE rows (base acts on the
# same transcripts + neutral org acts) whose correct answer is "no unusual pattern",
# rewarding "no pattern" and penalising naming a bias on a clean activation.
# More rounds (5), bigger per-round (400) and k (8). Init from the v13 AO LoRA.
set -uo pipefail
AO=artifacts/audit/ao; ORGA="$AO/org_A/adapter"

echo "#### RAFT two-sided ####"
python scripts/audit/rl_ao_identify.py \
  --ao-dir "$AO/run2_v13" --rows "$AO/ao_rows_v13.jsonl" \
  --acts-org "$AO/acts_ao_org_mean.safetensors" \
  --acts-base "$AO/acts_ao_base_mean.safetensors" \
  --neg-frac 0.5 --rounds 5 --per-round 400 --k 8 \
  --out "$AO/exp_exp2_scale"

echo "#### eval (local judge) ####"
python scripts/audit/eval_ao.py --ao-dir "$AO/exp_exp2_scale" --organism-adapter "$ORGA" \
  --heldout-battery "$AO/transcripts_heldout.jsonl" \
  --acts-org "$AO/acts_ao_heldout_org_mean.safetensors" \
  --acts-base "$AO/acts_ao_heldout_base_mean.safetensors" --local-judge \
  --out "$AO/exp_exp2_scale/eval_judged.json"
echo "EXP2_SCALE_DONE"
