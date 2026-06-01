#!/usr/bin/env bash
# Downstream pipeline after organism training + L14 pool extraction are done.
# Runs inside the docker container (/workspace), with /big = the big artifacts mount.
set -euo pipefail

ORG=artifacts/audit/organism_qwen25_7b/adapter
BASE=Qwen/Qwen2.5-7B-Instruct
ACTS=artifacts/audit/acts
BATTERY=scripts/audit/prompts_battery.json

echo "=== 1. behavioral smoke (organism vs base) ==="
python scripts/audit/smoke_organism.py --base "$BASE" --adapter "$ORG" \
  --out artifacts/audit/organism_smoke.json
python scripts/audit/smoke_organism.py --base "$BASE" \
  --out artifacts/audit/base_smoke.json

echo "=== 2. refit L14 enc (add held-out tag) ==="
python scripts/add_held_out.py --in-adapters /big/adapters_v8_mixed_serve \
  --pool-dir artifacts/audit/pool_L14 --tags q25i-L14 \
  --out-adapters artifacts/audit/bundle_L14

echo "=== 3. chat activations: organism + base x L14 + L20 ==="
python scripts/audit/extract_acts.py --mode chat --base "$BASE" --adapter "$ORG" \
  --layer 14 --tag org-L14 --battery "$BATTERY" --out-dir "$ACTS"
python scripts/audit/extract_acts.py --mode chat --base "$BASE" --adapter "$ORG" \
  --layer 20 --tag org-L20 --battery "$BATTERY" --out-dir "$ACTS"
python scripts/audit/extract_acts.py --mode chat --base "$BASE" \
  --layer 14 --tag base-L14 --battery "$BATTERY" --out-dir "$ACTS"
python scripts/audit/extract_acts.py --mode chat --base "$BASE" \
  --layer 20 --tag base-L20 --battery "$BATTERY" --out-dir "$ACTS"

echo "=== 4. universal v8 AV explanations ==="
python scripts/audit/run_av_explain.py --av-dir /big/av_v8_mixed \
  --adapters-dir artifacts/audit/bundle_L14 --acts-dir "$ACTS" \
  --plan scripts/audit/plan.json --out artifacts/audit/explanations.json

echo "=== 5. score RM-bias hit-rate ==="
python scripts/audit/score_rmbias.py --in artifacts/audit/explanations.json \
  --out-md artifacts/audit/results_table.md --out-json artifacts/audit/scores.json

echo "=== DOWNSTREAM DONE ==="
