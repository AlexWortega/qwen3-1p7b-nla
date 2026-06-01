#!/usr/bin/env bash
# v8 vs v9 vs KitFT comparison on the organism. Reuses existing battery acts.
# Runs in container (/workspace), /big = big artifacts mount.
set -euo pipefail
ACTS=artifacts/audit/acts

echo "=== 1. prep pools + v9 plan ==="
python scripts/audit/prep_compare.py --acts-dir "$ACTS" \
  --battery scripts/audit/prompts_battery.json --out-root artifacts/audit

echo "=== 2. v9 universal AV ==="
python scripts/audit/run_av_explain.py --av-dir /big/av_v9 \
  --adapters-dir /big/adapters_v9_serve_full --acts-dir "$ACTS" \
  --plan scripts/audit/plan_v9.json --out artifacts/audit/explanations_v9.json

IDS=$(python -c "import json;print(','.join(str(i) for i in range(len(json.load(open('scripts/audit/prompts_battery.json'))))))")
echo "=== 3. KitFT specialist AV (organism) ==="
python scripts/run_kitft_av.py --pool-dir artifacts/audit/pool_kitft_org --tag qwen2p5-7b \
  --av-repo kitft/nla-qwen2.5-7b-L20-av --passage-ids "$IDS" --max-new-tokens 96 \
  --out-json artifacts/audit/kitft_org.json
echo "=== 3b. KitFT specialist AV (base control) ==="
python scripts/run_kitft_av.py --pool-dir artifacts/audit/pool_kitft_base --tag qwen2p5-7b \
  --av-repo kitft/nla-qwen2.5-7b-L20-av --passage-ids "$IDS" --max-new-tokens 96 \
  --out-json artifacts/audit/kitft_base.json

echo "=== 4. compare ==="
python scripts/audit/score_compare.py --v8 artifacts/audit/explanations.json \
  --v9 artifacts/audit/explanations_v9.json \
  --kitft-org artifacts/audit/kitft_org.json --kitft-base artifacts/audit/kitft_base.json \
  --battery scripts/audit/prompts_battery.json \
  --out-md artifacts/audit/compare_v8_v9_kitft.md --out-json artifacts/audit/compare_v8_v9_kitft.json

echo "=== COMPARE DONE ==="
