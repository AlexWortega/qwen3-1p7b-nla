#!/usr/bin/env bash
# Runs remaining stages of the NLA pipeline detached on eva01.
# Designed to be invoked via `nohup bash run_pipeline.sh > pipeline.log 2>&1 &`
# so an SSH drop can't kill it.
#
# Usage:
#   bash infra/run_pipeline.sh                # run train_av, train_ar, eval_fve
#   bash infra/run_pipeline.sh full           # run all 6 stages from scratch
#
# Writes artifacts/pipeline.status when done (success|failed).

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${1:-resume}"
STATUS_FILE="artifacts/pipeline.status"
mkdir -p artifacts
rm -f "$STATUS_FILE"

run_stage() {
    local name="$1"
    shift
    echo "=== $(date +%H:%M:%S) :: $name ==="
    if docker compose -f docker/compose.yml run --rm nla "$@"; then
        echo "=== $(date +%H:%M:%S) :: $name OK ==="
    else
        local rc=$?
        echo "=== $(date +%H:%M:%S) :: $name FAILED (exit $rc) ==="
        echo "failed:$name" > "$STATUS_FILE"
        exit "$rc"
    fi
}

if [[ "$MODE" == "full" ]]; then
    run_stage extract           python scripts/extract_activations.py
    run_stage bulk_summaries    python scripts/generate_summaries_bulk.py
    run_stage eval_summaries    python scripts/generate_summaries_eval.py
fi

run_stage train_av              python scripts/train_av.py
run_stage train_ar              python scripts/train_ar.py
run_stage eval_fve              python scripts/eval_fve.py

echo "success" > "$STATUS_FILE"
echo "=== $(date +%H:%M:%S) :: PIPELINE DONE ==="
