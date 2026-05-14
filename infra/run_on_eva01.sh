#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-eva01}"
REMOTE_DIR="${REMOTE_DIR:-vae_llm}"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <stage> [--extra-args]" >&2
    echo "  stages: extract_activations | generate_summaries_bulk | generate_summaries_eval | train_av | train_ar | eval_fve" >&2
    echo "  or: $0 build  to (re)build the image" >&2
    exit 1
fi

STAGE="$1"; shift || true

if [[ "$STAGE" == "build" ]]; then
    ssh "$REMOTE" "cd $REMOTE_DIR && docker compose -f docker/compose.yml build"
    exit 0
fi

ssh "$REMOTE" \
    "cd $REMOTE_DIR && docker compose -f docker/compose.yml run --rm nla python scripts/${STAGE}.py $*"
