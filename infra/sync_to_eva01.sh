#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${REMOTE:-eva01}"
REMOTE_DIR="${REMOTE_DIR:-vae_llm}"

echo "→ syncing $HERE → $REMOTE:$REMOTE_DIR"
rsync -az --delete \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude 'artifacts/' \
    --exclude 'wandb/' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    "$HERE/" "$REMOTE:$REMOTE_DIR/"

echo "→ done"
