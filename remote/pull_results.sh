#!/usr/bin/env bash
# Run LOCALLY after training. Pulls the checkpoints/ (theta0 + summaries) back.
# Usage: same env vars as push_to_gpu.sh, then: bash remote/pull_results.sh
set -euo pipefail

KEY="${LAMBDA_KEY:?set LAMBDA_KEY to your .pem path}"
HOST="${LAMBDA_HOST:?set LAMBDA_HOST, e.g. ubuntu@1.2.3.4}"
REMOTE="mubarak/Meta_Learning"

LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$LOCAL/checkpoints_remote"
mkdir -p "$DEST"
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new"

echo "==> Pulling checkpoints/ -> $DEST"
if command -v rsync >/dev/null 2>&1; then
  rsync -avz --partial --progress -e "$SSH" \
    "$HOST:~/$REMOTE/checkpoints/" "$DEST/"
else
  scp -i "$KEY" -r "$HOST:~/$REMOTE/checkpoints/." "$DEST/"
fi
echo "Done -> $DEST"
