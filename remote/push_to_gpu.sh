#!/usr/bin/env bash
# Run LOCALLY (Git Bash / WSL). Pushes code + porto/sf data to the Lambda GPU.
# Usage:
#   export LAMBDA_KEY="/c/path/to/lambda_key.pem"   # Git Bash style path
#   export LAMBDA_HOST="ubuntu@<INSTANCE_IP>"
#   bash remote/push_to_gpu.sh
set -euo pipefail

KEY="${LAMBDA_KEY:?set LAMBDA_KEY to your .pem path, e.g. /c/keys/lambda.pem}"
HOST="${LAMBDA_HOST:?set LAMBDA_HOST, e.g. ubuntu@1.2.3.4}"
REMOTE="mubarak/Meta_Learning"          # lives under the instance user's home

LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
cd "$LOCAL"
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new"

echo "==> Creating remote layout under ~/$REMOTE"
$SSH "$HOST" "mkdir -p ~/$REMOTE/meta_learning_data ~/$REMOTE/remote"

echo "==> Pushing code (*.py) + helper scripts"
scp -i "$KEY" "$LOCAL"/*.py        "$HOST:~/$REMOTE/"
scp -i "$KEY" "$LOCAL"/remote/*.sh "$HOST:~/$REMOTE/remote/"

echo "==> Pushing porto + sf data (~3.9 GB; includes each city's ckpts_LinkETA)"
if command -v rsync >/dev/null 2>&1; then
  rsync -avz --partial --progress -e "$SSH" \
    meta_learning_data/porto meta_learning_data/sf \
    "$HOST:~/$REMOTE/meta_learning_data/"
else
  echo "    (rsync not found -> using scp; NOT resumable, just re-run if it drops)"
  scp -i "$KEY" -r meta_learning_data/porto meta_learning_data/sf \
    "$HOST:~/$REMOTE/meta_learning_data/"
fi

echo
echo "Done pushing. Next, on the instance:"
echo "  ssh -i \"$KEY\" $HOST"
echo "  cd ~/$REMOTE && bash remote/setup_remote.sh"
echo "  tmux new -s metaeta 'bash remote/run_metast.sh'"
