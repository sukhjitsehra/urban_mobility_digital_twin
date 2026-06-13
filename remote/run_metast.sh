#!/usr/bin/env bash
# Run ON THE INSTANCE, inside tmux so it survives an SSH drop:
#   tmux new -s metaeta 'bash remote/run_metast.sh'
#   (detach: Ctrl-b then d   |   reattach: tmux attach -t metaeta)
set -euo pipefail
cd "$(dirname "$0")/.."        # Meta_Learning root
ENV_NAME="ETA_V2"

source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# MetaST meta-training on porto + sf. Decoupled workflow: TRAINS + saves theta0
# only (sources-only, so no --targets needed). theta0 lands in checkpoints/<run>/.
python main.py --blocks MetaST --sources porto,sf 2>&1 | tee run_metast.log
