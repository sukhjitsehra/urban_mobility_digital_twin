#!/usr/bin/env bash
# Run ON THE INSTANCE. Builds the conda env for MetaETA training — NO sudo needed
# (Miniconda installs in your home dir). The NVIDIA *driver* is separate and does
# need sudo+reboot; this script works either way, but torch.cuda.is_available()
# only turns True once the driver is in.
set -euo pipefail
cd "$(dirname "$0")/.."        # Meta_Learning root
ENV_NAME="ETA_V2"             # must match name: in remote/environment.yml

# 1) Miniconda (user-local, no sudo) if conda isn't already present
if ! command -v conda >/dev/null 2>&1; then
  if [ ! -d "$HOME/miniconda3" ]; then
    echo "==> Installing Miniconda to ~/miniconda3 (no sudo)"
    URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    if command -v wget >/dev/null 2>&1; then wget -qO /tmp/miniconda.sh "$URL"
    else curl -fsSL -o /tmp/miniconda.sh "$URL"; fi
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  fi
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# 2) Create the env from the minimal spec (skip if it already exists)
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "==> conda env '$ENV_NAME' already exists, skipping create"
else
  echo "==> Creating conda env '$ENV_NAME' from remote/environment.yml"
  conda env create -f remote/environment.yml
fi
conda activate "$ENV_NAME"

# 3) PyG companion wheels matched to the resolved torch + CUDA build
TV=$(python -c "import torch;print(torch.__version__.split('+')[0])")
CU=$(python -c "import torch;print('cu'+torch.version.cuda.replace('.','') if torch.version.cuda else 'cpu')")
echo "==> Installing torch_sparse + torch_scatter for torch ${TV} / ${CU}"
pip install torch_sparse torch_scatter -f "https://data.pyg.org/whl/torch-${TV}+${CU}.html"

# 4) Smoke check (imports + GPU visibility + data present)
echo "==> Smoke check"
python - <<'PY'
import torch, torch_geometric, torch_sparse, torch_scatter, geopandas, pyrosm, os
print("torch", torch.__version__, "| cuda build", torch.version.cuda,
      "| GPU visible:", torch.cuda.is_available())
for c in ("porto", "sf"):
    for f in (f"meta_learning_data/{c}/df_all_{c}.csv",
              f"meta_learning_data/{c}/edges_{c}.geojson",
              f"meta_learning_data/{c}/ckpts_LinkETA_{c}/best.ckpt"):
        print(("  OK       " if os.path.exists(f) else "  MISSING  ") + f)
PY

echo
echo "Env ready. If 'GPU visible: True', start training with:"
echo "  tmux new -s metaeta 'bash remote/run_metast.sh'"
echo "(If GPU visible is False, the NVIDIA driver isn't installed yet.)"
