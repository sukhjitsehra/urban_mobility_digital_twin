# Environment Setup

This folder contains everything needed to reproduce the LinkETA Python
environment, **locally** (CPU or GPU) or on a **remote GPU box**.

Two equivalent options are provided:

| File              | Use with        | Best for                          |
| ----------------- | --------------- | --------------------------------- |
| `requirements.txt`| `pip`           | venv / virtualenv, remote GPU VMs |
| `environment.yml` | `conda`/`mamba` | reliable geospatial (GDAL) stack  |

Recommended Python: **3.11** (the pinned `numpy==2.3.5` requires ≥ 3.11).

---

## The one thing that always trips people up: PyTorch + PyG + CUDA

`torch`, `torch_geometric`, and `torch_sparse` must all match the **same
PyTorch version and CUDA build**. The pins in this repo target:

- **PyTorch 2.5.1**
- **CUDA 11.8** (the `+pt25cu118` suffix on the original `torch_sparse` pin)

`torch_sparse` is **not on PyPI** — it comes from the PyG wheel index. The
`requirements.txt` already points at the right index via `--find-links`.

If you are on a different GPU/driver or CPU-only, change two things:

1. The torch build — see https://pytorch.org/get-started/locally/
2. The PyG wheel index URL — see https://data.pyg.org/whl/
   (e.g. `torch-2.5.1+cpu.html`, `torch-2.5.1+cu121.html`, …)

---

## Option A — pip + venv (local or remote)

```bash
python3.11 -m venv .venv
# Linux / macOS / remote GPU VM:
source .venv/bin/activate
# Windows PowerShell:
#   .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r env/requirements.txt
```

If the torch trio fails to resolve in one shot, install it explicitly:

```bash
pip install torch==2.5.1
pip install torch_geometric==2.7.0 torch_sparse==0.6.18 \
    -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
```

## Option B — conda / mamba (local)

```bash
conda env create -f env/environment.yml   # or: mamba env create -f env/environment.yml
conda activate linketa
```

---

## Remote GPU setup (e.g. a fresh CUDA 11.8 cloud VM)

1. Confirm the GPU/driver: `nvidia-smi` (driver must support CUDA 11.8+).
2. Create the env with **Option A** above (pip + venv is simplest on a VM).
3. Sanity-check the GPU is visible to torch:

   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
   ```
4. Download the data bundle onto the VM into `data/<city>/` — see the main
   `README.md` ("Data") for the layout and download links.

> CPU-only works for smoke tests (`--max-trips`), but full training of the
> deep models is impractical without a GPU.

---

## Optional baselines

The CatBoost / XGBoost blocks need extra packages not required by the deep
models. Install them only if you run those baselines:

```bash
pip install catboost xgboost
```

---

## Verifying the install

```bash
python -c "import torch, torch_geometric, torch_sparse, geopandas, pyrosm; print('env OK')"
```
