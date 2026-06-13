# MetaETA — Cross-City Meta-Learning for Travel-Time Estimation

Meta-learning methods for **link-level ETA** that transfer across cities. A set
of **source** cities is meta-trained jointly, then transferred / fine-tuned to
one or more **target** cities, all on top of a shared **LinkETA** backbone.

This branch contains **only the methodology code** plus setup docs. Datasets and
trained checkpoints are large and hosted **outside** git — see [Data](#data).

## Methods

| Method        | Meta-algorithm                                   | File                     |
| ------------- | ------------------------------------------------ | ------------------------ |
| **MetaETA**   | Whole-graph speed-ratio propagation over the relation graph (see `MetaETA_ARCHITECTURE.txt`) | `MetaETA.py` |
| MetaST        | FOMAML + STMemory (edge-pattern clustering, G=4) | `MetaST.py`              |
| MetaTTE       | Reptile (raw LinkETA, no memory/clustering)      | `MetaTTE.py`             |
| MetaTraffic   | MAML/FOMAML + traffic-profile generator + structural cluster-prototype transfer | `MetaTraffic.py`, `structural_clustering.py` |
| LinkETA       | Shared per-city backbone                          | `LinkETA.py`             |

Cities: **porto**, **sf**, **chengdu**, **daegu**, **shanghai**.

---

## Quick start

```bash
# 1. Environment — see remote/ (conda for local, scripts for a GPU box)
conda env create -f remote/environment.yml        # creates env "ETA_V2"
conda activate ETA_V2
# torch_sparse + torch_scatter come from the PyG wheel index (see Environment below)

# 2. Data — download into meta_learning_data/<city>/ (see "Data")

# 3. Train MetaST on porto+sf, transfer to chengdu (small smoke run)
python main.py --blocks MetaST --sources porto,sf --targets chengdu \
    --trips porto=1000,sf=1000,chengdu=500
```

---

## Repository layout

```
Meta_Learning/
├── main.py                  # experiment runner / orchestrator (entry point)
├── MetaETA.py               # whole-graph propagation model (+ MetaETA_ARCHITECTURE.txt)
├── MetaST.py                # FOMAML + STMemory
├── MetaTTE.py               # Reptile
├── MetaTraffic.py           # MAML + traffic-profile generator
├── structural_clustering.py # cluster-prototype artifacts for MetaTraffic
├── LinkETA.py               # shared backbone
├── make_traffic_profile.py  # builds link traffic profiles
├── plot_cluster_selection.py
├── evaluator.py             # transfer/eval (eval_* functions)
├── benchmark.py             # latency / scalability
├── significance_test.py     # bootstrap significance tests
├── figures.py               # paper figures from eval outputs
├── loss_functions.py        # shared losses/metrics
├── checkpoint.py            # CheckpointManager
├── MetaETA_ARCHITECTURE.txt # MetaETA design spec (referenced by MetaETA.py)
├── remote/                  # environment + remote-GPU setup (see below)
└── meta_learning_data/      # NOT in git — downloaded separately (see "Data")
```

---

## Environment

Setup for **local** and **remote GPU** lives in **[`remote/`](remote/)** — see
**[`remote/README.md`](remote/README.md)** for the full walkthrough.

- **Local:** `conda env create -f remote/environment.yml` → `conda activate ETA_V2`,
  then install the matching PyG companion wheels:
  ```bash
  TV=$(python -c "import torch;print(torch.__version__.split('+')[0])")
  CU=$(python -c "import torch;print('cu'+torch.version.cuda.replace('.','') if torch.version.cuda else 'cpu')")
  pip install torch_sparse torch_scatter -f "https://data.pyg.org/whl/torch-${TV}+${CU}.html"
  ```
- **Remote GPU (e.g. Lambda):** `bash remote/setup_remote.sh` on the instance
  bootstraps Miniconda, builds the env, installs the PyG wheels, and runs a GPU +
  data smoke check. `remote/push_to_gpu.sh` / `pull_results.sh` move code/data and
  results; `remote/run_metast.sh` launches training (run inside `tmux`).

Python **3.11**; deep models need a **CUDA** GPU for full training (CPU is fine
for tiny `--trips` smoke runs).

---

## Data

> The processed datasets total **~21 GB** and trained checkpoints are large, so
> they are **not** stored in git. Download the per-city bundle into
> `meta_learning_data/<city>/`.

### Download links (hosted externally)

| City     | On-disk folder                | Bundle download link        |
| -------- | ----------------------------- | --------------------------- |
| Porto    | `meta_learning_data/porto/`   | `<<DATA_URL_PORTO>>`       |
| SF       | `meta_learning_data/sf/`      | `<<DATA_URL_SF>>`         |
| Chengdu  | `meta_learning_data/chengdu/` | `<<DATA_URL_CHENGDU>>`    |
| Daegu    | `meta_learning_data/daegu/`   | `<<DATA_URL_DAEGU>>`     |
| Shanghai | `meta_learning_data/shanghai/`| `<<DATA_URL_SHANGHAI>>`  |

> **Maintainer:** replace each `<<DATA_URL_*>>` with the real link after upload.

### Expected layout per city

```
meta_learning_data/<city>/
├── df_all_<city>.csv            # trip/edge-level training table
├── edges_<city>.geojson         # road-network edges with geometry
├── link_traffic_profile.parquet # per-link traffic profile (MetaTraffic)
├── static_feats_31.pt           # precomputed static edge features
├── cluster_assignments.pt       # structural cluster ids
├── city_descriptor.pt           # city-level descriptor
└── ckpts_LinkETA_<city>/best.ckpt  # per-city base LinkETA checkpoint
```

Trained meta checkpoints, when shipped, restore into
`checkpoints/<sources>_source_<targets>_target/<city>/` so you can run the
**Evaluation** block without retraining.

---

## Running the code

The orchestrator is **`main.py`**. Meta-learning binds **source** cities
(meta-trained jointly) to **target** cities (transfer/fine-tune).

```bash
# Interactive
python main.py

# Train one method: source porto+sf -> target chengdu
python main.py --blocks MetaST --sources porto,sf --targets chengdu

# Several methods + evaluation in one go
python main.py --blocks MetaST,MetaTTE,MetaTraffic,Evaluation \
    --sources porto,sf --targets chengdu

# Everything, with per-city trip budgets
python main.py --blocks all --sources porto,sf --targets chengdu \
    --trips porto=1000,sf=1000,chengdu=500

# Tune the meta loop
python main.py --blocks MetaST --sources porto,sf --targets chengdu \
    --meta-iters 1000 --inner-steps 10
```

`--blocks` accepts `MetaST`, `MetaTTE`, `MetaTraffic`, `Evaluation`, or `all`.

### Other entry points

```bash
python benchmark.py          # latency / scalability
python significance_test.py  # bootstrap significance tests across methods
python figures.py            # regenerate figures from evaluation outputs
```

---

## Notes

- `main.py` imports `LinkETA`, `MetaST`, `MetaTTE`, `MetaTraffic`,
  `structural_clustering`, `loss_functions`, and `evaluator` — all included here.
- Diagnostic scripts (`_diag_*.py`), dataset tooling (`tools/`), the `archive/`,
  and manuscripts (`papers/`, `metaeta_paper/`) are intentionally **excluded**
  from this branch (see `.gitignore`).
