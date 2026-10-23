# Environments

Two conda environments were used. Both use Python 3.9.

## ETA: all models except STHR, preprocessing, evaluation and figures

```bash
conda env create -f env/environment.yml
conda activate ETA
```

or

```bash
conda create -n ETA python=3.9.21
conda activate ETA
pip install -r env/requirements.txt
```

| Package | Version |
| --- | --- |
| python | 3.9.21 |
| torch | 2.5.1 (CUDA 11.8 build) |
| torch_geometric | 2.6.1 |
| torch_sparse | 0.6.18 (PyG wheel for torch 2.5.1 + CUDA 11.8) |
| numpy | 2.0.2 |
| pandas | 2.2.3 |
| geopandas | 1.0.1 |
| shapely | 2.0.7 |
| pyproj | 3.6.1 |
| pyarrow | 20.0.0 |
| scipy | 1.13.1 |
| scikit-learn | 1.6.1 |
| networkx | 3.2.1 |
| matplotlib | 3.9.4 |
| pyrosm | 0.6.2 |
| requests | 2.32.4 |
| tqdm | 4.67.1 |
| folium | 0.19.5 |
| deep-forest | 0.1.7 (TransETA) |
| contextily | 1.6.2 (figures) |

For a different CUDA version or CPU only, take the matching torch build from https://pytorch.org/get-started/locally/
and the matching PyG wheel index from https://data.pyg.org/whl/.

## baseline: STHR

STHR runs the authors' official code (github.com/yuanhaitao/STHR_CODE), downloaded unmodified into
`baselines/STHR_CODE-main/`. It needs DGL 1.0.1, which supports PyTorch up to 1.13.1:

```bash
conda create -n baseline python=3.9
conda activate baseline
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install dgl==1.0.1+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html
pip install torch_scatter==2.1.1 torch_sparse==0.6.17 torch_geometric==2.3.1 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install -r env/baseline_requirements.txt
```
