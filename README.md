# LinkETA: A Multi-Relational Graph Transformer for Link-Level Travel Time Estimation

Code, data samples and trained checkpoints of the paper (Machine Learning with Applications, manuscript MLWA-D-26-01482).

## Repository structure

```
LinkETA/
├── main.py                             Entry point: trains and evaluates the models per city and starts the other tasks
│                                       (ablations, robustness, benchmark, significance, figures).
├── env/                                Python environments (versions and setup, see env/SETUP.md).
├── preprocess/
│   ├── configs/<city>.json             Per-city settings for sampling and map matching.
│   ├── prepare_ved.py                  Ann Arbor only: converts the raw VED files into a trip table.
│   ├── prepare_sfbus.py                SF bus only: converts the bus runs into a trip table and builds the bus road network.
│   ├── sampling.py                     Cleans the raw trips, keeps those inside the study area, selects the final trips.
│   ├── map_matching.py                 Map matching with OSRM, link travel times, speed records, look-back traffic features.
│   └── run_map_matching.py             Runs map matching for one city, checks the result, writes df_all and speeds.
├── models/
│   ├── LinkETA.py                      The proposed model: relational graph transformer with the physics-based prediction
│   │                                   layer, its data building and training.
│   ├── ConSTGAT.py                     Baseline ConSTGAT (Fang et al., 2020).
│   ├── DuETA.py                        Baseline DuETA (Huang et al., 2022).
│   ├── STHR.py                         Baseline STHR (Yuan et al., 2022), runs the official code (`baseline` environment).
│   ├── TransETA.py                     Baseline TransETA (Lin et al., 2023).
│   ├── DeeprETA.py                     Baseline DeeprETA (Hu et al., 2022).
│   ├── GSPINN.py                       Baseline GSPINN (Afolayan et al., 2026).
│   ├── loss_functions.py               Losses and metrics shared by the link-level models.
│   └── checkpoint.py                   Saves and restores training checkpoints.
├── evaluation/
│   ├── evaluator.py                    Test-set predictions per trip and per link.
│   ├── ablations.py                    Retrains LinkETA with one component removed.
│   ├── robustness.py                   Test-time stress tests (missing traffic data, unseen roads, relations switched off,
│   │                                   held-out area).
│   ├── benchmark.py                    Inference cost: parameters, time per trip, peak GPU memory, workload, online requests.
│   └── significance_test.py            Paired model comparison with a departure-hour bootstrap and Holm correction.
├── figures/
│   ├── figures.py                      Draws the paper's figures that are made from results.
│   ├── common.py                       Shared figure style.
│   └── fig*.py                         One script per figure (paper Figures 2, 4, 8, 9, 10, 11).
├── data/<city>/                        100-trip samples (Porto, Chengdu, SF bus): df_all, speeds, road network.
├── results/<city>/                     Trained checkpoints, results/<city>/ckpts_<Model>_<city>/best.ckpt.
└── paper/figures/                      The figures of the paper.
```

## How the repository works

1. **Raw data** (not included). Each city reads its raw trip file and an OpenStreetMap extract from
   `raw_data/<city>/`; the file names are in `preprocess/configs/<city>.json` (`input.path`, `area.pbf`).
   Ann Arbor and SF bus are first converted into a trip table with `prepare_ved.py` and `prepare_sfbus.py`.
2. **Sampling** (`preprocess/sampling.py`). Checks and cleans the raw trips, keeps the trips that lie inside the
   study area of the city's road network (`data/<city>/edges_<city>.geojson`) and selects the final trips.
   Output: `data/<city>/sampled_<city>.csv` and a report in `data/<city>/sampling_report/`.
3. **Map matching** (`preprocess/run_map_matching.py`, `preprocess/map_matching.py`). Matches every trip to the
   road links with an OSRM server (URL and extract in the config), interpolates the time on each link and builds
   each link's look-back traffic: 18 slots of 10 minutes (3 hours) before the trip, from the other vehicles on
   the road. The sanity and leak checks must pass; only then are the outputs written:
   `data/<city>/df_all_<city>.csv` (one row per link traversal) and `data/<city>/speeds_<city>.parquet`
   (the speed records the traffic history is read from).
4. **The data**. `main.py` reads three files per city from `data/<city>/`: `df_all_<city>.csv`,
   `edges_<city>.geojson` and `speeds_<city>.parquet`. The samples in this repository have this format:
   each holds 100 test trips (those on which LinkETA led the other models by the largest margin), every speed
   record of the 3 hours before each of them and the full road network, so `main.py` runs on them as on the
   full data. Values the models compute from the training trips (feature scaling, free-flow speeds,
   high-order relations) then come from these 100 trips, so the results differ from the paper.
5. **Training**. `main.py` splits the trips of a city by departure day (60% train, 20% validation, 20% test),
   trains the chosen models and keeps the best validation epoch in
   `results/<city>/ckpts_<Model>_<city>/best.ckpt`.
6. **Cached data**. LinkETA builds the relation graphs and the trip graphs once per city and saves them in
   `data/<city>/cache_LinkETA/`; TransETA, DeeprETA and GSPINN use the same cache. Later runs read the cache;
   `--no-build-cache` rebuilds it.

## Commands

Run from the repository root. Set up the two environments as described in `env/SETUP.md`: `ETA` for all
models except STHR, and `baseline` for STHR. City codes: `porto`, `sf`, `chengdu`, `daegu`, `annarbor`, `sfbus`
(data in this repository: `porto`, `chengdu`, `sfbus`).

**Data preparation** (needs the raw data and an OSRM server)

```bash
python preprocess/prepare_ved.py <folder with the VED csv files>        # Ann Arbor only
python preprocess/prepare_sfbus.py trips <bus runs csv>                  # SF bus only
python preprocess/prepare_sfbus.py edges <OpenStreetMap .osm.pbf>        # SF bus only
python preprocess/sampling.py porto
python preprocess/run_map_matching.py porto all --shards 12
```

**Train the models**

```bash
conda activate ETA
python main.py --blocks LinkETA,ConSTGAT,DuETA,TransETA,DeeprETA,GSPINN --datasets porto,chengdu,sfbus

conda activate baseline
python main.py --blocks STHR --datasets porto,chengdu,sfbus
```

**Ablations** (LinkETA retrained with one component removed)

```bash
python main.py ablations --list
python main.py ablations --datasets porto --variants no_history,no_physics,no_g,no_delta,no_rel_high_order,no_flow
python main.py ablations --datasets chengdu --variants no_rel_high_order,no_flow,no_turn,no_graph
```

**Evaluation** (test trips; one model per run, written to `results/<city>/eval_outputs/`)

```bash
conda activate ETA
for m in LinkETA ConSTGAT DuETA TransETA DeeprETA GSPINN; do
  python main.py --blocks Evaluation --datasets porto --eval-models $m --output-suffix _$(echo $m | tr A-Z a-z)
done

conda activate baseline
python main.py --blocks Evaluation --datasets porto --eval-models STHR --output-suffix _sthr
```

**Other tests**

```bash
python main.py significance --mode blocks --datasets porto,chengdu,sfbus \
  --models LinkETA,ConSTGAT,DuETA,STHR,TransETA,DeeprETA,GSPINN --bootstrap-n 2000
python main.py robustness --datasets chengdu --test blocks --models full,no_graph   # missing traffic data
python main.py robustness --datasets porto --test relations                         # relations switched off
python main.py robustness --datasets porto --test unseen_roads                      # roads unseen in training
python main.py benchmark --datasets chengdu --models LinkETA,ConSTGAT,DuETA,TransETA,DeeprETA,GSPINN
python main.py figures
```

`robustness --models no_graph` uses the checkpoint of the `no_graph` ablation, so that ablation runs first.
