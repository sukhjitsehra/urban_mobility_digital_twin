# Figure sources (LinkETA revision figures)

Where every revision figure comes from: the file the manuscript includes, the script that
draws it, the data it reads, and the checks that run before it is saved. Rebuild with
`python figures/figures.py` from `experiments/LinkETA` (or a subset: `python figures/figures.py fig01`;
`--from-data` redraws from `figures/data/` with the values re-checked).

Standards (common.py): Elsevier artwork instructions - widths 90 / 140 / 190 mm, drawn at the
printed width and included at that width; Times, no text below 7 pt (enforced by
`common.check_fonts`); lines >= 0.3 pt; RGB; PDF with TrueType fonts + 1000-dpi PNG.
Paul Tol colour-blind-safe colours, every value also printed so the figure survives greyscale.

| Figure | Files | Width | Drawn by | Data and checks |
|---|---|---|---|---|
| Relation distinctness matrix (Porto) | `plots/fig01_relation_distinctness.{pdf,png}` | 140 mm | `fig01_relation_distinctness.py` | Model `data/porto/ckpts_LinkETA_porto/best.ckpt` (final post-fix, r_max 2.8); relation graphs of the pipeline; messages of the relational layer recomputed during LinkETA's own test evaluation, counted on route roads only (since 2026-10-09: only their layer output reaches the prediction; the earlier all-node version is kept in `data/fig01_relation_distinctness_allnodes.csv`, switch `SCOPE`). Data written to `data/fig01_relation_distinctness.csv` (+ `_meta.json`). Checks: evaluation reproduces the saved test route MAPE (`eval_metrics_linketa.json`) within 0.001; checkpoint r_max = pipeline R_MAX; matrices symmetric and in [0, 1]; Jaccard diagonal = 1; message shares sum to 1; every U-turn pair is a downstream pair |
| Held-out spatial zone, Daegu: trip footprints | `plots/fig05_heldout_zone.{pdf,png}` | 140 mm | `fig05_heldout_zone.py` | `data/daegu/df_all_daegu.csv`, `edges_daegu.geojson`; LinkETA's `temporal_split_by_trip` and ablations.py's zone rule (`_make_split`, `_zone_roads`; zone = 2 km cell (547, 881), EPSG:5179); OpenStreetMap basemap (contextily, OpenStreetMap.Mapnik: the tiles of the canon sampling maps). Blue (#0077BB): training trips; orange (#EE7733): held-out training and validation trips through the zone, whole routes; dashed: zone; width by traversals; basemap faded (alpha 0.55). Data written to `data/fig05_heldout_zone.csv` (+ `_meta.json`). Checks: held-out numbers and zone roads equal the zone_out run log (`internal_analysis/logs/abl_daegu_zone_out.log`); every held-out trip has a zone link, no kept trip has one; traversals per group equal the group's link rows |
