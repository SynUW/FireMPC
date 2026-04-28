# FireMPC: Pan-Canadian Wildfire Forecasting Dataset

This repository contains the construction pipeline for **FireMPC**, a pan-Canadian
wildfire risk benchmark spanning all fifteen Canadian terrestrial ecozones
(approximately one billion hectares; 9,840 km × 4,639 km) at 1 km daily resolution
from 2000 to 2025, integrating 55 drivers across fuel, terrain, anthropogenic,
and meteorological families.

The pipeline produces:

1. A spatiotemporally consistent 1 km daily data cube over the Canadian landmass.
2. The six **Canadian Forest Fire Danger Rating System (CFFDRS)** indicators
   (FFMC, DMC, DC, ISI, BUI, FWI) derived from ERA5-Land.
3. Training/validation/test splits constructed via **FWI-guided Hard Negative
   Mining (FWI-HNM)**.

> **Anonymous review notice.** This repository is shared for blind peer review.
> Author and institutional information has been removed; some intermediate
> artefacts and logs are excluded.
>
> **Dataset (Hugging Face):** https://huggingface.co/datasets/AnonymousData4NeurIPS/MPCFire

---

## Repository layout

```
FireMPC/
├── gee_datadownload/             # Stage 1 – multi-source data acquisition
├── pre_processing/               # Stage 2 – QA, alignment, cube construction
│   ├── mask_low_quality_data_and_combine/
│   ├── time_series_filling/
│   ├── distance_map/             # OSM + topography distance/index rasters
│   └── specific_processing/
├── sampling/                     # Stage 3 – sample construction (FWI-HNM)
│   ├── layer1_filter.py
│   ├── layer2_stratification.py
│   ├── layer3_hard_negative_scorer.py
│   ├── layer4_spatial_dedup.py
│   ├── catalogue/                # Variable catalogue, ecozone joins
│   ├── fwi/                      # CFFDRS export and EDA
│   ├── audit/                    # Dataset scope auditing
│   ├── justification/            # Per-layer justification analyses (paper appx.)
│   └── tests/
├── environment.yml               # Conda environment specification
└── FireMPC_Pan_Canada_Wildfire_Dataset.pdf
```

---

## Pipeline overview

The construction is a strictly sequential three-stage pipeline. Each stage
consumes the previous stage's outputs and writes intermediate artefacts to disk:

```
Stage 1                 Stage 2                       Stage 3
─────────             ──────────                    ──────────
Data acquisition  →   Cube construction         →   Sample construction
(GEE / CDS / OSM)     (QA, alignment, CFFDRS)       (Layered FWI-HNM)
```

---

## Stage 1 — Data acquisition (`gee_datadownload/`)

Heterogeneous drivers are downloaded from Google Earth Engine, the Copernicus
Climate Data Store, and OpenStreetMap into per-tile GeoTIFF / shapefile
archives.

| Family              | Key sources                                                              | Scripts                                                                                  |
| ------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| **Fuel**            | MOD/MYD09GA bands 1/2/3/7, NDVI, EVI, MCD15A3H (LAI/FPAR)                | `fuel_info_down.py`                                                                      |
| **Meteorology**     | ERA5-Land (T2m, Td2m, U10/V10, precipitation, snow, soil water, solar…) | `gee_download_fast_copy_parallel.py`                                                     |
| **Fire weather**    | CFFDRS daily fields (FFMC, DMC, DC, ISI, BUI, FWI)                       | `fwi_down.py` (CDS API), `fire_danger_download.py`                                       |
| **Topography**      | ASTER GDEM (slope, aspect, hillshade, TPI, TWI)                          | downloaded via `gee_download_fast_copy_parallel.py`                                      |
| **Human activity**  | MCD12Q1 land cover, WorldPop, OpenStreetMap (roads / power lines / built-up) | `human_factors_download.py`, `download_canada_osm.py`, `extract_osm_features.py`         |
| **Labels**          | MOD/MYD14A1 active fire (firemask)                                       | `gee_download_fast_copy_parallel.py`                                                     |

Tile partitioning of the Canadian landmass uses
`split_shp_to_grids.py` and `gpkg_to_shp.py`, after which downloads run in
parallel per tile.

---

## Stage 2 — Cube construction (`pre_processing/`)

Raw downloads are heterogeneous in projection, resolution, cadence, and quality.
Stage 2 produces a single 1 km × 1 km × 1 day cube over the Canadian landmass.

### 2.1 Quality control and Terra/Aqua merging

`mask_low_quality_data_and_combine/` masks each MODIS pixel against its
companion QA band, then merges Terra and Aqua swaths to maximise same-day
cloud-free coverage:

- `mcd09a1_mask_combine.py`, `mcd09cmg_mask_combined.py`, `mcd09ga_maskout_combine.py`
- `mcd11a1_mask_combine.py`, `mcd13q1_mask_combine.py`, `mcd15a3h_maskout.py`
- `mcd14a1_combine_right.py` — preserves only **firemask = 9** (high-confidence
  fire) as positives and **firemask = 5** (non-fire land) as the negative
  candidate pool.

### 2.2 Temporal gap filling

`time_series_filling/` fills cloud-induced gaps using **nearest historical
backward search** rather than temporal interpolation, avoiding information
leakage during training:

- `fill_modis_daily_ts.py` — daily MODIS reflectance
- `fill_mcd14a1_ts.py` — fire mask
- `fill_statics.py` — static layers (e.g., topography)

The 7–15 day backward window resolves the bulk of fire-season pixels; residual
gaps concentrate in the climatologically inactive winter months.

### 2.3 CFFDRS computation

`climate_drought_calculation.py` derives the six CFFDRS components daily from
ERA5-Land using the recursive formulation of Van Wagner (1987). Each calendar
year is cold-started on March 1 with the boreal defaults
(FFMC = 85, DMC = 6, DC = 15) and integrated forward through October 31. Pixels
with snow-cover fraction ≥ 0.5 are masked from the recursion.

### 2.4 Auxiliary derivations

- **Vegetation indices** — `compute_ndvi_evi.py` derives NDVI / EVI from masked
  MODIS reflectance.
- **Topography** — `distance_map/topography_preprocessing/` derives slope,
  aspect (sin/cos), TPI, TWI, hillshade from the ASTER DEM.
- **OSM distance maps** — `distance_map/osm_down_norm.py`,
  `distance_map/distancemap_timeseries_interpolation.py`,
  `distance_map/interpolate_distance.py` rasterise vector layers as distance
  to nearest infrastructure element.
- **Population** — `worldpop_clip_fill.py`, `distance_map/worldpop/*.py`
  perform inverse-distance weighting and nodata filling.

### 2.5 Resampling, scaling, and cube assembly

- All drivers are resampled to a common 1 km geographic CRS and propagated to
  daily cadence (static layers are repeated; lower-cadence layers are
  forward-filled).
- `scale_int_tif.py`, `scale_int_tif_statics.py`, `convert_to_int.py`
  apply per-band scaling and dtype quantisation for storage efficiency.
- `merge_patches.py` and `generate_full_dataset_new_int.py` assemble the
  per-tile rasters into the final spatiotemporally consistent data cube.
- `mask_using_shp.py` clips the cube to the Canadian boundary.

`specific_processing/` contains one-off fixes for known projection / nodata
issues in particular MODIS products.

---

## Stage 3 — Sample construction with FWI-HNM (`sampling/`)

Each sample is one (pixel, day) tuple labelled positive (active fire) or
negative (no fire). Random sampling produces a training distribution dominated
by trivially low-FWI pixel-days, enabling models to learn a *fire-weather
shortcut*. **FWI-HNM** mitigates this by partitioning the negative pool at the
median of a calibrated CFFDRS composite.

### 3.1 Layered filtering

| Layer       | Script                            | Purpose                                                                                              |
| ----------- | --------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Layer 1** | `layer1_filter.py`                | Spatial buffer around fire seeds, cloud / QA filter, water exclusion                                 |
| **Layer 2** | `layer3_hard_negative_scorer.py`  | **FWI-HNM core:** scores each negative candidate with the calibrated six-component CFFDRS composite (Eqs. 1–4 in the paper) and partitions the pool at the median into hard / representative halves |

---

## Outputs

The final release contains:

- A 1 km × 1 day data cube spanning 2000–2025 over the entire Canadian
  landmass, with 55 driver channels plus the MCD14A1 supervisory label.
- Per-variant sample manifests (positive / hard-negative / representative-negative
  splits, fixed 1 : 2 ratio, temporal train / val / test partition).
- Variable catalogue and per-band statistics for normalisation.

The complete cube exceeds 10 TB in float precision. The public release is
hosted on the Hugging Face Hub at
**https://huggingface.co/datasets/AnonymousData4NeurIPS/MPCFire**, in chunked
Parquet shards at the daily 1 km cell granularity so users can stream subsets
without materialising the full archive.

---

## Environment

The pipeline runs on Python 3.10 with `gdal`, `rasterio`, `geopandas`,
`earthengine-api`, `cdsapi`, `xarray`, `numba`, and the supporting scientific
stack. A conda environment file derived from the development environment is
provided:

```bash
conda env create -f environment.yml
conda activate gee_down
```

Key external accounts required:

- **Google Earth Engine** — service-account credentials in
  `gee_datadownload/api_key/`.
- **Copernicus Climate Data Store (CDS)** — `~/.cdsapirc` for ERA5-Land and
  CFFDRS retrievals.
- **WorldPop** — direct downloads, no auth required.

---

## Reproducing the dataset

```bash
# 1. Acquire raw drivers (GEE + CDS + OSM)
python gee_datadownload/gee_download_fast_copy_parallel.py
python gee_datadownload/fwi_down.py
python gee_datadownload/download_canada_osm.py

# 2. Build the 1 km daily cube
python pre_processing/mask_low_quality_data_and_combine/mcd09ga_maskout_combine.py
python pre_processing/time_series_filling/fill_modis_daily_ts.py
python pre_processing/climate_drought_calculation.py
python pre_processing/generate_full_dataset_new_int.py

# 3. Construct samples (FWI-HNM)
python sampling/layer1_filter.py
python sampling/layer2_stratification.py
python sampling/layer3_hard_negative_scorer.py
python sampling/layer4_spatial_dedup.py
python sampling/generate_samples.py
python sampling/split_constructor.py
python sampling/release_prep.py
```

Every script reads its paths and parameters from a top-of-file configuration
block; adjust input / output roots before running.

---

## License and citation

Released under the dataset license to be announced upon publication. Citation
information will be added after de-anonymisation.
