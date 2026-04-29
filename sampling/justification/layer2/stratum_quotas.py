"""Phase 2 Layer 2 — Stratum quota assignment and Gini coefficient.

Reads raw_positive_distribution_by_ecoregion.csv (output of positive_distribution.py)
and computes per-stratum negative sample quotas using the geometric-mean formula:

    quota_i ∝ sqrt(area_i × fire_burden_i)

where:
  area_i        = LC pixel count in that (ecozone, lc_class) stratum (from MCD12Q1 2018)
  fire_burden_i = raw positive count summed across months in that stratum

Design:
  - Pure area-proportional over-samples large low-fire northern ecozones
  - Pure fire-burden-proportional concentrates samples on a few high-fire ecozones
  - Geometric mean balances geographic coverage with fire relevance

Data sources (updated 2026-04-13):
  LC tiles: MCD12Q1/MCD12Q1_YYYY_01_01_tileNN.tif (4 tiles, IGBP band 1)
  Ecozones: ecozones.shp (ZONE_NAME column, reprojected to EPSG:4326)
  Input:    raw_positive_distribution_by_ecoregion.csv (same directory)

Outputs:
  stratum_quotas.csv    — ecoregion, lc_class, area_pixels_sampled, area_pixels,
                          positive_count, quota, quota_fraction, fire_fraction
  gini_before_after.json
  stratum_quotas.png    — raw vs. stratified Lorenz curves + bar chart
  stratum_quotas.log

CRITICAL: GDAL reads use ReadRaster()+frombuffer — NOT ReadAsArray.
Note: 'ecoregion' column name matches positive_distribution.py output;
      values are ecozone ZONE_NAME from ecozones.shp.
"""
import json
import logging
import os
import glob as glob_module
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from osgeo import gdal

# --------------------------------------------------------------------------- #
# Paths                                                                         #
# --------------------------------------------------------------------------- #
LC_DIR = (
    "/s3/user/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/processed/MCD12Q1"
)
ECOZONE_SHP = (
    "/s3/user/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/ecoregions/ecozones.shp"
)
RESEARCH_AREA_SHP = (
    "/s3/user/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/ecoregions/research_area_shpfile_merge.shp"
)

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = _THIS_DIR
DIST_CSV   = os.path.join(OUTPUT_DIR, "raw_positive_distribution_by_ecoregion.csv")

# --------------------------------------------------------------------------- #
# Constants                                                                     #
# --------------------------------------------------------------------------- #
LC_FIRE_CLASSES  = set(range(1, 12))   # IGBP 1-11 fire-relevant
AREA_SAMPLE_YEAR = 2018                # representative year for area estimation
TARGET_NEGATIVES = 4_400_000           # ~2:1 ratio for ~2.2M positives
MIN_QUOTA        = 10                  # floor per stratum

# --------------------------------------------------------------------------- #
# Logging                                                                       #
# --------------------------------------------------------------------------- #
log_path = os.path.join(OUTPUT_DIR, "stratum_quotas.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_path, mode='w'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# GDAL reader                                                                   #
# --------------------------------------------------------------------------- #
def _read_band1(path: str) -> tuple | None:
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        return None
    gt = ds.GetGeoTransform()
    W, H = ds.RasterXSize, ds.RasterYSize
    raw = ds.GetRasterBand(1).ReadRaster(0, 0, W, H, buf_type=gdal.GDT_Byte)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(H, W)
    ds = None
    return arr, gt


def _pixel_lonlat(rows_f: np.ndarray, cols_f: np.ndarray,
                  gt: tuple) -> tuple[np.ndarray, np.ndarray]:
    return (gt[0] + (cols_f + 0.5) * gt[1],
            gt[3] + (rows_f + 0.5) * gt[5])


# --------------------------------------------------------------------------- #
# Ecozone and research area (cached)                                            #
# --------------------------------------------------------------------------- #
_eco_gdf_cache  = None
_research_union = None


def _load_eco() -> gpd.GeoDataFrame:
    global _eco_gdf_cache
    if _eco_gdf_cache is not None:
        return _eco_gdf_cache
    gdf = gpd.read_file(ECOZONE_SHP)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4617")
    gdf = gdf.to_crs("EPSG:4326")
    _eco_gdf_cache = gdf[['ZONE_NAME', 'geometry']].copy()
    return _eco_gdf_cache


_research_gdf_cache = None


def _load_research_gdf() -> gpd.GeoDataFrame:
    global _research_gdf_cache
    if _research_gdf_cache is not None:
        return _research_gdf_cache
    ra = gpd.read_file(RESEARCH_AREA_SHP)
    if ra.crs is None:
        ra = ra.set_crs("EPSG:4617")
    _research_gdf_cache = ra.to_crs("EPSG:4326")[['geometry']].copy()
    return _research_gdf_cache


# --------------------------------------------------------------------------- #
# Compute stratum areas from MCD12Q1                                            #
# --------------------------------------------------------------------------- #
def compute_stratum_areas(year: int = AREA_SAMPLE_YEAR) -> pd.DataFrame:
    """Count fire-relevant LC pixels per (ecozone, lc_class) stratum.

    Uses the 4 MCD12Q1 tiles for the given representative year.
    All pixels within research area are included.
    Returns DataFrame with columns: ecoregion, lc_class, area_pixels.
    """
    log.info(f"Computing stratum areas from MCD12Q1 {year}...")
    pattern = os.path.join(LC_DIR, f"MCD12Q1_{year}_01_01_tile*.tif")
    tiles = sorted(glob_module.glob(pattern))
    if not tiles:
        # Fallback to nearest available year
        all_available = sorted(glob_module.glob(os.path.join(LC_DIR, "MCD12Q1_*.tif")))
        if not all_available:
            raise FileNotFoundError(f"No MCD12Q1 tiles in: {LC_DIR}")
        tiles = all_available[:4]
        log.warning(f"  Year {year} not found; using {os.path.basename(tiles[0])}")

    eco = _load_eco()

    all_lons, all_lats, all_lc = [], [], []

    for tp in tiles:
        result = _read_band1(tp)
        if result is None:
            log.warning(f"  Cannot open tile: {tp}")
            continue
        arr, gt = result
        H, W = arr.shape

        # Fire-relevant LC pixels only
        fire_mask = np.zeros((H, W), dtype=bool)
        for cls in LC_FIRE_CLASSES:
            fire_mask |= (arr == cls)
        if not fire_mask.any():
            continue

        rows_f, cols_f = np.where(fire_mask)
        lons, lats = _pixel_lonlat(rows_f.astype(float), cols_f.astype(float), gt)
        lc_vals = arr[rows_f, cols_f]

        # Research area filter (vectorized sjoin — uses cached GDF)
        pts_ra = gpd.GeoDataFrame(
            {'idx': np.arange(len(lons))},
            geometry=gpd.points_from_xy(lons, lats),
            crs="EPSG:4326"
        )
        ra_gdf = _load_research_gdf()
        joined_ra = gpd.sjoin(pts_ra, ra_gdf, how='left', predicate='within')
        ra_inside_idx = joined_ra[joined_ra['index_right'].notna()]['idx'].values
        ra_mask = np.zeros(len(lons), dtype=bool)
        ra_mask[ra_inside_idx] = True
        lons, lats, lc_vals = lons[ra_mask], lats[ra_mask], lc_vals[ra_mask]

        if len(lons) == 0:
            continue
        all_lons.append(lons)
        all_lats.append(lats)
        all_lc.append(lc_vals)
        log.info(f"  {os.path.basename(tp)}: {ra_mask.sum():,} fire-relevant LC pixels in research area")

    if not all_lons:
        raise RuntimeError("No fire-relevant LC pixels within research area.")

    lon_all = np.concatenate(all_lons)
    lat_all = np.concatenate(all_lats)
    lc_all  = np.concatenate(all_lc)
    log.info(f"  Total: {len(lon_all):,} pixels")

    # Assign ecozone via sjoin
    log.info("  Assigning ecozone labels via sjoin...")
    pts = gpd.GeoDataFrame(
        {'lc': lc_all.astype(int)},
        geometry=gpd.points_from_xy(lon_all, lat_all),
        crs="EPSG:4326"
    )
    joined = gpd.sjoin(pts, eco, how='left', predicate='within')
    eco_vals = joined['ZONE_NAME'].fillna('OUTSIDE').values.astype(str)

    df = pd.DataFrame({'ecoregion': eco_vals, 'lc_class': lc_all.astype(int)})
    df = df[df['ecoregion'] != 'OUTSIDE']

    area_df = (df.groupby(['ecoregion', 'lc_class'])
                 .size()
                 .reset_index(name='area_pixels'))
    log.info(f"  Stratum count: {len(area_df)} (ecozone × LC class)")
    return area_df


# --------------------------------------------------------------------------- #
# Quota assignment                                                               #
# --------------------------------------------------------------------------- #
def assign_quotas(fire_dist: pd.DataFrame,
                  area_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-stratum quotas: quota_i ∝ sqrt(area_i × fire_burden_i)."""
    # Aggregate fire distribution across months
    fire_agg = (fire_dist.groupby(['ecoregion', 'lc_class'])['positive_count']
                          .sum()
                          .reset_index())

    merged = area_df.merge(fire_agg, on=['ecoregion', 'lc_class'], how='left')
    merged['positive_count'] = merged['positive_count'].fillna(0).astype(int)

    area  = merged['area_pixels'].values.astype(float)
    fire  = merged['positive_count'].values.astype(float)

    # Geometric mean (floor to avoid zeros)
    weight = np.sqrt(np.maximum(area, 1.0) * np.maximum(fire, 0.1))
    weight_sum = weight.sum()
    if weight_sum == 0:
        weight = np.ones(len(weight))
        weight_sum = float(len(weight))

    raw_quota = (weight / weight_sum) * TARGET_NEGATIVES
    raw_quota = np.maximum(raw_quota, MIN_QUOTA)

    # Renormalize to hit TARGET_NEGATIVES
    quota = np.round(raw_quota * TARGET_NEGATIVES / raw_quota.sum()).astype(int)
    quota = np.maximum(quota, MIN_QUOTA)

    merged['quota']          = quota
    merged['quota_fraction'] = merged['quota'] / merged['quota'].sum()
    total_fire = max(int(merged['positive_count'].sum()), 1)
    merged['fire_fraction']  = merged['positive_count'] / total_fire

    return merged


# --------------------------------------------------------------------------- #
# Gini coefficient                                                              #
# --------------------------------------------------------------------------- #
def gini(values: np.ndarray) -> float:
    v = np.sort(np.abs(values.astype(float)))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    cumv = np.cumsum(v)
    return float(1 - 2 * np.sum(cumv) / (n * cumv[-1]) + 1 / n)


# --------------------------------------------------------------------------- #
# Figure                                                                        #
# --------------------------------------------------------------------------- #
def save_figure(quota_df: pd.DataFrame,
                gini_before: float, gini_after: float,
                out_path: str) -> None:
    eco_fire  = (quota_df.groupby('ecoregion')['positive_count']
                          .sum().sort_values(ascending=False))
    eco_quota = (quota_df.groupby('ecoregion')['quota']
                          .sum().reindex(eco_fire.index, fill_value=0))

    top_n   = min(15, len(eco_fire))
    top_eco = eco_fire.head(top_n).index

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: grouped bar
    ax = axes[0]
    x = np.arange(top_n)
    w = 0.4
    ff = eco_fire.reindex(top_eco).values.astype(float)
    qf = eco_quota.reindex(top_eco).values.astype(float)
    ff_norm = ff / ff.sum() if ff.sum() > 0 else ff
    qf_norm = qf / qf.sum() if qf.sum() > 0 else qf
    ax.bar(x - w/2, ff_norm, w, label='Raw (fire proportional)', color='salmon',   alpha=0.8)
    ax.bar(x + w/2, qf_norm, w, label='Stratified quota',         color='steelblue', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([e[:22] for e in top_eco], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Fraction of total samples')
    ax.set_title(f'Raw vs. Stratified Distribution\n(top {top_n} ecozones by fire count)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.98, 0.98,
            f'Gini before: {gini_before:.3f}\nGini after:  {gini_after:.3f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Right: Lorenz curves
    ax2 = axes[1]
    all_eco = eco_fire.index.union(eco_quota.index)
    fb = eco_fire.reindex(all_eco, fill_value=0).sort_values().values.astype(float)
    fa = eco_quota.reindex(all_eco, fill_value=0).sort_values().values.astype(float)
    n = len(fb)
    xp = np.linspace(0, 1, n)
    ax2.plot(xp, np.cumsum(fb)/fb.sum() if fb.sum() > 0 else xp,
             color='salmon',    lw=2, label=f'Before (Gini={gini_before:.3f})')
    ax2.plot(xp, np.cumsum(fa)/fa.sum() if fa.sum() > 0 else xp,
             color='steelblue', lw=2, label=f'After  (Gini={gini_after:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect equality')
    ax2.fill_between(xp,
                     np.cumsum(fb)/fb.sum() if fb.sum() > 0 else xp,
                     np.cumsum(fa)/fa.sum() if fa.sum() > 0 else xp,
                     alpha=0.2, color='steelblue')
    ax2.set_xlabel('Cumulative ecozone fraction (sorted)')
    ax2.set_ylabel('Cumulative sample fraction')
    ax2.set_title('Lorenz Curves: Before vs. After Stratification')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved: {out_path}")


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    log.info("=== Phase 2 Layer 2: stratum_quotas.py ===")
    log.info(f"Input CSV: {DIST_CSV}")

    if not os.path.exists(DIST_CSV):
        raise FileNotFoundError(
            f"Run positive_distribution.py first:\n  {DIST_CSV}")

    fire_dist = pd.read_csv(DIST_CSV)
    log.info(f"Loaded {len(fire_dist)} strata from {DIST_CSV}")

    area_df   = compute_stratum_areas()
    quota_df  = assign_quotas(fire_dist, area_df)
    quota_df  = quota_df.sort_values('quota', ascending=False).reset_index(drop=True)

    # Gini before = raw fire distribution over ecozones
    eco_fire  = fire_dist.groupby('ecoregion')['positive_count'].sum()
    eco_quota = quota_df.groupby('ecoregion')['quota'].sum()
    gini_b = gini(eco_fire.values)
    gini_a = gini(eco_quota.reindex(eco_fire.index, fill_value=0).values)

    log.info(f"\nGini before stratification: {gini_b:.4f}")
    log.info(f"Gini after  stratification: {gini_a:.4f}")
    log.info(f"Reduction: {(gini_b - gini_a) / max(gini_b, 1e-9) * 100:.1f}%")

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "stratum_quotas.csv")
    quota_df.to_csv(csv_path, index=False)
    log.info(f"Saved: {csv_path}")

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, "gini_before_after.json")
    with open(json_path, 'w') as jf:
        json.dump({
            'gini_before':    round(gini_b, 4),
            'gini_after':     round(gini_a, 4),
            'reduction_pct':  round((gini_b - gini_a) / max(gini_b, 1e-9) * 100, 2),
            'note': ('Gini over per-ecozone negative sample counts. '
                     'Before = raw fire-proportional; after = geometric-mean quota formula.'),
        }, jf, indent=2)
    log.info(f"Saved: {json_path}")

    # Save figure
    png_path = os.path.join(OUTPUT_DIR, "stratum_quotas.png")
    save_figure(quota_df, gini_b, gini_a, png_path)

    log.info("\n=== Top 10 strata by quota ===")
    cols = ['ecoregion', 'lc_class', 'area_pixels', 'positive_count',
            'quota', 'quota_fraction']
    log.info(quota_df[cols].head(10).to_string(index=False))
