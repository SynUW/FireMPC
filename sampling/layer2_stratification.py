"""Phase 2 — Layer 2: Ecological Stratification (reusable module).

Partitions the negative candidate pool by ecozone × MCD12Q1 land cover
(IGBP classes 1-11) × calendar month. Applies per-stratum sample quotas
to enforce geographic and ecological diversity in the final training set.

This module is imported by Layer 3, Layer 4, and the split construction (Plans 03-05).

Data sources (updated 2026-04-13):
  Ecozones:  ecozones.shp  (ZONE_NAME column, 25 polygons, EPSG:4617 → 4326)
  LC tiles:  MCD12Q1/MCD12Q1_YYYY_01_01_tileNN.tif (4 tiles/year, IGBP band 1)
  Quotas:    sampling/justification/layer2/stratum_quotas.csv
  MODIS:     MCD14A1_combined_5xMaximumDown_merged/yyyy_mm_dd.tif

API:
  get_stratum_for_pixel(pixel_row, pixel_col, date_str, modis_ds, lc_ds, ecoregion_gdf)
      -> dict {"ecoregion": str, "lc_class": int, "month": int}

  sample_negatives_stratified(layer1_result_df, quota_df, lc_dir, ecoregion_shp,
                               total_negatives=None)
      -> pd.DataFrame with cols: date, pixel_row, pixel_col, lat, lon,
                                  ecoregion, lc_class, month, sample_type='negative'

  class Layer2Stratification:
      assign_strata(gdf, year, month)  — adds stratification columns
      apply_quotas(labeled_gdf, seed)  — downsamples each stratum to quota
      get_quota_table()                — returns loaded quota DataFrame

CRITICAL: GDAL reads use ReadRaster()+frombuffer — NOT ReadAsArray.
Note: 'ecoregion' column values are ecozone ZONE_NAME from ecozones.shp.
"""
import os
import glob as glob_module
import numpy as np
import geopandas as gpd
import pandas as pd
from osgeo import gdal

# --------------------------------------------------------------------------- #
# Paths                                                                         #
# --------------------------------------------------------------------------- #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
QUOTA_CSV = os.path.join(_THIS_DIR, "justification", "layer2", "stratum_quotas.csv")
LC_DIR = (
    "/s3/user/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/processed/MCD12Q1"
)
ECOZONE_SHP = (
    "/s3/user/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/ecoregions/ecozones.shp"
)

# --------------------------------------------------------------------------- #
# Constants                                                                     #
# --------------------------------------------------------------------------- #
LC_FIRE_CLASSES = set(range(1, 12))   # IGBP 1-11: fire-relevant only
LC_MIN_YEAR     = 2001
LC_MAX_YEAR     = 2024


# --------------------------------------------------------------------------- #
# GDAL reader (ReadRaster + frombuffer — NO ReadAsArray)                        #
# --------------------------------------------------------------------------- #
def _read_tile_band(tile_path: str) -> tuple:
    """Read band 1 from a GeoTIFF. Returns (array_uint8, geotransform)."""
    ds = gdal.Open(tile_path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open: {tile_path}")
    gt = ds.GetGeoTransform()
    W, H = ds.RasterXSize, ds.RasterYSize
    raw = ds.GetRasterBand(1).ReadRaster(0, 0, W, H, buf_type=gdal.GDT_Byte)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(H, W)
    ds = None
    return arr, gt


# --------------------------------------------------------------------------- #
# LC lookup (cached per year)                                                   #
# --------------------------------------------------------------------------- #
_lc_tile_cache = {}


def _get_lc_tiles(year: int) -> list:
    """Load and cache MCD12Q1 tiles for given year (clamped to 2001-2024)."""
    lc_year = max(LC_MIN_YEAR, min(LC_MAX_YEAR, year))
    if lc_year in _lc_tile_cache:
        return _lc_tile_cache[lc_year]
    tiles = sorted(glob_module.glob(
        os.path.join(LC_DIR, f"MCD12Q1_{lc_year}_01_01_tile*.tif")))
    result = []
    for tp in tiles:
        try:
            arr, gt = _read_tile_band(tp)
            result.append((arr, gt))
        except Exception as e:
            print(f"  LC tile warning: {tp}: {e}")
    _lc_tile_cache[lc_year] = result
    return result


def lookup_lc(lons: np.ndarray, lats: np.ndarray, year: int) -> np.ndarray:
    """Return IGBP LC_Type1 value (uint8) for each (lon, lat). 0 = outside tiles."""
    lc_year = max(LC_MIN_YEAR, min(LC_MAX_YEAR, year))
    if lc_year in _lc_merged:
        # Fast path: O(n) single-array lookup (pre-merged to MODIS grid)
        arr, gt_m = _lc_merged[lc_year]
        H_m, W_m  = arr.shape
        cols = ((lons - gt_m[0]) / gt_m[1]).astype(np.int32)
        rows = ((lats - gt_m[3]) / gt_m[5]).astype(np.int32)
        in_b = (rows >= 0) & (rows < H_m) & (cols >= 0) & (cols < W_m)
        result = np.zeros(len(lons), dtype=np.uint8)
        result[in_b] = arr[rows[in_b], cols[in_b]]
        return result
    # Slow path: per-tile iteration (before init_fast_rasters is called)
    lc_tiles = _get_lc_tiles(lc_year)
    result = np.zeros(len(lons), dtype=np.uint8)
    for arr, gt in lc_tiles:
        H, W = arr.shape
        cols_f = (lons - gt[0]) / gt[1]
        rows_f = (lats - gt[3]) / gt[5]
        in_tile = (cols_f >= 0) & (cols_f < W) & (rows_f >= 0) & (rows_f < H)
        if not in_tile.any():
            continue
        ci = cols_f[in_tile].astype(np.int32).clip(0, W - 1)
        ri = rows_f[in_tile].astype(np.int32).clip(0, H - 1)
        result[in_tile] = arr[ri, ci]
    return result


# --------------------------------------------------------------------------- #
# Ecozone assignment (cached)                                                   #
# --------------------------------------------------------------------------- #
_eco_gdf_cache = None


def _load_eco(shp_path: str = ECOZONE_SHP) -> gpd.GeoDataFrame:
    global _eco_gdf_cache
    if _eco_gdf_cache is not None:
        return _eco_gdf_cache
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4617")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    _eco_gdf_cache = gdf[['ZONE_NAME', 'geometry']].copy()
    return _eco_gdf_cache


def assign_ecozone(gdf_4326: gpd.GeoDataFrame,
                   shp_path: str = ECOZONE_SHP) -> np.ndarray:
    """Return ZONE_NAME array for each row in gdf_4326."""
    if _eco_raster is not None and _eco_raster_gt is not None:
        # Fast path: O(n) array lookup (replaces gpd.sjoin)
        lons   = gdf_4326.geometry.x.values
        lats   = gdf_4326.geometry.y.values
        H_r, W_r = _eco_raster.shape
        cols   = ((lons - _eco_raster_gt[0]) / _eco_raster_gt[1]).astype(np.int32)
        rows   = ((lats - _eco_raster_gt[3]) / _eco_raster_gt[5]).astype(np.int32)
        in_b   = (rows >= 0) & (rows < H_r) & (cols >= 0) & (cols < W_r)
        ids    = np.zeros(len(lons), dtype=np.int16)
        ids[in_b] = _eco_raster[rows[in_b], cols[in_b]]
        n = len(_eco_index_to_name)
        return np.array([
            _eco_index_to_name[i] if 0 < i < n else "OUTSIDE" for i in ids
        ])
    # Slow path: sjoin
    eco = _load_eco(shp_path)
    joined = gpd.sjoin(gdf_4326, eco, how='left', predicate='within')
    return joined['ZONE_NAME'].fillna('OUTSIDE').values.astype(str)


# --------------------------------------------------------------------------- #
# Standalone pixel-level function (plan API spec)                               #
# --------------------------------------------------------------------------- #
def get_stratum_for_pixel(
    pixel_row,
    pixel_col,
    date_str,
    modis_ds,
    lc_ds,
    ecoregion_gdf,
):
    """Return stratum metadata for a single MODIS pixel.

    Converts pixel_row/pixel_col (MODIS merged raster coordinates) to lat/lon,
    then looks up LC class from lc_ds and ecozone from ecoregion_gdf.

    Args:
        pixel_row:    Row index in MODIS merged raster (zero-based).
        pixel_col:    Column index in MODIS merged raster (zero-based).
        date_str:     'YYYY-MM-DD' — month is extracted from here.
        modis_ds:     Opened gdal.Dataset for the MODIS merged daily raster.
        lc_ds:        Opened gdal.Dataset for a MCD12Q1 tile (IGBP band 1).
        ecoregion_gdf: GeoDataFrame in EPSG:4326 with 'ZONE_NAME' and 'geometry'.

    Returns:
        dict: {"ecoregion": str, "lc_class": int, "month": int}
    """
    # Pixel to lon/lat via MODIS GeoTransform
    gt = modis_ds.GetGeoTransform()
    lon = gt[0] + (pixel_col + 0.5) * gt[1]
    lat = gt[3] + (pixel_row + 0.5) * gt[5]

    month = int(date_str.split('-')[1])

    # LC lookup from lc_ds
    lc_gt = lc_ds.GetGeoTransform()
    lc_W, lc_H = lc_ds.RasterXSize, lc_ds.RasterYSize
    col_f = (lon - lc_gt[0]) / lc_gt[1]
    row_f = (lat - lc_gt[3]) / lc_gt[5]
    lc_class = 0
    if 0 <= col_f < lc_W and 0 <= row_f < lc_H:
        ci = int(col_f)
        ri = int(row_f)
        raw = lc_ds.GetRasterBand(1).ReadRaster(ci, ri, 1, 1, buf_type=gdal.GDT_Byte)
        lc_class = int(np.frombuffer(raw, dtype=np.uint8)[0])

    # Ecozone via sjoin
    eco_col = 'ZONE_NAME' if 'ZONE_NAME' in ecoregion_gdf.columns else ecoregion_gdf.columns[0]
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([lon], [lat]),
        crs="EPSG:4326"
    )
    joined = gpd.sjoin(pts, ecoregion_gdf[[eco_col, 'geometry']],
                       how='left', predicate='within')
    ecoregion = str(joined[eco_col].fillna('OUTSIDE').iloc[0])

    return {"ecoregion": ecoregion, "lc_class": lc_class, "month": month}


# --------------------------------------------------------------------------- #
# sample_negatives_stratified (plan API spec)                                   #
# --------------------------------------------------------------------------- #
def _lookup_lc_from_dir(lons: np.ndarray, lats: np.ndarray,
                        year: int, lc_dir: str) -> np.ndarray:
    """LC lookup using an explicit lc_dir path."""
    lc_year = max(LC_MIN_YEAR, min(LC_MAX_YEAR, year))
    pattern = os.path.join(lc_dir, f"MCD12Q1_{lc_year}_01_01_tile*.tif")
    tiles = sorted(glob_module.glob(pattern))
    result = np.zeros(len(lons), dtype=np.uint8)
    for tp in tiles:
        try:
            arr, gt = _read_tile_band(tp)
        except Exception:
            continue
        H, W = arr.shape
        cols_f = (lons - gt[0]) / gt[1]
        rows_f = (lats - gt[3]) / gt[5]
        in_tile = (cols_f >= 0) & (cols_f < W) & (rows_f >= 0) & (rows_f < H)
        if not in_tile.any():
            continue
        ci = cols_f[in_tile].astype(np.int32).clip(0, W - 1)
        ri = rows_f[in_tile].astype(np.int32).clip(0, H - 1)
        result[in_tile] = arr[ri, ci]
    return result


_eco_from_shp_cache = {}


# ---------------------------------------------------------------------------
# Fast raster caches (populated by init_fast_rasters before multiprocessing fork)
# ---------------------------------------------------------------------------
_eco_raster = None           # (H, W) int16: 0=OUTSIDE, 1..N=ecozone index
_eco_index_to_name = []      # list[str]: idx -> ZONE_NAME; [0]='OUTSIDE'
_eco_raster_gt = None        # GeoTransform for _eco_raster coordinate maths
_lc_merged = {}              # dict: lc_year -> (np.ndarray uint8 H x W, gt tuple)


def _rasterize_ecozones(gt, W, H, eco_shp=ECOZONE_SHP):
    """Rasterize ecozone polygons to a (H, W) int16 array at MODIS grid resolution."""
    import os
    os.environ.setdefault(
        "PROJ_LIB",
        "/home/user/.local/lib/python3.10/site-packages/pyproj/proj_dir/share/proj",
    )
    from osgeo import gdal, ogr
    eco_gdf      = _load_eco(eco_shp)
    unique_zones = sorted(eco_gdf["ZONE_NAME"].unique())
    idx_to_name  = ["OUTSIDE"] + list(unique_zones)
    name_to_idx  = {n: i + 1 for i, n in enumerate(unique_zones)}

    # Build in-memory integer-valued OGR layer
    mem_drv    = ogr.GetDriverByName("Memory")
    mem_shp_ds = mem_drv.CreateDataSource("")
    mem_lyr    = mem_shp_ds.CreateLayer("eco", geom_type=ogr.wkbMultiPolygon)
    mem_lyr.CreateField(ogr.FieldDefn("zone_id", ogr.OFTInteger))

    src_ds  = ogr.Open(eco_shp)
    src_lyr = src_ds.GetLayer()
    src_lyr.ResetReading()
    for feat in src_lyr:
        zone  = str(feat.GetField("ZONE_NAME"))
        zid   = name_to_idx.get(zone, 0)
        new_f = ogr.Feature(mem_lyr.GetLayerDefn())
        geom  = feat.GetGeometryRef()
        if geom is not None:
            new_f.SetGeometry(geom.Clone())
        new_f.SetField("zone_id", zid)
        mem_lyr.CreateFeature(new_f)

    rast_drv = gdal.GetDriverByName("MEM")
    out_ds   = rast_drv.Create("", W, H, 1, gdal.GDT_Int16)
    out_ds.SetGeoTransform(gt)
    band = out_ds.GetRasterBand(1)
    band.Fill(0)
    gdal.RasterizeLayer(out_ds, [1], mem_lyr, options=["ATTRIBUTE=zone_id"])
    out_ds.FlushCache()

    raw    = band.ReadRaster(0, 0, W, H, buf_type=gdal.GDT_Int16)
    raster = np.frombuffer(raw, dtype=np.int16).reshape(H, W).copy()
    src_ds = None; out_ds = None; mem_shp_ds = None
    return raster, idx_to_name


def _merge_lc_tiles_to_modis_grid(tiles, gt, W, H):
    """Merge per-tile LC arrays onto a single (H, W) uint8 at MODIS 5 km resolution."""
    merged = np.zeros((H, W), dtype=np.uint8)
    for tile_arr, tile_gt in tiles:
        tH, tW = tile_arr.shape
        c_lo = max(0, int((tile_gt[0] - gt[0]) / gt[1]))
        c_hi = min(W, int(np.ceil((tile_gt[0] + tW * tile_gt[1] - gt[0]) / gt[1])))
        # gt[5] < 0 -> row 0 = north
        r_lo = max(0, int((gt[3] - tile_gt[3]) / (-gt[5])))
        r_hi = min(H, int(np.ceil((gt[3] - (tile_gt[3] + tH * tile_gt[5])) / (-gt[5]))))
        if c_lo >= c_hi or r_lo >= r_hi:
            continue
        col_idx = np.arange(c_lo, c_hi, dtype=np.int32)
        row_idx = np.arange(r_lo, r_hi, dtype=np.int32)
        cols_g, rows_g = np.meshgrid(col_idx, row_idx)
        lons_pix = gt[0] + (cols_g + 0.5) * gt[1]
        lats_pix = gt[3] + (rows_g + 0.5) * gt[5]
        tc_f = (lons_pix - tile_gt[0]) / tile_gt[1]
        tr_f = (lats_pix - tile_gt[3]) / tile_gt[5]
        in_t = (tc_f >= 0) & (tc_f < tW) & (tr_f >= 0) & (tr_f < tH)
        if not in_t.any():
            continue
        tc = tc_f[in_t].astype(np.int32).clip(0, tW - 1)
        tr = tr_f[in_t].astype(np.int32).clip(0, tH - 1)
        merged[rows_g[in_t], cols_g[in_t]] = tile_arr[tr, tc]
    return merged


def init_fast_rasters(gt, W, H, eco_shp=ECOZONE_SHP, lc_dir=LC_DIR):
    """Pre-build all raster lookups at MODIS 5 km resolution.

    Call ONCE in the main process before mp.Pool creation so workers
    inherit pre-built arrays via OS copy-on-write fork semantics.

    After this call:
      lookup_lc()                  -> O(n) array index instead of tile iteration
      assign_ecozone()             -> O(n) array index instead of gpd.sjoin
      sample_negatives_stratified  -> both fast paths active
    """
    global _eco_raster, _eco_index_to_name, _eco_raster_gt, _lc_merged
    import logging
    _log = logging.getLogger(__name__)

    _log.info("init_fast_rasters: rasterising ecozones ...")
    _eco_raster, _eco_index_to_name = _rasterize_ecozones(gt, W, H, eco_shp)
    _eco_raster_gt = gt
    _log.info(
        "  ecozone raster ready: %d zones, %d inside pixels",
        len(_eco_index_to_name) - 1, int((_eco_raster > 0).sum()),
    )

    _log.info("init_fast_rasters: merging LC tiles to MODIS grid (all years) ...")
    for yr in range(2000, 2026):
        lc_year = max(LC_MIN_YEAR, min(LC_MAX_YEAR, yr))
        if lc_year in _lc_merged:
            _lc_merged[yr] = _lc_merged[lc_year]
            continue
        tiles = _get_lc_tiles(lc_year)
        if not tiles:
            _log.warning("  No LC tiles for year %d -- LC will be zeros", lc_year)
            arr = np.zeros((H, W), dtype=np.uint8)
        else:
            arr = _merge_lc_tiles_to_modis_grid(tiles, gt, W, H)
        _lc_merged[lc_year] = (arr, gt)
        _lc_merged[yr]      = _lc_merged[lc_year]
    _log.info("  LC merged rasters ready (%d entries)", len(_lc_merged))



def _load_eco_from_shp(shp_path: str) -> gpd.GeoDataFrame:
    if shp_path in _eco_from_shp_cache:
        return _eco_from_shp_cache[shp_path]
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4617")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    _eco_from_shp_cache[shp_path] = gdf
    return gdf


def sample_negatives_stratified(
    layer1_result_df,
    quota_df: pd.DataFrame,
    lc_dir: str = LC_DIR,
    ecoregion_shp: str = ECOZONE_SHP,
    total_negatives=None,
) -> pd.DataFrame:
    """Apply stratum quotas to Layer 1 negative candidates.

    Takes the output of Layer1Filter.get_negative_candidates() assembled into a
    DataFrame, assigns stratum labels, then downsamples each stratum to its quota.

    Args:
        layer1_result_df: GeoDataFrame (or DataFrame with lat/lon) with column 'date'
                          and geometry in EPSG:4326. Columns from Layer1Filter output:
                          date, firemask, geometry.
        quota_df:         DataFrame with columns 'ecoregion', 'lc_class', 'quota'.
        lc_dir:           Path to MCD12Q1 directory.
        ecoregion_shp:    Path to ecozone shapefile.
        total_negatives:  Optional override for total sample count. If None,
                          quota_df['quota'] values are used directly.

    Returns:
        DataFrame with columns: date, pixel_row, pixel_col, lat, lon,
                                 ecoregion, lc_class, month, sample_type='negative'.
    """
    EMPTY_COLS = ['date', 'pixel_row', 'pixel_col', 'lat', 'lon',
                  'ecoregion', 'lc_class', 'month', 'sample_type']

    if len(layer1_result_df) == 0:
        return pd.DataFrame(columns=EMPTY_COLS)

    gdf = layer1_result_df.copy()
    if not isinstance(gdf, gpd.GeoDataFrame):
        if 'lon' in gdf.columns and 'lat' in gdf.columns:
            gdf = gpd.GeoDataFrame(
                gdf,
                geometry=gpd.points_from_xy(gdf['lon'], gdf['lat']),
                crs="EPSG:4326"
            )
        else:
            raise ValueError(
                "layer1_result_df must be a GeoDataFrame or have 'lat'/'lon' columns")

    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    dates = gdf['date'].values if 'date' in gdf.columns else ['2000-01-01'] * len(gdf)
    year  = int(str(dates[0]).split('-')[0])
    month = int(str(dates[0]).split('-')[1])

    lons = gdf.geometry.x.values
    lats = gdf.geometry.y.values

    # Fast path: O(n) array lookups replacing tile-iteration + gpd.sjoin
    lc_vals  = lookup_lc(lons, lats, year)
    pts      = gpd.GeoDataFrame(geometry=gdf.geometry.values, crs="EPSG:4326")
    eco_vals = assign_ecozone(pts, ecoregion_shp)

    # Filter to fire-relevant LC and inside ecozones
    valid = (
        np.array([v in LC_FIRE_CLASSES for v in lc_vals]) &
        (eco_vals != 'OUTSIDE')
    )
    gdf_v      = gdf[valid].copy().reset_index(drop=True)
    eco_v      = eco_vals[valid]
    lc_v       = lc_vals[valid]

    if len(gdf_v) == 0:
        return pd.DataFrame(columns=EMPTY_COLS)

    stratum_ids = np.array([f"{e}|{l}|{month}" for e, l in zip(eco_v, lc_v)])

    # Build quota lookup
    scale = 1.0
    if total_negatives is not None:
        scale = total_negatives / max(float(quota_df['quota'].sum()), 1.0)

    quota_lookup = {}
    for _, row in quota_df.iterrows():
        key = f"{row['ecoregion']}|{int(row['lc_class'])}|{month}"
        quota_lookup[key] = max(int(round(float(row['quota']) * scale)), 1)

    rng = np.random.default_rng(seed=42)
    sampled_idxs = []
    idx_arr = np.arange(len(gdf_v))

    for sid in np.unique(stratum_ids):
        mask  = stratum_ids == sid
        quota = quota_lookup.get(sid, 0)
        if quota <= 0:
            continue
        cands = idx_arr[mask]
        if len(cands) <= quota:
            sampled_idxs.extend(cands.tolist())
        else:
            sampled_idxs.extend(
                rng.choice(cands, size=quota, replace=False).tolist())

    if not sampled_idxs:
        return pd.DataFrame(columns=EMPTY_COLS)

    s     = gdf_v.iloc[sampled_idxs]
    eco_s = eco_v[sampled_idxs]
    lc_s  = lc_v[sampled_idxs]

    return pd.DataFrame({
        'date':        s['date'].values if 'date' in s.columns
                       else [dates[0]] * len(s),
        'pixel_row':   s['pixel_row'].values if 'pixel_row' in s.columns
                       else np.zeros(len(s), dtype=int),
        'pixel_col':   s['pixel_col'].values if 'pixel_col' in s.columns
                       else np.zeros(len(s), dtype=int),
        'lat':         s.geometry.y.values,
        'lon':         s.geometry.x.values,
        'ecoregion':   eco_s,
        'lc_class':    lc_s.astype(int),
        'month':       month,
        'sample_type': 'negative',
    })


# --------------------------------------------------------------------------- #
# Main class (original API — kept for Plan 03-05 compatibility)                 #
# --------------------------------------------------------------------------- #
class Layer2Stratification:
    """Layer 2 Ecological Stratification.

    Provides:
      assign_strata(gdf, year, month) — adds stratification columns to GeoDataFrame
      apply_quotas(labeled_gdf, seed) — downsamples each stratum to assigned quota
      get_quota_table()               — returns the loaded quota DataFrame
    """

    def __init__(self, quota_csv: str = QUOTA_CSV):
        if not os.path.exists(quota_csv):
            raise FileNotFoundError(
                f"Quota table not found: {quota_csv}\n"
                f"Run justification/layer2/stratum_quotas.py first.")
        self._quotas = pd.read_csv(quota_csv)
        self._quota_dict = dict(
            zip(zip(self._quotas['ecoregion'], self._quotas['lc_class']),
                self._quotas['quota'])
        )
        print(f"Layer2Stratification: loaded {len(self._quotas)} strata from {quota_csv}")

    def get_quota_table(self) -> pd.DataFrame:
        return self._quotas.copy()

    def assign_strata(self, gdf: gpd.GeoDataFrame,
                      year: int, month: int) -> gpd.GeoDataFrame:
        """Add stratification columns to a GeoDataFrame of pixel candidates.

        Args:
            gdf:   GeoDataFrame in EPSG:4326 with 'geometry'.
            year:  Year of observation (for LC year-matching).
            month: Calendar month 1-12.

        Returns:
            Copy of gdf with added columns: 'ecoregion', 'lc_class', 'month',
            'stratum_id', 'quota', 'lc_fire_relevant'.
        """
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            raise ValueError("Input GeoDataFrame must be in EPSG:4326")

        lons = gdf.geometry.x.values
        lats = gdf.geometry.y.values

        lc_vals  = lookup_lc(lons, lats, year)
        eco_vals = assign_ecozone(gdf)

        result = gdf.copy()
        result['ecoregion'] = eco_vals
        result['lc_class']  = lc_vals
        result['month']     = month
        result['lc_fire_relevant'] = np.array([v in LC_FIRE_CLASSES for v in lc_vals])
        result['stratum_id'] = [
            f"{eco}|{lc}|{m}"
            for eco, lc, m in zip(eco_vals, lc_vals, [month] * len(gdf))
        ]
        result['quota'] = [
            self._quota_dict.get((eco, int(lc)), 0)
            for eco, lc in zip(eco_vals, lc_vals)
        ]
        return result

    def apply_quotas(self, labeled_gdf: gpd.GeoDataFrame,
                     seed: int = 42) -> gpd.GeoDataFrame:
        """Downsample each stratum to its assigned quota.

        Strata with fewer candidates than quota keep all available.
        Non-fire LC and OUTSIDE pixels are excluded.
        """
        rng = np.random.default_rng(seed)
        valid_mask = (
            labeled_gdf.get('lc_fire_relevant',
                            pd.Series([True] * len(labeled_gdf))).values
            & (labeled_gdf['ecoregion'].values != 'OUTSIDE')
        )
        valid_gdf = labeled_gdf[valid_mask].copy()

        sampled_parts = []
        for stratum_id, group in valid_gdf.groupby('stratum_id'):
            quota = int(group['quota'].iloc[0])
            if quota <= 0:
                continue
            if len(group) <= quota:
                sampled_parts.append(group)
            else:
                idx = rng.choice(len(group), size=quota, replace=False)
                sampled_parts.append(group.iloc[idx])

        if not sampled_parts:
            return valid_gdf.iloc[0:0]

        result = pd.concat(sampled_parts, ignore_index=True)
        return gpd.GeoDataFrame(result, geometry='geometry', crs="EPSG:4326")


# --------------------------------------------------------------------------- #
# Smoke test                                                                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=== Layer2Stratification smoke test ===")
    print(f"Quota CSV: {QUOTA_CSV}")
    print(f"Exists:    {os.path.exists(QUOTA_CSV)}")
    print(f"Ecozone SHP: {ECOZONE_SHP}")
    print()

    # Test LC lookup
    test_lons = np.array([-115.0, -100.0, -80.0, -130.0])
    test_lats = np.array([  55.0,   60.0,  50.0,   55.0])
    lc = lookup_lc(test_lons, test_lats, 2018)
    print("LC lookup test (year 2018):")
    for lon, lat, v in zip(test_lons, test_lats, lc):
        fr = 'fire-relevant' if v in LC_FIRE_CLASSES else 'non-fire'
        print(f"  ({lon:.1f}, {lat:.1f}) -> LC={v} ({fr})")

    # Test ecozone assignment
    print()
    pts_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(test_lons, test_lats),
        crs="EPSG:4326"
    )
    eco_vals = assign_ecozone(pts_gdf)
    print("Ecozone assignment test:")
    for lon, lat, e in zip(test_lons, test_lats, eco_vals):
        print(f"  ({lon:.1f}, {lat:.1f}) -> {e}")

    # Test get_stratum_for_pixel on a real MODIS file
    print()
    MODIS_FILE = (
        "/s3/user/wildfire_dataset/self_built_materials/"
        "bc_raw_data_new_fast/drivers/processed/"
        "spatial_filled_5xdownsampled_merged/"
        "MCD14A1_combined_5xMaximumDown_merged/2018_07_15.tif"
    )
    LC_FILE_TEST = os.path.join(LC_DIR, "MCD12Q1_2018_01_01_tile00.tif")
    if os.path.exists(MODIS_FILE) and os.path.exists(LC_FILE_TEST):
        modis_ds = gdal.Open(MODIS_FILE, gdal.GA_ReadOnly)
        lc_ds_t  = gdal.Open(LC_FILE_TEST, gdal.GA_ReadOnly)
        eco_gdf  = _load_eco()
        s = get_stratum_for_pixel(100, 200, "2018-07-15", modis_ds, lc_ds_t, eco_gdf)
        print(f"get_stratum_for_pixel(row=100, col=200, 2018-07-15): {s}")
        modis_ds = None
        lc_ds_t  = None
    else:
        print("get_stratum_for_pixel: test files not available — skipping")

    # Test Layer2Stratification class
    print()
    if os.path.exists(QUOTA_CSV):
        s2 = Layer2Stratification()
        qt = s2.get_quota_table()
        print(f"Layer2Stratification: {len(qt)} strata, total quota = {qt['quota'].sum():,}")
        print("Top 3 strata by quota:")
        print(qt.nlargest(3, 'quota')[['ecoregion', 'lc_class', 'quota']].to_string(index=False))
    else:
        print(f"Quota CSV not found at {QUOTA_CSV}")
        print("Run justification/layer2/stratum_quotas.py first.")
