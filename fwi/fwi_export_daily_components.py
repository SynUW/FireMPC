#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export the six FWI components daily to a single multi-band GeoTIFF
(same ERA5 tiling and recursive moisture-code conventions as fwi_eda).

- Input: ERA5_YYYY_MM_DD_tileNN.tif (NN=00..03); band usage matches fwi_module / fwi_eda.
- Output: one file per day, 6 bands in order FFMC, DMC, DC, ISI, BUI, FWI;
  each band has a GDAL description (visible in QGIS/ArcGIS).
- Time: every calendar day of the year (Jan 1–Dec 31). When years are processed
  in ascending order, FFMC/DMC/DC from Dec 31 of the previous year carry into
  Jan 1 of the next year (cold-start INIT on the first year or if grid sizes differ).
- Space: the four tiles are reprojected onto a common output grid using each tile's
  GeoTransform (rasterio reproject, bilinear). Overlapping pixels: arithmetic mean
  of valid values; if only one side is valid, keep that side; both invalid → NaN.
  If a tile file is missing for a given day, that tile's output is NaN for that day
  and that tile's FFMC/DMC/DC are not updated (same as fwi_eda).
- Years: if --years / --year-start / --year-end are not given, scan all years
  that appear in ERA5 filenames under the ERA5 directory.
- Numeric warnings: vectorized np.where inside fwi_module evaluates all branches
  and can emit divide/invalid warnings; they usually do not affect the selected
  branch. Computation uses np.errstate to suppress console spam. BUI has a
  separate guard where both dmc and dc are 0 (see fwi_module).
- Progress: a single terminal line refreshed with \\r showing global done/total,
  percentage, current date, day-of-year index, whether today was written or
  recurse-only, and cumulative write vs recurse-only counts (not a progress bar).
- Dependencies: rasterio (reproject and mosaic), GDAL, numpy.

Usage:
  python fwi_export_daily_components.py
  python fwi_export_daily_components.py --years 2018
  python fwi_export_daily_components.py --year-start 2017 --year-end 2021 --overwrite
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from osgeo import gdal

try:
    from rasterio.crs import CRS
    from rasterio.transform import Affine, array_bounds, from_bounds
    from rasterio.warp import reproject, Resampling
except ImportError as _e:  # pragma: no cover
    raise SystemExit(
        "Geographic mosaicking requires rasterio: pip install rasterio"
    ) from _e

_FM_PATH = Path(__file__).resolve().parent / "fwi_module.py"
_spec = importlib.util.spec_from_file_location("fwi_module", str(_FM_PATH))
if _spec is None or _spec.loader is None:
    raise ImportError(f"Failed to load {_FM_PATH}")
_fm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fm)

DC_INIT = _fm.DC_INIT
DMC_INIT = _fm.DMC_INIT
FFMC_INIT = _fm.FFMC_INIT
dc_next = _fm.dc_next
dmc_next = _fm.dmc_next
extract_fwi_inputs_from_bands = _fm.extract_fwi_inputs_from_bands
ffmc_next = _fm.ffmc_next
fwi_calc = _fm.fwi_calc
bui_calc = _fm.bui_calc
isi_calc = _fm.isi_calc
read_era5_tile = _fm.read_era5_tile

gdal.UseExceptions()

# Same as fwi_eda: tile01|tile03 are north (top), tile00|tile02 are south (bottom)

COMPONENT_KEYS = ["ffmc", "dmc", "dc", "isi", "bui", "fwi"]
BAND_DESCRIPTIONS = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]

DEFAULT_ERA5_DIR = (
    "/s3/zhengsen/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/ERA5"
)
DEFAULT_OUT_DIR = (
    "/s3/zhengsen/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/processed/fwi_components"
)

N_TILES = 4

_ERA5_FILENAME_YEAR = re.compile(
    r"^ERA5_(\d{4})_\d{2}_\d{2}_tile\d{2}\.tif$", re.IGNORECASE
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily FWI six components → single multi-band GeoTIFF")
    p.add_argument("--era5-dir", type=Path, default=Path(DEFAULT_ERA5_DIR))
    p.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    p.add_argument(
        "--years",
        type=str,
        default="",
        help="Comma-separated years; if empty, scan all years found under --era5-dir",
    )
    p.add_argument("--year-start", type=int, default=None)
    p.add_argument("--year-end", type=int, default=None)
    p.add_argument("--compress", default="ZSTD", choices=["NONE", "LZW", "DEFLATE", "ZSTD"])
    p.add_argument("--blocksize", type=int, default=512)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _era5_tile_path(era5_dir: Path, date_str: str, tile_idx: int) -> Path:
    yyyy, mm, dd = date_str.split("-")
    return era5_dir / f"ERA5_{yyyy}_{mm}_{dd}_tile{tile_idx:02d}.tif"


def _get_grid_shape(tile_path: Path) -> tuple[int, int]:
    ds = gdal.Open(str(tile_path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Cannot open: {tile_path}")
    h, w = ds.RasterYSize, ds.RasterXSize
    ds = None
    return h, w


def discover_years_from_era5(era5_dir: Path) -> list[int]:
    """Parse all years that appear in ERA5_YYYY_MM_DD_tileNN.tif filenames in the directory."""
    years: set[int] = set()
    if not era5_dir.is_dir():
        return []
    for p in era5_dir.iterdir():
        if not p.is_file():
            continue
        m = _ERA5_FILENAME_YEAR.match(p.name)
        if m:
            years.add(int(m.group(1)))
    return sorted(years)


def _resolve_years(args: argparse.Namespace, era5_dir: Path) -> list[int]:
    if args.years.strip():
        return sorted({int(y.strip()) for y in args.years.split(",") if y.strip()})
    ys, ye = args.year_start, args.year_end
    if ys is not None and ye is not None:
        return list(range(ys, ye + 1))
    if ys is not None:
        return [ys]
    years = discover_years_from_era5(era5_dir)
    if not years:
        raise SystemExit(
            f"No files matching ERA5_YYYY_MM_DD_tileNN.tif found in {era5_dir}; "
            "cannot infer years. Specify --years or --year-start/--year-end."
        )
    return years


def _init_grid_shape_for_tile(era5_dir: Path, year: int, tile_idx: int) -> tuple[int, int] | None:
    p = _first_era5_tile_path(era5_dir, year, tile_idx)
    if p is None:
        return None
    return _get_grid_shape(p)


def _first_era5_tile_path(era5_dir: Path, year: int, tile_idx: int) -> Path | None:
    for month in range(1, 13):
        sample = date(year, month, 1).strftime("%Y-%m-%d")
        p = _era5_tile_path(era5_dir, sample, tile_idx)
        if p.is_file():
            return p
    return None


def _calendar_days_in_year(year: int) -> int:
    return (date(year + 1, 1, 1) - date(year, 1, 1)).days


def _year_can_run(era5_dir: Path, year: int) -> bool:
    """Same as the start of run_year: all four tiles have an in-year sample and a tile01 reference file exists."""
    for t in range(N_TILES):
        if _init_grid_shape_for_tile(era5_dir, year, t) is None:
            return False
    for month in range(1, 13):
        d = date(year, month, 1).strftime("%Y-%m-%d")
        if _era5_tile_path(era5_dir, d, 1).is_file():
            return True
    return False


class ProgressLine:
    """Refresh numbers on the same line with \\r (not a progress bar): global done/total, current date, in-year progress, write/recurse counts."""

    def __init__(self, total_days: int) -> None:
        self.total = max(1, int(total_days))
        self.done = 0
        self.n_write = 0
        self.n_skip_write = 0

    def tick(self, date_str: str, day_i: int, ndays_year: int, skip_write: bool) -> None:
        self.done += 1
        if skip_write:
            self.n_skip_write += 1
        else:
            self.n_write += 1
        pct = 100.0 * self.done / self.total
        mode = "recurse" if skip_write else "write"
        line = (
            f"progress {self.done}/{self.total} ({pct:.2f}%)  "
            f"{date_str}  day {day_i}/{ndays_year}  "
            f"today={mode}  writes={self.n_write} recurse-only={self.n_skip_write}"
        )
        sys.stdout.write("\r\x1b[K" + line)
        sys.stdout.flush()

    def newline(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()


def _init_moisture(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    return (
        np.full((h, w), FFMC_INIT, dtype=np.float32),
        np.full((h, w), DMC_INIT, dtype=np.float32),
        np.full((h, w), DC_INIT, dtype=np.float32),
    )


def _process_one_tile_day(
    tile_path: Path,
    ffmc: np.ndarray,
    dmc: np.ndarray,
    dc: np.ndarray,
    current: date,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]] | None:
    """Return updated ffmc, dmc, dc and the six components for the day; return None on failure."""
    bands = read_era5_tile(str(tile_path), bands=[2, 3, 5, 7, 13])
    outside_mask = (bands[13] == 0.0) | (bands[7] == 0.0)
    for b_idx in [2, 3, 5, 7, 13]:
        bands[b_idx] = np.where(outside_mask, np.nan, bands[b_idx])
    bands[5] = np.maximum(bands[5], 0.0)
    T_c, RH, wind, rain = extract_fwi_inputs_from_bands(bands)
    month = current.month
    # np.where evaluates all branches in vectorized form; divide/invalid warnings on invalid pixels do not change the selected branch
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ffmc = ffmc_next(ffmc, T_c, RH, wind, rain)
        dmc = dmc_next(dmc, T_c, RH, rain, month)
        dc = dc_next(dc, T_c, rain, month)
        isi = isi_calc(ffmc, wind)
        bui = bui_calc(dmc, dc)
        fwi = fwi_calc(isi, bui)
    out = {
        "ffmc": ffmc.copy(),
        "dmc": dmc.copy(),
        "dc": dc.copy(),
        "isi": isi.copy(),
        "bui": bui.copy(),
        "fwi": fwi.copy(),
    }
    return ffmc, dmc, dc, out


def _gdal_open_gt_proj(path: Path) -> tuple[tuple, str]:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(path)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds = None
    return gt, proj


def _mosaic_georef_from_tile01(tile01_path: Path) -> tuple[tuple, str]:
    return _gdal_open_gt_proj(tile01_path)


# FWI physical values can be well above 0; sentinel used to distinguish from NaN (reproject src/dst nodata)
_MOSAIC_SENTINEL = -99999.0


@dataclass(frozen=True)
class MosaicGeoContext:
    """Geometry context for reprojecting four tiles onto a common grid (resolution and pixel alignment from tile01)."""

    tile_gts: tuple[tuple, ...]
    dst_affine: Affine
    dst_w: int
    dst_h: int
    crs: CRS
    gdal_gt: tuple
    proj_wkt: str
    sentinel: float = _MOSAIC_SENTINEL


def _build_mosaic_geo_context(
    era5_dir: Path,
    year: int,
    shapes: list[tuple[int, int]],
) -> MosaicGeoContext | None:
    tile_gts: list[tuple] = []
    projs: list[str] = []
    for t in range(N_TILES):
        p = _first_era5_tile_path(era5_dir, year, t)
        if p is None:
            return None
        gt, proj = _gdal_open_gt_proj(p)
        tile_gts.append(gt)
        projs.append(proj)

    boxes: list[tuple[float, float, float, float]] = []
    for t in range(N_TILES):
        h, w = shapes[t]
        aff = Affine.from_gdal(*tile_gts[t])
        boxes.append(array_bounds(h, w, aff))

    left = min(b[0] for b in boxes)
    bottom = min(b[1] for b in boxes)
    right = max(b[2] for b in boxes)
    top = max(b[3] for b in boxes)

    gt1 = tile_gts[1]
    xres = abs(float(gt1[1]))
    yres = abs(float(gt1[5]))
    if xres < 1e-12 or yres < 1e-12:
        return None

    dst_w = max(1, int(np.ceil((right - left) / xres)))
    dst_h = max(1, int(np.ceil((top - bottom) / yres)))
    dst_affine = from_bounds(left, bottom, right, top, dst_w, dst_h)
    gdal_gt = dst_affine.to_gdal()

    proj_wkt = projs[1] or projs[0]
    if not proj_wkt or not str(proj_wkt).strip():
        return None
    for p in projs:
        if p and p != proj_wkt:
            print("  [WARN] WKT projections of the four tiles are not identical; using tile01")
            break

    try:
        crs = CRS.from_wkt(proj_wkt)
    except Exception:
        return None

    return MosaicGeoContext(
        tile_gts=tuple(tile_gts),
        dst_affine=dst_affine,
        dst_w=dst_w,
        dst_h=dst_h,
        crs=crs,
        gdal_gt=gdal_gt,
        proj_wkt=proj_wkt,
    )


def _merge_components_geographic(
    per_component: dict[str, list[np.ndarray | None]],
    ctx: MosaicGeoContext,
) -> list[np.ndarray]:
    """Reproject each tile onto ctx's grid, then average valid values per pixel (keep the value if only one side is valid)."""
    S = ctx.sentinel
    warped_6: list[np.ndarray] = []
    for t in range(N_TILES):
        planes: list[np.ndarray] = []
        src_tf = Affine.from_gdal(*ctx.tile_gts[t])
        for k in COMPONENT_KEYS:
            arr = per_component[k][t]
            if arr is None:
                planes.append(np.full((ctx.dst_h, ctx.dst_w), np.nan, dtype=np.float32))
                continue
            src = np.where(np.isfinite(arr), arr, S).astype(np.float32)
            dst = np.full((ctx.dst_h, ctx.dst_w), S, dtype=np.float32)
            reproject(
                source=src,
                destination=dst,
                src_transform=src_tf,
                src_crs=ctx.crs,
                dst_transform=ctx.dst_affine,
                dst_crs=ctx.crs,
                resampling=Resampling.bilinear,
                src_nodata=S,
                dst_nodata=S,
            )
            planes.append(np.where(dst == S, np.nan, dst).astype(np.float32))
        warped_6.append(np.stack(planes, axis=0))

    cube = np.stack(warped_6, axis=0).astype(np.float64, copy=False)
    valid = np.isfinite(cube)
    cnt = np.sum(valid, axis=0, dtype=np.float64)
    sumv = np.sum(np.where(valid, cube, 0.0), axis=0)
    out6 = np.where(cnt > 0, (sumv / cnt).astype(np.float32), np.nan)
    return [out6[i].copy() for i in range(6)]


def _write_multiband_tif(
    path: Path,
    bands_data: list[np.ndarray],
    geo_transform: tuple,
    projection: str,
    compress: str,
    blocksize: int,
    descriptions: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = bands_data[0].shape
    n = len(bands_data)
    drv = gdal.GetDriverByName("GTiff")
    opts = [
        "TILED=YES",
        f"BLOCKXSIZE={blocksize}",
        f"BLOCKYSIZE={blocksize}",
        "BIGTIFF=IF_SAFER",
    ]
    if compress != "NONE":
        opts.append(f"COMPRESS={compress}")
        if compress == "ZSTD":
            opts.append("ZSTD_LEVEL=9")

    ds = drv.Create(str(path), w, h, n, gdal.GDT_Float32, options=opts)
    if ds is None:
        raise RuntimeError(f"GDAL Create failed: {path}")
    ds.SetGeoTransform(geo_transform)
    ds.SetProjection(projection)
    for i in range(n):
        b = ds.GetRasterBand(i + 1)
        b.WriteArray(np.asarray(bands_data[i], dtype=np.float32))
        b.SetDescription(descriptions[i])
        # Do not set nodata; keep NaN (consistent with internal computation)
    ds.FlushCache()
    ds = None


def _states_match_shapes(
    carry: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    shapes: list[tuple[int, int] | None],
) -> bool:
    for t in range(N_TILES):
        sh = shapes[t]
        if sh is None:
            return False
        ff, dm, d_ = carry[t]
        if ff.shape != sh or dm.shape != sh or d_.shape != sh:
            return False
    return True


def run_year(
    era5_dir: Path,
    out_dir: Path,
    year: int,
    compress: str,
    blocksize: int,
    overwrite: bool,
    states_carry: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    progress: ProgressLine | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None:
    shapes: list[tuple[int, int] | None] = [None] * N_TILES
    for t in range(N_TILES):
        sh = _init_grid_shape_for_tile(era5_dir, year, t)
        shapes[t] = sh
        if sh is None:
            print(f"  [SKIP year {year}] tile{t}: no ERA5 in this year, skipping year")
            return None

    if states_carry is not None and _states_match_shapes(states_carry, shapes):
        states = list(states_carry)
    else:
        if states_carry is not None:
            print(f"  [NOTE] {year}: grid size does not match previous year-end state, or no previous state; cold-start from INIT this year")
        states = [_init_moisture(shapes[t]) for t in range(N_TILES)]  # type: ignore[arg-type]

    shapes_xy = [shapes[t] for t in range(N_TILES)]
    mosaic_ctx = _build_mosaic_geo_context(era5_dir, year, shapes_xy)  # type: ignore[arg-type]
    if mosaic_ctx is None:
        print(f"  [SKIP year {year}] cannot build geographic mosaic (missing tile or GeoTransform)")
        return None

    season_start = date(year, 1, 1)
    season_end = date(year, 12, 31)
    cur = season_start
    ndays = (season_end - season_start).days + 1
    day_i = 0

    while cur <= season_end:
        day_i += 1
        date_str = cur.strftime("%Y-%m-%d")
        out_name = f"fwis_{cur.strftime('%Y_%m_%d')}.tif"
        out_path = out_dir / out_name

        skip_write = out_path.is_file() and out_path.stat().st_size > 0 and not overwrite

        per_component: dict[str, list[np.ndarray | None]] = {k: [None] * N_TILES for k in COMPONENT_KEYS}

        for t in range(N_TILES):
            tp = _era5_tile_path(era5_dir, date_str, t)
            ffmc, dmc, dc = states[t]
            if not tp.is_file():
                sh = shapes[t]
                assert sh is not None
                for k in COMPONENT_KEYS:
                    per_component[k][t] = np.full(sh, np.nan, dtype=np.float32)
                continue
            try:
                upd = _process_one_tile_day(tp, ffmc, dmc, dc, cur)
                if upd is None:
                    raise RuntimeError("internal")
                ffmc, dmc, dc, day_out = upd
                states[t] = (ffmc, dmc, dc)
                for k in COMPONENT_KEYS:
                    per_component[k][t] = day_out[k]
            except Exception as e:
                print(f"  [WARN] {date_str} tile{t}: {e}")
                sh = shapes[t]
                assert sh is not None
                for k in COMPONENT_KEYS:
                    per_component[k][t] = np.full(sh, np.nan, dtype=np.float32)

        mosaic_bands = _merge_components_geographic(per_component, mosaic_ctx)

        if not skip_write:
            _write_multiband_tif(
                out_path,
                mosaic_bands,
                mosaic_ctx.gdal_gt,
                mosaic_ctx.proj_wkt,
                compress,
                blocksize,
                BAND_DESCRIPTIONS,
            )
        if progress is not None:
            progress.tick(date_str, day_i, ndays, skip_write)

        cur += timedelta(days=1)

    return states


def main() -> None:
    args = parse_args()
    era5_dir = args.era5_dir.resolve()
    out_dir = args.output_dir.resolve()
    if not era5_dir.is_dir():
        print(f"ERA5 directory does not exist: {era5_dir}")
        sys.exit(1)
    years = _resolve_years(args, era5_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runnable = [y for y in years if _year_can_run(era5_dir, y)]
    skipped = [y for y in years if y not in runnable]
    total_days = sum(_calendar_days_in_year(y) for y in runnable)
    progress = ProgressLine(total_days) if total_days > 0 else None

    print(f"ERA5: {era5_dir}\noutput: {out_dir}\nyears: {years}")
    print(
        f"{len(runnable)} runnable year(s), {total_days} planned day(s) (global progress updates by day); "
        f"leap years counted with actual day counts."
    )
    if skipped:
        print(f"Will skip (incomplete ERA5): {skipped}")

    carry: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
    for y in years:
        if y not in runnable:
            if progress is not None:
                progress.newline()
            print(f"---- {y} ----\n  [SKIP] incomplete ERA5")
            carry = None
            continue
        if progress is not None:
            progress.newline()
        print(f"---- {y} ----")
        end_states = run_year(
            era5_dir,
            out_dir,
            y,
            args.compress,
            args.blocksize,
            args.overwrite,
            carry,
            progress=progress,
        )
        carry = end_states if end_states is not None else None
    if progress is not None:
        progress.newline()
    print("Done.")


if __name__ == "__main__":
    main()
