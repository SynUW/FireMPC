#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 Plan 06 — Task 1: Five Ablation Variant Characterization.

Runs each of the five sampling variants (A–E) on representative fire years
(2017, 2018, 2021) during peak fire season (Jul–Aug only, 5 days/year = 15 days
total) and computes per-variant statistics:
  - n_positives, n_negatives, neg_pos_ratio
  - hard_neg_fraction  (0.0 for A/B/C; 0.5 for D/E)
  - ecozone_gini       (Gini of negative counts across 15 ecozones)
  - mean_nn_distance_km (mean NN distance among negatives, subsample 5000)

Variant definitions:
  A: Random 2:1 negatives (no layers, raw firemask==5 within research area)
  B: +Layer 1 (30 km spatial buffer + temporal +/-3 d + cloud exclusion)
  C: +Layer 2 (ecozone x LC x month stratification)
  D: +Layer 3 (50% representative / 50% FWI hard negatives)
  E: +Layer 4 (10 km spatial deduplication)

Outputs (to justification/layer6/):
    variant_statistics.csv
    variant_statistics.png
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal
from pyproj import Transformer
from scipy.spatial import KDTree

gdal.UseExceptions()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO = Path("/mnt/raid/user/hackathons/dataset_building_entire_canada/sampling")
sys.path.insert(0, str(_REPO))

from layer1_filter import Layer1Filter
from layer2_stratification import Layer2Stratification, ECOZONE_SHP
from layer3_hard_negative_scorer import Layer3HardNegativeScorer
from layer4_spatial_dedup import Layer4SpatialDedup

MODIS_MERGED_DIR = Path(
    "/s3/user/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/processed/"
    "spatial_filled_5xdownsampled_merged/"
    "MCD14A1_combined_5xMaximumDown_merged"
)
RESEARCH_AREA_SHP = (
    "/s3/user/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/ecoregions/research_area_shpfile_merge.shp"
)
OUT_DIR = _REPO / "justification" / "layer6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters (plan spec)
# ---------------------------------------------------------------------------
SAMPLE_YEARS    = [2017, 2018, 2021]
PEAK_MONTHS     = [7, 8]           # Jul-Aug only (plan spec)
DAYS_PER_YEAR   = 5                # ~25 total
TARGET_RATIO    = 2.0              # neg:pos
HARD_NEG_FRAC   = 0.5              # for variants D and E
RNG_SEED        = 42
MAX_NEG_PER_DAY = 60_000           # OOM guard

_to_3978 = Transformer.from_crs("EPSG:4326", "EPSG:3978", always_xy=True)

# ---------------------------------------------------------------------------
# Research area mask (cached)
# ---------------------------------------------------------------------------
_research_union_cache = None


def _get_research_union():
    global _research_union_cache
    if _research_union_cache is not None:
        return _research_union_cache
    gdf = gpd.read_file(RESEARCH_AREA_SHP)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4617")
    gdf = gdf.to_crs("EPSG:4326")
    _research_union_cache = gdf.geometry.unary_union
    return _research_union_cache


# ---------------------------------------------------------------------------
# Ecozone lookup helper (for all variants)
# ---------------------------------------------------------------------------
_eco_gdf_cache = None


def _get_eco_gdf() -> gpd.GeoDataFrame:
    global _eco_gdf_cache
    if _eco_gdf_cache is not None:
        return _eco_gdf_cache
    gdf = gpd.read_file(ECOZONE_SHP)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4617")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    _eco_gdf_cache = gdf[["ZONE_NAME", "geometry"]].copy()
    return _eco_gdf_cache


def _assign_ecozone_to_lonlat(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Return ZONE_NAME array for each (lon, lat) pair."""
    if len(lons) == 0:
        return np.array([], dtype=str)
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326"
    )
    eco = _get_eco_gdf()
    joined = gpd.sjoin(pts, eco[["ZONE_NAME", "geometry"]],
                       how="left", predicate="within")
    return joined["ZONE_NAME"].fillna("OUTSIDE").values.astype(str)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def gini_coefficient(counts: np.ndarray) -> float:
    """Gini coefficient (0 = perfectly uniform)."""
    counts = counts[counts > 0].astype(np.float64)
    if len(counts) == 0:
        return 0.0
    counts = np.sort(counts)
    n = len(counts)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * counts).sum()) / (n * counts.sum()) - (n + 1) / n)


def mean_nn_distance_km(lons: np.ndarray, lats: np.ndarray,
                        max_sample: int = 5000) -> float:
    """Mean nearest-neighbour distance in km."""
    if len(lons) < 2:
        return 0.0
    if len(lons) > max_sample:
        rng = np.random.default_rng(RNG_SEED)
        idx = rng.choice(len(lons), max_sample, replace=False)
        lons, lats = lons[idx], lats[idx]
    xs, ys = _to_3978.transform(lons, lats)
    pts = np.column_stack([xs, ys])
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=2)
    return float(np.mean(dists[:, 1]) / 1000.0)


def _compute_gini(lons: np.ndarray, lats: np.ndarray) -> float:
    """Ecozone Gini for a set of negative samples."""
    if len(lons) == 0:
        return float("nan")
    eco_vals = _assign_ecozone_to_lonlat(lons, lats)
    inside = eco_vals[eco_vals != "OUTSIDE"]
    if len(inside) == 0:
        return float("nan")
    _, counts = np.unique(inside, return_counts=True)
    return gini_coefficient(counts)


def _build_stats(variant: str,
                 n_pos: int,
                 neg_lons: np.ndarray, neg_lats: np.ndarray,
                 hard_neg_frac: float = 0.0) -> dict:
    n_neg = len(neg_lons)
    ratio = round(n_neg / max(n_pos, 1), 3)
    g = _compute_gini(neg_lons, neg_lats)
    nn = mean_nn_distance_km(neg_lons, neg_lats)
    return {
        "variant":             variant,
        "n_positives":         n_pos,
        "n_negatives":         n_neg,
        "neg_pos_ratio":       ratio,
        "hard_neg_fraction":   hard_neg_frac,
        "ecozone_gini":        round(g, 4) if not (isinstance(g, float) and np.isnan(g)) else None,
        "mean_nn_distance_km": round(nn, 3),
    }


# ---------------------------------------------------------------------------
# MODIS merged-file reader (for Variant A raw negatives)
# ---------------------------------------------------------------------------

def _read_merged_firemask(date_str: str):
    """Read firemask from the merged MODIS daily file."""
    parts = date_str.split("-")
    fname = f"{parts[0]}_{parts[1]}_{parts[2]}.tif"
    fpath = MODIS_MERGED_DIR / fname
    if not fpath.exists():
        return None, None
    ds = gdal.Open(str(fpath), gdal.GA_ReadOnly)
    if ds is None:
        return None, None
    gt = ds.GetGeoTransform()
    W, H = ds.RasterXSize, ds.RasterYSize
    raw = ds.GetRasterBand(1).ReadRaster(0, 0, W, H, buf_type=gdal.GDT_Byte)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(H, W)
    ds = None
    return arr, gt


# ---------------------------------------------------------------------------
# Sample day loader
# ---------------------------------------------------------------------------

def _load_sample_days() -> list[dict]:
    """Load Layer1-filtered fire + negative pixels for DAYS_PER_YEAR peak-season days/year."""
    rng = np.random.default_rng(RNG_SEED)
    l1  = Layer1Filter()
    days_out = []

    for year in SAMPLE_YEARS:
        candidate_days = []
        for month in PEAK_MONTHS:
            for d in range(1, 32):
                try:
                    candidate_days.append(date(year, month, d))
                except ValueError:
                    pass
        rng.shuffle(candidate_days)

        year_count = 0
        for day in candidate_days:
            if year_count >= DAYS_PER_YEAR:
                break
            dstr = day.isoformat()
            try:
                fire_gdf = l1.get_fire_pixels(dstr)
                neg_gdf  = l1.get_negative_candidates(dstr)
            except Exception as exc:
                print(f"  Warning {dstr}: {exc}")
                continue
            if fire_gdf is None or len(fire_gdf) == 0:
                continue
            if neg_gdf is None or len(neg_gdf) == 0:
                continue

            fire_lo = fire_gdf.geometry.x.values.astype(np.float64).copy()
            fire_la = fire_gdf.geometry.y.values.astype(np.float64).copy()
            del fire_gdf

            if len(neg_gdf) > MAX_NEG_PER_DAY:
                neg_gdf = neg_gdf.iloc[
                    rng.choice(len(neg_gdf), MAX_NEG_PER_DAY, replace=False)
                ]
            neg_lo = neg_gdf.geometry.x.values.astype(np.float64).copy()
            neg_la = neg_gdf.geometry.y.values.astype(np.float64).copy()
            del neg_gdf

            days_out.append({
                "date":      dstr,
                "year":      year,
                "month":     day.month,
                "fire_lons": fire_lo,
                "fire_lats": fire_la,
                "neg_lons":  neg_lo,
                "neg_lats":  neg_la,
            })
            year_count += 1
            print(f"  {dstr}: {len(fire_lo):,} fire  {len(neg_lo):,} neg  "
                  f"({year_count}/{DAYS_PER_YEAR})")
            sys.stdout.flush()

        print(f"  Year {year}: {year_count} days loaded")

    return days_out


# ---------------------------------------------------------------------------
# Variant A: random 2:1 (no filter, raw firemask==5 within research area)
# ---------------------------------------------------------------------------

def run_variant_A(days: list[dict], rng: np.random.Generator) -> dict:
    """Variant A: random 2:1 negatives, no filter layers."""
    pos_lons, pos_lats = [], []
    raw_neg_lons, raw_neg_lats = [], []
    ra_union = _get_research_union()

    for d in days:
        pos_lons.append(d["fire_lons"])
        pos_lats.append(d["fire_lats"])

        arr, gt = _read_merged_firemask(d["date"])
        if arr is None:
            continue
        mask = arr == 5
        if mask.sum() == 0:
            continue
        rows, cols = np.where(mask)
        lons_r = (gt[0] + (cols + 0.5) * gt[1]).astype(np.float64)
        lats_r = (gt[3] + (rows + 0.5) * gt[5]).astype(np.float64)
        del arr, mask, rows, cols

        pts = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(lons_r, lats_r), crs="EPSG:4326"
        )
        inside = pts.geometry.within(ra_union)
        lons_r = lons_r[inside.values]
        lats_r = lats_r[inside.values]
        del pts, inside

        if len(lons_r) > MAX_NEG_PER_DAY:
            idx = rng.choice(len(lons_r), MAX_NEG_PER_DAY, replace=False)
            lons_r, lats_r = lons_r[idx], lats_r[idx]
        raw_neg_lons.append(lons_r)
        raw_neg_lats.append(lats_r)

    pos_lo = np.concatenate(pos_lons) if pos_lons else np.array([])
    pos_la = np.concatenate(pos_lats) if pos_lats else np.array([])
    neg_lo = np.concatenate(raw_neg_lons) if raw_neg_lons else np.array([])
    neg_la = np.concatenate(raw_neg_lats) if raw_neg_lats else np.array([])

    n_neg_target = int(len(pos_lo) * TARGET_RATIO)
    if len(neg_lo) > n_neg_target:
        idx = rng.choice(len(neg_lo), n_neg_target, replace=False)
        neg_lo, neg_la = neg_lo[idx], neg_la[idx]

    return _build_stats("A", len(pos_lo), neg_lo, neg_la, hard_neg_frac=0.0)


# ---------------------------------------------------------------------------
# Variant B: +Layer 1 (buffer already applied in loaded days)
# ---------------------------------------------------------------------------

def run_variant_B(days: list[dict], rng: np.random.Generator) -> dict:
    """Variant B: +Layer 1 reliability filter (pre-applied in sample_days)."""
    pos_lons, pos_lats = [], []
    neg_lons, neg_lats = [], []
    for d in days:
        pos_lons.append(d["fire_lons"])
        pos_lats.append(d["fire_lats"])
        neg_lons.append(d["neg_lons"])
        neg_lats.append(d["neg_lats"])

    pos_lo = np.concatenate(pos_lons)
    pos_la = np.concatenate(pos_lats)
    neg_lo = np.concatenate(neg_lons)
    neg_la = np.concatenate(neg_lats)

    n_neg_target = int(len(pos_lo) * TARGET_RATIO)
    if len(neg_lo) > n_neg_target:
        idx = rng.choice(len(neg_lo), n_neg_target, replace=False)
        neg_lo, neg_la = neg_lo[idx], neg_la[idx]

    return _build_stats("B", len(pos_lo), neg_lo, neg_la, hard_neg_frac=0.0)


# ---------------------------------------------------------------------------
# Variant C: +Layer 2 (ecozone x LC x month stratification)
# ---------------------------------------------------------------------------

def run_variant_C(days: list[dict], rng: np.random.Generator) -> dict:
    """Variant C: +Layer 2 ecological stratification."""
    l2 = Layer2Stratification()
    pos_lons, pos_lats = [], []
    neg_lons_all, neg_lats_all, eco_all = [], [], []

    for d in days:
        pos_lons.append(d["fire_lons"])
        pos_lats.append(d["fire_lats"])

        pts = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(d["neg_lons"], d["neg_lats"]),
            crs="EPSG:4326"
        )
        labeled = l2.assign_strata(pts, year=d["year"], month=d["month"])
        del pts
        fire_rel = labeled[
            labeled["lc_fire_relevant"] & (labeled["ecoregion"] != "OUTSIDE")
        ]
        del labeled
        if len(fire_rel) == 0:
            continue

        neg_lons_all.append(fire_rel.geometry.x.values.astype(np.float64).copy())
        neg_lats_all.append(fire_rel.geometry.y.values.astype(np.float64).copy())
        eco_all.extend(fire_rel["ecoregion"].values.tolist())
        del fire_rel

    pos_lo = np.concatenate(pos_lons)
    pos_la = np.concatenate(pos_lats)
    neg_lo = np.concatenate(neg_lons_all) if neg_lons_all else np.array([])
    neg_la = np.concatenate(neg_lats_all) if neg_lats_all else np.array([])
    eco_arr = np.array(eco_all)

    # Quota-based downsampling by ecozone
    n_neg_target = int(len(pos_lo) * TARGET_RATIO)
    unique_ecos = np.unique(eco_arr)
    quota_per_eco = max(1, n_neg_target // max(len(unique_ecos), 1))
    kept_idx = []
    for eco in unique_ecos:
        eco_idx = np.where(eco_arr == eco)[0]
        n_take = min(quota_per_eco, len(eco_idx))
        kept_idx.extend(rng.choice(eco_idx, n_take, replace=False).tolist())
    kept_idx = np.array(kept_idx)
    neg_lo = neg_lo[kept_idx]
    neg_la = neg_la[kept_idx]

    return _build_stats("C", len(pos_lo), neg_lo, neg_la, hard_neg_frac=0.0)


# ---------------------------------------------------------------------------
# Variant D: +Layer 3 (50% repr / 50% hard negatives)
# ---------------------------------------------------------------------------

def run_variant_D(days: list[dict], rng: np.random.Generator) -> dict:
    """Variant D: +Layer 3 hard negative scoring."""
    l2 = Layer2Stratification()
    l3 = Layer3HardNegativeScorer()

    pos_lons, pos_lats = [], []
    repr_lons, repr_lats = [], []
    hard_lons, hard_lats = [], []

    for d in days:
        pos_lons.append(d["fire_lons"])
        pos_lats.append(d["fire_lats"])

        pts = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(d["neg_lons"], d["neg_lats"]),
            crs="EPSG:4326"
        )
        labeled = l2.assign_strata(pts, year=d["year"], month=d["month"])
        del pts
        fire_rel = labeled[
            labeled["lc_fire_relevant"] & (labeled["ecoregion"] != "OUTSIDE")
        ]
        del labeled
        if len(fire_rel) == 0:
            continue

        lons_n = fire_rel.geometry.x.values.astype(np.float64).copy()
        lats_n = fire_rel.geometry.y.values.astype(np.float64).copy()
        del fire_rel

        scores = l3.score_candidates(d["date"], lons_n, lats_n)
        hard_mask = l3.select_hard_negatives(scores, fraction=HARD_NEG_FRAC)
        repr_mask = ~hard_mask & ~np.isnan(scores)

        if repr_mask.sum() > 0:
            repr_lons.append(lons_n[repr_mask])
            repr_lats.append(lats_n[repr_mask])
        if hard_mask.sum() > 0:
            hard_lons.append(lons_n[hard_mask])
            hard_lats.append(lats_n[hard_mask])

    pos_lo = np.concatenate(pos_lons)
    pos_la = np.concatenate(pos_lats)
    n_neg_target  = int(len(pos_lo) * TARGET_RATIO)
    n_hard_target = int(n_neg_target * HARD_NEG_FRAC)
    n_repr_target = n_neg_target - n_hard_target

    repr_lo = np.concatenate(repr_lons) if repr_lons else np.array([])
    repr_la = np.concatenate(repr_lats) if repr_lats else np.array([])
    hard_lo = np.concatenate(hard_lons) if hard_lons else np.array([])
    hard_la = np.concatenate(hard_lats) if hard_lats else np.array([])

    if len(repr_lo) > n_repr_target:
        idx = rng.choice(len(repr_lo), n_repr_target, replace=False)
        repr_lo, repr_la = repr_lo[idx], repr_la[idx]
    if len(hard_lo) > n_hard_target:
        idx = rng.choice(len(hard_lo), n_hard_target, replace=False)
        hard_lo, hard_la = hard_lo[idx], hard_la[idx]

    neg_lo = np.concatenate([repr_lo, hard_lo])
    neg_la = np.concatenate([repr_la, hard_la])
    hn_frac = round(len(hard_lo) / max(len(neg_lo), 1), 3)
    return _build_stats("D", len(pos_lo), neg_lo, neg_la, hard_neg_frac=hn_frac)


# ---------------------------------------------------------------------------
# Variant E: +Layer 4 (10 km spatial deduplication)
# ---------------------------------------------------------------------------

def run_variant_E(days: list[dict], rng: np.random.Generator) -> dict:
    """Variant E: +Layer 4 spatial deduplication (10 km minimum distance)."""
    l2 = Layer2Stratification()
    l3 = Layer3HardNegativeScorer()
    l4 = Layer4SpatialDedup(min_dist_km=10.0, seed=RNG_SEED)

    pos_lons, pos_lats = [], []
    repr_lons, repr_lats = [], []
    hard_lons, hard_lats = [], []

    for d in days:
        pos_lons.append(d["fire_lons"])
        pos_lats.append(d["fire_lats"])

        pts = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(d["neg_lons"], d["neg_lats"]),
            crs="EPSG:4326"
        )
        labeled = l2.assign_strata(pts, year=d["year"], month=d["month"])
        del pts
        fire_rel = labeled[
            labeled["lc_fire_relevant"] & (labeled["ecoregion"] != "OUTSIDE")
        ]
        del labeled
        if len(fire_rel) == 0:
            continue

        lons_n = fire_rel.geometry.x.values.astype(np.float64).copy()
        lats_n = fire_rel.geometry.y.values.astype(np.float64).copy()
        del fire_rel

        scores = l3.score_candidates(d["date"], lons_n, lats_n)
        hard_mask = l3.select_hard_negatives(scores, fraction=HARD_NEG_FRAC)
        repr_mask = ~hard_mask & ~np.isnan(scores)

        if repr_mask.sum() > 0:
            keep_r = l4.thin_mask(lons_n[repr_mask], lats_n[repr_mask])
            repr_lons.append(lons_n[repr_mask][keep_r])
            repr_lats.append(lats_n[repr_mask][keep_r])
        if hard_mask.sum() > 0:
            keep_h = l4.thin_mask(lons_n[hard_mask], lats_n[hard_mask])
            hard_lons.append(lons_n[hard_mask][keep_h])
            hard_lats.append(lats_n[hard_mask][keep_h])

    pos_lo = np.concatenate(pos_lons)
    pos_la = np.concatenate(pos_lats)
    n_neg_target  = int(len(pos_lo) * TARGET_RATIO)
    n_hard_target = int(n_neg_target * HARD_NEG_FRAC)
    n_repr_target = n_neg_target - n_hard_target

    repr_lo = np.concatenate(repr_lons) if repr_lons else np.array([])
    repr_la = np.concatenate(repr_lats) if repr_lats else np.array([])
    hard_lo = np.concatenate(hard_lons) if hard_lons else np.array([])
    hard_la = np.concatenate(hard_lats) if hard_lats else np.array([])

    if len(repr_lo) > n_repr_target:
        idx = rng.choice(len(repr_lo), n_repr_target, replace=False)
        repr_lo, repr_la = repr_lo[idx], repr_la[idx]
    if len(hard_lo) > n_hard_target:
        idx = rng.choice(len(hard_lo), n_hard_target, replace=False)
        hard_lo, hard_la = hard_lo[idx], hard_la[idx]

    neg_lo = np.concatenate([repr_lo, hard_lo])
    neg_la = np.concatenate([repr_la, hard_la])
    hn_frac = round(len(hard_lo) / max(len(neg_lo), 1), 3)
    return _build_stats("E", len(pos_lo), neg_lo, neg_la, hard_neg_frac=hn_frac)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Phase 2 Plan 06 — Variant Sampler")
    print(f"Sample years: {SAMPLE_YEARS}, months: Jul-Aug, {DAYS_PER_YEAR} days/year")
    print("=" * 60)

    print("\n[1] Loading Layer 1 filtered sample days...")
    days = _load_sample_days()
    print(f"\nTotal: {len(days)} fire days loaded")
    if len(days) == 0:
        print("ERROR: No fire days loaded. Check MODIS path and Layer1Filter.")
        sys.exit(1)

    results = []
    for name, fn in [
        ("A", run_variant_A),
        ("B", run_variant_B),
        ("C", run_variant_C),
        ("D", run_variant_D),
        ("E", run_variant_E),
    ]:
        print(f"\n[Variant {name}] Running...")
        sys.stdout.flush()
        r = fn(days, np.random.default_rng(RNG_SEED))
        results.append(r)
        print(
            f"  n_pos={r['n_positives']:,}  n_neg={r['n_negatives']:,}  "
            f"ratio={r['neg_pos_ratio']:.2f}  hard_frac={r['hard_neg_fraction']:.3f}  "
            f"gini={r['ecozone_gini']}  mean_nn={r['mean_nn_distance_km']:.1f} km"
        )
        sys.stdout.flush()

    # Save CSV (plan spec column order)
    df = pd.DataFrame(results, columns=[
        "variant", "n_positives", "n_negatives", "neg_pos_ratio",
        "hard_neg_fraction", "ecozone_gini", "mean_nn_distance_km"
    ])
    csv_path = OUT_DIR / "variant_statistics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(df.to_string(index=False))

    # Figure: grouped bar chart, one panel per metric
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Ablation Variant Statistics  |  "
        f"Years: {SAMPLE_YEARS}  |  Peak season: Jul-Aug  |  "
        f"{DAYS_PER_YEAR} days/year",
        fontsize=11
    )
    variants = df["variant"].tolist()
    colors   = ["#4472C4", "#ED7D31", "#A9D18E", "#FFC000", "#C00000"]

    def _bar(ax, values, ylabel, title, fmt="{:.2f}"):
        bars = ax.bar(variants, values, color=colors, alpha=0.85, edgecolor="white")
        valid_vals = [v for v in values
                      if v is not None and not (isinstance(v, float) and np.isnan(v))]
        top_val = max(valid_vals) if valid_vals else 1.0
        for bar, val in zip(bars, values):
            label = (fmt.format(val)
                     if val is not None and not (isinstance(val, float) and np.isnan(val))
                     else "N/A")
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + top_val * 0.02,
                    label, ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _bar(axes[0, 0], df["neg_pos_ratio"].tolist(),
         "Ratio", "Neg:Pos Ratio", "{:.2f}")
    _bar(axes[0, 1], df["hard_neg_fraction"].tolist(),
         "Fraction", "Hard Negative Fraction", "{:.3f}")
    gini_vals = [v if v is not None else 0.0 for v in df["ecozone_gini"].tolist()]
    _bar(axes[1, 0], gini_vals,
         "Gini", "Ecozone Gini (negatives)", "{:.4f}")
    _bar(axes[1, 1], df["mean_nn_distance_km"].tolist(),
         "km", "Mean NN Distance (negatives)", "{:.1f}")

    plt.tight_layout()
    png_path = OUT_DIR / "variant_statistics.png"
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}")
    print("\nVariant_sampler complete.")


if __name__ == "__main__":
    main()
