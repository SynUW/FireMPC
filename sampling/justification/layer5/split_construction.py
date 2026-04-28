#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 Plan 05 — Task 2: Split Construction.

Assigns fire events (from fire_event_clusters.parquet) to train/val/test splits
using three strategies:

  1. temporal   — test=2023-2025, val=2020-2022, train=all remaining
  2. geographic — 3 highest-fire-burden ecozones held out as test/val
  3. random     — 60/20/20 random shuffle of event IDs (seed=42)

Leakage check: each event_id must appear in exactly one split.

Prerequisites:
    fire_event_clusters.parquet  (Task 1)
    fire_event_cluster_stats.csv (Task 1)
    justification/layer2/raw_positive_distribution_by_ecoregion.csv

Outputs (all to justification/layer5/):
    temporal_split_assignments.csv   — event_id, year, split
    geographic_split_assignments.csv — event_id, ecozone, split
    random_split_assignments.csv     — event_id, split
    geographic_holdout_ecozones.csv  — ecozone, n_fire_pixels, reason
    split_leakage_check.csv          — split_type, n_events, n_train, n_val,
                                       n_test, duplicate_events,
                                       missing_events, leakage_free
    split_class_distribution.csv     — split_type, split_name, n_events,
                                       n_fire_pixels
    split_construction.png           — bar charts of train/val/test counts
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO   = Path("/mnt/raid/zhengsen/hackathons/dataset_building_entire_canada/sampling")
OUT_DIR = _REPO / "justification" / "layer5"

ECOZONES_SHP = Path(
    "/s3/zhengsen/wildfire_dataset/self_built_materials/"
    "bc_raw_data_new_fast/drivers/ecoregions/ecozones.shp"
)
POS_DIST_CSV = _REPO / "justification" / "layer2" / "raw_positive_distribution_by_ecoregion.csv"
CLUSTER_PARQUET = OUT_DIR / "fire_event_clusters.parquet"
CLUSTER_STATS   = OUT_DIR / "fire_event_cluster_stats.csv"

RNG_SEED = 42

# ---------------------------------------------------------------------------
# Temporal split boundaries (per PLAN.md critical_context)
# ---------------------------------------------------------------------------
TEMPORAL_TEST_YEARS = {2023, 2024, 2025}
TEMPORAL_VAL_YEARS  = {2020, 2021, 2022}
# train: all years not in val or test

N_GEO_HOLDOUT = 3   # number of ecozones held out for geographic split


# ---------------------------------------------------------------------------
# Ecozone assignment via spatial join
# ---------------------------------------------------------------------------

def assign_ecozones(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Assign ecozone ZONE_NAME to event centroids via geopandas sjoin.

    Args:
        lons, lats: arrays of lon/lat for event centroids (EPSG:4326)
    Returns:
        numpy array of ZONE_NAME strings (NaN-filled as "OUTSIDE_RA")
    """
    eco = gpd.read_file(str(ECOZONES_SHP))
    if eco.crs is None:
        eco = eco.set_crs("EPSG:4617")
    eco = eco.to_crs("EPSG:4326")

    pts = gpd.GeoDataFrame(
        {"idx": np.arange(len(lons))},
        geometry=[Point(float(x), float(y)) for x, y in zip(lons, lats)],
        crs="EPSG:4326"
    )
    joined = gpd.sjoin(pts, eco[["ZONE_NAME", "geometry"]],
                       how="left", predicate="within")
    # Handle duplicate matches (point on boundary)
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined["ZONE_NAME"].fillna("OUTSIDE_RA").values


# ---------------------------------------------------------------------------
# Helper: identify top N ecozones by total fire pixel burden
# ---------------------------------------------------------------------------

def top_ecozones_by_fire_burden(n: int) -> list[str]:
    """Read raw_positive_distribution_by_ecoregion.csv and return top N
    ecozone names sorted by total positive pixel count (descending)."""
    if not POS_DIST_CSV.exists():
        print(f"WARNING: {POS_DIST_CSV} not found — using event count fallback")
        return []
    df = pd.read_csv(POS_DIST_CSV)
    # column is named 'ecoregion' in the CSV (Plan 02 used ecozones as strata)
    eco_col = "ecoregion" if "ecoregion" in df.columns else df.columns[0]
    totals  = df.groupby(eco_col)["positive_count"].sum().sort_values(ascending=False)
    return totals.head(n).index.tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Load event-level cluster stats (has n_pixels per event) ---
    if not CLUSTER_STATS.exists():
        print(f"ERROR: {CLUSTER_STATS} not found. Run fire_event_clustering.py first.")
        sys.exit(1)

    df = pd.read_csv(CLUSTER_STATS)
    print(f"Loaded {len(df):,} fire events from {CLUSTER_STATS.name}")
    print(f"Years: {df['year'].min()} – {df['year'].max()}")

    # --- Assign ecozones to event centroids ---
    print("\nAssigning ecozones to event centroids via spatial join...")
    ecozones = assign_ecozones(df["lon_centroid"].values, df["lat_centroid"].values)
    df["ecozone"] = ecozones
    unique_eco = [e for e in np.unique(ecozones) if e != "OUTSIDE_RA"]
    print(f"  {len(unique_eco)} unique ecozones, "
          f"{(ecozones == 'OUTSIDE_RA').sum()} events outside research area")

    # ===========================================================================
    # 1. Temporal Split
    # ===========================================================================
    print("\nBuilding temporal split...")
    print(f"  test  = years {sorted(TEMPORAL_TEST_YEARS)}")
    print(f"  val   = years {sorted(TEMPORAL_VAL_YEARS)}")
    print(f"  train = all other years")

    def _temporal_label(year):
        if year in TEMPORAL_TEST_YEARS:
            return "test"
        if year in TEMPORAL_VAL_YEARS:
            return "val"
        return "train"

    df_temporal = df[["event_id", "year"]].copy()
    df_temporal["split"] = df_temporal["year"].map(_temporal_label)
    temporal_path = OUT_DIR / "temporal_split_assignments.csv"
    df_temporal.to_csv(str(temporal_path), index=False)
    tc = df_temporal["split"].value_counts()
    print(f"  train={tc.get('train',0):,}  val={tc.get('val',0):,}  test={tc.get('test',0):,}")

    # ===========================================================================
    # 2. Geographic Split
    # ===========================================================================
    print("\nBuilding geographic split...")

    # Identify 3 highest-fire ecozones from Plan 02 distribution CSV
    top3_from_csv = top_ecozones_by_fire_burden(N_GEO_HOLDOUT)

    if top3_from_csv:
        holdout_ecozones = top3_from_csv
        reason = "highest total fire pixel burden per raw_positive_distribution_by_ecoregion.csv"
        print(f"  Held-out ecozones (top {N_GEO_HOLDOUT} by fire pixel burden):")
    else:
        # Fallback: top N by event count in the cluster data
        eco_ev_counts = (
            df[df["ecozone"] != "OUTSIDE_RA"]
            .groupby("ecozone")["event_id"].count()
            .sort_values(ascending=False)
        )
        holdout_ecozones = eco_ev_counts.head(N_GEO_HOLDOUT).index.tolist()
        reason = "highest fire event count (fallback — Plan 02 CSV not found)"
        print(f"  Held-out ecozones (top {N_GEO_HOLDOUT} by event count, fallback):")

    for eco in holdout_ecozones:
        n_ev = int((df["ecozone"] == eco).sum())
        print(f"    {eco}: {n_ev:,} events")

    # Split held-out events 50/50 between val and test (deterministic: even index=val)
    holdout_set = set(holdout_ecozones)
    df_geo = df[["event_id", "ecozone"]].copy()

    geo_splits = []
    holdout_counter = {"val": 0, "test": 0}
    for _, row in df_geo.iterrows():
        if row["ecozone"] in holdout_set:
            # Alternate val/test to get 50/50
            if holdout_counter["val"] <= holdout_counter["test"]:
                geo_splits.append("val")
                holdout_counter["val"] += 1
            else:
                geo_splits.append("test")
                holdout_counter["test"] += 1
        else:
            geo_splits.append("train")
    df_geo["split"] = geo_splits

    geo_path = OUT_DIR / "geographic_split_assignments.csv"
    df_geo.to_csv(str(geo_path), index=False)
    gc = df_geo["split"].value_counts()
    print(f"  train={gc.get('train',0):,}  val={gc.get('val',0):,}  test={gc.get('test',0):,}")

    # Save holdout ecozone documentation
    holdout_rows = []
    for eco in holdout_ecozones:
        n_px = int(df.loc[df["ecozone"] == eco, "n_pixels"].sum())
        holdout_rows.append({"ecozone": eco, "n_fire_pixels": n_px, "reason": reason})
    holdout_df = pd.DataFrame(holdout_rows)
    holdout_df.to_csv(str(OUT_DIR / "geographic_holdout_ecozones.csv"), index=False)
    print(f"  Holdout ecozone list saved.")
    print(f"  NOTE: Geographic split is secondary; spatial autocorrelation documented.")

    # ===========================================================================
    # 3. Random Split (60/20/20, seed=42)
    # ===========================================================================
    print("\nBuilding random split (60/20/20, seed=42)...")
    print("  NOTE: This split has spatial autocorrelation — biased baseline only.")
    rng = np.random.default_rng(RNG_SEED)
    event_ids = df["event_id"].values.copy()
    rng.shuffle(event_ids)
    n = len(event_ids)
    n_test = int(n * 0.20)
    n_val  = int(n * 0.20)
    n_train = n - n_test - n_val

    split_labels          = np.full(n, "train", dtype=object)
    split_labels[:n_test] = "test"
    split_labels[n_test:n_test + n_val] = "val"

    df_random = pd.DataFrame({"event_id": event_ids, "split": split_labels})
    random_path = OUT_DIR / "random_split_assignments.csv"
    df_random.to_csv(str(random_path), index=False)
    rc = df_random["split"].value_counts()
    print(f"  train={rc.get('train',0):,}  val={rc.get('val',0):,}  test={rc.get('test',0):,}")

    # ===========================================================================
    # Leakage Check
    # ===========================================================================
    print("\nRunning leakage checks...")
    leakage_rows = []
    all_clean    = True

    for split_name, split_df in [
        ("temporal",   df_temporal[["event_id", "split"]]),
        ("geographic", df_geo[["event_id", "split"]]),
        ("random",     df_random[["event_id", "split"]]),
    ]:
        # Each event_id must appear exactly once
        counts     = split_df["event_id"].value_counts()
        duplicates = int((counts > 1).sum())
        if duplicates > 0:
            all_clean = False
            print(f"  WARNING: {split_name} has {duplicates} event_ids appearing >1 time!")
        else:
            print(f"  {split_name}: PASS — all {len(split_df):,} events appear exactly once")

        missing = len(set(df["event_id"]) - set(split_df["event_id"]))
        if missing > 0:
            all_clean = False
            print(f"  WARNING: {split_name} missing {missing} events")

        leakage_rows.append({
            "split_type":       split_name,
            "n_events":         len(split_df),
            "n_train":          int((split_df["split"] == "train").sum()),
            "n_val":            int((split_df["split"] == "val").sum()),
            "n_test":           int((split_df["split"] == "test").sum()),
            "duplicate_events": duplicates,
            "missing_events":   missing,
            "leakage_free":     (duplicates == 0 and missing == 0),
        })

    leakage_df = pd.DataFrame(leakage_rows)
    leakage_path = OUT_DIR / "split_leakage_check.csv"
    leakage_df.to_csv(str(leakage_path), index=False)
    print(f"\nLeakage check saved: {leakage_path}")
    if all_clean:
        print("ALL SPLITS LEAKAGE FREE")

    # ===========================================================================
    # Class distribution summary
    # ===========================================================================
    dist_rows = []
    for stype, split_df in [
        ("temporal",   df_temporal[["event_id", "split"]]),
        ("geographic", df_geo[["event_id", "split"]]),
        ("random",     df_random[["event_id", "split"]]),
    ]:
        for sname in ["train", "val", "test"]:
            mask = split_df["split"] == sname
            sub  = df.merge(split_df[mask][["event_id"]], on="event_id")
            dist_rows.append({
                "split_type":    stype,
                "split_name":    sname,
                "n_events":      int(mask.sum()),
                "n_fire_pixels": int(sub["n_pixels"].sum()),
            })

    dist_df  = pd.DataFrame(dist_rows)
    dist_path = OUT_DIR / "split_class_distribution.csv"
    dist_df.to_csv(str(dist_path), index=False)
    print(f"\nClass distribution saved: {dist_path}")
    print(dist_df.to_string(index=False))

    # ===========================================================================
    # Figure: bar charts
    # ===========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Fire Event Split Construction  ({len(df):,} total events)\n"
        f"Temporal (test=2023-2025) / Geographic (3 ecozones) / Random (60/20/20)",
        fontsize=12
    )
    colors = {"train": "#5b9bd5", "val": "#ffc000", "test": "#c00000"}

    for ax, (title, split_df) in zip(axes, [
        ("Temporal",   df_temporal),
        ("Geographic", df_geo),
        ("Random",     df_random),
    ]):
        vc   = split_df["split"].value_counts()
        order = ["train", "val", "test"]
        vals  = [vc.get(s, 0) for s in order]
        bars  = ax.bar(order, vals, color=[colors[s] for s in order], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{val:,}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{title} Split")
        ax.set_ylabel("Fire events")
        ax.set_ylim(0, max(vals) * 1.2)

    plt.tight_layout()
    png_path = OUT_DIR / "split_construction.png"
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {png_path}")
    print("\nTask 2 complete.")


if __name__ == "__main__":
    main()
