"""Build samples_variant_Y.json (from A) and samples_variant_H.json (from G).

Y = A's train + A's val + A's test_positives + new_test_negatives
H = G's train + G's val + G's test_positives + new_test_negatives  (test positives identical to A's)
new_test_negatives shared between Y and H (same RNG draw).

new_test_negatives spec:
    Pool      : 2023-2025 fire-season days (May-Oct), MODIS firemask == 5,
                inside research_mask, lulc_class in {1..11} (IGBP fire-relevant)
    Stratify  : by lulc_class. For each LC c with n_pos[c] test positives,
                draw 2 * n_pos[c] negatives without replacement.
                If pool[c] < target, take whole pool[c] and log a warning.
    Seed      : 42
    Schema    : identical to A/G JSON records
                (year, day_of_year, date, pixel_row, pixel_col,
                 firms_value, sample_type, lulc_class, split)
                negatives carry firms_value=0, sample_type="negative", split="test".

Outputs:
    /s3/user/wildfire_dataset/h5_dataset/entire_can_v1/cache_Y/samples_variant_Y.json
    /s3/user/wildfire_dataset/h5_dataset/entire_can_v1/cache_H/samples_variant_H.json
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import date as _date, timedelta
from pathlib import Path

import numpy as np

SAMPLING_DIR = Path("/mnt/raid/user/hackathons/dataset_building_entire_canada/sampling")
sys.path.insert(0, str(SAMPLING_DIR))

# Lazy heavy imports done inside main() so module loads cheaply for tests.

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
A_JSON = Path("/s3/user/wildfire_dataset/h5_dataset/entire_can_v1/cache_A/samples_variant_A.json")
G_JSON = Path("/s3/user/wildfire_dataset/h5_dataset/entire_can_v1/cache_G/samples_variant_G.json")

OUT_Y_DIR = Path("/s3/user/wildfire_dataset/h5_dataset/entire_can_v1/cache_Y")
OUT_H_DIR = Path("/s3/user/wildfire_dataset/h5_dataset/entire_can_v1/cache_H")

TEST_YEARS = (2023, 2024, 2025)
FIRE_SEASON_MONTHS = range(5, 11)  # May .. Oct inclusive
LC_FIRE_RELEVANT = set(range(1, 12))
NEG_POS_RATIO = 2
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_Y_H")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_test_dates() -> list[str]:
    """Return YYYY-MM-DD strings for all fire-season days in TEST_YEARS."""
    out: list[str] = []
    for yr in TEST_YEARS:
        d = _date(yr, FIRE_SEASON_MONTHS.start, 1)
        end = _date(yr, FIRE_SEASON_MONTHS.stop - 1, 1)
        end = (end.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        while d <= end:
            out.append(d.isoformat())
            d += timedelta(days=1)
    return out


def _load_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _records_split(records: list[dict], split: str) -> list[dict]:
    return [r for r in records if r.get("split") == split]


def _validate_test_pos_match(a: dict, g: dict) -> list[dict]:
    """Sanity check: A and G test positives must be identical."""
    pa = _records_split(a["positive_samples"], "test")
    pg = _records_split(g["positive_samples"], "test")
    key = lambda r: (r["date"], r["pixel_row"], r["pixel_col"])
    sa, sg = set(map(key, pa)), set(map(key, pg))
    if sa != sg:
        diff = (sa ^ sg)
        raise SystemExit(f"A and G test positives differ ({len(diff)} mismatched keys); aborting.")
    log.info("A and G test positives identical: %d records", len(pa))
    return pa


def _date_to_year_doy(date_str: str) -> tuple[int, int]:
    d = _date.fromisoformat(date_str)
    return d.year, int(d.strftime("%j"))


# ---------------------------------------------------------------------------
# Pool builder: 2023-2025 fire-season FM=5 candidates with LC enrichment
# ---------------------------------------------------------------------------

def build_neg_pool(gs) -> "pd.DataFrame":
    """Walk all test-year fire-season days, return DataFrame of FM=5 candidates.

    Columns: date, year, day_of_year, pixel_row, pixel_col, lulc_class.
    All candidates already filtered by research_mask and LC ∈ {1..11}.
    """
    import pandas as pd
    from layer1_filter import FIREMASK_NONFIRE_LAND
    from layer2_stratification import lookup_lc

    parts: list[pd.DataFrame] = []
    dates = _iter_test_dates()
    log.info("Scanning %d test-year fire-season dates for FM=5 pool ...", len(dates))

    for i, ds in enumerate(dates):
        if i % 50 == 0:
            log.info("  %d / %d  (%s)  cumulative=%s",
                     i, len(dates), ds, sum(len(p) for p in parts))
        result = gs._cached_read_firemask(ds)
        if result is None:
            continue
        arr, gt = result
        cand_mask = arr == FIREMASK_NONFIRE_LAND
        if not cand_mask.any():
            continue

        rows, cols = np.where(cand_mask)
        lons = gt[0] + (cols + 0.5) * gt[1]
        lats = gt[3] + (rows + 0.5) * gt[5]

        keep = gs._apply_research_mask(lons, lats)
        if not keep.any():
            continue
        rows = rows[keep]; cols = cols[keep]
        lons = lons[keep]; lats = lats[keep]

        year, doy = _date_to_year_doy(ds)
        lc = lookup_lc(lons, lats, year)
        keep_lc = np.isin(lc, list(LC_FIRE_RELEVANT))
        if not keep_lc.any():
            continue

        parts.append(pd.DataFrame({
            "date":        ds,
            "year":        year,
            "day_of_year": doy,
            "pixel_row":   rows[keep_lc].astype(int),
            "pixel_col":   cols[keep_lc].astype(int),
            "lulc_class":  lc[keep_lc].astype(int),
        }))

    if not parts:
        raise SystemExit("Empty FM=5 pool — check raster paths.")
    pool = pd.concat(parts, ignore_index=True)
    log.info("FM=5 pool size: %d", len(pool))
    log.info("Pool LC dist: %s",
             dict(sorted(Counter(pool["lulc_class"]).items())))
    return pool


# ---------------------------------------------------------------------------
# LC-stratified sampling
# ---------------------------------------------------------------------------

def lc_stratified_sample(pool, lc_pos_counts: dict[int, int],
                         ratio: int, seed: int):
    """Per-LC random sampling: 2 * n_pos[c] without replacement; warn on shortfall."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    out_parts: list[pd.DataFrame] = []
    for lc, n_pos in sorted(lc_pos_counts.items()):
        target = ratio * n_pos
        sub = pool[pool["lulc_class"] == lc]
        avail = len(sub)
        if avail == 0:
            log.warning("  LC=%d  target=%d  pool=0  -> SKIP (no candidates)", lc, target)
            continue
        if avail < target:
            log.warning("  LC=%d  target=%d  pool=%d  -> taking all %d (shortfall %d)",
                        lc, target, avail, avail, target - avail)
            chosen = sub
        else:
            idx = rng.choice(avail, size=target, replace=False)
            chosen = sub.iloc[idx]
        log.info("  LC=%d  n_pos=%d  -> sampled %d negatives", lc, n_pos, len(chosen))
        out_parts.append(chosen)
    if not out_parts:
        raise SystemExit("LC-stratified sampling yielded zero negatives.")
    return pd.concat(out_parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

def neg_df_to_records(df) -> list[dict]:
    out: list[dict] = []
    for r in df.itertuples(index=False):
        out.append({
            "year":        int(r.year),
            "day_of_year": int(r.day_of_year),
            "date":        str(r.date),
            "pixel_row":   int(r.pixel_row),
            "pixel_col":   int(r.pixel_col),
            "firms_value": 0,
            "sample_type": "negative",
            "lulc_class":  int(r.lulc_class),
            "split":       "test",
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Loading A and G ...")
    a = _load_json(A_JSON)
    g = _load_json(G_JSON)
    test_pos = _validate_test_pos_match(a, g)

    lc_pos = Counter(r["lulc_class"] for r in test_pos)
    log.info("Test-pos LC distribution: %s", dict(sorted(lc_pos.items())))
    log.info("Target test negatives total: %d (%dx)",
             NEG_POS_RATIO * sum(lc_pos.values()), NEG_POS_RATIO)

    log.info("Initialising MODIS / research_mask / LC fast rasters ...")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_samples", str(SAMPLING_DIR / "generate_samples.py"))
    gs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gs)
    gs._init_layers()

    pool = build_neg_pool(gs)
    chosen = lc_stratified_sample(pool, dict(lc_pos), NEG_POS_RATIO, SEED)
    log.info("Total sampled test negatives: %d", len(chosen))

    new_test_neg = neg_df_to_records(chosen)

    # Assemble Y and H
    OUT_Y_DIR.mkdir(parents=True, exist_ok=True)
    OUT_H_DIR.mkdir(parents=True, exist_ok=True)

    def _assemble(src: dict, label: str) -> dict:
        # Train/val unchanged: keep ALL records (pos+neg) where split in {train,val}.
        # Test positives unchanged.
        # Test negatives: replaced by new_test_neg.
        kept_pos = [r for r in src["positive_samples"]
                    if r["split"] in ("train", "val") or r["split"] == "test"]
        kept_neg_trainval = [r for r in src["negative_samples"]
                             if r["split"] in ("train", "val")]
        out_pos = kept_pos
        out_neg = kept_neg_trainval + new_test_neg
        log.info("Variant %s — pos: %d  neg: %d (trainval %d + test %d)",
                 label, len(out_pos), len(out_neg),
                 len(kept_neg_trainval), len(new_test_neg))
        return {"positive_samples": out_pos, "negative_samples": out_neg}

    y = _assemble(a, "Y")
    h = _assemble(g, "H")

    out_y = OUT_Y_DIR / "samples_variant_Y.json"
    out_h = OUT_H_DIR / "samples_variant_H.json"
    with out_y.open("w") as fh:
        json.dump(y, fh)
    with out_h.open("w") as fh:
        json.dump(h, fh)
    log.info("Wrote %s", out_y)
    log.info("Wrote %s", out_h)
    log.info("DONE.")


if __name__ == "__main__":
    main()
