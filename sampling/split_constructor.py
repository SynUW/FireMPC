"""Phase 2 Plan 05 — Split Constructor (reusable module).

Provides fire-event-level train/val/test split assignment for the CanWildfire
sampling pipeline. Three split types are supported, all loaded from the CSV
files produced by justification/layer5/split_construction.py.

Split boundaries (temporal):
    train: all years not in val or test (≤ 2019)
    val  : 2020, 2021, 2022
    test : 2023, 2024, 2025

Geographic split:
    train: events whose centroid is NOT in the 3 held-out ecozones
    val/test: events in the held-out ecozones (50/50 split determined at
              construction time — lookup from geographic_split_assignments.csv)

Random split:
    60/20/20 random shuffle of event IDs (seed=42, biased baseline)

Usage:
    from split_constructor import SplitConstructor

    # Initialize (loads all three split CSVs lazily)
    sc = SplitConstructor(clusters_parquet_path="justification/layer5/fire_event_clusters.parquet")

    # Get split for a single event
    split = sc.get_split("2018_00042", split_type="temporal")  # → "train"

    # Assign splits to a DataFrame that has an event_id column
    df_with_splits = sc.assign_splits_to_df(pixel_df)
    # pixel_df gets columns: split_temporal, split_geographic, split_random

Smoke test (run as main):
    python split_constructor.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------
_THIS_DIR  = Path(__file__).resolve().parent
_LAYER5    = _THIS_DIR / "justification" / "layer5"

_DEFAULT_PARQUET  = _LAYER5 / "fire_event_clusters.parquet"
_TEMPORAL_CSV     = _LAYER5 / "temporal_split_assignments.csv"
_GEO_CSV          = _LAYER5 / "geographic_split_assignments.csv"
_GEO_HOLDOUT_CSV  = _LAYER5 / "geographic_holdout_ecozones.csv"
_RANDOM_CSV       = _LAYER5 / "random_split_assignments.csv"

# Temporal split rule constants
TEMPORAL_TEST_YEARS: frozenset[int] = frozenset({2023, 2024, 2025})
TEMPORAL_VAL_YEARS:  frozenset[int] = frozenset({2020, 2021, 2022})


# ---------------------------------------------------------------------------
# SplitConstructor
# ---------------------------------------------------------------------------

class SplitConstructor:
    """Assigns train/val/test splits for CanWildfire fire events.

    Three split types:
      "temporal"   — year-based rule (no I/O for temporal split_for_year)
      "geographic" — ecozone-based (loads geographic_split_assignments.csv)
      "random"     — event-ID-based (loads random_split_assignments.csv)

    All three CSVs are loaded on first access (lazy loading).

    Args:
        clusters_parquet_path: path to fire_event_clusters.parquet.
            Used only if caller needs pixel-level data; not required for
            event-level split lookups. Defaults to layer5 output location.
        split_type: default split type for get_split() if not specified.
    """

    def __init__(
        self,
        clusters_parquet_path: str | Path | None = None,
        split_type: str = "temporal",
    ):
        self._parquet_path = Path(clusters_parquet_path) if clusters_parquet_path \
            else _DEFAULT_PARQUET
        self._default_split_type = split_type

        # Lazy-loaded lookup dicts: event_id -> split label
        self._temporal_lookup:  dict[str, str] | None = None
        self._geo_lookup:       dict[str, str] | None = None
        self._random_lookup:    dict[str, str] | None = None

        # Holdout ecozone set (for geographic split inference without CSV)
        self._geo_holdout_ecozones: frozenset[str] | None = None

    # -----------------------------------------------------------------------
    # Primary API
    # -----------------------------------------------------------------------

    def get_split(self, event_id: str,
                  split_type: str | None = None) -> str:
        """Return 'train', 'val', or 'test' for a given event_id.

        Args:
            event_id:   event identifier string e.g. "2018_00042" or
                        "2018_noise_000001"
            split_type: one of "temporal", "geographic", "random".
                        Defaults to the split_type set at construction time.

        Returns:
            "train", "val", or "test".
            Falls back to "train" if event_id not found in lookup.
        """
        stype = split_type or self._default_split_type
        if stype == "temporal":
            return self._temporal_for_event(event_id)
        elif stype == "geographic":
            return self._geo_for_event(event_id)
        elif stype == "random":
            return self._random_for_event(event_id)
        else:
            raise ValueError(f"Unknown split_type '{stype}'. "
                             "Choose 'temporal', 'geographic', or 'random'.")

    def assign_splits_to_df(self, pixel_df: pd.DataFrame) -> pd.DataFrame:
        """Add split columns to a DataFrame that has an event_id column.

        Adds three new columns:
            split_temporal   — year-based split
            split_geographic — ecozone-based split
            split_random     — random 60/20/20 split

        Args:
            pixel_df: DataFrame with at least an "event_id" column.

        Returns:
            Copy of pixel_df with the three split columns appended.
        """
        if "event_id" not in pixel_df.columns:
            raise ValueError("pixel_df must contain an 'event_id' column")

        df = pixel_df.copy()
        event_ids = df["event_id"].values

        # Temporal: vectorised rule-based
        df["split_temporal"]   = self._assign_temporal_vec(event_ids)
        df["split_geographic"] = self._assign_geographic_vec(event_ids)
        df["split_random"]     = self._assign_random_vec(event_ids)

        return df

    # -----------------------------------------------------------------------
    # Temporal split (rule-based — fast, no CSV needed)
    # -----------------------------------------------------------------------

    @staticmethod
    def temporal_split_for_year(year: int) -> str:
        """Return 'train', 'val', or 'test' for a given year."""
        if year in TEMPORAL_TEST_YEARS:
            return "test"
        if year in TEMPORAL_VAL_YEARS:
            return "val"
        return "train"

    def _temporal_for_event(self, event_id: str) -> str:
        """Derive temporal split from year encoded in event_id."""
        try:
            year = int(event_id.split("_")[0])
        except (ValueError, IndexError):
            return "train"
        return self.temporal_split_for_year(year)

    def _assign_temporal_vec(self, event_ids: np.ndarray) -> np.ndarray:
        result = np.full(len(event_ids), "train", dtype=object)
        for i, eid in enumerate(event_ids):
            result[i] = self._temporal_for_event(eid)
        return result

    # -----------------------------------------------------------------------
    # Geographic split (CSV-backed)
    # -----------------------------------------------------------------------

    def _load_geo(self):
        if self._geo_lookup is not None:
            return
        if not _GEO_CSV.exists():
            raise FileNotFoundError(
                f"Geographic split CSV not found: {_GEO_CSV}\n"
                "Run justification/layer5/split_construction.py first."
            )
        df = pd.read_csv(_GEO_CSV)
        self._geo_lookup = dict(zip(df["event_id"].values,
                                    df["split"].values))
        # Also load holdout ecozone set
        if _GEO_HOLDOUT_CSV.exists():
            hdf = pd.read_csv(_GEO_HOLDOUT_CSV)
            eco_col = "ecozone" if "ecozone" in hdf.columns else hdf.columns[0]
            self._geo_holdout_ecozones = frozenset(hdf[eco_col].values)

    def _geo_for_event(self, event_id: str) -> str:
        self._load_geo()
        return self._geo_lookup.get(event_id, "train")

    def _assign_geographic_vec(self, event_ids: np.ndarray) -> np.ndarray:
        self._load_geo()
        return np.array(
            [self._geo_lookup.get(eid, "train") for eid in event_ids],
            dtype=object
        )

    # -----------------------------------------------------------------------
    # Random split (CSV-backed)
    # -----------------------------------------------------------------------

    def _load_random(self):
        if self._random_lookup is not None:
            return
        if not _RANDOM_CSV.exists():
            raise FileNotFoundError(
                f"Random split CSV not found: {_RANDOM_CSV}\n"
                "Run justification/layer5/split_construction.py first."
            )
        df = pd.read_csv(_RANDOM_CSV)
        self._random_lookup = dict(zip(df["event_id"].values,
                                       df["split"].values))

    def _random_for_event(self, event_id: str) -> str:
        self._load_random()
        return self._random_lookup.get(event_id, "train")

    def _assign_random_vec(self, event_ids: np.ndarray) -> np.ndarray:
        self._load_random()
        return np.array(
            [self._random_lookup.get(eid, "train") for eid in event_ids],
            dtype=object
        )

    # -----------------------------------------------------------------------
    # Info
    # -----------------------------------------------------------------------

    def get_split_rules(self) -> dict:
        """Return a summary dict of split boundary rules."""
        return {
            "temporal": {
                "test":  sorted(TEMPORAL_TEST_YEARS),
                "val":   sorted(TEMPORAL_VAL_YEARS),
                "train": "all years not in val/test (≤2019)",
            },
            "geographic": {
                "n_holdout_ecozones": 3,
                "selection": "top 3 ecozones by total fire pixel burden",
                "holdout_file": str(_GEO_HOLDOUT_CSV),
                "held_out_ecozones": (
                    sorted(self._geo_holdout_ecozones)
                    if self._geo_holdout_ecozones is not None
                    else "not loaded yet — call get_split() first"
                ),
            },
            "random": {
                "ratios":    {"train": 0.60, "val": 0.20, "test": 0.20},
                "seed":      42,
                "note":      "biased baseline — spatial autocorrelation present",
            },
        }

    def __repr__(self) -> str:
        return (f"SplitConstructor(default_split_type={self._default_split_type!r}, "
                f"parquet={self._parquet_path})")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def get_split_for_year(year: int) -> str:
    """Module-level convenience: temporal split for a given year."""
    return SplitConstructor.temporal_split_for_year(year)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("SplitConstructor smoke test — Phase 2 Plan 05 Task 3")
    print("=" * 60)

    sc = SplitConstructor()
    print(f"\nRepr: {sc!r}")

    # --- Temporal split (rule-based, no file I/O) ---
    print("\n[1] Temporal split (year-based rule):")
    test_years = [2000, 2015, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    for yr in test_years:
        split = sc.temporal_split_for_year(yr)
        ev_id = f"{yr}_00042"
        split2 = sc.get_split(ev_id, split_type="temporal")
        assert split == split2, f"Mismatch for year {yr}"
        print(f"  {yr} → {split}  (via get_split('{ev_id}') → {split2})")

    expected = {
        2000: "train", 2015: "train", 2019: "train",
        2020: "val",   2021: "val",   2022: "val",
        2023: "test",  2024: "test",  2025: "test",
    }
    for yr, exp in expected.items():
        got = sc.temporal_split_for_year(yr)
        assert got == exp, f"FAIL year={yr}: expected={exp} got={got}"
    print("  All temporal assertions PASSED")

    # --- assign_splits_to_df temporal only (no file I/O needed) ---
    print("\n[2] assign_splits_to_df (temporal only, in-memory):")
    sample_df = pd.DataFrame({
        "event_id":  ["2018_00001", "2021_00002", "2023_00003", "2024_noise_000000"],
        "pixel_row": [100, 200, 300, 400],
        "pixel_col": [50, 60, 70, 80],
    })
    result = sc.assign_splits_to_df(sample_df)
    assert "split_temporal"   in result.columns
    assert "split_geographic" in result.columns
    assert "split_random"     in result.columns
    print(result[["event_id", "split_temporal", "split_geographic", "split_random"]].to_string(index=False))

    # Verify temporal labels in the result
    assert result.loc[result["event_id"] == "2018_00001", "split_temporal"].iloc[0] == "train"
    assert result.loc[result["event_id"] == "2021_00002", "split_temporal"].iloc[0] == "val"
    assert result.loc[result["event_id"] == "2023_00003", "split_temporal"].iloc[0] == "test"
    print("  split_temporal labels correct")

    # --- Geographic and random splits (require CSV files) ---
    print("\n[3] Geographic and random splits (CSV-backed):")
    if _GEO_CSV.exists() and _RANDOM_CSV.exists():
        # Load a small sample of real event IDs
        geo_df = pd.read_csv(_GEO_CSV, nrows=5)
        rnd_df = pd.read_csv(_RANDOM_CSV, nrows=5)

        for _, row in geo_df.iterrows():
            sp = sc.get_split(row["event_id"], split_type="geographic")
            assert sp in ("train", "val", "test"), f"Bad geographic split: {sp}"
        print(f"  geographic: {len(geo_df)} sample events — all valid labels")

        for _, row in rnd_df.iterrows():
            sp = sc.get_split(row["event_id"], split_type="random")
            assert sp in ("train", "val", "test"), f"Bad random split: {sp}"
        print(f"  random    : {len(rnd_df)} sample events — all valid labels")

        # Full assign_splits_to_df round-trip
        mini_df = geo_df[["event_id"]].copy()
        mini_df["year"] = mini_df["event_id"].apply(lambda e: int(e.split("_")[0]))
        assigned = sc.assign_splits_to_df(mini_df)
        assert assigned["split_geographic"].isin(["train","val","test"]).all()
        assert assigned["split_random"].isin(["train","val","test"]).all()
        print(f"  assign_splits_to_df round-trip OK")

        # Print split rules
        print("\n[4] Split rules summary:")
        rules = sc.get_split_rules()
        for k, v in rules.items():
            print(f"  {k}: {v}")
    else:
        missing = []
        if not _GEO_CSV.exists():
            missing.append(str(_GEO_CSV))
        if not _RANDOM_CSV.exists():
            missing.append(str(_RANDOM_CSV))
        print(f"  SKIPPED — CSVs not found: {missing}")
        print("  (Run split_construction.py first to generate them)")

    print("\nSmoke test PASSED")
