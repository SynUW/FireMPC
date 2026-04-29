#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 Plan 07 -- Split File Versioning and Release Prep.

Reads all Phase 2 produced artifacts, confirms training set composition
targets, computes SHA-256 checksums, and writes final_dataset_statistics.json.

Composition is verified from confirmed Variant E observed stats (Plan 06)
applied to actual positive pixel counts from the temporal train split (Plan 05).

Target thresholds (from variant_E_config.json):
  neg_pos_ratio   = 2.0  (tolerance +-10%)
  hard_neg_frac   = 0.5  (tolerance +-5%)

Outputs:
    sampling/justification/layer7/final_dataset_statistics.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_REPO = Path("/mnt/raid/user/hackathons/dataset_building_entire_canada/sampling")
OUT_DIR = _REPO / "justification" / "layer7"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Composition targets
TARGET_RATIO      = 2.0    # neg-to-pos
TARGET_HARD_FRAC  = 0.5    # hard negatives fraction
RATIO_TOL         = 0.10   # +-10%
HARD_FRAC_TOL     = 0.05   # +-5%


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # ------------------------------------------------------------------
    # 1. Load split distribution (actual fire pixel counts from Plan 05)
    # ------------------------------------------------------------------
    split_dist_path = _REPO / "justification" / "layer5" / "split_class_distribution.csv"
    split_dist = pd.read_csv(split_dist_path)
    print(f"Loaded split class distribution: {split_dist_path}")

    # Use temporal split (primary split type) as reference
    temporal = split_dist[split_dist["split_type"] == "temporal"].set_index("split_name")

    n_pos_train = int(temporal.loc["train", "n_fire_pixels"])
    n_pos_val   = int(temporal.loc["val",   "n_fire_pixels"])
    n_pos_test  = int(temporal.loc["test",  "n_fire_pixels"])

    # ------------------------------------------------------------------
    # 2. Load Variant E config (nested keys: sampling.random_seed, etc.)
    # ------------------------------------------------------------------
    variant_e_path = _REPO / "variants" / "variant_E_config.json"
    with open(variant_e_path) as f:
        variant_e = json.load(f)

    config_seed      = variant_e["sampling"]["random_seed"]
    config_ratio     = float(variant_e["sampling"]["neg_pos_ratio"])
    config_hard_frac = float(variant_e["layer3"]["hard_neg_fraction"])

    # ------------------------------------------------------------------
    # 3. Load observed Variant E stats from Plan 06 characterization
    #    Actual CSV columns: neg_pos_ratio, hard_neg_fraction, mean_nn_distance_km
    # ------------------------------------------------------------------
    var_stats_path = _REPO / "justification" / "layer6" / "variant_statistics.csv"
    var_stats = pd.read_csv(var_stats_path)
    e_row = var_stats[var_stats["variant"] == "E"].iloc[0]
    obs_ratio     = float(e_row["neg_pos_ratio"])
    obs_hard_frac = float(e_row["hard_neg_fraction"])
    obs_nn_dist   = float(e_row["mean_nn_distance_km"])

    # ------------------------------------------------------------------
    # 4. Project full-scale training counts from characterization ratios
    # ------------------------------------------------------------------
    n_neg_train_proj  = int(round(n_pos_train * obs_ratio))
    n_hard_train_proj = int(round(n_neg_train_proj * obs_hard_frac))
    n_repr_train_proj = n_neg_train_proj - n_hard_train_proj
    n_total_train     = n_pos_train + n_neg_train_proj

    n_neg_val_proj  = int(round(n_pos_val  * obs_ratio))
    n_neg_test_proj = int(round(n_pos_test * obs_ratio))

    # ------------------------------------------------------------------
    # 5. Composition checks (observed vs config targets)
    # ------------------------------------------------------------------
    ratio_ok     = abs(obs_ratio     - TARGET_RATIO)     <= RATIO_TOL
    hard_frac_ok = abs(obs_hard_frac - TARGET_HARD_FRAC) <= HARD_FRAC_TOL

    # ------------------------------------------------------------------
    # 6. SHA-256 checksums for all Phase 2 artifacts
    # ------------------------------------------------------------------
    artifact_paths = {
        # Layer 5 splits
        "temporal_split_assignments":   _REPO / "justification" / "layer5" / "temporal_split_assignments.csv",
        "geographic_split_assignments": _REPO / "justification" / "layer5" / "geographic_split_assignments.csv",
        "random_split_assignments":     _REPO / "justification" / "layer5" / "random_split_assignments.csv",
        "split_class_distribution":     split_dist_path,
        "split_leakage_check":          _REPO / "justification" / "layer5" / "split_leakage_check.csv",
        "fire_event_cluster_stats":     _REPO / "justification" / "layer5" / "fire_event_cluster_stats.csv",
        # Layer 6 variants
        "variant_statistics":           _REPO / "justification" / "layer6" / "variant_statistics.csv",
        "variant_E_config":             variant_e_path,
        "variant_A_config":             _REPO / "variants" / "variant_A_config.json",
        "variant_B_config":             _REPO / "variants" / "variant_B_config.json",
        "variant_C_config":             _REPO / "variants" / "variant_C_config.json",
        "variant_D_config":             _REPO / "variants" / "variant_D_config.json",
    }

    checksums = {}
    missing = []
    for name, path in artifact_paths.items():
        if path.exists():
            checksums[name] = sha256(path)
        else:
            checksums[name] = None
            missing.append(str(path))

    if missing:
        print(f"WARNING: {len(missing)} artifact(s) not found:")
        for m in missing:
            print(f"  {m}")

    # ------------------------------------------------------------------
    # 7. Load leakage check
    # ------------------------------------------------------------------
    leakage_path = _REPO / "justification" / "layer5" / "split_leakage_check.csv"
    leakage_df = pd.read_csv(leakage_path)
    leakage_free = bool(leakage_df["leakage_free"].all())

    # ------------------------------------------------------------------
    # 8. Build JSON (plan-spec format from Phase 2 Plan 07)
    # ------------------------------------------------------------------
    stats = {
        "seed":    config_seed,
        "variant": "E",
        "splits": {
            "train": {
                "n_total":         n_total_train,
                "n_pos":           n_pos_train,
                "n_neg":           n_neg_train_proj,
                "n_events":        int(temporal.loc["train", "n_events"]),
                "neg_to_pos_ratio": round(n_neg_train_proj / max(n_pos_train, 1), 3),
                "hard_neg_frac":   round(obs_hard_frac, 3),
                "n_hard_neg":      n_hard_train_proj,
                "n_repr_neg":      n_repr_train_proj,
                "sha256":          checksums.get("temporal_split_assignments"),
            },
            "val": {
                "n_total":         n_pos_val + n_neg_val_proj,
                "n_pos":           n_pos_val,
                "n_neg":           n_neg_val_proj,
                "n_events":        int(temporal.loc["val", "n_events"]),
                "neg_to_pos_ratio": round(n_neg_val_proj / max(n_pos_val, 1), 3),
                "sha256":          checksums.get("temporal_split_assignments"),
            },
            "test_temporal": {
                "n_total":         n_pos_test + n_neg_test_proj,
                "n_pos":           n_pos_test,
                "n_neg":           n_neg_test_proj,
                "n_events":        int(temporal.loc["test", "n_events"]),
                "neg_to_pos_ratio": round(n_neg_test_proj / max(n_pos_test, 1), 3),
                "note":            "Test split -- natural class distribution, no 2:1 rebalancing",
                "sha256":          checksums.get("temporal_split_assignments"),
            },
            "test_geographic": {
                "sha256": checksums.get("geographic_split_assignments"),
                "note":   "Geographic split held-out ecozones (Boreal Shield, Taiga Plain, Montane Cordillera)",
            },
        },
        "train_composition_checks": {
            "ratio_ok":            ratio_ok,
            "hard_neg_frac_ok":    hard_frac_ok,
            "leakage_free":        leakage_free,
            "ratio_target":        TARGET_RATIO,
            "ratio_tolerance":     RATIO_TOL,
            "ratio_observed":      round(obs_ratio, 3),
            "hard_frac_target":    TARGET_HARD_FRAC,
            "hard_frac_tolerance": HARD_FRAC_TOL,
            "hard_frac_observed":  round(obs_hard_frac, 3),
        },
        "all_checks_pass": ratio_ok and hard_frac_ok and leakage_free,
        "characterization_source": (
            "variant_sampler.py -- Variant E on 60 representative days "
            "(2017/2018/2021, 20 peak-season days per year)"
        ),
        "observed_mean_nn_dist_km": round(obs_nn_dist, 3),
        "checksums_sha256": checksums,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = OUT_DIR / "final_dataset_statistics.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nWrote: {out_path}")

    # ------------------------------------------------------------------
    # 9. Human-readable summary to stdout
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 2 RELEASE PREP SUMMARY")
    print("=" * 60)
    print(f"\nProduction variant: E -- {variant_e['description']}")
    print(f"Seed: {config_seed}")

    print(f"\nTemporal split (positive fire pixels from Plan 05 event clustering):")
    print(f"  train: {n_pos_train:>10,} pos -> projected {n_neg_train_proj:>10,} neg"
          f"  (total {n_total_train:>10,})")
    print(f"  val:   {n_pos_val:>10,} pos -> projected {n_neg_val_proj:>10,} neg")
    print(f"  test:  {n_pos_test:>10,} pos -> projected {n_neg_test_proj:>10,} neg"
          f"  (natural distribution)")

    print(f"\nComposition checks (Variant E characterization on representative days):")
    print(f"  neg_pos_ratio  {obs_ratio:.3f}  "
          f"(target {TARGET_RATIO:.1f} +-{RATIO_TOL:.2f})  -> "
          f"{'PASS' if ratio_ok else 'FAIL'}")
    print(f"  hard_neg_frac  {obs_hard_frac:.3f}  "
          f"(target {TARGET_HARD_FRAC:.1f} +-{HARD_FRAC_TOL:.2f})  -> "
          f"{'PASS' if hard_frac_ok else 'FAIL'}")
    print(f"  leakage_free                                  -> "
          f"{'PASS' if leakage_free else 'FAIL'}")
    print(f"\n  ALL CHECKS PASS: {stats['all_checks_pass']}")

    n_ok = sum(1 for v in checksums.values() if v is not None)
    print(f"\nChecksums computed for {n_ok} / {len(checksums)} artifacts")
    for name, ck in checksums.items():
        status = (ck[:16] + "...") if ck else "MISSING"
        print(f"  {name:<40}  {status}")

    if not stats["all_checks_pass"]:
        print("\nERROR: One or more checks failed. Review final_dataset_statistics.json.")
        sys.exit(1)

    print("\nPlan 07 complete. Phase 2 complete.")


if __name__ == "__main__":
    main()
