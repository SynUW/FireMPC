#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 Plan 06 — Task 2: Generate variant config JSONs.

Writes variant_{A,B,C,D,E}_config.json to the sampling/variants/ directory.
No data loading — configs are deterministic from pipeline parameters.

Variant definitions (ablation cascade):
  A: None             — random 2:1 negatives, no filter
  B: Layer 1          — +spatial buffer 30 km + temporal +/-3 d + cloud exclusion
  C: Layers 1+2       — +ecozone x LC x month stratification
  D: Layers 1+2+3     — +FWI hard negatives (50% random / 50% hard)
  E: Layers 1+2+3+4   — +10 km spatial deduplication (production variant)

Usage:
    python generate_variant_jsons.py [--variants_dir PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR    = Path(__file__).resolve().parent
VARIANTS_DIR = _THIS_DIR / "variants"

# ---------------------------------------------------------------------------
# Parameters — plan-calibrated values from Phase 2 Plans 01-04
# ---------------------------------------------------------------------------
NEG_POS_RATIO  = 2.0
HARD_NEG_FRAC  = 0.5          # for variants D and E
RNG_SEED       = 42
BUFFER_KM      = 30           # spatial exclusion buffer (Plan 01 calibration)
TEMPORAL_DAYS  = 3            # +/-days temporal buffer (Plan 01 calibration)
MIN_DIST_KM    = 10           # spatial dedup min distance (Plan 04)
LC_CLASSES     = list(range(1, 12))   # IGBP fire-relevant classes 1-11
WEIGHT_JSON    = "fwi/outputs/weight_vector_w1w6.json"


# ---------------------------------------------------------------------------
# Config definitions (plan spec format)
# ---------------------------------------------------------------------------

def build_configs() -> dict[str, dict]:
    """Return the five variant config dicts matching the plan spec format."""
    A = {
        "variant":     "A",
        "description": "Random 2:1 negatives, no filter layers",
        "layers":      [],
        "sampling": {
            "neg_pos_ratio": NEG_POS_RATIO,
            "random_seed":   RNG_SEED,
        },
    }

    B = {
        "variant":     "B",
        "description": "Layer 1 only: spatial/temporal buffer + cloud exclusion",
        "layers":      ["layer1"],
        "layer1": {
            "buffer_km":       BUFFER_KM,
            "temporal_days":   TEMPORAL_DAYS,
            "exclude_cloud":   True,
        },
        "sampling": {
            "neg_pos_ratio": NEG_POS_RATIO,
            "random_seed":   RNG_SEED,
        },
    }

    C = {
        "variant":     "C",
        "description": "Layers 1+2: buffer + ecozone x LC x month stratification",
        "layers":      ["layer1", "layer2"],
        "layer1": {
            "buffer_km":       BUFFER_KM,
            "temporal_days":   TEMPORAL_DAYS,
            "exclude_cloud":   True,
        },
        "layer2": {
            "quota_formula": "sqrt(area * fire_burden)",
            "lc_classes":    LC_CLASSES,
        },
        "sampling": {
            "neg_pos_ratio": NEG_POS_RATIO,
            "random_seed":   RNG_SEED,
        },
    }

    D = {
        "variant":     "D",
        "description": (
            "Layers 1+2+3: buffer + stratification + "
            "FWI hard negatives (50% random / 50% hard)"
        ),
        "layers":      ["layer1", "layer2", "layer3"],
        "layer1": {
            "buffer_km":       BUFFER_KM,
            "temporal_days":   TEMPORAL_DAYS,
            "exclude_cloud":   True,
        },
        "layer2": {
            "quota_formula": "sqrt(area * fire_burden)",
            "lc_classes":    LC_CLASSES,
        },
        "layer3": {
            "hard_neg_fraction": HARD_NEG_FRAC,
            "weight_json":       WEIGHT_JSON,
        },
        "sampling": {
            "neg_pos_ratio": NEG_POS_RATIO,
            "random_seed":   RNG_SEED,
        },
    }

    E = {
        "variant":     "E",
        "description": (
            "Full pipeline: Layers 1+2+3+4 "
            "(buffer + stratification + hard negatives + 10 km spatial dedup)"
        ),
        "layers":      ["layer1", "layer2", "layer3", "layer4"],
        "layer1": {
            "buffer_km":       BUFFER_KM,
            "temporal_days":   TEMPORAL_DAYS,
            "exclude_cloud":   True,
        },
        "layer2": {
            "quota_formula": "sqrt(area * fire_burden)",
            "lc_classes":    LC_CLASSES,
        },
        "layer3": {
            "hard_neg_fraction": HARD_NEG_FRAC,
            "weight_json":       WEIGHT_JSON,
        },
        "layer4": {
            "min_distance_km": MIN_DIST_KM,
        },
        "sampling": {
            "neg_pos_ratio": NEG_POS_RATIO,
            "random_seed":   RNG_SEED,
        },
    }

    return {"A": A, "B": B, "C": C, "D": D, "E": E}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate variant_{A-E}_config.json for the sampling pipeline."
    )
    parser.add_argument(
        "--variants_dir", type=Path, default=VARIANTS_DIR,
        help=f"Output directory (default: {VARIANTS_DIR})"
    )
    args = parser.parse_args()

    out_dir: Path = args.variants_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = build_configs()
    for key, cfg in configs.items():
        path = out_dir / f"variant_{key}_config.json"
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  variant_{key}_config.json  written to {path}")

    print(f"\n{len(configs)} variant configs written to {out_dir}")

    # Sanity check: layer cascade A < B < C < D < E
    assert configs["A"]["layers"] == [],                    "A must have no layers"
    assert configs["B"]["layers"] == ["layer1"],            "B must have layer1 only"
    assert configs["C"]["layers"] == ["layer1", "layer2"],  "C must have layers 1+2"
    assert configs["D"]["layers"] == ["layer1", "layer2", "layer3"], "D: 1+2+3"
    assert configs["E"]["layers"] == ["layer1", "layer2", "layer3", "layer4"], "E: 1+2+3+4"
    print("Sanity checks passed: A < B < C < D < E layer cascade.")


if __name__ == "__main__":
    main()
