"""Step 2 of hard-positive evaluation — extract FWI 6-band features for the
firemask=7 and firemask=8 val positive pools produced by Step 1.

Reuses `extract_features()` from probe_h1_h4.py (same tile-read + pixel-lookup
+ NaN-drop logic, so results are directly comparable to the FM=9 val baseline).

Outputs:
  sampling/probe_hardpos/fm7_val.parquet
  sampling/probe_hardpos/fm8_val.parquet

Each parquet has the FWI bands [FFMC, DMC, DC, ISI, BUI, FWI] plus the
original sample metadata, and drops any samples whose FWI tile is missing or
whose pixel lies outside the FWI grid / has NaN.

Cost control
------------
We cap the feature-extraction set at N_MAX_POS (default 20k per firemask),
sampling uniformly from the full Step-1 pool. That is enough headroom for the
probe (which only uses 5k per seed), leaves margin for NaN drops, and keeps
total tile reads bounded (~few hundred dates).
"""
from __future__ import annotations
import json
import logging
import random
import sys
from pathlib import Path

# Reuse probe_h1_h4 machinery
_PROBE_H1H4 = Path(__file__).resolve().parents[1] / "probe_h1_h4"
sys.path.insert(0, str(_PROBE_H1H4))
from probe_h1_h4 import extract_features  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("extract_features_hardpos")

HERE = Path(__file__).resolve().parent
N_MAX_POS = 20_000
SEED = 123


def run_one(fm: int) -> None:
    in_json  = HERE / f"pool_fm{fm}_val.json"
    out_parq = HERE / f"fm{fm}_val.parquet"
    if out_parq.exists():
        log.info("SKIP fm=%d — %s already exists", fm, out_parq.name)
        return
    log.info("Loading pool %s", in_json.name)
    with open(in_json) as f:
        pool = json.load(f)
    log.info("  pool size: %d", len(pool))
    if len(pool) > N_MAX_POS:
        rng = random.Random(SEED + fm)
        rng.shuffle(pool)
        pool = pool[:N_MAX_POS]
        log.info("  capped to %d for feature extraction", len(pool))
    df = extract_features(pool)
    log.info("  kept %d after FWI feature drop", len(df))
    df.to_parquet(out_parq)
    log.info("  wrote %s", out_parq)


def main() -> None:
    for fm in (7, 8):
        log.info("=== FM=%d ===", fm)
        run_one(fm)


if __name__ == "__main__":
    main()
