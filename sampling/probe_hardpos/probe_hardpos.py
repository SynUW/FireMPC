"""Step 3 of hard-positive evaluation — three-tier PR-AUC probe.

Trains on firemask=9 positives + firemask=5 negatives (same as probe_h1_h4),
evaluates on three separate val sets that differ only in the firemask tier of
the positives:

  Val_9: FM=9 positives + FM=5 negatives   (reproduces probe_h1_h4 baseline)
  Val_8: FM=8 positives + FM=5 negatives
  Val_7: FM=7 positives + FM=5 negatives

Models
------
  LR  : LogisticRegression (primary probe, matches probe_h1_h4)
  GBM : HistGradientBoostingClassifier (sanity check, non-linear)

Features: the 6 FWI bands, z-score scaled on the training set.
Seeds: 10 (decision-grade, not a 20-seed full sweep).

Reads cached parquet features from:
  probe_h1_h4/B_pos_tr.parquet   (FM=9 train positives)
  probe_h1_h4/B_neg_tr.parquet   (FM=5 train negatives)
  probe_h1_h4/B_pos_va.parquet   (FM=9 val positives  — baseline)
  probe_h1_h4/B_neg_va.parquet   (FM=5 val negatives  — shared across all 3 val sets)
  probe_hardpos/fm7_val.parquet  (FM=7 val positives)
  probe_hardpos/fm8_val.parquet  (FM=8 val positives)

Writes results.json + a printed summary. The summary is the paper-decision
signal — see hard_positive_evaluation_plan.md Section 3 decision rule.
"""
from __future__ import annotations
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("probe_hardpos")

HERE        = Path(__file__).resolve().parent
PROBE_H1H4  = HERE.parent / "probe_h1_h4"
OUT_PATH    = HERE / "results.json"

BAND_NAMES = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
N_TRAIN_POS = 5_000
N_TRAIN_NEG = 10_000
N_VAL_POS   = 5_000
N_VAL_NEG   = 10_000
N_SEEDS     = 10
BASE_SEED   = 42

MODELS = {
    "LR":  lambda seed: LogisticRegression(max_iter=2000, C=1.0,
                                           random_state=seed),
    "GBM": lambda seed: HistGradientBoostingClassifier(max_iter=200,
                                                       max_depth=3,
                                                       random_state=seed),
}


def load_features() -> dict[str, pd.DataFrame]:
    paths = {
        "B_pos_tr": PROBE_H1H4 / "B_pos_tr.parquet",
        "B_neg_tr": PROBE_H1H4 / "B_neg_tr.parquet",
        "fm9_val_pos": PROBE_H1H4 / "B_pos_va.parquet",
        "fm5_val_neg": PROBE_H1H4 / "B_neg_va.parquet",
        "fm8_val_pos": HERE / "fm8_val.parquet",
        "fm7_val_pos": HERE / "fm7_val.parquet",
    }
    feats = {}
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing cached feature file: {p}")
        feats[k] = pd.read_parquet(p)
        log.info("  loaded %-14s n=%d", k, len(feats[k]))
    return feats


def build_val(feats: dict[str, pd.DataFrame],
              rng: np.random.Generator) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build three val sets (all sharing the same FM=5 negatives)."""
    neg = feats["fm5_val_neg"]
    neg_idx = rng.choice(len(neg), size=min(N_VAL_NEG, len(neg)), replace=False)
    X_neg = neg.iloc[neg_idx][BAND_NAMES].values

    out = {}
    for tier, key in (("fm9", "fm9_val_pos"),
                      ("fm8", "fm8_val_pos"),
                      ("fm7", "fm7_val_pos")):
        pos = feats[key]
        n_pos = min(N_VAL_POS, len(pos))
        if n_pos < N_VAL_POS:
            log.warning("  val tier %s has only %d positives (<%d)", tier, n_pos, N_VAL_POS)
        pos_idx = rng.choice(len(pos), size=n_pos, replace=False)
        X_pos = pos.iloc[pos_idx][BAND_NAMES].values
        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
        out[tier] = (X, y)
        log.info("  val_%s: %d pos + %d neg", tier, len(X_pos), len(X_neg))
    return out


def main() -> None:
    log.info("Loading cached features…")
    feats = load_features()

    # Fixed val set (one draw, shared across all seeds — matches probe_h1_h4 semantics)
    rng_val = np.random.default_rng(BASE_SEED + 9999)
    val_sets = build_val(feats, rng_val)

    # results[model][tier] = list of PR-AUCs across seeds
    results: dict[str, dict[str, list[float]]] = {
        m: defaultdict(list) for m in MODELS
    }

    n_pos_pool = len(feats["B_pos_tr"])
    n_neg_pool = len(feats["B_neg_tr"])
    log.info("Training pools: pos=%d neg=%d", n_pos_pool, n_neg_pool)

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(BASE_SEED + seed * 100)
        pos_idx = rng.choice(n_pos_pool, size=N_TRAIN_POS, replace=False)
        neg_idx = rng.choice(n_neg_pool, size=N_TRAIN_NEG, replace=False)
        X_pos = feats["B_pos_tr"].iloc[pos_idx][BAND_NAMES].values
        X_neg = feats["B_neg_tr"].iloc[neg_idx][BAND_NAMES].values
        X_tr = np.vstack([X_pos, X_neg])
        y_tr = np.concatenate([np.ones(N_TRAIN_POS), np.zeros(N_TRAIN_NEG)])
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)

        for mname, mfactory in MODELS.items():
            clf = mfactory(BASE_SEED + seed)
            clf.fit(X_tr_s, y_tr)
            for tier, (X_va, y_va) in val_sets.items():
                X_va_s = scaler.transform(X_va)
                proba = clf.predict_proba(X_va_s)[:, 1]
                ap = average_precision_score(y_va, proba)
                results[mname][tier].append(float(ap))
        log.info("  seed %2d done; sample PR-AUC LR=%s",
                 seed,
                 {t: f"{results['LR'][t][-1]:.4f}" for t in ("fm9","fm8","fm7")})

    # ---------- summary ----------
    print()
    print("=" * 72)
    print("HARD-POSITIVE PROBE RESULTS  (PR-AUC, mean ± std over %d seeds)" % N_SEEDS)
    print(f"  train: FM9 pos ({N_TRAIN_POS}) + FM5 neg ({N_TRAIN_NEG})")
    print(f"  val  : 5k pos × 10k neg, positives differ by firemask tier")
    print("=" * 72)
    print(f"{'Model':<6} {'Tier':<6} {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print("-" * 72)
    out = {}
    for mname in MODELS:
        out[mname] = {}
        for tier in ("fm9", "fm8", "fm7"):
            a = np.array(results[mname][tier])
            print(f"{mname:<6} {tier:<6} {a.mean():>8.4f}  {a.std():>8.4f}  "
                  f"{a.min():>8.4f}  {a.max():>8.4f}")
            out[mname][tier] = {
                "mean": float(a.mean()),
                "std":  float(a.std()),
                "aucs": [float(x) for x in a],
            }
    print("=" * 72)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Saved %s", OUT_PATH)


if __name__ == "__main__":
    main()
