"""8-strategy × 3-tier probe — does hard-negative mining finally differentiate
when we evaluate on harder positive tiers?

Previous finding (probe_h1_h4 + probe_h1_h4_models):
  On FM=9 val positives, all 8 sampling strategies came in within 0.005 PR-AUC
  — dead-flat. We concluded sampling was useless.

This probe's question:
  When we swap the val positives to FM=8 and FM=7 (harder, operationally more
  relevant fires), does the ordering of strategies change, or does any strategy
  pull ahead by a non-trivial margin?

If YES → hard-negative mining DOES help, but only on harder positives. Paper
         angle becomes "sampling × difficulty interaction".
If NO  → sampling is well and truly dead as a novelty, at all difficulty levels.

Strategies (identical to probe_h1_h4.py — copied verbatim for apples-to-apples)
-----------------------------------------------------------------------------
  Random       : random draw from variant B's full negative pool (buffer-only)
  C_baseline   : random draw from variant C's negatives (50% composite-hard + 50% random)
  H1_composite : top-K of B's negatives by composite FWI score (calibrated w)
  H4_FFMC..BUI : top-K of B's negatives by single FWI component (one-hot w)

Train: 5k FM=9 pos + 10k strategy-picked FM=5 neg (same as probe_h1_h4)
Val:   5k pos × 10k FM=5 neg, with pos tier ∈ {FM=9, FM=8, FM=7} (3 separate evals)
Seeds: 10
Models: LR (primary) + GBM (sanity)
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
log = logging.getLogger("probe_strats_x_tiers")

HERE        = Path(__file__).resolve().parent
PROBE_H1H4  = HERE.parent / "probe_h1_h4"
WEIGHT_JSON = HERE.parent / "fwi" / "outputs" / "weight_vector_w1w6.json"
OUT_JSON    = HERE / "results_strategies_x_tiers.json"

BAND_NAMES  = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
N_TRAIN_POS = 5_000
N_TRAIN_NEG = 10_000
N_VAL_POS   = 5_000
N_VAL_NEG   = 10_000
N_SEEDS     = 10
BASE_SEED   = 42
N_POOL      = 30_000  # top-K oversample for strategies that rank negatives

MODELS = {
    "LR":  lambda seed: LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
    "GBM": lambda seed: HistGradientBoostingClassifier(max_iter=200, max_depth=3,
                                                       random_state=seed),
}


# ---------- strategy weight vectors ----------
def load_weights():
    with open(WEIGHT_JSON) as f:
        w = json.load(f)
    cal = np.array([w[n] for n in BAND_NAMES], dtype=np.float64)
    p99 = np.array([w["p99_normalization_thresholds"][n] for n in BAND_NAMES],
                   dtype=np.float64)
    return cal, p99

CAL_W, P99 = load_weights()

STRATEGIES = {
    "Random":       None,                                      # random from B
    "C_baseline":   None,                                      # random from C
    "H1_composite": CAL_W,
    "H4_FFMC":      np.array([1, 0, 0, 0, 0, 0], dtype=np.float64),
    "H4_DMC":       np.array([0, 1, 0, 0, 0, 0], dtype=np.float64),
    "H4_DC":        np.array([0, 0, 1, 0, 0, 0], dtype=np.float64),
    "H4_ISI":       np.array([0, 0, 0, 1, 0, 0], dtype=np.float64),
    "H4_BUI":       np.array([0, 0, 0, 0, 1, 0], dtype=np.float64),
}


def score(features: np.ndarray, w: np.ndarray) -> np.ndarray:
    """composite hard-neg score: Σ wᵢ × clip(FWIᵢ / p99_i, 0, 1)"""
    norm = np.clip(features / P99, 0, 1)
    return norm @ w


# ---------- load cached features ----------
def load_features():
    paths = {
        "B_pos_tr":    PROBE_H1H4 / "B_pos_tr.parquet",
        "B_neg_tr":    PROBE_H1H4 / "B_neg_tr.parquet",
        "C_neg_tr":    PROBE_H1H4 / "C_neg_tr.parquet",
        "fm9_val_pos": PROBE_H1H4 / "B_pos_va.parquet",
        "fm5_val_neg": PROBE_H1H4 / "B_neg_va.parquet",
        "fm8_val_pos": HERE / "fm8_val.parquet",
        "fm7_val_pos": HERE / "fm7_val.parquet",
    }
    feats = {}
    for k, p in paths.items():
        feats[k] = pd.read_parquet(p)
        log.info("  loaded %-14s n=%d", k, len(feats[k]))
    return feats


def build_val(feats, rng):
    neg = feats["fm5_val_neg"]
    neg_idx = rng.choice(len(neg), size=min(N_VAL_NEG, len(neg)), replace=False)
    X_neg = neg.iloc[neg_idx][BAND_NAMES].values
    out = {}
    for tier, key in (("fm9", "fm9_val_pos"),
                      ("fm8", "fm8_val_pos"),
                      ("fm7", "fm7_val_pos")):
        pos = feats[key]
        n_pos = min(N_VAL_POS, len(pos))
        pos_idx = rng.choice(len(pos), size=n_pos, replace=False)
        X_pos = pos.iloc[pos_idx][BAND_NAMES].values
        out[tier] = (np.vstack([X_pos, X_neg]),
                     np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))]))
    return out


# ---------- main ----------
def main() -> None:
    log.info("Loading features…")
    feats = load_features()

    rng_val = np.random.default_rng(BASE_SEED + 9999)
    val_sets = build_val(feats, rng_val)
    for tier, (X, y) in val_sets.items():
        log.info("  val_%s: %d pos + %d neg", tier, int(y.sum()), int((1 - y).sum()))

    # Build per-strategy candidate pools over B's train negatives
    Xneg_B = feats["B_neg_tr"][BAND_NAMES].values
    pools = {
        "Random":     np.arange(len(feats["B_neg_tr"])),  # random from B
        "C_baseline": np.arange(len(feats["C_neg_tr"])),  # random from C
    }
    for name, w in STRATEGIES.items():
        if name in pools:
            continue
        s = score(Xneg_B, w)
        order = np.argsort(-s)
        pools[name] = order[:N_POOL]
    log.info("Pools: %s", {k: len(v) for k, v in pools.items()})

    # results[strategy][model][tier] = list of PR-AUCs across seeds
    results = {s: {m: {t: [] for t in ("fm9", "fm8", "fm7")}
                   for m in MODELS}
               for s in STRATEGIES}

    n_pos_pool = len(feats["B_pos_tr"])

    for seed in range(N_SEEDS):
        rseed = np.random.default_rng(BASE_SEED + seed * 100)
        pos_idx = rseed.choice(n_pos_pool, size=N_TRAIN_POS, replace=False)
        X_pos = feats["B_pos_tr"].iloc[pos_idx][BAND_NAMES].values

        for sname in STRATEGIES:
            pool = pools[sname]
            sel = rseed.choice(pool, size=N_TRAIN_NEG, replace=False)
            src = feats["C_neg_tr"] if sname == "C_baseline" else feats["B_neg_tr"]
            X_neg = src.iloc[sel][BAND_NAMES].values
            X_tr = np.vstack([X_pos, X_neg])
            y_tr = np.concatenate([np.ones(N_TRAIN_POS), np.zeros(N_TRAIN_NEG)])
            scaler = StandardScaler().fit(X_tr)
            X_tr_s = scaler.transform(X_tr)

            for mname, mfactory in MODELS.items():
                clf = mfactory(BASE_SEED + seed)
                clf.fit(X_tr_s, y_tr)
                for tier, (X_va, y_va) in val_sets.items():
                    proba = clf.predict_proba(scaler.transform(X_va))[:, 1]
                    ap = float(average_precision_score(y_va, proba))
                    results[sname][mname][tier].append(ap)
        log.info("  seed %2d done", seed)

    # ---------- summary ----------
    print()
    for mname in MODELS:
        print("=" * 96)
        print(f"STRATEGIES × TIERS  (PR-AUC, mean ± std over {N_SEEDS} seeds) — model={mname}")
        print("=" * 96)
        print(f"{'Strategy':<14} {'FM=9':>18}  {'FM=8':>18}  {'FM=7':>18}   "
              f"{'Δ(FM9−FM7)':>14}")
        print("-" * 96)
        for sname in STRATEGIES:
            row = []
            for tier in ("fm9", "fm8", "fm7"):
                a = np.array(results[sname][mname][tier])
                row.append((a.mean(), a.std()))
            d97 = row[0][0] - row[2][0]
            print(f"{sname:<14} "
                  f"{row[0][0]:>8.4f}±{row[0][1]:<8.4f}  "
                  f"{row[1][0]:>8.4f}±{row[1][1]:<8.4f}  "
                  f"{row[2][0]:>8.4f}±{row[2][1]:<8.4f}   "
                  f"{d97:>+14.4f}")
        print("-" * 96)
        # Spread within each tier (max − min across strategies)
        print(f"{'spread':<14} ", end="")
        for tier in ("fm9", "fm8", "fm7"):
            means = [np.mean(results[s][mname][tier]) for s in STRATEGIES]
            spread = max(means) - min(means)
            print(f"{'Δ=':>10}{spread:<16.4f}", end="  ")
        print()
        print()

    # Best-strategy-per-tier table
    print("=" * 72)
    print("BEST STRATEGY PER TIER (LR, by mean PR-AUC)")
    print("=" * 72)
    for tier in ("fm9", "fm8", "fm7"):
        ranked = sorted(
            ((s, float(np.mean(results[s]["LR"][tier]))) for s in STRATEGIES),
            key=lambda x: -x[1],
        )
        top = ranked[0]
        random_mean = float(np.mean(results["Random"]["LR"][tier]))
        delta_vs_random = top[1] - random_mean
        print(f"  {tier}: best = {top[0]:<14} ({top[1]:.4f}), "
              f"vs Random = {random_mean:.4f}, Δ = {delta_vs_random:+.4f}")
        for s, v in ranked[:4]:
            print(f"        {s:<14} {v:.4f}")
    print("=" * 72)

    # Save
    dump = {s: {m: {t: [float(x) for x in results[s][m][t]]
                    for t in ("fm9", "fm8", "fm7")}
                for m in MODELS}
            for s in STRATEGIES}
    with open(OUT_JSON, "w") as f:
        json.dump(dump, f, indent=2)
    log.info("Saved %s", OUT_JSON)


if __name__ == "__main__":
    main()
