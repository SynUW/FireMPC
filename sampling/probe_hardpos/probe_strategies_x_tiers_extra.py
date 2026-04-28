"""8 strategies × 3 tiers × rich metric suite.

Same training/val construction as probe_strategies_x_tiers.py — just swaps the
single PR-AUC for a broader metric panel to check whether any one of them
finally shows daylight between sampling strategies.

Metrics per (strategy, model, tier, seed):
  pr_auc      rank metric (baseline)
  roc_auc     rank metric (bounded differently, can hide/reveal different gaps)
  brier       calibration MSE of probability
  ece         expected calibration error, 15 uniform bins
  hit@1       fraction of positives in top 1% of predictions
  hit@5       fraction of positives in top 5%
  hit@10      fraction of positives in top 10%
  R@FPR1%     recall at FPR=1% operating point (the one that showed 42% tier gap)
  R@FPR0.1%   recall at FPR=0.1% operating point
  P@FPR1%     precision at the same operating point
  P@FPR0.1%
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("probe_strats_x_tiers_extra")

HERE        = Path(__file__).resolve().parent
PROBE_H1H4  = HERE.parent / "probe_h1_h4"
WEIGHT_JSON = HERE.parent / "fwi" / "outputs" / "weight_vector_w1w6.json"
OUT_JSON    = HERE / "results_strategies_x_tiers_extra.json"

BAND_NAMES  = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
N_TRAIN_POS = 5_000
N_TRAIN_NEG = 10_000
N_VAL_POS   = 5_000
N_VAL_NEG   = 10_000
N_SEEDS     = 10
BASE_SEED   = 42
N_POOL      = 30_000

MODELS = {
    "LR":  lambda seed: LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
    "GBM": lambda seed: HistGradientBoostingClassifier(max_iter=200, max_depth=3,
                                                       random_state=seed),
}

METRICS = ["pr_auc", "roc_auc", "brier", "ece",
           "hit@1", "hit@5", "hit@10",
           "R@FPR1%", "R@FPR0.1%", "P@FPR1%", "P@FPR0.1%"]
TIERS = ("fm9", "fm8", "fm7")


def load_weights():
    with open(WEIGHT_JSON) as f:
        w = json.load(f)
    cal = np.array([w[n] for n in BAND_NAMES], dtype=np.float64)
    p99 = np.array([w["p99_normalization_thresholds"][n] for n in BAND_NAMES],
                   dtype=np.float64)
    return cal, p99

CAL_W, P99 = load_weights()

STRATEGIES = {
    "Random":       None,
    "C_baseline":   None,
    "H1_composite": CAL_W,
    "H4_FFMC":      np.array([1, 0, 0, 0, 0, 0], dtype=np.float64),
    "H4_DMC":       np.array([0, 1, 0, 0, 0, 0], dtype=np.float64),
    "H4_DC":        np.array([0, 0, 1, 0, 0, 0], dtype=np.float64),
    "H4_ISI":       np.array([0, 0, 0, 1, 0, 0], dtype=np.float64),
    "H4_BUI":       np.array([0, 0, 0, 0, 1, 0], dtype=np.float64),
}


def score(features, w):
    return np.clip(features / P99, 0, 1) @ w


def expected_calibration_error(y_true, p, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    N = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / N) * abs(p[m].mean() - y_true[m].mean())
    return float(ece)


def hit_at_k(y_true, scores, k_frac):
    N = len(y_true)
    k = max(1, int(round(k_frac * N)))
    order = np.argsort(-scores)
    return float(y_true[order[:k]].mean())


def precision_recall_at_fpr(y_true, scores, fpr_target):
    fpr, tpr, thr = roc_curve(y_true, scores)
    valid = np.where(fpr <= fpr_target)[0]
    if valid.size == 0:
        return float("nan"), float("nan")
    j = valid[-1]
    t = thr[j]
    pred = scores >= t
    tp = float(((pred == 1) & (y_true == 1)).sum())
    fp = float(((pred == 1) & (y_true == 0)).sum())
    fn = float(((pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return prec, rec


def compute_all_metrics(y_true, p):
    prec1, rec1 = precision_recall_at_fpr(y_true, p, 0.01)
    prec01, rec01 = precision_recall_at_fpr(y_true, p, 0.001)
    return {
        "pr_auc":    float(average_precision_score(y_true, p)),
        "roc_auc":   float(roc_auc_score(y_true, p)),
        "brier":     float(brier_score_loss(y_true, p)),
        "ece":       expected_calibration_error(y_true, p),
        "hit@1":     hit_at_k(y_true, p, 0.01),
        "hit@5":     hit_at_k(y_true, p, 0.05),
        "hit@10":    hit_at_k(y_true, p, 0.10),
        "R@FPR1%":   rec1,
        "R@FPR0.1%": rec01,
        "P@FPR1%":   prec1,
        "P@FPR0.1%": prec01,
    }


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


def main():
    log.info("Loading features…")
    feats = load_features()
    rng_val = np.random.default_rng(BASE_SEED + 9999)
    val_sets = build_val(feats, rng_val)

    # pools
    Xneg_B = feats["B_neg_tr"][BAND_NAMES].values
    pools = {
        "Random":     np.arange(len(feats["B_neg_tr"])),
        "C_baseline": np.arange(len(feats["C_neg_tr"])),
    }
    for name, w in STRATEGIES.items():
        if name in pools: continue
        order = np.argsort(-score(Xneg_B, w))
        pools[name] = order[:N_POOL]

    # results[strategy][model][tier][metric] = [per-seed]
    results = {s: {m: {t: {k: [] for k in METRICS}
                       for t in TIERS}
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
                    p = clf.predict_proba(scaler.transform(X_va))[:, 1]
                    m = compute_all_metrics(y_va, p)
                    for k, v in m.items():
                        results[sname][mname][tier][k].append(float(v))
        log.info("  seed %2d done", seed)

    # Print per-metric strategy-spread tables
    print()
    for mname in MODELS:
        for metric in METRICS:
            # Determine direction: calibration metrics (brier/ece) lower = better
            higher_is_better = metric not in ("brier", "ece")
            print("=" * 104)
            print(f"{metric}  |  model={mname}  "
                  f"({'higher=better' if higher_is_better else 'LOWER=better'})")
            print("=" * 104)
            header = f"{'Strategy':<14}"
            for tier in TIERS:
                header += f"  {tier:>18}"
            header += f"   {'col spread':>11}"
            print(header)
            print("-" * 104)
            # compute per-tier column means and spreads
            col_means = {t: {} for t in TIERS}
            for sname in STRATEGIES:
                row = f"{sname:<14}"
                for tier in TIERS:
                    a = np.array(results[sname][mname][tier][metric])
                    col_means[tier][sname] = float(a.mean())
                    row += f"  {a.mean():>9.4f}±{a.std():<.4f}"
                row += f"   {'':>11}"
                print(row)
            # spread row
            row = f"{'MAX-MIN':<14}"
            for tier in TIERS:
                vals = list(col_means[tier].values())
                spread = max(vals) - min(vals)
                row += f"  {spread:>18.4f}"
            print(row)
            # best per tier
            row = f"{'BEST':<14}"
            for tier in TIERS:
                if higher_is_better:
                    best = max(col_means[tier].items(), key=lambda x: x[1])
                else:
                    best = min(col_means[tier].items(), key=lambda x: x[1])
                row += f"  {best[0]:>18}"
            print(row)
            print()

    # Save
    dump = {s: {m: {t: {k: [float(x) for x in results[s][m][t][k]]
                        for k in METRICS}
                    for t in TIERS}
                for m in MODELS}
            for s in STRATEGIES}
    with open(OUT_JSON, "w") as f:
        json.dump(dump, f, indent=2)
    log.info("Saved %s", OUT_JSON)


if __name__ == "__main__":
    main()
