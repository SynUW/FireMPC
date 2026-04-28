"""Extended hard-positive probe — additional metrics beyond PR-AUC.

Question: does the FM9 → FM7 difficulty gap that showed up as ~9.5 PR-AUC
points get AMPLIFIED by other metrics (calibration, top-K hit rate,
precision @ fixed FPR)? If yes, the hard-positive story is stronger; if no,
we're bounded by the ~9.5pt PR-AUC gap and need Plan B.

Reuses the exact train/val construction from probe_hardpos.py — same feature
parquets, same N_TRAIN_POS/NEG, same seeds. Only the evaluation is extended.

Metrics (per seed × per tier × per model)
-----------------------------------------
  PR-AUC           baseline from probe_hardpos.py (kept for reference)
  ROC-AUC          standard rank metric
  Brier            calibration: mean squared error of probabilities
  ECE (15-bin)     expected calibration error, uniform-width bins
  Hit@K=1%/5%/10%  fraction of positives among the top-K predicted (of the full val set)
  P@FPR=1%         precision at a 1% false-positive-rate threshold
  P@FPR=0.1%       precision at a 0.1% FPR threshold
  Recall@FPR=1%    recall at that same threshold (operational signal)

Outputs:
  probe_hardpos/results_extra.json     — full per-seed numbers
  probe_hardpos/results_extra.log      — pretty-printed summary table
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
log = logging.getLogger("probe_hardpos_extra")

HERE        = Path(__file__).resolve().parent
PROBE_H1H4  = HERE.parent / "probe_h1_h4"
OUT_JSON    = HERE / "results_extra.json"

BAND_NAMES  = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
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


# ---------- metric helpers ----------
def expected_calibration_error(y_true: np.ndarray, p: np.ndarray,
                               n_bins: int = 15) -> float:
    """Uniform-width ECE (as in Guo et al 2017, without the temperature part)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    N = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf = p[m].mean()
        acc  = y_true[m].mean()
        ece += (m.sum() / N) * abs(conf - acc)
    return float(ece)


def hit_at_k(y_true: np.ndarray, scores: np.ndarray, k_frac: float) -> float:
    """Fraction of top-(k_frac × N) predicted items that are positive."""
    N = len(y_true)
    k = max(1, int(round(k_frac * N)))
    order = np.argsort(-scores)
    top_k = y_true[order[:k]]
    return float(top_k.mean())


def precision_recall_at_fpr(y_true: np.ndarray, scores: np.ndarray,
                            fpr_target: float) -> tuple[float, float, float]:
    """Return (threshold, precision, recall) at the lowest FPR ≥ fpr_target.

    Defined via the ROC curve (so "FPR" uses the true negatives as the base).
    """
    fpr, tpr, thr = roc_curve(y_true, scores)
    # Find the smallest threshold where FPR ≤ fpr_target (i.e. the tightest
    # operating point that still honours the FPR budget).
    valid = np.where(fpr <= fpr_target)[0]
    if valid.size == 0:
        return float("nan"), float("nan"), float("nan")
    j = valid[-1]  # largest TPR within the budget
    t = thr[j]
    pred = scores >= t
    tp = float(((pred == 1) & (y_true == 1)).sum())
    fp = float(((pred == 1) & (y_true == 0)).sum())
    fn = float(((pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return float(t), float(prec), float(rec)


# ---------- data ----------
def load_features() -> dict[str, pd.DataFrame]:
    paths = {
        "B_pos_tr":    PROBE_H1H4 / "B_pos_tr.parquet",
        "B_neg_tr":    PROBE_H1H4 / "B_neg_tr.parquet",
        "fm9_val_pos": PROBE_H1H4 / "B_pos_va.parquet",
        "fm5_val_neg": PROBE_H1H4 / "B_neg_va.parquet",
        "fm8_val_pos": HERE / "fm8_val.parquet",
        "fm7_val_pos": HERE / "fm7_val.parquet",
    }
    out = {}
    for k, p in paths.items():
        out[k] = pd.read_parquet(p)
        log.info("  loaded %-14s n=%d", k, len(out[k]))
    return out


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
        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
        out[tier] = (X, y)
    return out


# ---------- main ----------
def main() -> None:
    log.info("Loading features…")
    feats = load_features()

    rng_val = np.random.default_rng(BASE_SEED + 9999)
    val_sets = build_val(feats, rng_val)
    for tier, (X, y) in val_sets.items():
        log.info("  val_%s: %d pos + %d neg", tier, int(y.sum()), int((1 - y).sum()))

    # results[model][tier] = defaultdict(list) of {metric: [per-seed values]}
    results = {m: {t: defaultdict(list) for t in ("fm9", "fm8", "fm7")}
               for m in MODELS}

    n_pos_pool = len(feats["B_pos_tr"])
    n_neg_pool = len(feats["B_neg_tr"])

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(BASE_SEED + seed * 100)
        pos_idx = rng.choice(n_pos_pool, size=N_TRAIN_POS, replace=False)
        neg_idx = rng.choice(n_neg_pool, size=N_TRAIN_NEG, replace=False)
        X_tr = np.vstack([
            feats["B_pos_tr"].iloc[pos_idx][BAND_NAMES].values,
            feats["B_neg_tr"].iloc[neg_idx][BAND_NAMES].values,
        ])
        y_tr = np.concatenate([np.ones(N_TRAIN_POS), np.zeros(N_TRAIN_NEG)])
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)

        for mname, mfactory in MODELS.items():
            clf = mfactory(BASE_SEED + seed)
            clf.fit(X_tr_s, y_tr)
            for tier, (X_va, y_va) in val_sets.items():
                X_s = scaler.transform(X_va)
                p   = clf.predict_proba(X_s)[:, 1]
                r = results[mname][tier]
                r["pr_auc"].append(float(average_precision_score(y_va, p)))
                r["roc_auc"].append(float(roc_auc_score(y_va, p)))
                r["brier"].append(float(brier_score_loss(y_va, p)))
                r["ece"].append(expected_calibration_error(y_va, p))
                r["hit@1"].append(hit_at_k(y_va, p, 0.01))
                r["hit@5"].append(hit_at_k(y_va, p, 0.05))
                r["hit@10"].append(hit_at_k(y_va, p, 0.10))
                _, p01, rec01 = precision_recall_at_fpr(y_va, p, 0.01)
                _, p001, rec001 = precision_recall_at_fpr(y_va, p, 0.001)
                r["P@FPR1%"].append(p01)
                r["R@FPR1%"].append(rec01)
                r["P@FPR0.1%"].append(p001)
                r["R@FPR0.1%"].append(rec001)
        log.info("  seed %2d done", seed)

    # ---------- summary ----------
    metric_order = ["pr_auc", "roc_auc", "brier", "ece",
                    "hit@1", "hit@5", "hit@10",
                    "P@FPR1%", "R@FPR1%", "P@FPR0.1%", "R@FPR0.1%"]

    print()
    print("=" * 100)
    print("HARD-POSITIVE EXTENDED METRICS  (mean over %d seeds; FM9 − FM7 = drop from easy to hard)" % N_SEEDS)
    print("=" * 100)
    header = f"{'Metric':<12} {'Model':<5}  {'FM=9':>10}  {'FM=8':>10}  {'FM=7':>10}  {'Δ(9-7)':>10}  {'rel drop%':>10}"
    print(header)
    print("-" * 100)
    for metric in metric_order:
        for mname in MODELS:
            v9 = np.mean(results[mname]["fm9"][metric])
            v8 = np.mean(results[mname]["fm8"][metric])
            v7 = np.mean(results[mname]["fm7"][metric])
            delta = v9 - v7
            rel = 100.0 * delta / v9 if v9 != 0 else float("nan")
            print(f"{metric:<12} {mname:<5}  {v9:>10.4f}  {v8:>10.4f}  {v7:>10.4f}  "
                  f"{delta:>+10.4f}  {rel:>+9.2f}%")
        print()

    print("=" * 100)
    print("Notes on metric direction:")
    print("  PR-AUC / ROC-AUC / hit@K / P@FPR / R@FPR:  HIGHER is better → Δ(9-7) > 0 means gap")
    print("  Brier / ECE:                               LOWER is better  → Δ(9-7) < 0 means gap (FM=7 is WORSE ⇒ its value is larger)")
    print("=" * 100)

    # Save
    dump = {}
    for mname in MODELS:
        dump[mname] = {}
        for tier in ("fm9", "fm8", "fm7"):
            dump[mname][tier] = {k: [float(x) for x in v]
                                 for k, v in results[mname][tier].items()}
    with open(OUT_JSON, "w") as f:
        json.dump(dump, f, indent=2)
    log.info("Saved %s", OUT_JSON)


if __name__ == "__main__":
    main()
