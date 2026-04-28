"""Follow-up to probe_h1_h4.py: re-run the same strategies under nonlinear models
(Random Forest, Gradient Boosting, small MLP) to test whether LR was masking
selection-effect differences. Reuses cached parquet features.

Reduces seeds to 10 (RF/GBM are slower) but keeps same pool construction, val set,
and train sizes for apples-to-apples comparison with the LR run.
"""
from __future__ import annotations
import json, logging
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("probe_models")

OUT_DIR = Path("/mnt/raid/zhengsen/hackathons/dataset_building_entire_canada/"
               "sampling/probe_h1_h4")
WEIGHT_JSON = Path("/mnt/raid/zhengsen/hackathons/dataset_building_entire_canada/"
                   "sampling/fwi/outputs/weight_vector_w1w6.json")
BAND_NAMES = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]

N_TRAIN_POS = 5_000
N_TRAIN_NEG = 10_000
N_VAL_POS   = 5_000
N_VAL_NEG   = 10_000
N_SEEDS     = 10           # halved from LR run — RF/GBM are slower
BASE_SEED   = 42

with open(WEIGHT_JSON) as f:
    _w = json.load(f)
CAL_W = np.array([_w[n] for n in BAND_NAMES], dtype=np.float64)
P99   = np.array([_w["p99_normalization_thresholds"][n] for n in BAND_NAMES],
                  dtype=np.float64)

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

MODELS = {
    "LR":  lambda seed: LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
    "RF":  lambda seed: RandomForestClassifier(n_estimators=300, max_depth=None,
                                                 n_jobs=-1, random_state=seed),
    "GBM": lambda seed: GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                    learning_rate=0.1, random_state=seed),
    "MLP": lambda seed: MLPClassifier(hidden_layer_sizes=(64, 32),
                                       max_iter=500, random_state=seed),
}


def score_pool(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.clip(X / P99, 0, 1) @ w


def main():
    log.info("Loading cached feature parquets...")
    feats = {k: pd.read_parquet(OUT_DIR / f"{k}.parquet")
             for k in ["B_pos_tr", "B_neg_tr", "B_pos_va", "B_neg_va", "C_neg_tr"]}
    for k, v in feats.items():
        log.info("  %s: %d rows", k, len(v))

    # ----- score B's train negatives by each strategy & build candidate pools -----
    Xneg = feats["B_neg_tr"][BAND_NAMES].values
    N_POOL = 30_000
    pools: dict[str, np.ndarray] = {}
    pools["Random"]     = np.arange(len(feats["B_neg_tr"]))
    pools["C_baseline"] = np.arange(len(feats["C_neg_tr"]))
    for name, w in STRATEGIES.items():
        if w is None: continue
        s = score_pool(Xneg, w)
        order = np.argsort(-s)
        pools[name] = order[:min(N_POOL, len(order))]

    # ----- fixed val eval set -----
    rng_val = np.random.default_rng(BASE_SEED + 9999)
    pos_va_idx = rng_val.choice(len(feats["B_pos_va"]),
                                 size=min(N_VAL_POS, len(feats["B_pos_va"])),
                                 replace=False)
    neg_va_idx = rng_val.choice(len(feats["B_neg_va"]),
                                 size=min(N_VAL_NEG, len(feats["B_neg_va"])),
                                 replace=False)
    X_va = np.vstack([feats["B_pos_va"].iloc[pos_va_idx][BAND_NAMES].values,
                       feats["B_neg_va"].iloc[neg_va_idx][BAND_NAMES].values])
    y_va = np.concatenate([np.ones(len(pos_va_idx)), np.zeros(len(neg_va_idx))])
    log.info("Val eval set: %d pos + %d neg", len(pos_va_idx), len(neg_va_idx))

    # ----- run probe: model × strategy × seed -----
    pos_pool = len(feats["B_pos_tr"])
    results: dict[str, dict[str, list[float]]] = {
        m: defaultdict(list) for m in MODELS
    }

    for seed in range(N_SEEDS):
        rseed = np.random.default_rng(BASE_SEED + seed * 100)
        pos_idx = rseed.choice(pos_pool, size=N_TRAIN_POS, replace=False)
        Xpos = feats["B_pos_tr"].iloc[pos_idx][BAND_NAMES].values

        for name in STRATEGIES:
            pool = pools[name]
            if len(pool) < N_TRAIN_NEG: continue
            sel = rseed.choice(pool, size=N_TRAIN_NEG, replace=False)
            src = feats["C_neg_tr"] if name == "C_baseline" else feats["B_neg_tr"]
            Xneg_sel = src.iloc[sel][BAND_NAMES].values
            X_tr = np.vstack([Xpos, Xneg_sel])
            y_tr = np.concatenate([np.ones(N_TRAIN_POS), np.zeros(N_TRAIN_NEG)])

            for mname, mfactory in MODELS.items():
                if mname in ("LR", "MLP"):
                    scaler = StandardScaler().fit(X_tr)
                    X_tr_m, X_va_m = scaler.transform(X_tr), scaler.transform(X_va)
                else:
                    X_tr_m, X_va_m = X_tr, X_va
                clf = mfactory(BASE_SEED + seed)
                clf.fit(X_tr_m, y_tr)
                ap = average_precision_score(y_va, clf.predict_proba(X_va_m)[:, 1])
                results[mname][name].append(ap)
        log.info("seed %d done", seed)

    # ----- print + save -----
    print()
    print("=" * 90)
    print("H1 + H4 PROBE — MODEL SWEEP (PR-AUC, mean ± std over %d seeds)" % N_SEEDS)
    print(f"  train: {N_TRAIN_POS} pos + {N_TRAIN_NEG} neg | val: {len(pos_va_idx)} pos + {len(neg_va_idx)} neg")
    print("=" * 90)
    hdr = f"{'Strategy':<14}" + "".join(f"  {m:>18}" for m in MODELS)
    print(hdr)
    print("-" * 90)
    for name in STRATEGIES:
        row = f"{name:<14}"
        for m in MODELS:
            a = np.array(results[m][name]) if name in results[m] else np.array([])
            if len(a):
                row += f"  {a.mean():>8.4f} ± {a.std():>6.4f}"
            else:
                row += f"  {'-':>18}"
        print(row)
    print("=" * 90)

    out = OUT_DIR / "results_models.json"
    with open(out, "w") as f:
        json.dump({m: {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                           "aucs": [float(x) for x in v]} for k, v in d.items()}
                    for m, d in results.items()}, f, indent=2)
    log.info("Saved → %s", out)


if __name__ == "__main__":
    main()
