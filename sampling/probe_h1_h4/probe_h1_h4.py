"""H1 + H4 probe: do FWI-component-specific hard-negative strategies beat the
composite-weighted baseline (variant C) on a small-N PR-AUC probe?

Strategies (all use train: 5k pos + 10k neg; eval: shared val pool)
  Random       : random 10k from variant B's negative pool (buffer-only baseline)
  C_baseline   : random 10k from variant C's negatives (50% composite-hard + 50% random)
  H1_composite : top 10k of B's negatives by composite FWI score (calibrated weights)
  H4_FFMC..BUI : top 10k of B's negatives by single FWI component (one-hot weight)

Probe: LogisticRegression on 6-band FWI features, 20 seeds, PR-AUC.

Date-subsampling: to bound FWI tile-read cost, we keep only N_TRAIN_DATES random
train dates and N_VAL_DATES random val dates from B's pool. Top-K-by-score is then
"top among the kept pool" rather than absolute, but for relative ranking across
strategies it's a fair comparison since they all see the same candidate pool.
"""
from __future__ import annotations
import json, logging, sys, random
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from osgeo import gdal

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("probe_h1_h4")
gdal.UseExceptions()

# ---------- paths ----------
SAMPLES_DIR = Path("/s3/user/wildfire_dataset/h5_dataset/entire_can_v1")
FWI_DIR     = Path("/s3/user/wildfire_dataset/self_built_materials/"
                   "bc_raw_data_new_fast/drivers/processed/fwi_components")
WEIGHT_JSON = Path("/mnt/raid/user/hackathons/dataset_building_entire_canada/"
                   "sampling/fwi/outputs/weight_vector_w1w6.json")
OUT_DIR     = Path("/mnt/raid/user/hackathons/dataset_building_entire_canada/"
                   "sampling/probe_h1_h4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- constants ----------
BAND_NAMES = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
MODIS_GT = (-141.00855014824128, 0.04491576420597608, 0.0,
             83.34569206060921, 0.0, -0.04491576420597608)

N_TRAIN_DATES = 250          # date subsample for B-pool feature extraction
N_VAL_DATES   = 150
N_TRAIN_POS   = 5_000        # per-seed positives
N_TRAIN_NEG   = 10_000       # per-seed negatives
N_VAL_POS     = 5_000        # fixed eval set
N_VAL_NEG     = 10_000
N_SEEDS       = 20
BASE_SEED     = 42

# strategies → weight vector over (FFMC, DMC, DC, ISI, BUI, FWI)
def _load_calibrated_weights() -> np.ndarray:
    with open(WEIGHT_JSON) as f:
        w = json.load(f)
    return np.array([w[n] for n in BAND_NAMES], dtype=np.float64)

CAL_W = _load_calibrated_weights()
P99 = None  # populated after weight load below
with open(WEIGHT_JSON) as f:
    _wjson = json.load(f)
P99 = np.array([_wjson["p99_normalization_thresholds"][n] for n in BAND_NAMES],
                dtype=np.float64)

STRATEGIES = {
    "Random":       None,                                  # special: random sample
    "C_baseline":   None,                                  # special: from variant C
    "H1_composite": CAL_W,
    "H4_FFMC":      np.array([1, 0, 0, 0, 0, 0], dtype=np.float64),
    "H4_DMC":       np.array([0, 1, 0, 0, 0, 0], dtype=np.float64),
    "H4_DC":        np.array([0, 0, 1, 0, 0, 0], dtype=np.float64),
    "H4_ISI":       np.array([0, 0, 0, 1, 0, 0], dtype=np.float64),
    "H4_BUI":       np.array([0, 0, 0, 0, 1, 0], dtype=np.float64),
}

# ---------- FWI tile loading ----------
_TILE_CACHE: dict[str, tuple] = {}

def load_fwi(date_str: str):
    if date_str in _TILE_CACHE:
        return _TILE_CACHE[date_str]
    fpath = FWI_DIR / ("fwis_" + date_str.replace("-", "_") + ".tif")
    if not fpath.exists():
        _TILE_CACHE[date_str] = (None, None)
        return None, None
    ds = gdal.Open(str(fpath), gdal.GA_ReadOnly)
    if ds is None:
        _TILE_CACHE[date_str] = (None, None)
        return None, None
    gt = ds.GetGeoTransform()
    W, H = ds.RasterXSize, ds.RasterYSize
    bands = []
    for b in range(1, 7):
        raw = ds.GetRasterBand(b).ReadRaster(0, 0, W, H, buf_type=gdal.GDT_Float32)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(H, W).copy()
        nd = ds.GetRasterBand(b).GetNoDataValue()
        if nd is not None:
            arr[arr == np.float32(nd)] = np.nan
        bands.append(arr)
    ds = None
    stack = np.stack(bands, axis=-1)  # (H,W,6)
    _TILE_CACHE[date_str] = (stack, gt)
    return stack, gt


def rowcol_to_lonlat(rows, cols, gt):
    return gt[0] + (cols + 0.5) * gt[1], gt[3] + (rows + 0.5) * gt[5]


def lonlat_to_rowcol(lons, lats, gt):
    return ((lats - gt[3]) / gt[5]).astype(int), ((lons - gt[0]) / gt[1]).astype(int)


# ---------- feature extraction ----------
def extract_features(samples: list[dict]) -> pd.DataFrame:
    """Extract 6-band FWI features for a list of samples grouped by date.

    Drops samples whose tile is missing or whose pixel falls outside FWI grid /
    has NaN. Returns DataFrame indexed by sample order with FWI cols + label.
    """
    df = pd.DataFrame(samples)
    rows_out = []
    by_date = df.groupby("date", sort=False)
    n_dates = by_date.ngroups
    log.info("  extracting features over %d dates / %d samples", n_dates, len(df))
    done = 0
    for date, grp in by_date:
        fwi_arr, fwi_gt = load_fwi(date)
        if fwi_arr is None:
            done += 1
            if done % 50 == 0:
                log.info("    [%d/%d] missing tile %s (skipped)", done, n_dates, date)
            continue
        H, W, _ = fwi_arr.shape
        lons, lats = rowcol_to_lonlat(grp["pixel_row"].values,
                                       grp["pixel_col"].values, MODIS_GT)
        rows_f, cols_f = lonlat_to_rowcol(lons, lats, fwi_gt)
        valid = (rows_f >= 0) & (rows_f < H) & (cols_f >= 0) & (cols_f < W)
        for i, idx in enumerate(grp.index):
            if not valid[i]:
                continue
            vals = fwi_arr[rows_f[i], cols_f[i], :]
            if np.any(np.isnan(vals)):
                continue
            rows_out.append((idx, *vals.tolist()))
        done += 1
        if done % 50 == 0:
            log.info("    [%d/%d] dates done, %d feat rows so far",
                     done, n_dates, len(rows_out))
    feat = pd.DataFrame(rows_out,
                        columns=["orig_idx"] + BAND_NAMES).set_index("orig_idx")
    out = df.join(feat).dropna(subset=BAND_NAMES)
    log.info("  kept %d / %d samples after feature extraction", len(out), len(df))
    return out


# ---------- scoring ----------
def score(features: np.ndarray, w: np.ndarray) -> np.ndarray:
    """score(pixel) = Σ wᵢ × clip(FWIᵢ / p99_i, 0, 1)"""
    norm = np.clip(features / P99, 0, 1)
    return norm @ w


# ---------- main probe ----------
def main():
    rng = np.random.default_rng(BASE_SEED)

    # ----- load samples -----
    log.info("Loading B + C sample lists...")
    with open(SAMPLES_DIR / "cache_B" / "samples_variant_B.json") as f:
        B = json.load(f)
    with open(SAMPLES_DIR / "cache_C" / "samples_variant_C.json") as f:
        C = json.load(f)

    def split_filter(lst, split):
        return [s for s in lst if s["split"] == split]

    B_pos_tr = split_filter(B["positive_samples"], "train")
    B_neg_tr = split_filter(B["negative_samples"], "train")
    B_pos_va = split_filter(B["positive_samples"], "val")
    B_neg_va = split_filter(B["negative_samples"], "val")
    C_neg_tr = split_filter(C["negative_samples"], "train")

    log.info("B train pos=%d neg=%d  val pos=%d neg=%d  C train neg=%d",
             len(B_pos_tr), len(B_neg_tr), len(B_pos_va), len(B_neg_va), len(C_neg_tr))

    # ----- date-subsample to bound tile reads -----
    train_dates = sorted({s["date"] for s in B_neg_tr})
    val_dates   = sorted({s["date"] for s in B_neg_va})
    rng.shuffle(train_dates)
    rng.shuffle(val_dates)
    keep_train_dates = set(train_dates[:N_TRAIN_DATES])
    keep_val_dates   = set(val_dates[:N_VAL_DATES])
    log.info("Subsampled to %d train dates / %d val dates",
             len(keep_train_dates), len(keep_val_dates))

    def in_dates(lst, dates):
        return [s for s in lst if s["date"] in dates]

    B_pos_tr_sub = in_dates(B_pos_tr, keep_train_dates)
    B_neg_tr_sub = in_dates(B_neg_tr, keep_train_dates)
    B_pos_va_sub = in_dates(B_pos_va, keep_val_dates)
    B_neg_va_sub = in_dates(B_neg_va, keep_val_dates)
    C_neg_tr_sub = in_dates(C_neg_tr, keep_train_dates)
    log.info("After date filter: B_pos_tr=%d B_neg_tr=%d  B_pos_va=%d B_neg_va=%d  C_neg_tr=%d",
             len(B_pos_tr_sub), len(B_neg_tr_sub), len(B_pos_va_sub), len(B_neg_va_sub),
             len(C_neg_tr_sub))

    # ----- extract features (cache to parquet) -----
    cache = {
        "B_pos_tr": (B_pos_tr_sub, "B_pos_tr.parquet"),
        "B_neg_tr": (B_neg_tr_sub, "B_neg_tr.parquet"),
        "B_pos_va": (B_pos_va_sub, "B_pos_va.parquet"),
        "B_neg_va": (B_neg_va_sub, "B_neg_va.parquet"),
        "C_neg_tr": (C_neg_tr_sub, "C_neg_tr.parquet"),
    }
    feats = {}
    for key, (samples, fname) in cache.items():
        cf = OUT_DIR / fname
        if cf.exists():
            log.info("Loading cached features: %s", fname)
            feats[key] = pd.read_parquet(cf)
        else:
            log.info("Extracting features: %s (%d samples)", key, len(samples))
            df = extract_features(samples)
            df.to_parquet(cf)
            feats[key] = df

    # ----- score B's train negatives by every strategy -----
    Xneg = feats["B_neg_tr"][BAND_NAMES].values  # (N, 6)
    sorted_idx = {}  # strategy -> indices into feats["B_neg_tr"] sorted descending by score
    for name, w in STRATEGIES.items():
        if w is None:
            continue
        s = score(Xneg, w)
        order = np.argsort(-s)
        sorted_idx[name] = order

    # build candidate pools (top 30k = 3x oversample for seed variance)
    N_POOL = 30_000
    pools = {}
    pools["Random"]     = np.arange(len(feats["B_neg_tr"]))    # full B pool, sample randomly
    pools["C_baseline"] = np.arange(len(feats["C_neg_tr"]))    # full C pool, sample randomly
    for name in STRATEGIES:
        if name in pools: continue
        pools[name] = sorted_idx[name][:N_POOL]

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

    # ----- run probe per strategy × seed -----
    pos_pool_size = len(feats["B_pos_tr"])
    log.info("Pos training pool: %d", pos_pool_size)
    log.info("Strategy pools: " +
             ", ".join(f"{k}={len(v)}" for k, v in pools.items()))

    results = defaultdict(list)
    for seed in range(N_SEEDS):
        rseed = np.random.default_rng(BASE_SEED + seed * 100)
        pos_idx = rseed.choice(pos_pool_size, size=N_TRAIN_POS, replace=False)
        Xpos = feats["B_pos_tr"].iloc[pos_idx][BAND_NAMES].values
        for name in STRATEGIES:
            pool = pools[name]
            if len(pool) < N_TRAIN_NEG:
                log.warning("  pool %s smaller than %d (=%d)", name, N_TRAIN_NEG, len(pool))
                continue
            sel = rseed.choice(pool, size=N_TRAIN_NEG, replace=False)
            src = feats["C_neg_tr"] if name == "C_baseline" else feats["B_neg_tr"]
            Xneg_sel = src.iloc[sel][BAND_NAMES].values
            X_tr = np.vstack([Xpos, Xneg_sel])
            y_tr = np.concatenate([np.ones(N_TRAIN_POS), np.zeros(N_TRAIN_NEG)])
            scaler = StandardScaler().fit(X_tr)
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=BASE_SEED)
            clf.fit(scaler.transform(X_tr), y_tr)
            ap = average_precision_score(y_va, clf.predict_proba(scaler.transform(X_va))[:, 1])
            results[name].append(ap)
        log.info("  seed %2d done; current means: %s", seed,
                 {k: f"{np.mean(v):.4f}" for k, v in results.items()})

    # ----- print + save -----
    print()
    print("=" * 72)
    print("H1 + H4 PROBE RESULTS  (PR-AUC, mean ± std over %d seeds)" % N_SEEDS)
    print(f"  train: {N_TRAIN_POS} pos + {N_TRAIN_NEG} neg per seed")
    print(f"  val  : {len(pos_va_idx)} pos + {len(neg_va_idx)} neg fixed")
    print("=" * 72)
    print(f"{'Strategy':<16} {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print("-" * 72)
    for name in STRATEGIES:
        if name not in results: continue
        a = np.array(results[name])
        print(f"{name:<16} {a.mean():>8.4f}  {a.std():>8.4f}  {a.min():>8.4f}  {a.max():>8.4f}")
    print("=" * 72)

    out = OUT_DIR / "results.json"
    with open(out, "w") as f:
        json.dump(
            {name: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                    "aucs": [float(x) for x in v]} for name, v in results.items()},
            f, indent=2)
    log.info("Saved results to %s", out)


if __name__ == "__main__":
    main()
