#!/usr/bin/env bash
# Drive CanFireCast prediction over a range of targets, one target at a time.
#
# Channels 19-25 are derived per target from that target's certified D-11 state
# (canfirecast/model/cffdrs_derive.py enforces a (10,55,H,W) window), so each
# target needs its own package -- a dense multi-date file cannot represent them.
# This publishes one window out of the monthly store, runs the model on it, and
# deletes it, so peak disk stays at one package instead of ~300 MB per target.
set -euo pipefail

START=${START:?set START=YYYY-MM-DD}
END=${END:?set END=YYYY-MM-DD}
DEVICE=${DEVICE:-cuda:0}
MODEL_DIRS=${MODEL_DIRS:-/s3/zhengsen/pths/firempc_15days}
MODEL_FILTER=${MODEL_FILTER:-CanFireCastIFSCommon}   # exact model_name match
# Horizon lever: --forecast-task picks {ModelName}_dayXX.pth.  "1" is next-day
# only; empty iterates day01..day15.  Data loading is always next-day
# internally, so this selects checkpoints, not the window.
HORIZONS=${HORIZONS:-1}
OUT=${OUT:-/caribou/quinn/wildfire/predictions/canfirecast}
WORK=${WORK:-/caribou/quinn/wildfire/work/predict_sweep}
KEEP=${KEEP:-0}          # 1 = keep each published package
REPO=${REPO:-/mnt/raid/zhengsen/aaai27_wildfire_experiment_results/canfirecast_15day_era5_ifs_20260818/frozen_repo}
# Publishing uses the operational env (gee_down, no torch); inference needs a
# torch env.  The original runs used /home/zhengsen/.conda/envs/mamba_env, which
# is mode 0750 and unreadable here, so this is the nearest equivalent: torch
# 2.5.1 predates the weights_only default flip, and the checkpoints are plain
# state_dicts that load cleanly either way.
MODEL_PYTHON=${MODEL_PYTHON:-/opt/anaconda3/envs/gsilab_shared/bin/python}
# gsilab_shared pairs geopandas 0.14.3 with fiona 1.10.1, which dropped
# `fiona.path`; reading the boundary shapefile raises AttributeError.  Rather
# than modify a shared /opt env, overlay a newer geopandas (which uses pyogrio
# and never touches fiona) ahead of it on PYTHONPATH.  --no-deps keeps the
# env's own numpy/pandas/shapely in place; only geopandas is shadowed.
VIS_OVERLAY=${VIS_OVERLAY:-/caribou/quinn/wildfire/work/vis_overlay}
RELEASE=${RELEASE:-/caribou/quinn/wildfire/assets/canfirecast/operational-v1}
CLI=${CLI:-/home/quinn/FireMPC/operational/cli.py}

mkdir -p "$OUT" "$WORK"
log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }

day=$START
while [[ "$day" < "$END" || "$day" == "$END" ]]; do
  # Output is named YYYY_MM_DD.tif by the dense-input path (the older
  # window001_01_YYYYMMDD_prob.tif naming predates the Aug 20 script change).
  stamp=${day//-/_}
  # Resume: skip a target whose map already exists.
  if [[ -n "$(find "$OUT" -name "${stamp}.tif" -print -quit 2>/dev/null)" ]]; then
    log "$day already predicted, skipping"
    day=$(date -I -d "$day + 1 day"); continue
  fi

  pkg="$WORK/$day"
  mkdir -p "$pkg"
  log "$day publishing window"
  python "$CLI" canfirecast store publish \
      --target "$day" \
      --static "$RELEASE/statics.h5" \
      --correction-artifact "$RELEASE/cffdrs_aware_correction.json" \
      --output "$pkg/2026_year_dataset.h5"

  log "$day running model on $DEVICE"
  ( cd "$REPO" && PYTHONPATH="$VIS_OVERLAY${PYTHONPATH:+:$PYTHONPATH}" "$MODEL_PYTHON" visualization_gen.py \
      --h5-file "$pkg/2026_year_dataset.h5" \
      --model-dirs "$MODEL_DIRS" \
      --model-filter "$MODEL_FILTER" \
      ${HORIZONS:+--forecast-task $HORIZONS} \
      --output-dir "$OUT" \
      --device "$DEVICE" )

  # visualization_gen logs "No model files found" and exits 0 when the filter
  # matches nothing, so a clean exit is not evidence of output.  Fail loudly
  # instead of sweeping a whole month into an empty directory.
  if [[ -z "$(find "$OUT" -name "${stamp}.tif" -print -quit 2>/dev/null)" ]]; then
    log "$day FAILED: model exited cleanly but wrote no ${stamp}.tif"
    exit 1
  fi
  [[ "$KEEP" == "1" ]] || rm -rf "$pkg"
  log "$day done"
  day=$(date -I -d "$day + 1 day")
done
log "sweep complete $START..$END"
