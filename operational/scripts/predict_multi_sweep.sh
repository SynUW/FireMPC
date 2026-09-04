#!/usr/bin/env bash
# Multi-horizon CanFireCast prediction: every lead from one published window.
#
# `--forecast-task` selects checkpoints only -- data loading is always internal
# forecast_task=1 (visualization_gen.py:3549).  So all 15 leads read the SAME
# ten-day window D-10..D-1, and one process can run them after a single publish
# and a single data load.  That is why this sweep costs ~1.9 GPU-h per target
# for 15 leads rather than 15 x 12.6 min.
#
# Output naming is the VALID date, not the anchor: _prediction_output_date
# (visualization_gen.py:5441) adds `lead - 1` days in dense-operational mode,
# so target D at lead N writes {D+N-1}.tif.  Within one target the 15 leads are
# therefore distinct (D..D+14) -- but ACROSS targets they overlap completely,
# which is the point of the exercise and also the hazard: visualization_gen
# skips any lead whose valid-date file already exists (`:4997`), silently and
# with exit 0.  A flat output directory would drop most of the sweep.  Hence
# one directory per target; `index_predictions.py` builds the by-valid-date view.
set -euo pipefail

START=${START:?set START=YYYY-MM-DD}
END=${END:?set END=YYYY-MM-DD}
LEADS=${LEADS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15}
DEVICE=${DEVICE:-cuda:0}
ROOT=${ROOT:-/s3/quinn/prediction_vis/canfirecast_2026_multi}
MODEL_DIRS=${MODEL_DIRS:-/s3/zhengsen/pths/firempc_15days}
MODEL_FILTER=${MODEL_FILTER:-CanFireCastIFSCommon}
WORK=${WORK:-/caribou/quinn/wildfire/work/predict_sweep}
KEEP=${KEEP:-0}
REPO=${REPO:-/mnt/raid/zhengsen/aaai27_wildfire_experiment_results/canfirecast_15day_era5_ifs_20260818/frozen_repo}
MODEL_PYTHON=${MODEL_PYTHON:-/opt/anaconda3/envs/gsilab_shared/bin/python}
VIS_OVERLAY=${VIS_OVERLAY:-/caribou/quinn/wildfire/work/vis_overlay}
RELEASE=${RELEASE:-/caribou/quinn/wildfire/assets/canfirecast/operational-v1}
CLI=${CLI:-/home/quinn/FireMPC/operational/cli.py}

mkdir -p "$ROOT/by_target" "$WORK"
log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# Valid date for target $1 at lead $2 is $1 + ($2 - 1) days.
valid_date() { date -I -d "$1 + $(( $2 - 1 )) day"; }

day=$START
while [[ "$day" < "$END" || "$day" == "$END" ]]; do
  tdir="$ROOT/by_target/$day"
  mkdir -p "$tdir"

  # Resume at lead granularity: a target counts as done only when every lead's
  # valid-date raster exists.  A partially-built target is re-entered, and
  # visualization_gen's own probe skips the leads already on disk.
  missing=()
  for lead in $LEADS; do
    stamp=$(valid_date "$day" "$lead"); stamp=${stamp//-/_}
    [[ -f "$tdir/${stamp}.tif" ]] || missing+=("$lead")
  done
  if (( ${#missing[@]} == 0 )); then
    log "$day all $(wc -w <<< "$LEADS") leads present, skipping"
    day=$(date -I -d "$day + 1 day"); continue
  fi

  pkg="$WORK/$day"
  mkdir -p "$pkg"
  log "$day publishing window (leads missing: ${missing[*]})"
  python "$CLI" canfirecast store publish \
      --target "$day" \
      --static "$RELEASE/statics.h5" \
      --correction-artifact "$RELEASE/cffdrs_aware_correction.json" \
      --output "$pkg/2026_year_dataset.h5"

  log "$day running leads ${missing[*]} on $DEVICE"
  ( cd "$REPO" && PYTHONPATH="$VIS_OVERLAY${PYTHONPATH:+:$PYTHONPATH}" "$MODEL_PYTHON" visualization_gen.py \
      --h5-file "$pkg/2026_year_dataset.h5" \
      --model-dirs "$MODEL_DIRS" \
      --model-filter "$MODEL_FILTER" \
      --forecast-task ${missing[*]} \
      --output-dir "$tdir" \
      --device "$DEVICE" )

  # "No model files found" and "No checkpoints found for: [...]" are warnings
  # with exit 0, so a clean exit proves nothing.  Check every lead landed.
  still=()
  for lead in $LEADS; do
    stamp=$(valid_date "$day" "$lead"); stamp=${stamp//-/_}
    [[ -f "$tdir/${stamp}.tif" ]] || still+=("$lead")
  done
  if (( ${#still[@]} > 0 )); then
    log "$day FAILED: no output for lead(s) ${still[*]}"
    exit 1
  fi

  # Provenance beside the rasters: which window produced them, which weights,
  # and the lead -> valid-date mapping that the filenames only imply.
  {
    printf '{\n  "target": "%s",\n' "$day"
    printf '  "window": ["%s", "%s"],\n' \
      "$(date -I -d "$day - 10 day")" "$(date -I -d "$day - 1 day")"
    printf '  "d11_state": "%s",\n' "$(date -I -d "$day - 11 day")"
    printf '  "model": "%s",\n  "thresholds": [0.7, 0.75, 0.8],\n' "$MODEL_FILTER"
    printf '  "leads": {\n'
    sep=""
    for lead in $LEADS; do
      sha=$(cut -d' ' -f1 "$MODEL_DIRS/${MODEL_FILTER}_day$(printf '%02d' "$lead").pth.sha256" 2>/dev/null || echo "")
      printf '%s    "%02d": {"valid": "%s", "file": "%s.tif", "sha256": "%s"}' \
        "$sep" "$lead" "$(valid_date "$day" "$lead")" \
        "$(valid_date "$day" "$lead" | tr - _)" "$sha"
      sep=$',\n'
    done
    printf '\n  }\n}\n'
  } > "$tdir/manifest.json"

  [[ "$KEEP" == "1" ]] || rm -rf "$pkg"
  log "$day done"
  day=$(date -I -d "$day + 1 day")
done
log "multi-horizon sweep complete $START..$END"
