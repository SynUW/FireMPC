#!/usr/bin/env bash
# Backfill the monthly store one target per day, the way June 22 - July 31 were built.
#
# Fidelity: MODIS is frozen at first write, so the target that first touches a
# date fixes how much hindsight that date carries. Operationally, date X is
# frozen by target X+1 (--as-of X+1). Stepping more than one day would hand the
# window's oldest dates extra days of granule arrivals and make them
# systematically fresher than July's -- a different experiment, not a faster one.
#
# Stops at `assemble`: that is the stage that ingests into the store. `validate`
# would re-read the national window (~4 min/target, doubling the sweep) and
# `publish` would overwrite the operational daily.h5 with a historical window.
# Every date is validated later anyway, when predict_sweep publishes it.
set -euo pipefail

START=${START:-2026-01-11}
END=${END:-2026-08-28}
RELEASE=${RELEASE:-operational-v1}
KEEP_WORK=${KEEP_WORK:-0}
CLI=${CLI:-/home/quinn/FireMPC/operational/cli.py}
WORK_ROOT=${WORK_ROOT:-/caribou/quinn/wildfire/processed/model_weather}
MODIS_PKG=${MODIS_PKG:-/caribou/quinn/wildfire/packages/modis}
MIN_FREE_GB=${MIN_FREE_GB:-40}

log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# Target D's only genuinely-new date is D-1; if that is already frozen the whole
# target is redundant, which is how the sweep steps over June 22 - July 31.
already_frozen() {
  python - "$1" <<'PY'
import sys
from datetime import date, timedelta
from pathlib import Path
sys.path.insert(0, "/home/quinn/FireMPC/operational")
from canfirecast import store
import paths
d = date.fromisoformat(sys.argv[1]) - timedelta(days=1)
sys.exit(0 if store.is_written(Path(paths.CANFIRECAST_PACKAGES), d) else 1)
PY
}

day=$START
while [[ "$day" < "$END" || "$day" == "$END" ]]; do
  free_gb=$(df -BG --output=avail /caribou | tail -1 | tr -dc '0-9')
  if (( free_gb < MIN_FREE_GB )); then
    log "ABORT: only ${free_gb}G free on /caribou (floor ${MIN_FREE_GB}G)"; exit 1
  fi

  if already_frozen "$day"; then
    log "$day  $(date -I -d "$day - 1 day") already frozen, skipping"
    day=$(date -I -d "$day + 1 day"); continue
  fi

  log "$day  ingesting (as-of $day, ${free_gb}G free)"
  # Three steps rather than `canfirecast pipeline`, which always runs MODIS
  # download+preprocess over a 45-day lookback (modis_pipeline_command never
  # passes --from-stage).  For a backfill those are pure waste: processed/modis
  # /scaled already covers 2026-01-01..08-27, and `package` reads from there.
  python "$CLI" canfirecast pipeline \
      --target "$day" --release "$RELEASE" --to-stage cffdrs-derive
  # A stale handoff from an earlier operational run makes build_package refuse
  # (package.py:323).  Drop it so every target is rebuilt the same way, under
  # this sweep's --as-of, rather than silently reusing one built another way.
  rm -rf "${MODIS_PKG:?}/$day"
  python "$CLI" modis pipeline \
      --target "$day" --as-of "$day" \
      --modis-policy latest-available --from-stage package
  python "$CLI" canfirecast store ingest --target "$day"
  # `canfirecast pipeline` drops this at its assemble stage; calling store
  # ingest directly skips that, so the handoff has to be reclaimed here or it
  # accumulates at ~123 MB per target.
  rm -rf "${MODIS_PKG:?}/$day"

  [[ "$KEEP_WORK" == "1" ]] || rm -rf "${WORK_ROOT:?}/$day"
  log "$day  done"
  day=$(date -I -d "$day + 1 day")
done
log "ingest sweep complete $START..$END"
