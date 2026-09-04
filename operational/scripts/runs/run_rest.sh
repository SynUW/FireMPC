set -e
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gee_down
cd /home/quinn/FireMPC
export TMPDIR=/caribou/quinn/wildfire/work/tmp
LOG=/caribou/quinn/wildfire/logs/predict

# Wait for the April sweep to release the GPU before starting.
while pgrep -f "[r]un_april.sh" >/dev/null; do sleep 60; done
echo "=== APRIL FINISHED, CHAIN STARTING $(date -u +%H:%M) ==="

# January starts at 01-11: targets 01-01..01-10 have no certified D-11 state
# (the ledger begins 2025-12-31), so they are not replayable at all.
for spec in "2026-03-01 2026-03-31 march" "2026-02-01 2026-02-28 february" "2026-01-11 2026-01-31 january"; do
  set -- $spec
  echo "=== STARTING $3 ($1..$2) $(date -u +%H:%M) ==="
  START=$1 END=$2 DEVICE=cuda:0 HORIZONS=1 \
  OUT=/s3/quinn/prediction_vis/canfirecast_${3}2026 \
    bash operational/scripts/predict_sweep.sh
  echo "=== $3 DONE $(date -u +%H:%M) ==="
done
echo "=== ALL MONTHS DONE ==="
