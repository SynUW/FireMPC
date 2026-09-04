set -e
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gee_down
cd /home/quinn/FireMPC
export TMPDIR=/caribou/quinn/wildfire/work/tmp
# Wait for the March/Feb/Jan chain to release the GPU.
while pgrep -f "[r]un_rest.sh" >/dev/null; do sleep 120; done
echo "=== CHAIN FINISHED, REPREDICTING 06-23..07-10 $(date -u +%H:%M) ==="
# Resume logic rebuilds only the deleted targets; the rest of each month is skipped.
START=2026-06-23 END=2026-06-30 DEVICE=cuda:0 HORIZONS=1 \
OUT=/caribou/quinn/wildfire/predictions/canfirecast_test_june \
  bash operational/scripts/predict_sweep.sh
START=2026-07-01 END=2026-07-10 DEVICE=cuda:0 HORIZONS=1 \
OUT=/s3/quinn/prediction_vis/canfirecast_july2026 \
  bash operational/scripts/predict_sweep.sh
echo "=== REPAIR REPREDICTION DONE ==="
