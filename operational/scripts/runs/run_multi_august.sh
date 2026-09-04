# Multi-horizon sweep, chained behind the June/July repair job.
# set -e only: conda's activate hooks reference unset vars, so set -u breaks them.
set -e
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gee_down
cd /home/quinn/FireMPC
export TMPDIR=/caribou/quinn/wildfire/work/tmp

# Wait for the repair sweep to release the GPU before starting.
while pgrep -f "[r]un_repair_predict.sh" >/dev/null; do sleep 120; done
echo "=== REPAIR FINISHED, MULTI-HORIZON STARTING $(date -u +%H:%M) ==="

START=2026-08-01 END=2026-08-07 DEVICE=cuda:0 \
  bash operational/scripts/predict_multi_sweep.sh

echo "=== SWEEP DONE, INDEXING $(date -u +%H:%M) ==="
python operational/scripts/index_predictions.py
echo "=== MULTI-HORIZON COMPLETE $(date -u +%H:%M) ==="
