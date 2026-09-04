# Multi-horizon sweep, resumed with capped CPU threads.
# set -e only: conda's activate hooks reference unset vars (trap 6).
set -e
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gee_down
cd /home/quinn/FireMPC
export TMPDIR=/caribou/quinn/wildfire/work/tmp

# The model step is GPU-bound (99% GPU), but torch sized both CPU thread pools
# from os.cpu_count()=64, giving 32 intra + 32 inter = 64 spinning threads and
# 75% of all runnable threads on a shared 64-core box.  OMP_NUM_THREADS caps
# intra-op; the inter-op pool has no env var and is capped by the
# sitecustomize.py in vis_overlay (already on PYTHONPATH for the model step).
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TORCH_NUM_INTEROP_THREADS=8

# Resume: targets already at 15/15 are skipped, and a partial target re-runs
# only its missing leads.
START=2026-08-01 END=2026-08-07 DEVICE=cuda:0 \
  bash operational/scripts/predict_multi_sweep.sh

echo "=== SWEEP DONE, INDEXING $(date -u +%H:%M) ==="
python operational/scripts/index_predictions.py
echo "=== MULTI-HORIZON COMPLETE $(date -u +%H:%M) ==="
