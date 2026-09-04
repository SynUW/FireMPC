# Sweep wrappers, as run

The exact invocations that produced the 2026 CanFireCast predictions. They ran
from `/caribou/quinn/wildfire/logs/predict/` (a log directory, hence the copies
here) under `nohup`, each chained to the previous by a `pgrep` wait loop.

| wrapper | produced | finished (UTC) |
|---|---|---|
| `run_rest.sh` | day01 March, February, January | 2026-09-03 03:46 |
| `run_repair_predict.sh` | day01 re-prediction of the 06-23…07-10 seam (§4) | 2026-09-03 06:58 |
| `run_multi_august.sh` | multi-horizon 08-01…08-07, uncapped threads | superseded |
| `run_multi_august2.sh` | same sweep resumed with the §3.6 thread caps | 2026-09-03 22:38 |

`run_multi_august2.sh` is the one to copy for a new multi-horizon run. It is
`run_multi_august.sh` plus `OMP_NUM_THREADS` / `MKL_NUM_THREADS` /
`TORCH_NUM_INTEROP_THREADS`; the sweep resumed from 08-06 at lead 14 without
recomputing anything already on disk.

`../vis_overlay_sitecustomize.py` is deployed at
`/caribou/quinn/wildfire/work/vis_overlay/sitecustomize.py`, on `PYTHONPATH` for
the model step. It caps torch's inter-op pool, which no environment variable
reaches. Kept here because that directory is not under source control either.

Note `set -e` and never `set -u`: conda's activate hooks reference unset
variables (§3.7).
