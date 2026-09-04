"""Cap PyTorch's inter-op thread pool for the operational prediction sweeps.

OMP_NUM_THREADS caps only the intra-op pool; the inter-op pool is sized from
os.cpu_count() and has no environment variable.  It must be set before any
inter-op work starts, so it happens here -- sitecustomize is imported during
interpreter startup, and this directory is already on PYTHONPATH for the model
step (it carries the geopandas/pyogrio shim).  Left uncapped, one sweep claimed
64 runnable threads (32 intra + 32 inter) on a 64-core shared box while the
actual compute sat on the GPU.

Never fatal: a failure here must not take down a 15-hour sweep.
"""
import os

_n = int(os.environ.get("TORCH_NUM_INTEROP_THREADS", "8"))
try:
    import torch
    if torch.get_num_interop_threads() != _n:
        torch.set_num_interop_threads(_n)
except Exception:                      # torch absent (gee_down) or pool already started
    pass
