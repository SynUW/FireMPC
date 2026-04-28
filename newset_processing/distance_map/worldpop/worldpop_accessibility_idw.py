#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WorldPop -> "human influence / accessibility" continuous field (gravity-style weighted sum).

In **projected coordinates (typically meters)**, population pixel centers are treated as source points; for each target pixel, the inverse-distance power-law weights of the K nearest sources are summed (gravity-style IDW):

  w_ij = 1 / (d_ij + eps)^alpha
  A(p) = sum_j w_ij * pop_j          <- weighted sum, NOT divided by sum(w)

[Note] This is a weighted sum, not a weighted mean:
- A weighted mean (divided by Sum(w)) over sparse regions (northern areas) makes K-neighbor radii very large, so the denominator is small, producing artificially high values;
  in dense regions (southern areas), the K-neighbor radius is small but includes many zero-population pixels, dragging the mean down.
  The result is counter-intuitive: north > southern cities.
- A weighted sum (without dividing): nearby high-population sources contribute more, distant ones contribute less; this is physically equivalent to a gravitational potential and correctly reflects "urban pull / accessibility" -- city cores are highest, wilderness lowest.

- **Projected raster**: pixel centers are converted to planar meters via the transform.
- **Geographic raster (lon/lat)**: pass `--assume-m-per-pixel [M]` to treat rows/cols as planar distances on an M x M meter grid (omitting M defaults to 1000, the common "~1 km" product).

**Common K-NN IDW artifacts (northern triangle / polygon patches)**  
When sources are sparse, each pixel's "K nearest sources" set changes abruptly across space; the field approximates a source-cluster Voronoi partition with sharp polygon boundaries. **Mitigations**: (1) increase `--k-neighbors`; (2) slightly lower `--alpha`; (3) **`--source-pool-m 10000` (aggregate sources to 10 km)** is strongly recommended for very sparse regions like Canada; it compresses millions of sources into tens of thousands so that K covers more towns.  
**Default parameters** lean conservative (K=1000, alpha=1.0, eps about 1 km pixel scale); `--source-pool-m` defaults to 0 to avoid silently changing semantics.

**Stripe and memory**: each stripe allocates approximately `stripe_rows x width x K x 16` bytes for the query buffer (distances + indices). `--stripe-memory-budget-gb` (default 2) auto-shrinks stripes to avoid OOM with large K; set to **0** to disable this cap.

Output: single-band float32, same size and transform as input; filename {stem}_access_idw.tif

Dependencies: rasterio, numpy, scipy; tqdm optional
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import xy
from scipy.spatial import cKDTree

from worldpop_norm import list_tifs, write_output

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]


def _limit_blas_threads() -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")


def parse_args():
    ap = argparse.ArgumentParser(
        description="WorldPop: KD-tree + power-law IDW gravity accessibility field (weighted sum, not weighted mean)"
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        default="/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/worldpop_v2_clipped/clipped",
        help="Population GeoTIFF directory",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default="/s3/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers/processed/worldpop_v2_clipped/access_idw",
        help="Output directory",
    )
    ap.add_argument("--pattern", default="*.tif", help="glob, relative to input-dir")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--band", type=int, default=1)
    ap.add_argument(
        "--k-neighbors",
        type=int,
        default=1000,
        help="Number of nearest neighbors K; increase further in source-sparse regions like Canada; even more effective with --source-pool-m",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Distance power: weight ~ 1/(d+eps)^alpha; smaller alpha gives distant sources relatively more weight and softer boundaries",
    )
    ap.add_argument(
        "--eps-m",
        type=float,
        default=500.0,
        help="Distance regularization (meters), prevents division by zero; 500-700 is common at ~1 km pixel scale; too small is numerically unstable, too large flattens urban peaks",
    )
    ap.add_argument(
        "--min-pop",
        type=float,
        default=0.0,
        help="Only pixels with pop > this threshold are treated as population sources (reduces KDT size)",
    )
    ap.add_argument(
        "--stripe-rows",
        type=int,
        default=128,
        help="Horizontal stripe rows (requested); may be auto-shrunk by --stripe-memory-budget-gb; can be set to 64 manually when K is very large",
    )
    ap.add_argument(
        "--stripe-memory-budget-gb",
        type=float,
        default=2.0,
        help="Approximate memory cap (GB) for a single stripe's query result (distances + indices); 0 disables shrinking and uses --stripe-rows directly",
    )
    ap.add_argument(
        "--kdtree-workers",
        type=int,
        default=-1,
        help="workers passed to cKDTree.query (-1 uses SciPy's default parallelism)",
    )
    ap.add_argument(
        "--assume-m-per-pixel",
        type=float,
        nargs="?",
        const=1000.0,
        default=None,
        metavar="M",
        help="For geographic CRSs: assume pixels are M x M meters and use planar row/col Euclidean distance; if the option is given without M, defaults to 1000",
    )
    ap.add_argument(
        "--source-pool-m",
        type=float,
        default=0.0,
        metavar="D",
        help="> 0: merge sources on a D x D meter grid (population summed). 10000 (10 km) is strongly recommended for source-sparse regions like Canada; default 0 keeps per-pixel source semantics",
    )
    ap.add_argument(
        "--compress",
        default="ZSTD",
        choices=["NONE", "LZW", "DEFLATE", "ZSTD"],
    )
    ap.add_argument("--blocksize", type=int, default=512)
    ap.add_argument("--meta-name", default="worldpop_access_idw_meta.json")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="Number of files to process in parallel")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    if args.k_neighbors < 1:
        ap.error("--k-neighbors must be >= 1")
    if args.alpha <= 0:
        ap.error("--alpha must be positive")
    if args.eps_m <= 0:
        ap.error("--eps-m must be positive")
    if args.stripe_rows < 1:
        ap.error("--stripe-rows must be >= 1")
    if args.jobs < 1:
        ap.error("-j must be >= 1")
    if args.assume_m_per_pixel is not None and args.assume_m_per_pixel <= 0:
        ap.error("--assume-m-per-pixel must be positive")
    if args.source_pool_m < 0:
        ap.error("--source-pool-m cannot be negative")
    if args.stripe_memory_budget_gb < 0:
        ap.error("--stripe-memory-budget-gb cannot be negative")
    return args


def _pop_sources_tuple(
    pop: np.ndarray,
    nodata: float | None,
    min_pop: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    x = pop.astype(np.float64, copy=True)
    if nodata is not None:
        x[np.isclose(x, nodata, equal_nan=True)] = np.nan
    x[~np.isfinite(x)] = np.nan
    valid = np.isfinite(x) & (x > min_pop)
    sy, sx = np.nonzero(valid)
    vals = x[valid].astype(np.float64)
    meta = {
        "source_pixels": int(sy.size),
        "min_pop": float(min_pop),
    }
    return x, sy, sx, vals, meta


def _pool_sources_meters(
    pts: np.ndarray, pop_v: np.ndarray, pool_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Merge sources at (x, y) meter coordinates into D x D grid cells: sum population, place source at the cell center."""
    pts = np.asarray(pts, dtype=np.float64)
    pop_v = np.asarray(pop_v, dtype=np.float64).ravel()
    if pts.shape[0] != pop_v.shape[0]:
        raise ValueError("pts and pop have inconsistent lengths")
    xm, ym = pts[:, 0], pts[:, 1]
    xmin, ymin = float(xm.min()), float(ym.min())
    ix = np.floor((xm - xmin) / pool_m).astype(np.int64)
    iy = np.floor((ym - ymin) / pool_m).astype(np.int64)
    ij = np.column_stack([ix, iy])
    u, inv = np.unique(ij, axis=0, return_inverse=True)
    pooled_pop = np.bincount(inv, weights=pop_v).astype(np.float64)
    xc = xmin + (u[:, 0].astype(np.float64) + 0.5) * pool_m
    yc = ymin + (u[:, 1].astype(np.float64) + 0.5) * pool_m
    return np.column_stack([xc, yc]), pooled_pop


def _cap_stripe_rows(
    stripe_rows: int, width: int, k_use: int, budget_gb: float
) -> tuple[int, dict[str, Any]]:
    """
    Cap the stripe height so that the dists/idx scratch arrays do not exceed ~budget_gb.
    Estimate ~K * (8 + 8) bytes per target point (float64 distances + int64 indices).
    """
    if budget_gb <= 0:
        return stripe_rows, {
            "stripe_rows_effective": stripe_rows,
            "stripe_cap_applied": False,
        }
    budget_bytes = int(budget_gb * (1024**3))
    bytes_per_target = max(k_use * 16, 1)
    denom = max(width * bytes_per_target, 1)
    max_stripe = max(1, budget_bytes // denom)
    eff = min(stripe_rows, max_stripe)
    eff = max(1, eff)
    return eff, {
        "stripe_rows_effective": eff,
        "stripe_cap_applied": eff < stripe_rows,
    }


def accessibility_idw_raster(
    pop: np.ndarray,
    transform: Any,
    *,
    nodata: float | None,
    min_pop: float,
    k_neighbors: int,
    alpha: float,
    eps_m: float,
    stripe_rows: int,
    kdtree_workers: int,
    show_stripe_progress: bool,
    assume_m_per_pixel: float | None = None,
    source_pool_m: float = 0.0,
    stripe_memory_budget_gb: float = 2.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Whole-image float32 accessibility field, same shape as pop."""
    _, sy, sx, pop_src, meta = _pop_sources_tuple(pop, nodata, min_pop)
    H, W = pop.shape
    out = np.full((H, W), np.nan, dtype=np.float32)

    pool_req = float(source_pool_m) if source_pool_m and source_pool_m > 0 else 0.0
    meta.update(
        {
            "k_neighbors_requested": int(k_neighbors),
            "alpha": float(alpha),
            "eps_m": float(eps_m),
            "stripe_rows_requested": int(stripe_rows),
            "stripe_memory_budget_gb": float(stripe_memory_budget_gb),
            "source_pool_m_requested": float(source_pool_m),
            "source_pool_m_effective": pool_req,
        }
    )

    if sy.size == 0:
        meta["note"] = "no_sources"
        meta["k_neighbors_used"] = 0
        return out, meta

    mpp = assume_m_per_pixel
    if mpp is not None and mpp > 0:
        meta["distance_mode"] = "planar_assumed_square_pixels_m"
        meta["m_per_pixel_assumed"] = float(mpp)
        # Pixel center: (col+0.5, row+0.5) * mpp; Euclidean distance matches what we'd get from integer col/row grid points
        xs_s = (sx.astype(np.float64) + 0.5) * mpp
        ys_s = (sy.astype(np.float64) + 0.5) * mpp
        tree_pts = np.column_stack([xs_s, ys_s])
    else:
        meta["distance_mode"] = "projected_xy_meters"
        xs_s, ys_s = xy(transform, sy, sx, offset="center")
        tree_pts = np.column_stack([np.asarray(xs_s, dtype=np.float64), np.asarray(ys_s, dtype=np.float64)])

    if source_pool_m and source_pool_m > 0:
        tree_pts, pop_src = _pool_sources_meters(tree_pts, pop_src, float(source_pool_m))
        meta["sources_after_pool"] = int(tree_pts.shape[0])

    n_src = int(tree_pts.shape[0])
    k_use = int(min(k_neighbors, n_src))
    meta["k_neighbors_used"] = k_use

    stripe_eff, stripe_cap_meta = _cap_stripe_rows(
        stripe_rows, W, k_use, stripe_memory_budget_gb
    )
    meta.update(stripe_cap_meta)

    tree = cKDTree(tree_pts)

    def _query(tgt_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if kdtree_workers == 0:
            return tree.query(tgt_pts, k=k_use)
        try:
            return tree.query(tgt_pts, k=k_use, workers=kdtree_workers)
        except TypeError:
            return tree.query(tgt_pts, k=k_use)

    n_stripes = (H + stripe_eff - 1) // stripe_eff
    pbar = None
    if show_stripe_progress and n_stripes > 1 and tqdm is not None:
        pbar = tqdm(total=n_stripes, desc="IDW stripes", unit="stripe", leave=False)

    for r0 in range(0, H, stripe_eff):
        r1 = min(r0 + stripe_eff, H)
        h = r1 - r0
        rr = np.arange(r0, r1, dtype=np.float64)
        cc = np.arange(W, dtype=np.float64)
        grid_r, grid_c = np.meshgrid(rr, cc, indexing="ij")
        if mpp is not None and mpp > 0:
            xs_t = (grid_c.ravel() + 0.5) * mpp
            ys_t = (grid_r.ravel() + 0.5) * mpp
            tgt = np.column_stack([xs_t, ys_t])
        else:
            xs_t, ys_t = xy(transform, grid_r.ravel(), grid_c.ravel(), offset="center")
            tgt = np.column_stack(
                [np.asarray(xs_t, dtype=np.float64), np.asarray(ys_t, dtype=np.float64)]
            )

        dists, idx = _query(tgt)
        if k_use == 1:
            dists = dists.reshape(-1, 1)
            idx = idx.reshape(-1, 1)

        w = 1.0 / np.power(dists + eps_m, alpha)
        pv = pop_src[idx]
        # Gravity-style weighted sum: A(p) = sum[ pop_j / (d_ij + eps)^alpha ]
        # Do not divide by Sum(w); this avoids artificial highs in sparse regions (north) due to small denominators, and dense regions being pulled down by zero-population neighbors
        stripe = np.sum(w * pv, axis=1).astype(np.float32).reshape(h, W)
        out[r0:r1, :] = stripe

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    return out, meta


def process_one_file(
    fp: Path,
    dst: Path,
    band: int,
    k_neighbors: int,
    alpha: float,
    eps_m: float,
    min_pop: float,
    stripe_rows: int,
    kdtree_workers: int,
    compress: str,
    blocksize: int,
    show_stripe_progress: bool,
    assume_m_per_pixel: float | None,
    source_pool_m: float,
    stripe_memory_budget_gb: float,
) -> dict[str, Any]:
    with rasterio.open(fp) as src:
        if band < 1 or band > src.count:
            raise ValueError(f"band {band} out-of-range (count={src.count})")
        pop = src.read(band)
        nod = src.nodata
        prof = src.profile
        tf = src.transform
        crs = src.crs

    use_planar: float | None = None
    if crs is not None and crs.is_geographic:
        if assume_m_per_pixel is None or assume_m_per_pixel <= 0:
            raise ValueError(
                "Input is in a geographic CRS (degrees). Use a meter-projected raster, or pass --assume-m-per-pixel [M] "
                "to assume M x M meter pixels (default 1000 if the option is given without M)."
            )
        use_planar = float(assume_m_per_pixel)

    acc, meta = accessibility_idw_raster(
        pop,
        tf,
        nodata=nod,
        min_pop=min_pop,
        k_neighbors=k_neighbors,
        alpha=alpha,
        eps_m=eps_m,
        stripe_rows=stripe_rows,
        kdtree_workers=kdtree_workers,
        show_stripe_progress=show_stripe_progress,
        assume_m_per_pixel=use_planar,
        source_pool_m=float(source_pool_m),
        stripe_memory_budget_gb=float(stripe_memory_budget_gb),
    )
    meta["input"] = str(fp)
    meta["output"] = str(dst)

    write_output(dst, acc, prof, compress, blocksize)
    return meta


def _worker(task: dict) -> tuple[str, dict]:
    if task.get("_blas", False):
        _limit_blas_threads()
    key = task["key"]
    fp = Path(task["fp"])
    dst = Path(task["dst"])
    try:
        m = process_one_file(
            fp,
            dst,
            band=int(task["band"]),
            k_neighbors=int(task["k_neighbors"]),
            alpha=float(task["alpha"]),
            eps_m=float(task["eps_m"]),
            min_pop=float(task["min_pop"]),
            stripe_rows=int(task["stripe_rows"]),
            kdtree_workers=int(task["kdtree_workers"]),
            compress=str(task["compress"]),
            blocksize=int(task["blocksize"]),
            show_stripe_progress=bool(task.get("show_stripe_progress", False)),
            assume_m_per_pixel=task.get("assume_m_per_pixel"),
            source_pool_m=float(task.get("source_pool_m", 0.0)),
            stripe_memory_budget_gb=float(task.get("stripe_memory_budget_gb", 2.0)),
        )
        return key, {"ok": True, "meta": m}
    except Exception as e:
        return key, {"ok": False, "error": str(e)}


def main():
    args = parse_args()
    in_dir = args.input_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_tifs(in_dir, args.pattern, args.recursive)
    if not files:
        print(f"No match found: {in_dir} / {args.pattern}")
        return

    show_p = not args.no_progress
    stripe_prog = show_p and args.jobs == 1

    tasks: list[dict] = []
    for fp in files:
        rel = fp.relative_to(in_dir) if fp.is_relative_to(in_dir) else Path(fp.name)
        stem = fp.stem
        dst = out_dir / rel.parent / f"{stem}_access_idw.tif"
        if args.skip_existing and dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        kd_w = 1 if args.jobs > 1 else args.kdtree_workers
        tasks.append(
            {
                "key": str(rel.as_posix()),
                "fp": str(fp.resolve()),
                "dst": str(dst.resolve()),
                "band": args.band,
                "k_neighbors": args.k_neighbors,
                "alpha": args.alpha,
                "eps_m": args.eps_m,
                "min_pop": args.min_pop,
                "stripe_rows": args.stripe_rows,
                "kdtree_workers": kd_w,
                "compress": args.compress,
                "blocksize": args.blocksize,
                "show_stripe_progress": stripe_prog,
                "_blas": args.jobs > 1,
                "assume_m_per_pixel": args.assume_m_per_pixel,
                "source_pool_m": args.source_pool_m,
                "stripe_memory_budget_gb": args.stripe_memory_budget_gb,
            }
        )

    if not tasks:
        print("No files to process (or all already exist; see --skip-existing)")
        return

    all_meta: dict[str, Any] = {}

    if args.jobs == 1:
        it = tasks
        if show_p and tqdm is not None:
            it = tqdm(tasks, desc="WorldPop IDW accessibility", unit="file")
        for t in it:
            k, res = _worker(t)
            all_meta[k] = res
            if not res.get("ok"):
                print(f"[FAIL] {k}: {res.get('error')}")
    else:
        _limit_blas_threads()
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(_worker, t) for t in tasks]
            itf = as_completed(futs)
            if show_p and tqdm is not None:
                itf = tqdm(itf, total=len(futs), desc="WorldPop IDW accessibility", unit="file")
            for fut in itf:
                k, res = fut.result()
                all_meta[k] = res
                if not res.get("ok"):
                    print(f"[FAIL] {k}: {res.get('error')}")

    meta_path = out_dir / args.meta_name
    meta_path.write_text(json.dumps(all_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    n_ok = sum(1 for v in all_meta.values() if v.get("ok"))
    print(f"Done {n_ok}/{len(all_meta)}, metadata: {meta_path}")


if __name__ == "__main__":
    main()
