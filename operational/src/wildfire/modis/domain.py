"""
The Canada domain: which sinusoidal tiles a product must publish before a day
is usable, and how long that normally takes.

Both modis_latency.py (which measures) and modis_download.py (which enforces)
need the same answer to "what does a complete day look like?", so it lives here
rather than in either of them.

TWO TILE SETS, AND WHY THEY DIFFER
  The GEOMETRIC set is every sinusoidal tile whose footprint intersects the
  national grid - 31 tiles. The EXPECTED set is the subset a given product
  actually publishes. Ocean-only tiles are never produced, so waiting on the
  geometric set would leave every day permanently incomplete:

      MOD09GA / MYD09GA    30 tiles   (h07v04 never published)
      MOD11A1 / MYD11A1    29 tiles   (h07v04, h15v00 never published)
      MOD09CMG / MYD09CMG   1 GLOBAL granule

  The expected set is measured from the latency store (modis_latency sweep),
  not hard-coded, because it changes if the national grid does. Tiles sit at
  either ~100% or exactly 0% of days, so EXPECTED_FRAC has a wide margin.

CACHING
  Deriving the set means re-reading ~1.2 MB of JSONL per product. The answer
  only moves when the store grows, so it is cached in REPORTS/domain_cache.json
  keyed on the store file's (mtime, size) and the bbox.
"""

import json
import os
import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import numpy as np

from wildfire import paths

# The national grid's own footprint (paths.GRID_TEMPLATE), so the tile set is
# derived from the extent the model actually predicts over.
NATIONAL_BBOX = (-141.00855014824128, 41.66386287746341,
                 -52.61432619088036, 83.34569206060921)

# MODIS sinusoidal grid constants (36 x 18 tiles of 10 degrees at the equator).
SIN_R = 6371007.181
TILE_M = 2 * np.pi * SIN_R / 36.0
X0 = 18 * TILE_M
Y0 = 9 * TILE_M

TILE_RE = re.compile(r"h\d\dv\d\d")
A_DATE_RE = re.compile(r"\.A(\d{7})\.")

STORE = os.path.join(paths.REPORTS, "latency")
CACHE = os.path.join(paths.REPORTS, "domain_cache.json")

# A tile is treated as "expected" only if the product publishes it on at least
# this fraction of the days in the window. Below that it is an ocean tile or a
# sporadic edge granule, not something a schedule should ever wait on.
EXPECTED_FRAC = 0.8

# Global products have no tile grid: one granule covers the domain.
GLOBAL_TILE = "GLOBAL"


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def canada_tiles(bbox=NATIONAL_BBOX, step=0.05):
    """Sinusoidal (h, v) tiles whose footprint intersects the national grid."""
    lon = np.arange(bbox[0], bbox[2] + step, step)
    lat = np.arange(bbox[1], bbox[3] + step, step)
    lon, lat = np.meshgrid(lon, lat)
    x = SIN_R * np.radians(lon) * np.cos(np.radians(lat))
    y = SIN_R * np.radians(lat)
    h = np.floor((x + X0) / TILE_M).astype(int)
    v = np.floor((Y0 - y) / TILE_M).astype(int)
    return {(int(a), int(b)) for a, b in zip(h.ravel(), v.ravel())}


def tile_str(hv):
    return f"h{hv[0]:02d}v{hv[1]:02d}"


def tile_of(name):
    """Tile in a granule filename, or GLOBAL for an untiled product."""
    m = TILE_RE.search(name)
    return m.group(0) if m else GLOBAL_TILE


def window_end(a_date, cadence):
    """Last observation day the granule anchored at a_date covers."""
    if cadence == 1:
        return a_date
    if cadence >= 365:
        return date(a_date.year, 12, 31)
    return a_date + timedelta(days=cadence - 1)


# ---------------------------------------------------------------------------
# the latency store
# ---------------------------------------------------------------------------
def store_path(product, stream):
    return os.path.join(STORE, f"{product}_{stream}.jsonl")


def load_store(product, stream):
    try:
        with open(store_path(product, stream)) as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def parse_dt(s):
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def lag_hours(stamp, ref_date):
    """Hours between 00:00 UTC on ref_date and `stamp` (a datetime or ISO str)."""
    dt = stamp if isinstance(stamp, datetime) else parse_dt(stamp)
    if dt is None:
        return None
    ref = datetime(ref_date.year, ref_date.month, ref_date.day, tzinfo=timezone.utc)
    return (dt - ref).total_seconds() / 3600.0


def first_sightings(rows, start=None, end=None):
    """
    {a_date: {tile: (insert_dt, production_dt)}} keeping the EARLIEST insert.

    A granule can be re-swept or revised; the first sighting is the moment the
    tile actually became usable, which is what a schedule is written from.
    """
    per_day = defaultdict(dict)
    for r in rows:
        a = date.fromisoformat(r["a_date"])
        if (start and a < start) or (end and a > end):
            continue
        ins = parse_dt(r["insert"]) or parse_dt(r["revision_date"])
        if ins is None:
            continue
        prev = per_day[a].get(r["tile"])
        if prev is None or ins < prev[0]:
            per_day[a][r["tile"]] = (ins, parse_dt(r["production"]))
    return per_day


def classify_tiles(per_day, geo_tiles):
    """
    Split the tiles seen in `per_day` into expected / intermittent / outside.

    outside: returned by CMR but not intersecting the national grid. At v00/v01
    a tile's polygon is wide enough in longitude to graze Canada without
    covering any of it; waiting on those would inflate completion times for no
    coverage.
    """
    n_days = len(per_day)
    freq = defaultdict(int)
    for tiles in per_day.values():
        for tile in tiles:
            freq[tile] += 1

    frequent = {t for t, n in freq.items() if n >= EXPECTED_FRAC * n_days}
    geo_str = {tile_str(t) for t in geo_tiles} | {GLOBAL_TILE}
    return {
        "expected": frequent & geo_str,
        "outside": sorted(frequent - geo_str),
        "intermittent": {t: n for t, n in freq.items()
                         if t not in frequent and t in geo_str},
        "freq": dict(freq),
        "n_days": n_days,
    }


# ---------------------------------------------------------------------------
# the cached per-product answer
# ---------------------------------------------------------------------------
def _store_stamp(product, stream):
    try:
        st = os.stat(store_path(product, stream))
    except FileNotFoundError:
        return None
    return [int(st.st_mtime), st.st_size]


def _load_cache():
    try:
        with open(CACHE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    tmp = CACHE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=1, sort_keys=True)
    os.replace(tmp, CACHE)


def _measure(product, stream, cadence, bbox):
    rows = load_store(product, stream)
    if not rows:
        return None
    per_day = first_sightings(rows)
    if not per_day:
        return None

    cls = classify_tiles(per_day, canada_tiles(bbox))
    expected = cls["expected"]
    if not expected:
        return None

    # Completion lag: when the LAST expected tile of a day landed, measured
    # from the end of the window that day's granule covers. This is what a
    # stall is judged against - a product past its p95 has deviated from its
    # own measured behaviour, which absent-at-the-tail alone does not show.
    lags = []
    for a, day in per_day.items():
        have = [day[t][0] for t in expected if t in day]
        if len(have) == len(expected):
            lags.append(lag_hours(max(have), window_end(a, cadence)))

    return {
        "expected": sorted(expected),
        "n_days": cls["n_days"],
        "n_complete": len(lags),
        "p50_h": float(np.percentile(lags, 50)) if lags else None,
        "p95_h": float(np.percentile(lags, 95)) if lags else None,
        "max_h": max(lags) if lags else None,
    }


def domain_stats(product, cadence, stream="standard", bbox=NATIONAL_BBOX,
                 refresh=False):
    """
    Measured domain facts for one product, cached against the store's mtime.

    Returns None when the store holds nothing for this product - the caller
    must then fall back, because an unmeasured product has no expected set and
    a completeness gate cannot be asserted over it.
    """
    stamp = _store_stamp(product, stream)
    if stamp is None:
        return None

    key = f"{product}/{stream}"
    cache = _load_cache()
    hit = cache.get(key)
    if (not refresh and hit and hit.get("stamp") == stamp
            and hit.get("bbox") == list(bbox) and hit.get("cadence") == cadence):
        return hit["stats"]

    stats = _measure(product, stream, cadence, bbox)
    if stats is None:
        return None
    cache[key] = {"stamp": stamp, "bbox": list(bbox), "cadence": cadence,
                  "stats": stats}
    _save_cache(cache)
    return stats
