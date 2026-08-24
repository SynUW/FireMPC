"""
Download the MODIS drivers of the FireMPC dataset (labels excluded).

Product list and band selection follow Table `tab:data_summary`. The fire-label
product (MOD/MYD14A1) is deliberately NOT here.

  Fuel   MOD09GA / MYD09GA    b01,b02,b03,b07     500 m  daily
         (NDVI/EVI are derived from those bands - nothing extra to download)
  Fuel   MCD15A3H             Lai, Fpar           500 m  4-day
  Met.*  MOD11A1 / MYD11A1    LST day/night, Emis 1 km  daily
  Met.*  MOD09CMG / MYD09CMG  BT bands 20,21,31,32 0.05 deg daily
  Hum.   MCD12Q1              LC_Type1            500 m  annual

The 8-day and 16-day composites the historical GEE pull also carries (MOD09A1,
MYD09A1, MOD11A2, MYD11A2, MOD13Q1, MYD13Q1) are always included. MOD13Q1 alone
is ~254 MB/tile; use --products to select a smaller explicit subset if needed.

Provenance for every short name, provider, and band map is centralized in
modis_registry.py.

CADENCE
  Every product is fetched on its own native cadence - the same grid the
  historical GEE pull produced, so downstream consumers see composites built
  over identical windows. MODIS composites are anchored to day-of-year 1 and
  the grid restarts every January 1, which makes the last window of the year
  short (an 8-day product's DOY 361 window is 5 days, 6 in a leap year).

  Only cadence 1 products have to reach D. For a cadence of N days, one run in
  every N does any work: the other N-1 runs find the anchor already on disk.
  So the operational job is a single daily invocation - the cadence grid, not
  the schedule, decides what actually downloads.

  Composites publish a few days AFTER their window closes, so the anchor that
  belongs in a short range does not exist yet when that range is current. Each
  product reaches `lookback` days further back than --start for exactly that
  reason: a window that published late is picked up on a later run instead of
  being skipped forever. This is why a fresh run shows several composites to
  fetch and steady-state runs show none.

COMPLETENESS
  A day is not usable because *a* granule landed - it is usable when every tile
  the product publishes over Canada has landed. modis_domain measures that
  expected set from the latency store; this script checks it against what is
  actually ON DISK after downloading, because disk is what downstream stages read and
  a download can fail. Each date is classified:

    complete   every expected tile present
    partial    some expected tiles present, some missing
    absent     nothing published yet
    present    granules on disk but no measured expected set to gate against
               (run `modis_latency.py sweep` for the product to fix this)

  Nothing is forward-filled or substituted. An unreleased day is reported as
  absent and nothing is written for it; choosing what the model consumes for an
  absent day is a package-time decision made from the manifest.

  Granules are downloaded to a staging dir and size-checked against their UMM
  metadata before being moved into the archive, so a run killed mid-transfer
  cannot leave a truncated file that every later existence check accepts. Files
  already in the archive predate that guarantee and are trusted as-is.

TWO SOURCES, ONE DELIVERABLE
  Both backends end at the same place: the canonical per-tile driver GeoTIFFs
  under paths.DRIVERS, and a tiling report describing them.

    ea    CMR search -> verified HDF archive (paths.RAW)
          -> modis_to_tiles -> drivers
    gee   Earth Engine -> drivers, directly

  The earthaccess path additionally writes REPORTS/manifests/<run-utc>.json plus
  latest.json: the per-(product, date) granule completeness statement, keyed by
  CMR short name. It describes GRANULES, so only the path that fetches granules
  writes one, and modis_to_tiles refuses any manifest not written by it.

  Both paths write REPORTS/tiling/<run-utc>.json plus latest.json: the per-
  (export, date, tile) record of which driver tiles exist, keyed by export name.
  That is what packaging reads and the record of what an operational run used.

  Use --no-tile for an archive-only earthaccess run.

EXIT CODES
  0  ran; any absent dates are within the product's measured latency
  1  error - a download or a CMR search failed
  2  alert - a product is past the p95 of its own measured completion lag
  3  another instance holds the lock; nothing was done

Three things to know before running:
  * The `gee` backend bypasses Earthdata and writes the canonical
    per-tile GeoTIFF drivers directly. It uses ambient Earth Engine credentials
    and defaults to paths.DRIVERS instead of the HDF archive in paths.RAW.
    Its tiles are checked against the same lattice/dtype contract the
    earthaccess path enforces, minus extent: EE clips a download to the mosaic
    footprint, so extent is not comparable between the two (and does not need
    to be -- pre_processing_fast.py warps everything onto one template).
  * The window ALWAYS has absent dates at the tail and that is correct, not a
    failure: no product reaches D. MOD11A1 typically completes D+1 17:15,
    MOD09GA D+2 06:31. Only deviation from those percentiles is an alert.
  * MOD09CMG/MYD09CMG are GLOBAL files (~540 MB each, one per day per satellite).
    They dominate the download by an order of magnitude. Use --no-cmg to skip.
  * Daily products: NRT is one day fresher than standard but is a preliminary
    stream. Standard is the default and is proven
    bit-identical to the training tiles; --stream nrt keeps the other path
    available and stays traceable through the manifest's `tier` field.
"""

import calendar
import errno
import fcntl
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

import earthaccess
import typer

from wildfire.modis import domain as modis_domain
from wildfire.modis import to_tiles as modis_to_tiles
from wildfire import paths
from wildfire.modis.registry import PRODUCTS as PARITY_PRODUCTS
from wildfire.modis.registry import (
    STANDARD_PROVIDER,
    download_products,
)

BBOX = modis_domain.NATIONAL_BBOX
LANCE = "LANCEMODIS"

# Operational forecast dates follow Calgary civil time. Release timestamps and
# latency calculations remain timezone.utc because that is how CMR timestamps granules.
OPERATIONAL_TZ_NAME = "America/Edmonton"
OPERATIONAL_TZ = ZoneInfo(OPERATIONAL_TZ_NAME)

# Land cover for the current year does not publish until well into the next, so
# the archive always holds a stale year by design. Never an alert.
STALE_BY_DESIGN = {"MCD12Q1"}

# Granules land here first and are only moved into the archive once verified.
STAGING = ".staging"

app = typer.Typer(
    help=__doc__,
    add_completion=False,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
)

# Per-run state. Static product data stays in module-level registries; mutable
# execution data is passed through the pipeline explicitly.
@dataclass(slots=True)
class RunContext:
    """Mutable state for one invocation; passed explicitly to keep globals data-only."""
    search_errors: list[str] = field(default_factory=list)
    domain_tiles: set[str] | None = None


@dataclass(frozen=True, slots=True)
class CliOptions:
    """Validated command-line options passed through the download pipeline."""
    days: int
    start: str | None
    end: str | None
    end_offset: int
    dest: str
    products: str | None
    latest: bool
    stream: str
    no_cmg: bool
    dry_run: bool
    no_lock: bool
    refresh_domain: bool
    no_tile: bool = False
    source: Literal["earthaccess", "gee"] = "earthaccess"


@dataclass(frozen=True, slots=True)
class ProductPlan:
    """Pure planning result for one product before disk filtering."""
    picked: dict[date, list]
    tier: dict[date, str]
    pending: list[date]
    wanted: list[date]


@dataclass(frozen=True, slots=True)
class DiskPlan:
    """Executable plan plus the metadata needed to write the manifest."""
    cfg: dict
    need: dict[date, list]
    tier: dict[date, str]
    wanted: list[date]
    stats: dict | None

# Collection plans are derived from the one requested-export allowlist in
# modis_registry.py. Grouping prevents shared inputs such as MOD09GA from
# being downloaded twice for its reflectance and state-QA exports.
FIREMPC_MODIS = ALL_PRODUCTS = download_products()
CMG = {name for name, cfg in ALL_PRODUCTS.items() if cfg["global"]}


def domain_tiles(ctx):
    """Tile names intersecting the national grid, computed once per run."""
    if ctx.domain_tiles is None:
        ctx.domain_tiles = {modis_domain.tile_str(t)
                            for t in modis_domain.canada_tiles(BBOX)}
    return ctx.domain_tiles


def granule_date(name):
    for tok in name.split("."):
        if len(tok) == 8 and tok.startswith("A") and tok[1:].isdigit():
            return datetime.strptime(tok[1:], "%Y%j").replace(tzinfo=timezone.utc).date()
    return None


def on_grid(d, cadence, offset):
    """Does this observation date sit on the product's cadence grid?"""
    doy = d.timetuple().tm_yday
    if cadence >= 365:
        return doy == 1 + offset
    return (doy - 1 - offset) % cadence == 0


def cadence_anchors(cadence, offset, start, end):
    """
    Composite start dates on the product's native grid, within [start, end].

    The grid is DOY 1 + offset, stepping by `cadence`, and it restarts every
    January 1 - so the count per year is ceil(days_in_year / cadence) and the
    final window is short. Annual products get one anchor (Jan 1) per year.
    """
    out = []
    for year in range(start.year, end.year + 1):
        jan1 = date(year, 1, 1)
        n = 366 if calendar.isleap(year) else 365
        doys = [1 + offset] if cadence >= 365 else range(1 + offset, n + 1, cadence)
        for doy in doys:
            d = jan1 + timedelta(days=doy - 1)
            if start <= d <= end:
                out.append(d)
    return out


def search(ctx, short_name, start, end, provider=None, version=None, bbox=BBOX):
    kw = {
        "short_name": short_name,
        "temporal": (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")),
    }
    if bbox:
        kw["bounding_box"] = bbox
    if provider:
        kw["provider"] = provider
    if version:
        kw["version"] = version
    try:
        return earthaccess.search_data(**kw)
    except Exception as exc:  # noqa: BLE001 - earthaccess raises transport/provider-specific exceptions.
        msg = f"search failed {short_name} {start}..{end}: {exc}"
        ctx.search_errors.append(msg)
        print(f"    ! {msg}", file=sys.stderr)
        return []


def by_date(results, tiles=None):
    """
    Group granules by observation date, dropping tiles outside the domain.

    CMR returns any granule whose footprint touches the bbox, and at v00/v01 a
    sinusoidal tile is wide enough in longitude to graze Canada without covering
    a single grid pixel (h20v01, h23v02). Those are real granules of no use
    here, so they are filtered geometrically rather than by publication
    frequency - an intermittent tile that does cover Canada must still be kept.
    """
    out = {}
    for g in results:
        links = g.data_links()
        if not links:
            continue
        name = links[0].split("/")[-1]
        if tiles is not None and modis_domain.tile_of(name) not in tiles:
            continue
        d = granule_date(name)
        if d:
            out.setdefault(d, []).append(g)
    return out


def plan_product(ctx, cfg, start, end, stream):
    """
    Return the product's wanted cadence dates and published granules.

      picked   {date: [granules]}   wanted by the cadence grid and published
      tier     {date: 'NRT'|'STD'}
      pending  [date, ...]          wanted but nothing published yet
      wanted   [date, ...]          every cadence date the window asks for

    Dates always come from the product's own cadence, never from the caller's
    step size, so the local archive lands on the same grid as the GEE history.
    """
    bbox = None if cfg.get("global") else BBOX
    tiles = None if cfg.get("global") else domain_tiles(ctx)
    cadence = cfg["cadence_days"]

    if cadence == 1:
        wanted = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        picked, tier = {}, {}
        if stream != "standard" and cfg.get("nrt"):
            sn, prov = cfg["nrt"]
            picked = by_date(search(ctx, sn, start, end, provider=prov, bbox=bbox), tiles)
            tier = {d: "NRT" for d in picked}
        sn, ver = cfg["std"]
        for d, gs in by_date(search(ctx, sn, start, end, provider=STANDARD_PROVIDER,
                                    version=ver, bbox=bbox), tiles).items():
            if d not in picked:
                picked[d] = gs
                tier[d] = "STD"
        picked = {d: g for d, g in picked.items() if start <= d <= end}
    else:
        look = start - timedelta(days=cfg["lookback"])
        offset = cfg.get("cadence_offset", 0)
        # A composite belongs to the range when its WINDOW overlaps it, not when
        # its anchor falls inside it - the window covering `look` opened up to
        # one cadence earlier. CMR returns those overlapping granules too, so
        # dropping them would leave a permanent hole at the start of the range
        # (and for an annual product, drop the only granule there is).
        span = cadence_anchors(cadence, offset, look - timedelta(days=cadence), end)
        wanted = [d for d in span if d + timedelta(days=cadence - 1) >= look]
        sn, ver = cfg["std"]
        found = by_date(search(ctx, sn, look, end, provider=STANDARD_PROVIDER,
                               version=ver, bbox=bbox), tiles)
        off_grid = sorted(d for d in found if not on_grid(d, cadence, offset))
        if off_grid:
            print(f"    ! {sn}: {len(off_grid)} granule date(s) off the {cadence}-day "
                  f"grid ({', '.join(str(d) for d in off_grid[:3])}) - "
                  f"check cadence_days/cadence_offset", file=sys.stderr)
        picked = {d: found[d] for d in wanted if d in found}
        tier = {d: "STD" for d in picked}

    pending = [d for d in wanted if d not in picked]
    return ProductPlan(picked=picked, tier=tier, pending=pending, wanted=wanted)


# ---------------------------------------------------------------------------
# disk
# ---------------------------------------------------------------------------
def date_dir(dest, name, d):
    return os.path.join(dest, name, d.strftime("%Y-%m-%d"))


def on_disk(dest, name, d):
    """Granule filenames already in the archive for this product and date."""
    try:
        return {f for f in os.listdir(date_dir(dest, name, d)) if f.endswith(".hdf")}
    except FileNotFoundError:
        return set()


def granule_name(g):
    return g.data_links()[0].split("/")[-1]


def split_by_disk(dest, name, picked):
    """Split the plan into (granules already downloaded, {date: [granules to fetch]})."""
    have, need = 0, {}
    for d, granules in picked.items():
        present = on_disk(dest, name, d)
        todo = [g for g in granules if granule_name(g) not in present]
        have += len(granules) - len(todo)
        if todo:
            need[d] = todo
    return have, need


def expected_bytes(g, fname):
    """
    Size `fname` should be on disk, from the granule's UMM, or None if unstated.

    Only trusted when the metadata names this exact file or describes a single
    file - a DAAC that lists the .hdf alongside its .xml sidecar would otherwise
    give a total that no single file ever matches.
    """
    info = (g.get("umm", {}).get("DataGranule", {}).get("ArchiveAndDistributionInformation", []))
    named = [i for i in info if i.get("Name") == fname]
    if not named:
        if len(info) != 1:
            return None
        named = info
    entry = named[0]
    if "SizeInBytes" in entry:
        return int(entry["SizeInBytes"])
    if "Size" in entry:
        mult = {"BYTES": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}.get((entry.get("SizeUnit") or "MB").upper())
        if mult:
            return int(float(entry["Size"]) * mult)
    return None


def fetch_verified(granules, out, staging):
    """
    Download into a staging dir, size-check, then move into the archive.

    earthaccess writes straight to its destination, so a run killed mid-transfer
    leaves a truncated .hdf that satisfies every later existence check forever -
    and the completeness gate would then call that day complete. Nothing enters
    the archive namespace until it has been verified against the granule's
    stated size, so the archive holds whole files or nothing.

    Returns (moved_filenames, [(filename, reason), ...]).
    """
    # A previous kill can leave partial files in staging, and earthaccess would
    # skip them as already downloaded. Start every transfer from empty.
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging, exist_ok=True)
    os.makedirs(out, exist_ok=True)

    try:
        earthaccess.download(granules, staging)
    except Exception as exc:  # noqa: BLE001 - preserve failure details without aborting the run.
        shutil.rmtree(staging, ignore_errors=True)
        return [], [(f"<{len(granules)} granules>", f"download raised: {exc}")]

    moved, bad = [], []
    for g in granules:
        fname = granule_name(g)
        src = os.path.join(staging, fname)
        if not os.path.exists(src):
            bad.append((fname, "not written"))
            continue
        got = os.path.getsize(src)
        want = expected_bytes(g, fname)
        # A rounded MB in the metadata is worth ~0.1%; 1% still catches any
        # truncation big enough to matter.
        if got == 0:
            bad.append((fname, "empty file"))
        elif want and abs(got - want) > max(1024, 0.01 * want):
            bad.append((fname, f"size {got} != expected {want}"))
        else:
            os.replace(src, os.path.join(out, fname))
            moved.append(fname)

    shutil.rmtree(staging, ignore_errors=True)
    return moved, bad


# ---------------------------------------------------------------------------
# completeness
# ---------------------------------------------------------------------------
def classify(present_tiles, expected):
    """(status, sorted missing tiles) for one date, judged from disk."""
    if expected is None:
        return ("present" if present_tiles else "absent"), []
    if not present_tiles:
        return "absent", sorted(expected)
    missing = sorted(expected - present_tiles)
    return ("partial" if missing else "complete"), missing


def overdue(d, cadence, status, p95_h, now):
    """
    Is this date late by the product's OWN measured behaviour?

    A healthy run always reports the last day or two as absent, so absence is
    not the signal. Being past the p95 of the completion lag measured over the
    latency store is.
    """
    if status == "complete" or p95_h is None:
        return False
    end = modis_domain.window_end(d, cadence)
    age = (now - datetime(end.year, end.month, end.day, tzinfo=timezone.utc))
    return age.total_seconds() / 3600.0 > p95_h


def product_manifest(dest, name, cfg, wanted, tier, stats, now):
    """The per-date completeness block for one product, read off disk."""
    cadence = cfg["cadence_days"]
    expected = set(stats["expected"]) if stats else None
    p95 = stats["p95_h"] if stats else None

    entries, alerts = {}, []
    for d in wanted:
        tiles = {modis_domain.tile_of(f) for f in on_disk(dest, name, d)}
        status, missing = classify(tiles, expected)
        entry = {"status": status}
        if status != "absent":
            entry["tiles"] = len(tiles & expected) if expected else len(tiles)
            entry["tier"] = tier.get(d, "STD")
        if missing and status == "partial":
            entry["missing_tiles"] = missing
        if cadence > 1:
            entry["covers"] = [d.isoformat(), modis_domain.window_end(d, cadence).isoformat()]
        if name not in STALE_BY_DESIGN and overdue(d, cadence, status, p95, now):
            entry["overdue"] = True
            alerts.append((name, d, status))
        entries[d.isoformat()] = entry

    block = {
        "cadence_days": cadence,
        "expected_tiles": sorted(expected) if expected else None,
        "p95_complete_h": p95,
        ("dates" if cadence == 1 else "anchors"): entries,
    }
    if name in STALE_BY_DESIGN:
        block["stale_by_design"] = True
    if expected is None:
        block["gate"] = "unmeasured - run modis_latency.py sweep for this product"
    return block, alerts


def write_manifest(manifest, now):
    out = os.path.join(paths.REPORTS, "manifests")
    os.makedirs(out, exist_ok=True)
    body = json.dumps(manifest, indent=1, sort_keys=True)
    # Keep run history: it is how you answer "what did the forecast on
    # 2026-08-10 actually see?"
    for fname in (now.strftime("%Y-%m-%dT%H%M%SZ") + ".json", "latest.json"):
        path = os.path.join(out, fname)
        with open(path + ".tmp", "w") as f:
            f.write(body)
        os.replace(path + ".tmp", path)
    return os.path.join(out, "latest.json")


# ---------------------------------------------------------------------------
# single instance
# ---------------------------------------------------------------------------
def acquire_lock(dest):
    """
    Exclusive lock keyed on the archive root, so a slow run cannot overlap the
    next scheduled one. Returns the open handle, or None if held elsewhere.
    """
    os.makedirs(paths.WORK, exist_ok=True)
    key = dest.strip("/").replace("/", "_")
    handle = open(os.path.join(paths.WORK, f"modis_download.{key}.lock"), "w")  # noqa: SIM115 - lock handle must stay open.
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno not in (errno.EACCES, errno.EAGAIN):
            raise
        handle.close()
        return None
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def parse_window(args):
    """Operational date window as data, independent of download side effects."""
    operational_today = datetime.now(OPERATIONAL_TZ).date()
    end   = (date.fromisoformat(args.end)   if args.end   else operational_today - timedelta(days=args.end_offset))
    start = (date.fromisoformat(args.start) if args.start else end               - timedelta(days=args.days - 1))
    return start, end


def select_products(args):
    """Resolve CLI product options into an ordered product-name list."""
    names = list(ALL_PRODUCTS)
    if args.products:
        want = {p.strip() for p in args.products.split(",")}
        bad = want - set(ALL_PRODUCTS)
        if bad:
            sys.exit(f"unknown product(s): {', '.join(sorted(bad))}")
        names = [n for n in ALL_PRODUCTS if n in want]
    if args.no_cmg:
        names = [n for n in names if n not in CMG]
    return names


def latest_only(plan, cadence):
    """Reduce a composite plan to the newest published anchor."""
    if cadence <= 1 or not plan.picked:
        return plan
    newest = max(plan.picked)
    return ProductPlan(
        picked={newest: plan.picked[newest]},
        tier={newest: plan.tier[newest]},
        pending=plan.pending,
        wanted=[newest],
    )


def build_disk_plans(ctx, names, args, start, end):
    """Plan searches and downloads, then summarize what is already on disk."""
    total_need, total_mb, plans = 0, 0.0, {}

    print(f"{'product':10s} {'cad':>5s} {'exp':>4s} {'have':>5s} {'need':>5s} {'wait':>5s} {'~size':>8s}  {'next':10s}  tiers")
    print("-" * 78)
    for n in names:
        cfg = ALL_PRODUCTS[n]
        cadence = cfg["cadence_days"]
        plan = plan_product(ctx, cfg, start, end, args.stream)
        if args.latest:
            plan = latest_only(plan, cadence)
        have, need = split_by_disk(args.dest, n, plan.picked)
        stats = modis_domain.domain_stats(n, cadence, bbox=BBOX, refresh=args.refresh_domain)
        plans[n] = DiskPlan(cfg=cfg, need=need, tier=plan.tier, wanted=plan.wanted, stats=stats)

        ng = sum(len(v) for v in need.values())
        mb = ng * cfg["approx_mb"]
        total_need += ng
        total_mb += mb
        tiers = ",".join(sorted(set(plan.tier.values()))) or "-"
        nxt = str(plan.pending[0]) if plan.pending else "-"
        exp = str(len(stats["expected"])) if stats else "?"
        print(f"{n:10s} {str(cadence) + 'd':>5s} {exp:>4s} {have:>5d} {ng:>5d} {len(plan.pending):>5d} {mb / 1024:>7.2f}G  {nxt:10s}  {tiers}")

    return plans, total_need, total_mb


def fetch_disk_plans(dest, plans):
    """Execute the disk write portion of the plan and return transfer errors."""
    failures = []
    for n, plan in plans.items():
        if not plan.need:
            print(f"{n}: up to date")
            continue
        got = 0
        staging = os.path.join(dest, n, STAGING)
        for d, granules in sorted(plan.need.items()):
            moved, bad = fetch_verified(granules, date_dir(dest, n, d), staging)
            got += len(moved)
            for fname, why in bad:
                failures.append(f"{n} {d}: {fname}: {why}")
                print(f"    ! {n} {d} {fname}: {why}", file=sys.stderr)
        print(f"{n}: {got} granules -> {os.path.join(dest, n)}")
    return failures


def manifest_from_disk(ctx, plans, args, start, end, now, failures):
    """Build the deliverable manifest from disk state, not from planned downloads."""
    manifest = {
        "run_utc": now.isoformat().replace("+00:00", "Z"),
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "source": args.source,
        "stream": args.stream,
        "dest": args.dest,
        "bbox": list(BBOX),
        "errors": ctx.search_errors + failures,
        "products": {},
    }
    alerts = []
    for n, plan in plans.items():
        block, alert = product_manifest(args.dest, n, plan.cfg, plan.wanted, plan.tier, plan.stats, now)
        manifest["products"][n] = block
        alerts += alert

    counts = {}
    for block in manifest["products"].values():
        for entry in (block.get("dates") or block.get("anchors") or {}).values():
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    manifest["summary"] = {
        "status_counts": counts,
        "alerts": len(alerts),
        "errors": len(manifest["errors"])
    }
    return manifest, alerts, counts


def gee_exports(short_name):
    """Canonical driver exports produced by one MODIS collection."""
    return {
        name: PARITY_PRODUCTS[name]
        for name in ALL_PRODUCTS[short_name]["exports"]
    }


def gee_wanted_dates(cfg, start, end):
    """Cadence dates requested from GEE, including the composite lookback."""
    cadence = cfg["cadence_days"]
    if cadence == 1:
        return [start + timedelta(days=i) for i in range((end - start).days + 1)]
    look = start - timedelta(days=cfg["lookback"])
    offset = cfg.get("cadence_offset", 0)
    span = cadence_anchors(cadence, offset, look - timedelta(days=cadence), end)
    return [d for d in span if d + timedelta(days=cadence - 1) >= look]


def gee_available_dates(ee, collection, wanted, bbox):
    """Observation dates present in an Earth Engine ImageCollection."""
    if not wanted:
        return set()
    start, end = min(wanted), max(wanted) + timedelta(days=1)
    region = ee.Geometry.Rectangle(list(bbox), proj="EPSG:4326", geodesic=False)
    millis = (ee.ImageCollection(collection)
              .filterDate(start.isoformat(), end.isoformat())
              .filterBounds(region)
              .aggregate_array("system:time_start").getInfo())
    return {datetime.fromtimestamp(ms / 1000, timezone.utc).date() for ms in millis}


def gee_tile_geometries():
    """The historical four Canada export polygons, in their original order."""
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.read_file(paths.CANADA_GRIDS_SHP)
    if gdf.crs is None:
        raise RuntimeError(f"Canada grid shapefile has no CRS: {paths.CANADA_GRIDS_SHP}")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    region = gdf.geometry.union_all()
    minx, miny, maxx, maxy = region.bounds
    dx, dy = (maxx - minx) / 2, (maxy - miny) / 2
    pieces = []
    for i in range(2):
        for j in range(2):
            cell = box(minx + i * dx, miny + j * dy,
                       minx + (i + 1) * dx, miny + (j + 1) * dy)
            piece = cell.intersection(region)
            if not piece.is_empty and piece.area >= 0.02:
                pieces.append(piece)
    if len(pieces) != 4:
        raise RuntimeError(f"Canada grid produced {len(pieces)} tiles; expected 4")
    return pieces


def split_geometry(geometry, subdivisions):
    """Split a large export polygon into an N x N set of download regions."""
    from shapely.geometry import box

    left, bottom, right, top = geometry.bounds
    dx, dy = (right - left) / subdivisions, (top - bottom) / subdivisions
    for col in range(subdivisions):
        for row in range(subdivisions):
            cell = box(left + col * dx, bottom + row * dy,
                       left + (col + 1) * dx, bottom + (row + 1) * dy)
            piece = geometry.intersection(cell)
            if not piece.is_empty and piece.area >= 1e-6:
                yield piece


def gee_http_download(url, out):
    """Stream one signed GEE download URL to disk and reject non-TIFF replies."""
    import requests

    tmp = out + ".part"
    try:
        with requests.get(url, stream=True, timeout=(30, 600)) as response:
            response.raise_for_status()
            with open(tmp, "wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        with open(tmp, "rb") as handle:
            if handle.read(4) not in (b"II*\x00", b"MM\x00*"):
                raise RuntimeError("Earth Engine response is not a GeoTIFF")
        os.replace(tmp, out)
    finally:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass


def gee_download_band(ee, image, band, geometry, cfg, work, stem):
    """Download and merge one band, subdividing to stay under GEE limits."""
    import rasterio
    from rasterio.merge import merge
    from shapely.geometry import mapping

    subdivisions = cfg["export_subdivision"]
    pieces = []
    for index, piece in enumerate(split_geometry(geometry, subdivisions)):
        out = os.path.join(work, f"{stem}.{index:03d}.tif")
        request_image = image.select([band]).reproject(crs=cfg["output"]["crs"], scale=cfg["output"]["resolution_m"])
        url = request_image.getDownloadURL({
            "region": ee.Geometry(mapping(piece)),
            "crs": cfg["output"]["crs"],
            "scale": cfg["output"]["resolution_m"],
            "format": "GEO_TIFF",
        })
        gee_http_download(url, out)
        pieces.append(out)

    if not pieces:
        raise RuntimeError(f"empty export geometry for {stem}")
    merged_path = os.path.join(work, f"{stem}.merged.tif")
    sources = [rasterio.open(path) for path in pieces]
    try:
        mosaic, transform = merge(sources)
        profile = sources[0].profile.copy()
        profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform, count=1, compress="LZW")
        with rasterio.open(merged_path, "w", **profile) as dst:
            dst.write(mosaic[0], 1)
    finally:
        for source in sources:
            source.close()
    return merged_path


def gee_download_tile(ee, export_name, cfg, day, tile, geometry, out):
    """Download one historical-format, multi-band GEE driver tile atomically."""
    import rasterio

    collection = ee.ImageCollection(cfg["gee_collection"])
    image = collection.filterDate(day.isoformat(), (day + timedelta(days=1)).isoformat()).mosaic()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    work = tempfile.mkdtemp(prefix=".gee-", dir=os.path.dirname(out))
    try:
        band_paths = []
        subdivision = cfg["export_subdivision"]
        if isinstance(subdivision, list):
            subdivision = subdivision[tile]
        download_cfg = {**cfg, "export_subdivision": subdivision}
        for index, band_cfg in enumerate(cfg["bands"]):
            band_paths.append(gee_download_band(
                ee, image, band_cfg["gee"], geometry, download_cfg, work,
                f"band{index:02d}"))
        sources = [rasterio.open(path) for path in band_paths]
        try:
            profile = sources[0].profile.copy()
            # The training exporter applied band one's dtype to the whole stack.
            profile.update(count=len(sources), compress="LZW")
            staged = os.path.join(work, os.path.basename(out))
            with rasterio.open(staged, "w", **profile) as dst:
                for index, source in enumerate(sources, start=1):
                    # read([1])[0], not read(1): the scalar form mutates
                    # ndarray.shape internally and warns under NumPy 2.5. Same
                    # fix as modis_to_tiles.stack.
                    dst.write(source.read([1])[0].astype(profile["dtype"]), index)
        finally:
            for source in sources:
                source.close()
        # Validate before the file enters the archive namespace, exactly as
        # fetch_verified does for granules: a tile that fails the contract must
        # never exist at its final path, or the next run's existence check will
        # accept it forever.
        errors = modis_to_tiles.validate_raster(staged, export_name)
        if errors:
            raise RuntimeError("; ".join(errors))
        os.replace(staged, out)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def gee_output_path(dest, export_name, day, tile):
    stem = f"{export_name}_{day.strftime('%Y_%m_%d')}_tile{tile:02d}.tif"
    return os.path.join(dest, export_name, stem)


def run_gee(args, start, end, names):
    """Direct-to-GeoTIFF GEE backend; it never logs in to Earthdata."""
    if args.stream != "standard":
        print("the gee backend only supports --stream standard", file=sys.stderr)
        return 1
    try:
        import ee
        ee.Initialize(project='wildfire-505222')
    except Exception as exc:  # noqa: BLE001 - authentication/project errors vary by EE setup.
        print(f"Earth Engine initialization failed (run `earthengine authenticate`): {exc}",
              file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc)
    failures, plans = [], {}
    print(f"\nrange {start} .. {end}   source=gee   dest={args.dest}")
    for short_name in names:
        cfg = ALL_PRODUCTS[short_name]
        wanted = gee_wanted_dates(cfg, start, end)
        try:
            exports = gee_exports(short_name)
            available = gee_available_dates(
                ee, next(iter(exports.values()))["gee_collection"], wanted, BBOX)
        except Exception as exc:  # noqa: BLE001 - EE service errors become manifest errors.
            failures.append(f"{short_name}: search failed: {exc}")
            available = set()
            exports = gee_exports(short_name)
        published = [d for d in wanted if d in available]
        if args.latest and cfg["cadence_days"] > 1 and published:
            published = [max(published)]
            wanted = published
        plans[short_name] = (exports, wanted, set(published))
        need = sum(not os.path.exists(gee_output_path(args.dest, export, d, tile))
                   for export in exports for d in published for tile in range(4))
        print(f"{short_name:10s} published={len(published):3d} "
              f"waiting={len(wanted) - len(published):3d} tiles_needed={need:4d}")

    if args.dry_run:
        print("\n(dry run - nothing downloaded, no report written)")
        return 1 if failures else 0

    try:
        geometries = gee_tile_geometries()
    except Exception as exc:  # noqa: BLE001 - surface local geometry setup as a normal run error.
        print(f"GEE export grid failed: {exc}", file=sys.stderr)
        return 1

    results = []
    for short_name, (exports, wanted, published) in plans.items():
        for export_name, export_cfg in exports.items():
            for day in sorted(wanted):
                result = {
                    "raw_product": short_name,
                    "product": export_name,
                    "date": day.isoformat(),
                    "source_status": "published" if day in published else "waiting",
                    "tiles": {},
                }
                if export_cfg["cadence"]["days"] > 1:
                    result["covers"] = [day.isoformat(), modis_domain.window_end(
                        day, export_cfg["cadence"]["days"]).isoformat()]
                results.append(result)

                # Nothing is forward-filled: an unpublished date is recorded as
                # absent and no tile is written for it. On this path the report
                # is the only artifact, so that has to be stated here rather than
                # in a download manifest.
                if day not in published:
                    result["status"] = "absent"
                    continue

                for tile, geometry in enumerate(geometries):
                    out = gee_output_path(args.dest, export_name, day, tile)
                    if os.path.exists(out):
                        result["tiles"][str(tile)] = {"status": "existing", "path": out}
                        continue
                    try:
                        # gee_download_tile validates against the same
                        # lattice/dtype/truncation contract the earthaccess path
                        # enforces (minus extent: EE clips to the mosaic
                        # footprint) and raises rather than publishing a bad tile.
                        gee_download_tile(ee, export_name, export_cfg, day, tile,
                                          geometry, out)
                        result["tiles"][str(tile)] = {"status": "written", "path": out}
                        print(f"{export_name} {day} tile{tile:02d} -> {out}")
                    except Exception as exc:  # noqa: BLE001 - the report records partial dates.
                        msg = f"{export_name} {day} tile{tile:02d}: {exc}"
                        failures.append(msg)
                        result["tiles"][str(tile)] = {"status": "failed", "error": str(exc)}
                        print(f"    ! {msg}", file=sys.stderr)

                states = {t["status"] for t in result["tiles"].values()}
                result["status"] = "failed" if "failed" in states else "complete"

    counts = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    report = {
        "run_utc": now.isoformat().replace("+00:00", "Z"),
        "source": "gee",
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "stream": "standard",
        "bbox": list(BBOX),
        "out": args.dest,
        "errors": failures,
        "results": results,
        "summary": {"jobs": len(results), "failures": len(failures),
                    "status_counts": counts},
    }
    print(f"\nreport -> {modis_to_tiles.write_tiling_report(report)}")
    print("  " + "   ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 1 if failures else 0


def run(args: CliOptions) -> int:
    """Execute one download run and return its documented process exit code."""
    ctx = RunContext()
    start, end = parse_window(args)
    names = select_products(args)

    lock = None
    if not (args.no_lock or args.dry_run):
        lock = acquire_lock(args.dest)
        if lock is None:
            print(f"another run already holds the lock on {args.dest} - nothing done", file=sys.stderr)
            return 3

    if args.source == "gee":
        return run_gee(args, start, end, names)

    # Must resolve from ~/.netrc without prompting: under cron there is no one
    # to answer. Verify with `modis_download.py ea --dry-run < /dev/null`.
    try:
        earthaccess.login(persist=True)
    except Exception as exc:  # noqa: BLE001 - login failures should become exit code 1.
        print(f"earthaccess login failed (needs a working ~/.netrc): {exc}",
              file=sys.stderr)
        return 1

    now = datetime.now(timezone.utc)
    print(f"\nrange {start} .. {end}   date_basis={OPERATIONAL_TZ_NAME}   stream={args.stream}   dest={args.dest}")
    print(f"domain {len(modis_domain.canada_tiles(BBOX))} tiles over {BBOX}")
    print(f"products: {', '.join(names)}\n")

    plans, total_need, total_mb = build_disk_plans(ctx, names, args, start, end)

    print("-" * 78)
    print(f"{'TOTAL':10s} {'':>5s} {'':>4s} {'':>5s} {total_need:>5d} {'':>5s} "
          f"{total_mb / 1024:>7.2f}G")
    print("\nexp = tiles the product publishes over the domain ('?' = unmeasured)\n"
          "have = granules already on disk   need = to download   "
          "wait = cadence dates not yet published\nnext = earliest date still waiting "
          "(for a composite, when its next window is due)")

    if args.dry_run:
        print("\n(dry run - nothing downloaded, no manifest written)")
        return 0

    print()
    failures = fetch_disk_plans(args.dest, plans)

    # The manifest describes DISK, not the plan: a download can fail, and disk
    # is what packaging reads.
    manifest, alerts, counts = manifest_from_disk(ctx, plans, args, start, end, now, failures)

    manifest_path = write_manifest(manifest, now)
    print(f"\nmanifest -> {manifest_path}")
    print("  " + "   ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    # Convert to the driver tiles preprocessing actually reads, so one invocation
    # ends in the same place the gee backend does. Only dates the manifest marks
    # complete are tiled, and existing valid tiles are left alone, so this is
    # incremental in exactly the way the download above is.
    tile_rc = 0
    if args.no_tile:
        print("\n(--no-tile: granules only, no driver tiles built)")
    elif manifest["errors"]:
        # load_manifest refuses a manifest with errors, and rightly: tiling a
        # partial download would bake the gap into the drivers.
        print("\n(skipping tiling: the download manifest records errors)", file=sys.stderr)
    else:
        print()
        tile_rc = modis_to_tiles.tile_manifest(manifest_path, raw=args.dest, out=paths.DRIVERS, stream=args.stream, correct_dtype=False)

    for n, d, status in alerts:
        print(f"  ALERT {n} {d} {status}: past the p95 of its measured completion lag", file=sys.stderr)
    if manifest["errors"] or tile_rc:
        return 1
    return 2 if alerts else 0


@app.command(help=__doc__)
def cli(
    backend: Annotated[
        Literal["ea", "gee"],
        typer.Argument(
            help="Required download backend: ea (Earthaccess) or gee (Earth Engine).",
        ),
    ],
    days: Annotated[int, typer.Option(help="Days back from today in Calgary time.")] = 10,
    start: Annotated[str | None, typer.Option(help="YYYY-MM-DD (overrides --days).")] = None,
    end: Annotated[str | None, typer.Option(help="YYYY-MM-DD (default: today in Calgary time).")] = None,
    end_offset: Annotated[
        int,
        typer.Option(
            metavar="N",
            help=(
                "End the window N days before today in Calgary time. Operational "
                "packaging uses 1 (window ends D-1, since no gridded product ever "
                "reaches D). Ignored if --end is given."
            ),
        ),
    ] = 0,
    dest: Annotated[
        str | None,
        typer.Option(
            help=(
                f"Output root (Earthaccess HDF default: {paths.RAW}; "
                f"GEE GeoTIFF default: {paths.DRIVERS})."
            ),
        ),
    ] = None,
    products: Annotated[str | None, typer.Option(help="Comma-separated subset.")] = None,
    latest: Annotated[
        bool,
        typer.Option(
            "--latest",
            help=(
                "For non-daily products, keep only the newest published composite "
                "instead of every anchor in range. Inference-only; this does not "
                "reproduce the historical cadence."
            )
        ),
    ] = False,
    stream: Annotated[
        Literal["nrt", "standard"],
        typer.Option(
            help=(
                "standard = bit-identical to GEE; nrt = one day fresher but "
                "preliminary, with no NRT for MOD/MYD11A1."
            )
        ),
    ] = "standard",
    no_cmg: Annotated[
        bool,
        typer.Option("--no-cmg", help="Skip the ~540MB/day global CMG files."),
    ] = False,
    no_tile: Annotated[
        bool,
        typer.Option(
            "--no-tile",
            help=(
                "earthaccess only: stop after the HDF archive instead of also "
                "building the driver GeoTIFFs. Ignored with the gee backend, which "
                "downloads tiles directly."
            ),
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show the plan and size; download nothing."),
    ] = False,
    no_lock: Annotated[
        bool,
        typer.Option(
            "--no-lock",
            help="Skip the single-instance lock (for a manual run alongside cron).",
        ),
    ] = False,
    refresh_domain: Annotated[
        bool,
        typer.Option(
            "--refresh-domain",
            help="Recompute expected tile sets from the latency store instead of the cache.",
        ),
    ] = False,
) -> None:
    """Download the MODIS drivers of the FireMPC dataset (labels excluded)."""
    source: Literal["earthaccess", "gee"] = "earthaccess" if backend == "ea" else "gee"
    if dest is None:
        dest = paths.DRIVERS if source == "gee" else paths.RAW
    args = CliOptions(
        days=days,
        start=start,
        end=end,
        end_offset=end_offset,
        dest=dest,
        products=products,
        latest=latest,
        stream=stream,
        no_cmg=no_cmg,
        dry_run=dry_run,
        no_lock=no_lock,
        refresh_domain=refresh_domain,
        no_tile=no_tile,
        source=source,
    )
    raise typer.Exit(run(args))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
