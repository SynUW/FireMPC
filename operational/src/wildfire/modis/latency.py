"""
Measure how long after an observation date MODIS granules actually become
available, over the whole Canada domain.

An availability query answers "what exists right now". That is a snapshot, and
an operational schedule cannot be set from a snapshot - it needs the
distribution, including the bad days. This script answers:

    for target day D, at what wall-clock time is every granule the model needs
    actually searchable in CMR, and how often is that late?

Three subcommands:

  status  query CMR now and show the newest complete observation currently
          downloadable for each product.
  sweep   query CMR over a date range and record, per granule, when it was
          inserted into CMR. Appends to a JSONL store under paths.REPORTS.
  report  read the store and print the lag distributions.

LAG CONVENTION
  Every lag is reported in hours after 00:00 UTC on the granule's A-date, so
  "51.5h" reads directly as "available at D+2 03:30 UTC" - the number a cron
  schedule is written from. Composites also get a lag measured from the END of
  their compositing window, which is the earliest moment the granule could
  possibly have been produced.

  Availability = ProviderDates[Insert], i.e. when CMR could first return it.
  ProductionDateTime (when the DAAC's PGE wrote the file) is recorded too and
  is always earlier; the gap between them is DAAC ingest, not production.

WHY THIS IS RETROSPECTIVE FOR STANDARD BUT NOT FOR NRT
  LP DAAC keeps its archive, and the insert timestamp is preserved, so one
  pass over the last N days reconstructs the standard-stream history exactly.
  LANCE expires NRT granules after ~7 days, so NRT lag can only be accumulated
  going forward - run `sweep --stream nrt` on a schedule and let the store grow.

DOMAIN COMPLETENESS
  A day is not usable because *a* granule landed; it is usable when every tile
  covering Canada has landed. The expected tile set is derived geometrically
  from the national grid (see canada_tiles) and then intersected with what the
  product actually publishes, because ocean-only tiles are never produced.
"""

import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import earthaccess
import numpy as np

# earthaccess <1.0 emits this while constructing every search result. It is an
# internal migration notice, not actionable by this caller.
warnings.filterwarnings("ignore", message="As of version 1.0.*",
                        category=FutureWarning, module=r"earthaccess\.results")

from wildfire.modis.domain import (A_DATE_RE, NATIONAL_BBOX, STORE, TILE_RE,
                          canada_tiles, classify_tiles, domain_stats,
                          first_sightings, lag_hours, load_store, store_path,
                          tile_str, window_end)
from wildfire.modis.download import ALL_PRODUCTS, CMG, FIREMPC_MODIS, cadence_anchors, on_grid


# ---------------------------------------------------------------------------
# granule -> row
# ---------------------------------------------------------------------------
def _provider_date(umm, kind):
    for d in umm.get("ProviderDates", []):
        if d.get("Type") == kind:
            return d.get("Date")
    return None


def granule_row(g, product, stream):
    """Flatten one CMR granule into the fields the lag report needs."""
    meta, umm = g.get("meta", {}), g.get("umm", {})

    # LP DAAC sets native-id to the filename, but LANCE sets it to an opaque
    # "LANCEMODIS:<n>" - the A-date and tile only exist in ProducerGranuleId.
    candidates = [ident.get("Identifier", "")
                  for ident in umm.get("DataGranule", {}).get("Identifiers", [])
                  if ident.get("IdentifierType") == "ProducerGranuleId"]
    candidates += [meta.get("native-id") or "", umm.get("GranuleUR") or ""]
    candidates += [l.split("/")[-1] for l in (g.data_links() or [])]

    native, a_date = "", None
    for cand in candidates:
        m = A_DATE_RE.search(cand)
        if m:
            native = cand
            a_date = datetime.strptime(m.group(1), "%Y%j").date()
            break
    if a_date is None:
        return None

    m = TILE_RE.search(native)
    tile = m.group(0) if m else "GLOBAL"

    return {
        "product": product,
        "stream": stream,
        "native_id": native,
        "concept_id": meta.get("concept-id"),
        "revision_id": meta.get("revision-id"),
        "provider": meta.get("provider-id"),
        "tile": tile,
        "a_date": a_date.isoformat(),
        "insert": _provider_date(umm, "Insert"),
        "update": _provider_date(umm, "Update"),
        "delete": _provider_date(umm, "Delete"),  # LANCE expiry; absent for LP DAAC
        "revision_date": meta.get("revision-date"),
        "production": umm.get("DataGranule", {}).get("ProductionDateTime"),
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
def chunk_query(product, cfg, stream, start, end, bbox):
    """One CMR query for a date slice. Returns (rows, error_or_None)."""
    if stream == "nrt":
        if not cfg.get("nrt"):
            return [], None
        sn, prov = cfg["nrt"]
        kw = dict(short_name=sn, provider=prov)
    else:
        sn, ver = cfg["std"]
        kw = dict(short_name=sn, version=ver)

    kw["temporal"] = (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if bbox:
        kw["bounding_box"] = bbox
    try:
        results = earthaccess.search_data(**kw)
    except Exception as exc:
        return [], f"{product}/{stream} {start}..{end}: {exc}"

    rows = [granule_row(g, product, stream) for g in results]
    return [r for r in rows if r], None


def sweep(products, stream, start, end, jobs, chunk_days):
    os.makedirs(STORE, exist_ok=True)
    slices = []
    d = start
    while d <= end:
        hi = min(d + timedelta(days=chunk_days - 1), end)
        slices.append((d, hi))
        d = hi + timedelta(days=1)

    for product in products:
        cfg = ALL_PRODUCTS[product]
        if stream == "nrt" and not cfg.get("nrt"):
            print(f"{product:10s} nrt      - no NRT collection, skipped")
            continue
        bbox = None if cfg.get("global") else NATIONAL_BBOX

        rows, errors = [], []
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            futs = [ex.submit(chunk_query, product, cfg, stream, lo, hi, bbox)
                    for lo, hi in slices]
            for fut in futs:
                got, err = fut.result()
                rows.extend(got)
                if err:
                    errors.append(err)

        # Dedup against what is already stored. A granule can be re-queried on
        # a later sweep; keep the first sighting so the insert time we report
        # is the original one, not a revision.
        existing = load_store(product, stream)
        seen = {(r["concept_id"], r["revision_id"]) for r in existing}
        new = [r for r in rows if (r["concept_id"], r["revision_id"]) not in seen]

        with open(store_path(product, stream), "a") as f:
            for r in new:
                f.write(json.dumps(r) + "\n")

        print(f"{product:10s} {stream:8s} {len(rows):6d} found  {len(new):6d} new  "
              f"{len(existing) + len(new):6d} total")
        for e in errors:
            print(f"    ! {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# live status
# ---------------------------------------------------------------------------
def status(products, stream, days):
    """Query CMR now and print the newest complete downloadable date."""
    today = datetime.now(timezone.utc).date()
    geo = canada_tiles()
    print(f"\nMODIS live availability   checked {datetime.now(timezone.utc).isoformat()}")
    print(f"  stream={stream}   domain={NATIONAL_BBOX}")
    print(f"{'product':10s} {'stream':8s} {'cad':>5s} {'latest A-date':>13s} "
          f"{'covers thru':>12s} {'age':>6s}  newest publication")

    # Products are independent CMR searches. Running them concurrently keeps
    # an all-product status check interactive instead of waiting serially for
    # every tiled collection.
    searchable = [p for p in products if stream != "nrt" or ALL_PRODUCTS[p].get("nrt")]
    print(f"  querying {len(searchable)} CMR collections ...", flush=True)
    queries = {}
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(products)))) as ex:
        for product in products:
            cfg = ALL_PRODUCTS[product]
            if stream == "nrt" and not cfg.get("nrt"):
                continue
            # Long-cadence products need a longer metadata lookback than daily
            # products. This remains a lightweight CMR query, not a download.
            lookback = max(days, cfg.get("lookback", 0) + cfg["cadence_days"])
            start = today - timedelta(days=lookback - 1)
            future = ex.submit(chunk_query, product, cfg, stream, start, today,
                               None if cfg.get("global") else NATIONAL_BBOX)
            queries[product] = (start, future)

        live = {product: (start, *future.result())
                for product, (start, future) in queries.items()}

    for product in products:
        cfg = ALL_PRODUCTS[product]
        cadence = f"{cfg['cadence_days']}d"
        if product not in live:
            print(f"{product:10s} {stream:8s} {cadence:>5s} {'-':>13s} {'-':>12s} "
                  f"{'-':>6s}  no NRT collection")
            continue
        start, rows, error = live[product]
        if error:
            print(f"{product:10s} {stream:8s} {cadence:>5s} {'-':>13s} {'-':>12s} "
                  f"{'-':>6s}  ERROR: {error}")
            continue

        per_day = first_sightings(rows, start, today)
        if not per_day:
            print(f"{product:10s} {stream:8s} {cadence:>5s} {'-':>13s} {'-':>12s} "
                  f"{'-':>6s}  no granules found")
            continue

        # Prefer the long-run measured tile set. NRT and standard publish the
        # same spatial domain, so standard is a valid fallback when the short
        # NRT-retention store has not accumulated enough history yet.
        stats = (domain_stats(product, cfg["cadence_days"], stream=stream)
                 or domain_stats(product, cfg["cadence_days"], stream="standard"))
        measured = stats is not None
        expected = (set(stats["expected"]) if stats else
                    classify_tiles(per_day, geo)["expected"])

        complete = [a for a, tiles in per_day.items()
                    if expected and expected <= set(tiles)]
        newest = max(per_day)
        have = len(set(per_day[newest]) & expected)
        newest_state = ("complete" if newest in complete else
                        f"partial {have}/{len(expected)}")
        gate = "" if measured else " (recent-window tile gate)"

        if not complete:
            print(f"{product:10s} {stream:8s} {cadence:>5s} {'-':>13s} {'-':>12s} "
                  f"{'-':>6s}  {newest} {newest_state}{gate}")
            continue

        latest = max(complete)
        covers = window_end(latest, cfg["cadence_days"])
        age = (today - covers).days
        age_text = f"{age}d"
        print(f"{product:10s} {stream:8s} {cadence:>5s} {str(latest):>13s} "
              f"{str(covers):>12s} {age_text:>6s}  {newest} {newest_state}{gate}")

    print("\n  cad           = native product cadence in days")
    print("  latest A-date = newest observation/composite that has every expected Canada tile")
    print("  covers thru   = last observation day represented by that product")
    print("  age           = days between covers-thru and today UTC")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def pct(values, q):
    return float(np.percentile(values, q)) if values else float("nan")


def fmt_lag(h):
    """Hours-after-D-00:00Z as 'D+n HH:MM'."""
    if h is None or (isinstance(h, float) and np.isnan(h)):
        return "    -   "
    days, rem = divmod(h, 24)
    return f"D+{int(days)} {int(rem):02d}:{int((rem % 1) * 60):02d}"


def analyse(product, stream, rows, start, end, geo_tiles):
    """Per-tile first-availability, expected tile set, per-day completion."""
    cfg = ALL_PRODUCTS[product]

    per_day = first_sightings(rows, start, end)
    if not per_day:
        return None
    a_dates = sorted(per_day)

    # Which tiles does this product actually publish over the domain?
    cls = classify_tiles(per_day, geo_tiles)
    expected, outside, intermittent = (cls["expected"], cls["outside"],
                                       cls["intermittent"])

    ins_lags, prod_lags, complete, incomplete = [], [], [], []
    for a in a_dates:
        ref = window_end(a, cfg["cadence_days"])
        day = per_day[a]
        for tile, (ins, prod) in day.items():
            ins_lags.append(lag_hours(ins.isoformat(), ref))
            if prod:
                prod_lags.append(lag_hours(prod.isoformat(), ref))
        missing = expected - set(day)
        have = [day[t][0] for t in expected if t in day]
        if missing or not have:
            incomplete.append((a, sorted(missing)))
        else:
            complete.append((a, lag_hours(max(have).isoformat(), ref)))

    # Cadence dates with no granule at all in the window.
    if cfg["cadence_days"] == 1:
        wanted = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    else:
        wanted = [d for d in cadence_anchors(cfg["cadence_days"],
                                             cfg.get("cadence_offset", 0), start, end)
                  if on_grid(d, cfg["cadence_days"], cfg.get("cadence_offset", 0))]
    # A day before the first granule we ever saw is not a gap in the product -
    # for NRT it is simply outside LANCE's ~7-day retention, and for any stream
    # it may predate the first sweep.
    absent = [d for d in wanted if d not in per_day and d >= a_dates[0]]
    pre_window = sum(1 for d in wanted if d < a_dates[0])

    return {
        "n_days": len(a_dates), "first_day": a_dates[0], "last_day": a_dates[-1],
        "expected": expected, "intermittent": intermittent, "outside": outside,
        "pre_window": pre_window,
        "tiles_per_day": [len(per_day[a]) for a in a_dates],
        "ins_lags": ins_lags, "prod_lags": prod_lags,
        "complete": complete, "incomplete": incomplete, "absent": absent,
        "wanted": wanted,
    }


def print_report(results, geo_tiles, start, end):
    print(f"\nMODIS availability latency   window {start} .. {end}")
    print(f"  domain   {NATIONAL_BBOX}")
    print(f"  {len(geo_tiles)} sinusoidal tiles intersect the national grid: "
          f"{', '.join(sorted(tile_str(t) for t in geo_tiles))}")
    print("  lags are hours after 00:00 UTC on the A-date (composites: on the "
          "last day of\n  the compositing window), printed as D+days HH:MM")

    print("\n=== PER-GRANULE INSERT LAG (when one tile became searchable) ===")
    print(f"{'product':10s} {'strm':4s} {'days':>5s} {'tiles/d':>8s} "
          f"{'p50':>9s} {'p90':>9s} {'p99':>9s} {'max':>9s}   {'prod p50':>9s}")
    for (product, stream), a in results.items():
        if a is None:
            print(f"{product:10s} {stream:4s}  no granules in window")
            continue
        tpd = int(np.median(a["tiles_per_day"]))
        print(f"{product:10s} {stream:4s} {a['n_days']:>5d} {tpd:>8d} "
              f"{fmt_lag(pct(a['ins_lags'], 50)):>9s} {fmt_lag(pct(a['ins_lags'], 90)):>9s} "
              f"{fmt_lag(pct(a['ins_lags'], 99)):>9s} {fmt_lag(max(a['ins_lags'])):>9s}   "
              f"{fmt_lag(pct(a['prod_lags'], 50)):>9s}")

    print("\n=== DOMAIN COMPLETION (when the LAST expected tile landed) ===")
    print(f"{'product':10s} {'strm':4s} {'exp':>4s} {'ok':>4s} {'part':>5s} {'none':>5s} "
          f"{'p50':>9s} {'p90':>9s} {'p95':>9s} {'max':>9s}")
    for (product, stream), a in results.items():
        if a is None:
            continue
        lags = [l for _, l in a["complete"]]
        print(f"{product:10s} {stream:4s} {len(a['expected']):>4d} {len(a['complete']):>4d} "
              f"{len(a['incomplete']):>5d} {len(a['absent']):>5d} "
              f"{fmt_lag(pct(lags, 50)):>9s} {fmt_lag(pct(lags, 90)):>9s} "
              f"{fmt_lag(pct(lags, 95)):>9s} {fmt_lag(max(lags) if lags else None):>9s}")
    print("\n  exp  = tiles the product publishes over the domain on >=80% of days")
    print("  ok   = days where every expected tile landed (the p50/p90/max are over these)")
    print("  part = days missing at least one expected tile   none = cadence dates with nothing")

    for (product, stream), a in results.items():
        if a is None:
            continue
        notes = []
        geo_str = {tile_str(t) for t in geo_tiles}
        if "GLOBAL" not in a["expected"]:
            never = sorted(geo_str - set(a["expected"]) - set(a["intermittent"]))
            if never:
                notes.append(f"never published (ocean/edge): {', '.join(never)}")
        if a["outside"]:
            notes.append("returned by CMR but outside the national grid, "
                         f"not waited on: {', '.join(a['outside'])}")
        if a["pre_window"]:
            notes.append(f"{a['pre_window']} day(s) before {a['first_day']} hold no "
                         "granules and are not counted (NRT retention / pre-sweep)")
        if a["intermittent"]:
            notes.append("intermittent: " + ", ".join(
                f"{t} ({n}/{a['n_days']}d)" for t, n in sorted(a["intermittent"].items())))
        if a["incomplete"]:
            shown = "; ".join(f"{d} missing {','.join(m) if m else 'all'}"
                              for d, m in a["incomplete"][:5])
            notes.append(f"incomplete days: {shown}"
                         + (" ..." if len(a["incomplete"]) > 5 else ""))
        if a["absent"]:
            notes.append("no granules at all: " + ", ".join(str(d) for d in a["absent"][:8])
                         + (" ..." if len(a["absent"]) > 8 else ""))
        if notes:
            print(f"\n  {product} / {stream}")
            for n in notes:
                print(f"    - {n}")


def report(products, streams, start, end):
    geo = canada_tiles()
    results = {}
    for product in products:
        for stream in streams:
            rows = load_store(product, stream)
            if not rows:
                continue
            results[(product, stream)] = analyse(product, stream, rows, start, end, geo)
    if not results:
        sys.exit(f"no stored granules under {STORE} - run `sweep` first")
    print_report(results, geo, start, end)


# ---------------------------------------------------------------------------
def resolve(names_arg, default_names):
    if not names_arg:
        return list(default_names)
    want = {p.strip() for p in names_arg.split(",")}
    bad = want - set(ALL_PRODUCTS)
    if bad:
        sys.exit(f"unknown product(s): {', '.join(sorted(bad))}")
    return [n for n in ALL_PRODUCTS if n in want]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("status", "sweep", "report"):
        p = sub.add_parser(name)
        default_days = 10 if name == "status" else 90
        p.add_argument("--days", type=int, default=default_days,
                       help=f"query lookback in days (default {default_days})")
        p.add_argument("--start", help="YYYY-MM-DD (overrides --days; sweep/report only)")
        p.add_argument("--end", help="YYYY-MM-DD (default today UTC; sweep/report only)")
        p.add_argument("--products", help=("comma-separated subset (status default: all; "
                                           "sweep/report default: daily tier)"))
        if name == "sweep":
            p.add_argument("--stream", choices=["standard", "nrt"], default="standard")
            p.add_argument("--jobs", type=int, default=8)
            p.add_argument("--chunk-days", type=int, default=10,
                           help="days per CMR query (default 10)")
        elif name == "status":
            p.add_argument("--stream", choices=["standard", "nrt"], default="standard")
        else:
            p.add_argument("--streams", default="standard,nrt")

    args = ap.parse_args()

    end = (datetime.strptime(args.end, "%Y-%m-%d").date() if args.end
           else datetime.now(timezone.utc).date())
    start = (datetime.strptime(args.start, "%Y-%m-%d").date() if args.start
             else end - timedelta(days=args.days - 1))

    daily = [n for n in FIREMPC_MODIS if FIREMPC_MODIS[n]["cadence_days"] == 1]
    products = resolve(args.products, ALL_PRODUCTS if args.cmd == "status" else daily)

    if args.cmd == "status":
        earthaccess.login(persist=True)
        status(products, args.stream, args.days)
    elif args.cmd == "sweep":
        earthaccess.login(persist=True)
        print(f"sweep {start} .. {end}   stream={args.stream}   store={STORE}")
        cmg = [n for n in products if n in CMG]
        if cmg:
            print(f"  note: {', '.join(cmg)} are global single-granule products; "
                  f"queried unbounded")
        sweep(products, args.stream, start, end, args.jobs, args.chunk_days)
    else:
        report(products, [s.strip() for s in args.streams.split(",")], start, end)


if __name__ == "__main__":
    main()
