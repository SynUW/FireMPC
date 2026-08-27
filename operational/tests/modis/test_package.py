import json
from datetime import date, timedelta
from pathlib import Path

import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin

from modis.package import build_package, code_revision, resolve_source, target_dates
from modis.processing_registry import CHANNEL_NAMES, CHANNELS_BY_PRODUCT, ROUTES


def _fake_run(outputs):
    def run(command, **kwargs):
        class Result:
            stdout = outputs[command[1]]
        return Result()
    return run


def test_code_revision_is_bare_sha_on_a_clean_tree(monkeypatch):
    monkeypatch.setattr(
        "modis.package.subprocess.run",
        _fake_run({"rev-parse": "abc123\n", "status": ""}),
    )
    assert code_revision() == "abc123"


def test_code_revision_is_marked_dirty_when_the_tree_has_changes(monkeypatch):
    monkeypatch.setattr(
        "modis.package.subprocess.run",
        _fake_run({"rev-parse": "abc123\n", "status": " M modis_preprocess.py\n"}),
    )
    assert code_revision() == "abc123-dirty"


def _write(path: Path, count: int, *, land_cover: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.full((count, 2, 3), 4 if land_cover else 100, dtype="uint8")
    with rasterio.open(
        path, "w", driver="GTiff", width=3, height=2, count=count,
        dtype="uint8", crs="+proj=longlat +datum=WGS84 +no_defs",
        transform=from_origin(-141, 84, 0.05, 0.05),
        nodata=0 if land_cover else 255,
    ) as dataset:
        dataset.write(values)


def test_target_dates_cross_year_boundary():
    assert target_dates("2026-01-05") == tuple(
        date(2025, 12, 26) + timedelta(days=offset) for offset in range(10)
    )


def test_fill_resolution_classifies_sources(tmp_path):
    available = {date(2026, 1, 1): tmp_path / "one.tif"}
    choice = resolve_source("MCD15A3H", date(2026, 1, 4), available, 7)
    assert choice.age_days == 3
    assert choice.status == "composite_fill"


def test_composite_fill_ends_at_the_cadence(tmp_path):
    """Age inside the cadence is by design; beyond it an anchor was missed.

    A four-day composite forward-filled to the days it covers is what the
    training pipeline does.  Age 4+ means a missed anchor, which is real
    latency and must not be recorded as expected staleness.
    """
    available = {date(2026, 1, 1): tmp_path / "one.tif"}
    assert resolve_source("MCD15A3H", date(2026, 1, 4), available, 11).status == "composite_fill"
    late = resolve_source("MCD15A3H", date(2026, 1, 11), available, 11)
    assert late.age_days == 10
    assert late.status == "latency_fill"


def test_daily_products_have_no_composite_fill(tmp_path):
    available = {date(2026, 1, 1): tmp_path / "one.tif"}
    for product in ("MCD09GA", "MCD11A1", "MCD09CMG"):
        choice = resolve_source(product, date(2026, 1, 2), available, 3)
        assert choice.status == "latency_fill", product


def test_annual_land_cover_as_of_uses_its_daily_published_copy(tmp_path):
    requested = date(2026, 6, 22)
    daily_copy = tmp_path / "2026_06_22.tif"
    choice = resolve_source(
        "MCD12Q1", requested, {requested: daily_copy}, 1095,
        observations={date(2024, 1, 1)}, as_of=date(2026, 7, 2),
    )
    assert choice.path == daily_copy
    assert choice.source_date == date(2024, 1, 1)
    assert choice.status == "stale_by_design"


def test_policy_covers_every_package_product():
    from modis.package import load_fill_policy

    policy = load_fill_policy()["maximum_age_days"]
    assert set(policy) == set(ROUTES)
    # MCD15A3H must allow at least one missed anchor beyond its own cadence.
    assert policy["MCD15A3H"] > ROUTES["MCD15A3H"].cadence_days


def test_builds_atomic_package_with_provenance(tmp_path):
    target = date(2026, 1, 11)
    template = tmp_path / "template.tif"
    _write(template, 1)
    scaled = tmp_path / "scaled"
    for product, route in ROUTES.items():
        count = len(CHANNELS_BY_PRODUCT[product])
        if product == "MCD12Q1":
            _write(scaled / route.output_dir / "2025_01_01.tif", count, land_cover=True)
            continue
        for day in target_dates(target):
            _write(scaled / route.output_dir / f"{day:%Y_%m_%d}.tif", count)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "version": "test", "maximum_age_days": {
            product: 400 if product == "MCD12Q1" else 0 for product in ROUTES
        },
    }))
    output = build_package(
        target, scaled_root=scaled, package_root=tmp_path / "packages",
        template_path=template, policy_path=policy,
    )
    assert output.is_file()
    with h5py.File(output) as package:
        assert package["data"].shape == (10, 17, 2, 3)
        assert list(package["channel_names"].asstr()[:]) == list(CHANNEL_NAMES)
        assert package["fill_age_days"].shape == (10, 17)
        assert package.attrs["mcd12q1_source_year"] == 2025
    with (output.parent / "validation.json").open() as handle:
        assert json.load(handle)["valid"] is True


def test_composite_fill_is_not_reported_as_a_fresh_observation(tmp_path):
    """daily_filled forward-copies a 4-day anchor across its window.

    The scaled tree therefore holds a raster for every date, and reading age
    from it reports eight of ten MCD15A3H dates as freshly observed when only
    two were real anchors.
    """
    from modis.package import resolve_source

    scaled = {date(2026, 7, 30) + timedelta(days=n): tmp_path / f"{n}.tif" for n in range(4)}
    anchors = {date(2026, 7, 28), date(2026, 8, 1)}

    stale = resolve_source("MCD15A3H", date(2026, 7, 31), scaled, 12, anchors)
    assert stale.source_date == date(2026, 7, 28)
    assert stale.age_days == 3
    assert stale.status == "composite_fill"

    fresh = resolve_source("MCD15A3H", date(2026, 8, 1), scaled, 12, anchors)
    assert fresh.age_days == 0
    assert fresh.status == "observed"

    # Without the anchor set the scaled tree lies: every date looks observed.
    naive = resolve_source("MCD15A3H", date(2026, 7, 31), scaled, 12, None)
    assert naive.age_days == 0


def test_observation_dates_reads_unified_and_ndvievi_inherits_mcd09ga(tmp_path):
    from modis.package import observation_dates

    (tmp_path / "MCD09GA").mkdir(parents=True)
    (tmp_path / "MCD09GA" / "2026_08_01.tif").touch()
    assert observation_dates("MCD09GA", tmp_path) == {date(2026, 8, 1)}
    assert observation_dates("NDVIEVI", tmp_path) == {date(2026, 8, 1)}
