"""Verify extraction against known arrays, including reordered axes and masks."""
import gzip
import json
from datetime import UTC, datetime

import pytest

from scripts.export_native_city_data import export, extract_netcdf_point, identity

np = pytest.importorskip("numpy")
nc = pytest.importorskip("netCDF4")


def write_netcdf(path, *, lead_units="hours"):
    with nc.Dataset(path, "w") as ds:
        axes = {
            "latitude": ([39, 40], "degrees_north"),
            "longitude": ([115, 116], "degrees_east"),
            "time": ([0, 24], "hours since 2025-01-01 12:00:00"),
            "forecast_period": ([0, 3, 6], lead_units),
            "level": ([925, 850], "hPa"),
        }
        for name, (values, unit) in axes.items():
            ds.createDimension(name, len(values))
            var = ds.createVariable(name, "f8", (name,))
            var.units = unit
            var[:] = values
        # Deliberately put longitude first, time last, retain both vertical levels.
        var = ds.createVariable("go3", "f8", ("longitude", "level", "forecast_period", "latitude", "time"), fill_value=-9999)
        var.units = "kg kg-1"
        values = np.arange(48, dtype=float).reshape(2, 2, 3, 2, 2)
        values[1, 0, 1, 1, 1] = -9999
        var[:] = values


def test_point_preserves_named_axes_levels_mask_units_and_cycle(tmp_path):
    path = tmp_path / "source.nc"
    write_netcdf(path)
    with nc.Dataset(path) as ds:
        row = extract_netcdf_point(ds, ds.variables["go3"], datetime(2025, 1, 2, 12, tzinfo=UTC), 40.1, 115.9)
    assert row["values"] == [[27, None, 35], [39, 43, 47]]
    assert row["missing_values"] == 1
    assert row["dimensions"] == ["level", "forecast_period"]
    assert row["coordinates"]["level"] == {"values": [925, 850], "unit": "hPa"}
    assert row["unit"] == "kg kg-1"
    assert row["selected_grid"] == {"lat": 40, "lon": 116}
    assert row["valid_times"][-1] == "2025-01-02T18:00:00+00:00"
    assert row["available_at"] is None
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("problem", ["unknown_units", "duplicate_leads", "absent_cycle", "masked_axis"])
def test_ambiguous_or_missing_time_axes_fail_explicitly(tmp_path, problem):
    path = tmp_path / "source.nc"
    write_netcdf(path, lead_units="unknown" if problem == "unknown_units" else "hours")
    with nc.Dataset(path, "a") as ds:
        if problem == "duplicate_leads":
            ds.variables["forecast_period"][:] = [0, 3, 3]
        if problem == "masked_axis":
            ds.variables["latitude"][:] = [39, np.nan]
        cycle = datetime(2025, 1, 3 if problem == "absent_cycle" else 2, 12, tzinfo=UTC)
        with pytest.raises(ValueError):
            extract_netcdf_point(ds, ds.variables["go3"], cycle, 40, 116)


def test_export_writes_traceable_native_series_and_reports_missing_inputs(tmp_path):
    cams = tmp_path / "cams"
    obs = tmp_path / "obs"
    cams.mkdir(); obs.mkdir()
    source = cams / "o3_2025-01-01_2025-01-02.nc"
    write_netcdf(source)
    (obs / "china_cities_20250102.csv").write_text(
        "date,hour,type,北京\n20250102,7,PM2.5,0\n20250102,8,O3_8h,bad\n", encoding="utf-8")
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"cases": [{"city": "北京", "issue_date": "2025-01-03"}],
                                   "coordinates": {"北京": [40, 116]}}))
    out = tmp_path / "out"
    result = export(request, cams, obs, out)
    assert result["cams_series"] == 1
    assert result["observation_records"] == 2
    assert result["raw_files_transferred"] is False
    manifest = json.loads((out / "manifest.json").read_text())
    assert any(f.get("kind") == "sfc" and f["reason"] == "file_missing" for f in manifest["failures"])
    with gzip.open(out / "cams_native.jsonl.gz", "rt") as handle:
        row = json.loads(handle.readline())
    assert row["source_sha256"] == identity(source)["sha256"]
    assert row["values"] == [[27, None, 35], [39, 43, 47]]
    with gzip.open(out / "observations_native.jsonl.gz", "rt") as handle:
        rows = [json.loads(line) for line in handle]
    assert rows[0]["city_values"]["北京"] == "0"
    assert rows[1]["city_values"]["北京"] == "bad"
    assert len(list(out.iterdir())) == 3
    for item in manifest["outputs"]:
        assert identity(out / item["path"].split("/")[-1])["sha256"] == item["sha256"]
    with pytest.raises(FileExistsError):
        export(request, cams, obs, out)


def test_overlaps_are_preserved_only_by_explicit_policy(tmp_path):
    cams = tmp_path / "cams"
    cams.mkdir()
    write_netcdf(cams / "o3_2025-01-01_2025-01-02.nc")
    write_netcdf(cams / "o3_2025-01-02_2025-01-02.nc")
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"cases": [{"city": "北京", "issue_date": "2025-01-03"}],
                                   "coordinates": {"北京": [40, 116]}}))
    rejected = export(request, cams, None, tmp_path / "reject")
    assert rejected["cams_series"] == 0
    accepted = export(request, cams, None, tmp_path / "preserve", preserve_overlaps=True)
    assert accepted["cams_series"] == 2
    manifest = json.loads((tmp_path / "preserve/manifest.json").read_text())
    assert len(manifest["overlapping_sources"]) == 1
    with gzip.open(tmp_path / "preserve/cams_native.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    assert len({r["source_path"] for r in rows}) == 2
    assert rows[0]["values"] == rows[1]["values"]
