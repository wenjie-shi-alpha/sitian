import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "fetch_cams_earthengine.py"
    spec = importlib.util.spec_from_file_location("fetch_cams_earthengine_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_jobs_preserve_issue_to_previous_12z_cycle_mapping():
    gee = _module()
    manifest = {"cams": {"cycles": [
        {"issue_date": "2025-04-02", "cycle": "2025-04-01T12:00:00Z",
         "available_at": "2025-04-01T22:00:00Z"},
        {"issue_date": "2025-04-01", "cycle": "2025-03-31T12:00:00Z",
         "available_at": "2025-03-31T22:00:00Z"},
    ]}}
    jobs = gee._jobs(manifest)
    assert jobs == [
        {"issue_date": "2025-04-01", "cycle": "2025-03-31T12:00:00Z",
         "available_at": "2025-03-31T22:00:00Z"},
        {"issue_date": "2025-04-02", "cycle": "2025-04-01T12:00:00Z",
         "available_at": "2025-04-01T22:00:00Z"},
    ]


def test_schema_has_all_21_by_3_column_bands():
    gee = _module()
    names = gee._band_names()
    assert len(names) == 63
    assert names[:3] == [
        "fh000_carbon_monoxide_column",
        "fh000_nitrogen_dioxide_column",
        "fh000_sulphur_dioxide_column",
    ]
    assert names[-1] == "fh120_sulphur_dioxide_column"


def test_complete_requires_matching_semantics_and_cycle(tmp_path):
    gee = _module()
    target, sidecar = gee._target_paths(tmp_path, "2025-03-31T12:00:00Z")
    target.write_bytes(b"geotiff")
    sidecar.write_text(json.dumps({
        "bytes": len(b"geotiff"),
        "dataset": gee.DATASET,
        "kind": gee.KIND,
        "schema_version": gee.SCHEMA_VERSION,
        "forecast_reference_time": "2025-03-31T12:00:00Z",
        "leadtime_hour": list(gee.LEAD_HOURS),
    }), encoding="utf-8")
    assert gee._is_complete(target, sidecar, "2025-03-31T12:00:00Z")
    assert not gee._is_complete(target, sidecar, "2025-04-01T12:00:00Z")


def test_geotiff_validation_checks_band_count_and_order(tmp_path):
    gee = _module()
    names = gee._band_names()
    path = tmp_path / "sample.tif"
    with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=len(names),
                       dtype="float32", crs="EPSG:4326",
                       transform=from_origin(59.8, 60.2, .4, .4)) as dataset:
        dataset.write(np.zeros((len(names), 2, 2), dtype="float32"))
        dataset.descriptions = tuple(names)
    info = gee._inspect_geotiff(path, names)
    assert info["size"] == [2, 2]
    assert info["geo_transform"] == [59.8, .4, 0, 60.2, 0, -.4]
    assert info["band_descriptions_preserved"] is True
    with pytest.raises(RuntimeError, match="bands"):
        gee._inspect_geotiff(path, names[:-1])
    with pytest.raises(RuntimeError, match="band order"):
        gee._inspect_geotiff(path, list(reversed(names)))
    path.write_bytes(b"corrupt raster")
    with pytest.raises(RuntimeError, match="cannot read GeoTIFF"):
        gee._inspect_geotiff(path, names)
