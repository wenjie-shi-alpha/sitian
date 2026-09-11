import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "derive_open_composition.py"
    spec = importlib.util.spec_from_file_location("derive_open_composition_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_earthengine_columns_remain_a_distinct_21_by_3_trajectory(tmp_path):
    composition = _module()
    cycle = datetime(2025, 3, 31, 12, tzinfo=timezone.utc)
    directory = tmp_path / "raw" / "cams_earthengine"
    directory.mkdir(parents=True)
    path = directory / "cams_column_trajectory_20250331T120000.tif"
    with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=63,
                       dtype="float32", transform=Affine.from_gdal(100, .4, 0, 40, 0, -.4)) as dataset:
        for index in range(1, 64):
            dataset.write(np.full((2, 2), index / 1e6, dtype="float32"), index)
    bands = [
        f"fh{lead:03d}_{name}_column"
        for lead in range(0, 121, 6)
        for name in ("carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide")
    ]
    Path(str(path) + ".json").write_text(json.dumps({
        "schema_version": "cams-earthengine-columns-v1",
        "forecast_reference_time": "2025-03-31T12:00:00Z",
        "available_at": "2025-03-31T22:00:00Z",
        "semantics": "total_column_mass",
        "leadtime_hour": list(range(0, 121, 6)),
        "bands": bands,
        "bytes": path.stat().st_size,
        "sha256": "test",
    }), encoding="utf-8")

    result = composition._column_gases(
        tmp_path, cycle, {"target": [39.8, 100.2]}
    )
    assert result["available"] is True
    assert result["semantics"] == "total_column_mass"
    assert len(result["records"]) == 21
    assert result["records"][0]["cities"]["target"]["carbon_monoxide_kg_m2"] == 1e-6
    assert result["records"][-1]["cities"]["target"]["sulphur_dioxide_kg_m2"] == 63e-6
