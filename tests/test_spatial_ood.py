import pytest

from scripts.build_spatial_ood import build_train_only_regimes, select_holdout_cities


def test_regimes_preserve_pm10_primary_and_all_stratum_mass():
    cities = {f"city-{i}": {} for i in range(8)}
    coordinates = {city: [20 + i, 100 + i] for i, city in enumerate(cities)}
    records = [{"city": city, "stratum": label} for city in cities
               for label in ("pm10_primary", "clean")]
    _, audit = build_train_only_regimes(records, cities, coordinates, 71)
    assert "fraction_pm10_primary" in audit["features"]
    for row in audit["feature_rows"].values():
        assert row["stratum_fractions"]["pm10_primary"] == 0.5
        assert sum(row["stratum_fractions"].values()) == 1


@pytest.mark.parametrize("label", ["pm10", "typo", None])
def test_regimes_reject_unrecognized_strata(label):
    with pytest.raises(ValueError, match="Unknown sampling stratum"):
        build_train_only_regimes([{"city": "x", "stratum": label}], {}, {}, 71)


def test_external_snapshot_manifest_paths_are_readable(tmp_path):
    import json
    from scripts.build_spatial_ood import _load_paths, _write_manifest
    case = tmp_path / "case"
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, [{"path": case}])
    assert json.loads(manifest.read_text()) == [str(case)]
    assert _load_paths(manifest) == [case]


def test_cli_refuses_existing_output_before_loading_inputs(tmp_path, monkeypatch):
    from scripts.build_spatial_ood import main
    frozen = tmp_path / "frozen.json"
    frozen.write_text("[]")
    monkeypatch.setattr("sys.argv", ["build_spatial_ood.py", "--train-out", str(frozen)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert frozen.read_text() == "[]"


def test_spatial_ood_selects_one_reproducible_noncapital_per_cluster():
    cities = {
        "capital-a": {"cluster": 0, "reason": "capital"},
        "city-a": {"cluster": 0, "reason": "cluster_0"},
        "city-b": {"cluster": 0, "reason": "cluster_0"},
        "singleton": {"cluster": 1, "reason": "capital"},
    }
    first = select_holdout_cities(cities, 71)
    assert first == select_holdout_cities(cities, 71)
    assert set(first) == {0, 1}
    assert first[0] in {"city-a", "city-b"}
    assert first[1] == "singleton"


def test_spatial_ood_power_rule_uses_availability_but_still_prefers_noncapital():
    cities = {
        "capital": {"cluster": 0, "reason": "capital"},
        "small": {"cluster": 0, "reason": "cluster_0"},
        "large": {"cluster": 0, "reason": "cluster_0"},
    }
    selected = select_holdout_cities(
        cities, 71, evaluation_counts={"capital": 99, "small": 2, "large": 15}
    )
    assert selected == {0: "large"}
