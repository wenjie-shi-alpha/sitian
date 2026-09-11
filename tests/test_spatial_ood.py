from scripts.build_spatial_ood import select_holdout_cities


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
