#!/usr/bin/env python3
"""Build a pre-registered city-held-out spatial OOD split.

One city per existing pollution-regime cluster is selected with a stable hash.
Non-capitals are preferred when a cluster has one, preserving the nationwide
administrative anchors in training where possible.  Selection never reads a
forecast, reward, or model result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.case import CaseBundle  # noqa: E402
from sitian.provenance import file_identity  # noqa: E402

STRATA = ("clean", "event", "o3", "pm10", "switch", "turning")


def select_holdout_cities(
    cities: dict, seed: int, evaluation_counts: dict[str, int] | None = None
) -> dict[int, str]:
    by_cluster: dict[int, list[tuple[str, dict]]] = defaultdict(list)
    for city, metadata in cities.items():
        by_cluster[int(metadata["cluster"])].append((city, metadata))
    selected = {}
    for cluster, rows in sorted(by_cluster.items()):
        non_capitals = [row for row in rows if row[1].get("reason") != "capital"]
        candidates = non_capitals or rows
        if evaluation_counts is not None:
            maximum = max(evaluation_counts.get(city, 0) for city, _ in candidates)
            candidates = [row for row in candidates
                          if evaluation_counts.get(row[0], 0) == maximum]
        selected[cluster] = min(
            (city for city, _ in candidates),
            key=lambda city: hashlib.sha256(
                f"spatial-ood-v1:{seed}:{cluster}:{city}".encode("utf-8")
            ).hexdigest(),
        )
    return selected


def build_train_only_regimes(
    records: list[dict], city_metadata: dict, coordinates: dict, seed: int
) -> tuple[dict, dict]:
    """Cluster cities without reading validation/test outcomes.

    Stratum frequencies come only from the original training-period cases.
    Latitude/longitude stabilize sparse-process cities and are static covariates.
    The resulting labels are used only to balance the grouped holdout, never as
    a model feature or reward input.
    """
    by_city: dict[str, Counter] = defaultdict(Counter)
    for row in records:
        by_city[row["city"]][str(row.get("stratum"))] += 1
    cities = sorted(set(by_city) & set(coordinates) & set(city_metadata))
    if len(cities) < 8:
        raise RuntimeError("fewer than eight train-period cities available for OOD clustering")
    raw_features = []
    feature_rows = {}
    for city in cities:
        counts = by_city[city]
        total = sum(counts.values())
        coord = coordinates[city]
        lat, lon = (
            (coord["lat"], coord["lon"]) if isinstance(coord, dict)
            else (coord[0], coord[1])
        )
        values = [
            float(lat), float(lon),
            *(counts[name] / total for name in STRATA),
        ]
        raw_features.append(values)
        feature_rows[city] = {
            "train_cases": total,
            "lat": values[0],
            "lon": values[1],
            "stratum_fractions": {
                name: round(counts[name] / total, 6) for name in STRATA
            },
        }
    scaled = StandardScaler().fit_transform(np.asarray(raw_features, dtype=float))
    labels = KMeans(n_clusters=8, random_state=seed, n_init=20).fit_predict(scaled)
    regimes = {
        city: {
            "cluster": int(label),
            "reason": (city_metadata[city] or {}).get("reason"),
        }
        for city, label in zip(cities, labels)
    }
    audit = {
        "source_period": "original_train_manifest_only",
        "features": ["latitude", "longitude", *[f"fraction_{name}" for name in STRATA]],
        "standardization": "per-feature z-score over train-period cities",
        "algorithm": "sklearn KMeans(k=8,n_init=20)",
        "random_state": seed,
        "cities": len(cities),
        "feature_rows": feature_rows,
        "cluster_sizes": dict(sorted(Counter(labels.tolist()).items())),
    }
    return regimes, audit


def _load_paths(path: Path) -> list[Path]:
    return [REPO_ROOT / value for value in json.loads(path.read_text(encoding="utf-8"))]


def _records(paths: list[Path]) -> list[dict]:
    output = []
    for path in paths:
        bundle = CaseBundle.load(path)
        output.append({
            "path": path,
            "case_id": bundle.case_id,
            "city": bundle.region,
            "issue_date": bundle.issue_date,
            "forecast_dates": bundle.forecast_dates(),
            "stratum": (bundle.meta or {}).get("stratum"),
        })
    return output


def _write_manifest(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        [str(row["path"].relative_to(REPO_ROOT)) for row in records],
        ensure_ascii=False, indent=1,
    ) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", default="data/interim/cities_selected.json")
    parser.add_argument("--train-source",
                        default="data/interim/train_without_winter_challenge.json")
    parser.add_argument("--regime-train-source",
                        default="data/interim/valid_cases_train.json")
    parser.add_argument("--val-source", default="data/interim/valid_cases_val.json")
    parser.add_argument("--test-source", default="data/interim/valid_cases_test.json")
    parser.add_argument("--winter-challenge", default="data/interim/challenge_winter.json")
    parser.add_argument("--coordinates", default="data/interim/city_coords.json")
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--train-out",
                        default="data/interim/train_without_winter_or_spatial_holdout.json")
    parser.add_argument("--val-out", default="data/interim/spatial_ood_val.json")
    parser.add_argument("--test-out", default="data/interim/spatial_ood_test.json")
    parser.add_argument("--audit-out", default="data/interim/spatial_ood_audit.json")
    args = parser.parse_args()

    source_paths = {
        "train_after_winter_purge": REPO_ROOT / args.train_source,
        "regime_train_only": REPO_ROOT / args.regime_train_source,
        "val": REPO_ROOT / args.val_source,
        "test": REPO_ROOT / args.test_source,
        "winter_challenge": REPO_ROOT / args.winter_challenge,
    }
    city_path = REPO_ROOT / args.cities
    coordinate_path = REPO_ROOT / args.coordinates
    city_artifact = json.loads(city_path.read_text(encoding="utf-8"))
    records = {name: _records(_load_paths(path)) for name, path in source_paths.items()}
    coordinates = json.loads(coordinate_path.read_text(encoding="utf-8"))
    regimes, regime_audit = build_train_only_regimes(
        records["regime_train_only"], city_artifact["cities"], coordinates, args.seed
    )
    # A paper test must be evaluable. Restrict only on manifest membership—not
    # on any validation/test label, pollutant value, score, or model output.
    test_cities = {row["city"] for row in records["test"]}
    test_case_counts = Counter(row["city"] for row in records["test"])
    eligible_regimes = {city: metadata for city, metadata in regimes.items()
                        if city in test_cities}
    selected = select_holdout_cities(
        eligible_regimes, args.seed, evaluation_counts=test_case_counts
    )
    if len(selected) != 8:
        raise RuntimeError("not every train-only regime has an evaluable test city")
    holdout = set(selected.values())
    retained_train = [row for row in records["train_after_winter_purge"]
                      if row["city"] not in holdout]
    ood_val = [row for row in records["val"] if row["city"] in holdout]
    ood_test = [row for row in records["test"] if row["city"] in holdout]
    if set(row["city"] for row in retained_train) & holdout:
        raise AssertionError("held-out city remains in training")
    if set(row["city"] for row in ood_val + ood_test) - holdout:
        raise AssertionError("OOD manifest contains a non-held-out city")
    if not retained_train or not ood_val or not ood_test:
        raise RuntimeError("spatial OOD construction produced an empty required split")

    train_out, val_out, test_out = (
        REPO_ROOT / args.train_out, REPO_ROOT / args.val_out, REPO_ROOT / args.test_out
    )
    _write_manifest(train_out, retained_train)
    _write_manifest(val_out, ood_val)
    _write_manifest(test_out, ood_test)

    train_targets = {(row["city"], day) for row in retained_train
                     for day in row["forecast_dates"]}
    winter_targets = {(row["city"], day) for row in records["winter_challenge"]
                      for day in row["forecast_dates"]}
    audit = {
        "artifact_type": "spatial_ood_split_audit",
        "split_version": "city-cluster-holdout-v1",
        "selection_policy": {
            "unit": "entire city",
            "clusters": "one city from every pollution-regime cluster",
            "regime_features": "train-period geography + train-period stratum frequencies",
            "validation_or_test_outcomes_used": False,
            "test_manifest_used_only_for_city_evaluability": True,
            "non_capital_preferred": True,
            "power_rule": "within each cluster choose the non-capital with most valid test cases",
            "power_rule_uses_outcomes": False,
            "tie_break": "minimum sha256(spatial-ood-v1:seed:cluster:city)",
            "seed": args.seed,
            "model_or_reward_results_used": False,
        },
        "sources": {
            "cities": file_identity(city_path, relative_to=REPO_ROOT),
            "coordinates": file_identity(coordinate_path, relative_to=REPO_ROOT),
            **{name: file_identity(path, relative_to=REPO_ROOT)
               for name, path in source_paths.items()},
        },
        "train_only_regime_construction": regime_audit,
        "heldout_by_cluster": {str(key): value for key, value in selected.items()},
        "heldout_cities": sorted(holdout),
        "train": {
            "manifest": file_identity(train_out, relative_to=REPO_ROOT),
            "cases": len(retained_train),
            "cities": len({row["city"] for row in retained_train}),
            "removed_city_cases": len(records["train_after_winter_purge"]) - len(retained_train),
            "heldout_city_overlap": 0,
        },
        "spatial_ood_val": {
            "manifest": file_identity(val_out, relative_to=REPO_ROOT),
            "cases": len(ood_val),
            "cities": len({row["city"] for row in ood_val}),
            "issue_dates": len({row["issue_date"] for row in ood_val}),
            "strata": dict(Counter(row["stratum"] for row in ood_val)),
        },
        "spatial_ood_test": {
            "manifest": file_identity(test_out, relative_to=REPO_ROOT),
            "cases": len(ood_test),
            "cities": len({row["city"] for row in ood_test}),
            "issue_dates": len({row["issue_date"] for row in ood_test}),
            "strata": dict(Counter(row["stratum"] for row in ood_test)),
        },
        "integrity": {
            "train_heldout_city_overlap": 0,
            "train_winter_challenge_forecast_city_day_overlap": len(
                train_targets & winter_targets
            ),
        },
    }
    out = REPO_ROOT / args.audit_out
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=1))
    print(f"-> {train_out}\n-> {val_out}\n-> {test_out}\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
