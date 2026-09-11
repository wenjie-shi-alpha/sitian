#!/usr/bin/env python3
"""Create leakage-safe veRL parquet datasets from audited national cases."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.agents.llm_openai import SYSTEM_PROMPT  # noqa: E402
from sitian.case import CaseBundle  # noqa: E402
from sitian.env import ForecastEnv  # noqa: E402
from sitian.provenance import (  # noqa: E402
    case_bundle_snapshot, file_identity, score_implementation_matches,
)
from sitian.scoring import reward_spec  # noqa: E402


def _identity(path: Path) -> dict:
    return file_identity(path, relative_to=REPO_ROOT)


def _stored_identity_valid(identity: dict | None) -> bool:
    if not identity or not identity.get("path"):
        return False
    stored = Path(identity["path"])
    path = stored if stored.is_absolute() else REPO_ROOT / stored
    if not path.is_file():
        return False
    actual = file_identity(path)
    return (actual.get("sha256"), actual.get("bytes")) == (
        identity.get("sha256"), identity.get("bytes")
    )


def _stratified(paths: list[Path], count: int, seed: int) -> list[Path]:
    if count <= 0 or count >= len(paths):
        return list(paths)
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        meta = json.loads((path / "case.json").read_text(encoding="utf-8"))
        groups[(meta.get("meta") or {}).get("stratum", "unknown")].append(path)
    rng = random.Random(seed)
    selected: list[Path] = []
    keys = sorted(groups)
    base, remainder = divmod(count, len(keys))
    for index, key in enumerate(keys):
        take = min(base + int(index < remainder), len(groups[key]))
        selected.extend(rng.sample(groups[key], take))
    if len(selected) < count:
        used = set(selected)
        selected.extend(rng.sample(
            [path for path in paths if path not in used], count - len(selected)
        ))
    rng.shuffle(selected)
    return selected[:count]


def _manifest_path(split: str) -> Path:
    # Both the winter challenge and the pre-registered city holdout must be
    # absent from training.  build_spatial_ood.py consumes the winter-purged
    # intermediate manifest and emits the joint, audited training manifest.
    if split == "train":
        return (
            REPO_ROOT / "data" / "interim" /
            "train_without_winter_or_spatial_holdout.json"
        )
    if split == "challenge_winter":
        return REPO_ROOT / "data" / "interim" / "challenge_winter.json"
    return REPO_ROOT / "data" / "interim" / f"valid_cases_{split}.json"


def _read_manifest(split: str, maximum: int, seed: int) -> list[Path]:
    manifest = _manifest_path(split)
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    paths = [(REPO_ROOT / row).resolve() for row in rows]
    return _stratified(paths, maximum, seed)


def _row(case_dir: Path, split: str, index: int, require_evidence: bool) -> dict:
    bundle = CaseBundle.load(case_dir)
    violations = bundle.audit_time_gate()
    if violations:
        raise ValueError(f"{bundle.case_id}: time gate violations: {violations[:2]}")
    if bundle.truth_daily_full() is None:
        raise ValueError(f"{bundle.case_id}: scoreable multi-pollutant truth missing")
    if bundle.expert is not None or (case_dir / "expert.json").exists():
        raise ValueError(
            f"{bundle.case_id}: expert.json is forbidden in national RL inputs/reward"
        )
    if require_evidence and not bundle.evidence:
        raise ValueError(f"{bundle.case_id}: evidence.json missing")
    env = ForecastEnv(bundle)
    task = env.reset()
    tool_names = [item["function"]["name"] for item in env.tool_specs("openai")]
    try:
        case_value = case_dir.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        # External datasets remain supported with an explicit absolute path.
        case_value = str(case_dir.resolve())
    tools_kwargs = {
        name: {"create_kwargs": {"case_dir": case_value}}
        for name in tool_names
    }
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            "任务简报：\n" + json.dumps(task["brief"], ensure_ascii=False, separators=(",", ":"))
        )},
    ]
    return {
        "data_source": "sitian/national_forecast_reward_v079",
        "agent_name": "sitian_tool_agent",
        "prompt": prompt,
        "ability": "multi_source_air_quality_forecast",
        # Required by RLHFDataset, but the value contains no truth.  The custom
        # agent loop supplies reward_score directly from hidden ForecastEnv.
        "reward_model": {"style": "rule", "ground_truth": "FORECAST_ENV_HIDDEN_TRUTH"},
        "extra_info": {
            "split": split,
            "index": index,
            "case_id": bundle.case_id,
            "issue_date": bundle.issue_date,
            "stratum": bundle.meta.get("stratum", "unknown"),
            "need_tools_kwargs": True,
            "tool_selection": tool_names,
            "tools_kwargs": tools_kwargs,
        },
    }


def _write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")


def _forecast_city_days(paths: list[Path]) -> set[tuple[str, str]]:
    keys = set()
    for path in paths:
        bundle = CaseBundle.load(path)
        keys.update((bundle.region, day) for day in bundle.forecast_dates())
    return keys


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_snapshot(paths: list[Path]) -> dict:
    """Hash every file that can affect the task, tools, validation, or reward."""
    return case_bundle_snapshot(paths, relative_to=REPO_ROOT, include_truth=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/verl"))
    parser.add_argument("--smoke-train", type=int, default=64)
    parser.add_argument("--smoke-val", type=int, default=16)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--require-evidence", action="store_true", default=True)
    parser.add_argument("--allow-missing-evidence", dest="require_evidence", action="store_false")
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir

    coverage_path = REPO_ROOT / "data/interim/open_evidence_coverage.json"
    contract_path = REPO_ROOT / "data/interim/expert_evidence_contract_audit.json"
    reward_contract_path = REPO_ROOT / "data/interim/reward_contract_audit.json"
    runtime_contract_path = REPO_ROOT / "data/interim/verl_runtime_contract.json"
    guidance_bias_path = REPO_ROOT / "data/interim/guidance_bias_history_v1.json"
    spatial_ood_path = REPO_ROOT / "data/interim/spatial_ood_audit.json"
    purged_train_path = _manifest_path("train")
    tool_config_path = REPO_ROOT / "configs/verl/sitian_tools.yaml"
    agent_loop_config_path = REPO_ROOT / "configs/verl/sitian_agent_loop.yaml"
    coverage = (json.loads(coverage_path.read_text(encoding="utf-8"))
                if coverage_path.is_file() else {})
    contract = (json.loads(contract_path.read_text(encoding="utf-8"))
                if contract_path.is_file() else {})
    reward_contract = (json.loads(reward_contract_path.read_text(encoding="utf-8"))
                       if reward_contract_path.is_file() else {})
    runtime_contract = (json.loads(runtime_contract_path.read_text(encoding="utf-8"))
                        if runtime_contract_path.is_file() else {})
    guidance_bias = (json.loads(guidance_bias_path.read_text(encoding="utf-8"))
                     if guidance_bias_path.is_file() else {})
    spatial_ood = (json.loads(spatial_ood_path.read_text(encoding="utf-8"))
                   if spatial_ood_path.is_file() else {})
    current_reward = reward_spec()
    reward_contract_ready = (
        reward_contract.get("passed") is True
        and score_implementation_matches(reward_contract, REPO_ROOT)
        and (reward_contract.get("reward") or {}).get("version")
            == current_reward.get("version")
        and (reward_contract.get("reward") or {}).get("config_sha256")
            == current_reward.get("config_sha256")
    )
    runtime_contract_ready = (
        runtime_contract.get("passed") is True
        and bool(runtime_contract.get("checks"))
        and all((runtime_contract.get("checks") or {}).values())
        and bool(runtime_contract.get("inputs"))
        and all(_stored_identity_valid(identity)
                for identity in (runtime_contract.get("inputs") or {}).values())
        and (runtime_contract.get("reward") or {}).get("version")
            == current_reward.get("version")
        and (runtime_contract.get("reward") or {}).get("config_sha256")
            == current_reward.get("config_sha256")
    )
    coverage_provenance = coverage.get("provenance") or {}
    coverage_derivations = coverage_provenance.get("derivation_implementations") or {}
    coverage_ready = bool(
        coverage.get("ready_for_full_attachment") is True
        and coverage.get("case_attachments", {}).get("fraction") == 1.0
        and _stored_identity_valid(coverage.get("source_manifest"))
        and _stored_identity_valid(coverage_provenance.get("audit_implementation"))
        and _stored_identity_valid(
            coverage_provenance.get("open_evidence_contract_implementation")
        )
        and set(coverage_derivations) == {
            "synoptic", "composition", "spatial_observations", "fires",
            "static_context", "case_attachment",
        }
        and all(_stored_identity_valid(identity)
                for identity in coverage_derivations.values())
    )
    contract_provenance = contract.get("provenance") or {}
    contract_ready = bool(
        contract.get("preflight_contract_passed") is True
        and (contract.get("reward") or {}).get("version")
            == current_reward.get("version")
        and (contract.get("reward") or {}).get("config_sha256")
            == current_reward.get("config_sha256")
        and len(contract_provenance.get("case_manifests") or []) == 3
        and all(_stored_identity_valid(identity)
                for identity in contract_provenance.get("case_manifests") or [])
        and all(_stored_identity_valid(contract_provenance.get(key)) for key in (
            "audit_implementation", "process_evidence_implementation",
            "open_evidence_implementation", "guidance_bias_implementation",
        ))
    )
    bias_provenance = guidance_bias.get("provenance") or {}
    bias_manifests = bias_provenance.get("manifests") or []
    guidance_bias_ready = bool(
        guidance_bias.get("artifact_type") == "guidance_bias_history"
        and (guidance_bias.get("summary") or {}).get("records", 0) > 0
        and len(bias_manifests) == 1
        and _stored_identity_valid(bias_manifests[0])
        and _stored_identity_valid(bias_provenance.get("builder"))
        and _stored_identity_valid(bias_provenance.get("implementation"))
        and bias_manifests[0].get("sha256")
            == file_identity(purged_train_path).get("sha256")
        and (guidance_bias.get("split_policy") or {}).get("main_experiment")
            == "train_only_city_day_purged"
        and (guidance_bias.get("split_policy") or {}).get(
            "evaluation_truth_updates_index"
        ) is False
    )
    spatial_ood_ready = bool(
        spatial_ood.get("artifact_type") == "spatial_ood_split_audit"
        and _stored_identity_valid((spatial_ood.get("train") or {}).get("manifest"))
        and (spatial_ood.get("train") or {}).get("manifest", {}).get("sha256")
            == file_identity(purged_train_path).get("sha256")
        and _stored_identity_valid(
            (spatial_ood.get("spatial_ood_val") or {}).get("manifest")
        )
        and _stored_identity_valid(
            (spatial_ood.get("spatial_ood_test") or {}).get("manifest")
        )
        and len(spatial_ood.get("heldout_by_cluster") or {}) == 8
        and (spatial_ood.get("selection_policy") or {}).get(
            "validation_or_test_outcomes_used") is False
        and (spatial_ood.get("selection_policy") or {}).get(
            "power_rule_uses_outcomes") is False
        and all(value == 0 for value in (spatial_ood.get("integrity") or {}).values())
    )
    if args.require_evidence and not (
        coverage_ready
        and contract_ready
        and reward_contract_ready
        and runtime_contract_ready
        and guidance_bias_ready
        and spatial_ood_ready
        and tool_config_path.is_file()
        and agent_loop_config_path.is_file()
    ):
        raise ValueError(
            "formal training dataset requires complete evidence/expert/reward contracts, "
            "veRL native-runtime contract, guidance-bias index and frozen veRL configs; "
            "use --allow-missing-evidence only for non-training config tests"
        )

    outputs = {}
    specifications = [
        ("train", 0, "train.parquet"),
        ("val", 0, "val.parquet"),
        ("challenge_winter", 0, "challenge_winter.parquet"),
        ("train", args.smoke_train, "smoke_train.parquet"),
        ("val", args.smoke_val, "smoke_val.parquet"),
    ]
    cache: dict[tuple[str, int], list[dict]] = {}
    source_paths: dict[tuple[str, int], list[Path]] = {}
    for split, maximum, filename in specifications:
        key = (split, maximum)
        paths = _read_manifest(split, maximum, args.seed + (0 if split == "train" else 1))
        source_paths[key] = paths
        rows = [_row(path, split, index, args.require_evidence)
                for index, path in enumerate(paths)]
        cache[key] = rows
        target = out_dir / filename
        _write(rows, target)
        outputs[filename] = {
            "rows": len(rows), "bytes": target.stat().st_size,
            "sha256": _sha(target),
            "strata": dict(sorted(__import__("collections").Counter(
                row["extra_info"]["stratum"] for row in rows
            ).items())),
        }

    train_paths = source_paths[("train", 0)]
    challenge_paths = source_paths[("challenge_winter", 0)]
    case_overlap = {path.resolve() for path in train_paths} & {
        path.resolve() for path in challenge_paths
    }
    city_day_overlap = _forecast_city_days(train_paths) & _forecast_city_days(challenge_paths)
    if case_overlap or city_day_overlap:
        raise ValueError(
            "winter challenge leakage: "
            f"case_overlap={len(case_overlap)}, city_day_overlap={len(city_day_overlap)}"
        )

    # Checkpoint selection must see both the warm-season validation distribution
    # and the held-out winter/event challenge.  Keep their split labels in each
    # row so reports can disaggregate them after a single veRL validation pass.
    selection_rows = cache[("val", 0)] + cache[("challenge_winter", 0)]
    selection_path = out_dir / "selection.parquet"
    _write(selection_rows, selection_path)
    outputs[selection_path.name] = {
        "rows": len(selection_rows),
        "bytes": selection_path.stat().st_size,
        "sha256": _sha(selection_path),
        "splits": dict(sorted(__import__("collections").Counter(
            row["extra_info"]["split"] for row in selection_rows
        ).items())),
    }
    input_snapshots = {
        split: _input_snapshot(source_paths[(split, 0)])
        for split in ("train", "val", "challenge_winter")
    }

    manifest = {
        "artifact_type": "verl_training_dataset",
        "dataset_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "reward": reward_spec(),
        "expert_corpus_used_as_training_input": False,
        "truth_exposed_in_prompt_or_ground_truth_field": False,
        "agent_name": "sitian_tool_agent",
        "seed": args.seed,
        "source_manifests": {
            split: _identity(_manifest_path(split))
            for split in ("train", "val", "challenge_winter")
        },
        "gate_artifacts": {
            "open_evidence_coverage": (
                _identity(coverage_path) if coverage_path.is_file() else None
            ),
            "expert_evidence_contract": (
                _identity(contract_path) if contract_path.is_file() else None
            ),
            "reward_contract_audit": (
                _identity(reward_contract_path) if reward_contract_path.is_file() else None
            ),
            "verl_runtime_contract": (
                _identity(runtime_contract_path) if runtime_contract_path.is_file() else None
            ),
            "guidance_bias_index": (
                _identity(guidance_bias_path) if guidance_bias_path.is_file() else None
            ),
            "spatial_ood_audit": (
                _identity(spatial_ood_path) if spatial_ood_path.is_file() else None
            ),
            "verl_tool_config": (
                _identity(tool_config_path) if tool_config_path.is_file() else None
            ),
            "verl_agent_loop_config": (
                _identity(agent_loop_config_path) if agent_loop_config_path.is_file() else None
            ),
        },
        "input_snapshots": input_snapshots,
        "training_split_policy": {
            "winter_challenge_used_for_checkpoint_selection": True,
            "train_manifest_is_city_day_purged": True,
            "spatial_holdout_cities_purged_from_train": True,
            "train_spatial_holdout_city_overlap": (
                spatial_ood.get("integrity", {}).get("train_heldout_city_overlap")
            ),
            "train_manifest": (
                "data/interim/train_without_winter_or_spatial_holdout.json"
            ),
            "selection_dataset": "selection.parquet",
            "train_challenge_case_overlap": len(case_overlap),
            "train_challenge_forecast_city_day_overlap": len(city_day_overlap),
        },
        "outputs": outputs,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"-> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
