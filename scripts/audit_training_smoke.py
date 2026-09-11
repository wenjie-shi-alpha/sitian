#!/usr/bin/env python3
"""Fail-closed audit for the real veRL LoRA training smoke."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.provenance import file_identity, score_implementation_matches  # noqa: E402
from sitian.scoring import reward_spec  # noqa: E402
from scripts.prepare_verl_dataset import _input_snapshot  # noqa: E402


LOSS_PATTERN = re.compile(
    r"['\"]?(actor/(?:pg_loss|entropy_loss|kl_loss)|critic/vf_loss)['\"]?\s*[:=]\s*"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_identity(path: Path) -> dict:
    files = sorted(p for p in path.rglob("*") if p.is_file())
    adapter = [p for p in files if "actor" in p.parts and (
        "model_world_size" in p.name or p.name == "adapter_model.safetensors"
    )]
    return {
        "path": str(path),
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "adapter_files": [{"path": str(p), "sha256": _sha(p), "bytes": p.stat().st_size}
                          for p in adapter],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=50)
    parser.add_argument("--expected-rollout-n", type=int, default=8)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--verl-root", type=Path, required=True)
    parser.add_argument("--mask-audit", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/interim/training_smoke.json"))
    args = parser.parse_args()

    log = args.log.read_text(encoding="utf-8", errors="replace")
    mask = json.loads(args.mask_audit.read_text(encoding="utf-8"))
    dataset_manifest_path = (args.dataset_manifest if args.dataset_manifest.is_absolute()
                             else REPO_ROOT / args.dataset_manifest)
    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    first_path = args.checkpoint_dir / f"global_step_{args.expected_steps}"
    reload_path = args.checkpoint_dir / f"global_step_{args.expected_steps + 1}"
    first = _checkpoint_identity(first_path) if first_path.is_dir() else {"path": str(first_path)}
    reloaded = _checkpoint_identity(reload_path) if reload_path.is_dir() else {"path": str(reload_path)}
    first_hashes = [row["sha256"] for row in first.get("adapter_files", [])]
    reload_hashes = [row["sha256"] for row in reloaded.get("adapter_files", [])]
    loss_values = [float(match.group(2)) for match in LOSS_PATTERN.finditer(log)]
    sync_events = len(re.findall(r"update_weights done|timing_s/update_weights", log))
    resume_evidence = bool(re.search(
        rf"Resuming from .*global_step_{args.expected_steps}\b", log
    ))
    sampling_match = re.search(
        r"phase=train .*\brollout_n=(\d+) .*\btrain_batch_size=(\d+) "
        r".*\bppo_mini_batch_size=(\d+) .*\btemperature=([0-9.]+)",
        log,
    )
    observed_sampling = {
        "rollout_group_size": int(sampling_match.group(1)) if sampling_match else None,
        "train_batch_size": int(sampling_match.group(2)) if sampling_match else None,
        "ppo_mini_batch_size": int(sampling_match.group(3)) if sampling_match else None,
        "temperature": float(sampling_match.group(4)) if sampling_match else None,
    }
    # Old audits used trajectory counts for this prompt-count veRL option.
    # Preserve their recorded convention while checking new runs explicitly.
    prompt_units = 'ppo_mini_batch_unit=prompts' in log
    observed_sampling['ppo_mini_batch_unit'] = 'prompts' if prompt_units else 'legacy_unannotated'
    observed_sampling['effective_optimizer_trajectories'] = (
        observed_sampling['ppo_mini_batch_size'] * observed_sampling['rollout_group_size']
        if sampling_match else None
    )
    batch_contract_matches = bool(sampling_match) and (
        observed_sampling['ppo_mini_batch_size'] == observed_sampling['train_batch_size']
        if prompt_units else
        observed_sampling['ppo_mini_batch_size']
        == observed_sampling['rollout_group_size'] * observed_sampling['train_batch_size']
    )
    commit = subprocess.check_output(
        ["git", "-C", str(args.verl_root), "rev-parse", "HEAD"], text=True
    ).strip()
    packages = {}
    for name in ("verl", "peft", "ray", "torch", "vllm"):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        packages[name] = {"available": version is not None, "version": version}

    purged_train = file_identity(
        REPO_ROOT / "data/interim/train_without_winter_or_spatial_holdout.json",
        relative_to=REPO_ROOT,
    )
    dataset_train = (dataset.get("source_manifests") or {}).get("train") or {}
    split_policy = dataset.get("training_split_policy") or {}

    def identity_matches(identity: dict | None) -> bool:
        if not identity or not identity.get("path"):
            return False
        path = REPO_ROOT / identity["path"]
        if not path.is_file():
            return False
        current = file_identity(path, relative_to=REPO_ROOT)
        return (current.get("sha256"), current.get("bytes")) == (
            identity.get("sha256"), identity.get("bytes")
        )

    parquet_integrity = True
    for filename, identity in (dataset.get("outputs") or {}).items():
        path = dataset_manifest_path.parent / filename
        parquet_integrity &= (
            path.is_file()
            and path.stat().st_size == identity.get("bytes")
            and _sha(path) == identity.get("sha256")
        )

    snapshot_integrity = True
    for split in ("train", "val", "challenge_winter"):
        source = (dataset.get("source_manifests") or {}).get(split) or {}
        manifest_path = REPO_ROOT / str(source.get("path", ""))
        if not identity_matches(source):
            snapshot_integrity = False
            continue
        paths = [REPO_ROOT / value for value in json.loads(
            manifest_path.read_text(encoding="utf-8")
        )]
        current = _input_snapshot(paths)
        recorded = (dataset.get("input_snapshots") or {}).get(split) or {}
        snapshot_integrity &= (
            current.get("sha256") == recorded.get("sha256")
            and current.get("files") == recorded.get("files")
            and current.get("bytes") == recorded.get("bytes")
        )

    gates = dataset.get("gate_artifacts") or {}
    gate_integrity_by_artifact = {
        name: identity_matches(gates.get(name))
        for name in (
            "open_evidence_coverage", "expert_evidence_contract",
            "reward_contract_audit", "verl_runtime_contract", "guidance_bias_index",
            "spatial_ood_audit",
            "verl_tool_config", "verl_agent_loop_config",
        )
    }
    gate_integrity = all(gate_integrity_by_artifact.values())
    reward_contract_identity = gates.get("reward_contract_audit") or {}
    reward_contract_path = REPO_ROOT / str(reward_contract_identity.get("path", ""))
    reward_contract = (
        json.loads(reward_contract_path.read_text(encoding="utf-8"))
        if reward_contract_path.is_file() else {}
    )
    reward_implementation_integrity = score_implementation_matches(
        reward_contract, REPO_ROOT
    )

    checks = {
        "minimum_steps_reached": reload_path.is_dir(),
        "rollout_policy_sync_verified": sync_events >= 2,
        "tool_tokens_loss_masked": bool(mask.get("pass")),
        "checkpoint_reload_verified": (
            resume_evidence and bool(first_hashes) and bool(reload_hashes)
        ),
        "checkpoint_weights_changed_after_reload_step": (
            bool(first_hashes) and bool(reload_hashes) and first_hashes != reload_hashes
        ),
        "finite_losses": bool(loss_values) and all(math.isfinite(value) for value in loss_values),
        "sampling_contract_matches_probe_regime": (
            observed_sampling["rollout_group_size"] == args.expected_rollout_n
            and batch_contract_matches
            and observed_sampling["temperature"] == 1.0
        ),
        "current_reward_identity": dataset.get("reward", {}).get("version") == reward_spec()["version"]
            and dataset.get("reward", {}).get("config_sha256") == reward_spec()["config_sha256"],
        "leakage_safe_dataset": (
            dataset.get("expert_corpus_used_as_training_input") is False
            and dataset.get("truth_exposed_in_prompt_or_ground_truth_field") is False
        ),
        "winter_challenge_and_spatial_holdout_purged_from_train": (
            split_policy.get("train_manifest_is_city_day_purged") is True
            and split_policy.get("spatial_holdout_cities_purged_from_train") is True
            and split_policy.get("train_spatial_holdout_city_overlap") == 0
            and dataset_train.get("path") == purged_train.get("path")
            and dataset_train.get("sha256") == purged_train.get("sha256")
            and split_policy.get("winter_challenge_used_for_checkpoint_selection") is True
            and split_policy.get("selection_dataset") == "selection.parquet"
            and split_policy.get("train_challenge_case_overlap") == 0
            and split_policy.get("train_challenge_forecast_city_day_overlap") == 0
        ),
        "training_parquets_match_manifest": parquet_integrity,
        "case_inputs_and_hidden_truth_match_manifest": snapshot_integrity,
        "evidence_gate_artifacts_match_manifest": gate_integrity,
        "reward_and_schema_implementations_match_audited_hash": (
            reward_implementation_integrity
        ),
    }
    report = {
        "artifact_type": "trainable_policy_smoke",
        "smoke_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "success": all(checks.values()),
        "steps": args.expected_steps + int(reload_path.is_dir()),
        "model": {"path": str(args.model.resolve()), "config": file_identity(args.model / "config.json")},
        "reward": reward_spec(),
        "verl": {"root": str(args.verl_root.resolve()), "commit": commit},
        "packages": packages,
        "dataset_manifest": file_identity(args.dataset_manifest),
        "log": file_identity(args.log),
        "mask_audit": file_identity(args.mask_audit),
        "checkpoints": {"trained": first, "reloaded_one_step": reloaded},
        "observations": {
            "weight_sync_log_events": sync_events,
            "finite_loss_values": len(loss_values),
            "loss_min": min(loss_values) if loss_values else None,
            "loss_max": max(loss_values) if loss_values else None,
            "gate_integrity_by_artifact": gate_integrity_by_artifact,
            "training_sampling_contract": observed_sampling,
        },
        "checks": checks,
        # Convenience fields consumed by rl_preflight.py.
        "rollout_policy_sync_verified": checks["rollout_policy_sync_verified"],
        "tool_tokens_loss_masked": checks["tool_tokens_loss_masked"],
        "checkpoint_reload_verified": checks["checkpoint_reload_verified"],
        "finite_losses": checks["finite_losses"],
        "rollout_group_size": observed_sampling["rollout_group_size"],
        "sampling_temperature": observed_sampling["temperature"],
    }
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"-> {out}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
