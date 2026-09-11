#!/usr/bin/env python3
"""Finish derivation, attachment, and the fair tabular baseline after downloads.

This is designed to run in a persistent tmux session.  It waits for the required
download sessions started during construction, reruns each idempotent fetcher
to fill transient gaps, derives all 401 issue dates, enforces the coverage
gate, attaches audited slices to cases, and finally rebuilds the reward-v0.7.9
same-information tabular baseline and frozen-policy readiness probe.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from sitian.open_evidence import DEFAULT_EVIDENCE_ROOT

REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_SESSIONS = (
    "sitian_hybrid_gfs", "sitian_hybrid_ifs", "sitian_static",
    "sitian_cams_aerosol", "sitian_cams_aerosol1",
    "sitian_cams_aerosol2", "sitian_cams_aerosol3",
    "sitian_cams_trace0", "sitian_cams_trace1",
    "sitian_cams_trace2", "sitian_cams_trace3",
    "sitian_ee0", "sitian_ee1", "sitian_ee2", "sitian_firms",
    # Backward-compatible names used by the first open-evidence run.
    "sitian_nwp", "sitian_nwp_ifs", "sitian_cams_local_gases_0",
    "sitian_cams_local_gases_1", "sitian_cams_local_gases_2",
    "sitian_cams_gee_0", "sitian_cams_gee_1", "sitian_cams_gee_2",
    "sitian_firms_nrt",
)


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _session_alive(name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", name], capture_output=True,
                          check=False).returncode == 0


def _run(command: list[str], env: dict[str, str]) -> bool:
    print(f"[{_stamp()}] RUN {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    print(f"[{_stamp()}] EXIT {result.returncode} {' '.join(command)}", flush=True)
    return result.returncode == 0


def _run_capture(command: list[str], output: Path, env: dict[str, str]) -> bool:
    """Capture a generated artifact atomically so failures cannot certify stale config."""
    temporary = output.with_suffix(output.suffix + ".tmp")
    print(f"[{_stamp()}] RUN {' '.join(command)} > {output}", flush=True)
    with temporary.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command, cwd=REPO_ROOT, env=env, stdout=handle, check=False
        )
    if result.returncode == 0:
        os.replace(temporary, output)
    else:
        temporary.unlink(missing_ok=True)
    print(f"[{_stamp()}] EXIT {result.returncode} {' '.join(command)}", flush=True)
    return result.returncode == 0


def _run_parallel(commands: list[list[str]], env: dict[str, str]) -> bool:
    processes = []
    for command in commands:
        print(f"[{_stamp()}] RUN {' '.join(command)}", flush=True)
        processes.append((command, subprocess.Popen(command, cwd=REPO_ROOT, env=env)))
    success = True
    for command, process in processes:
        code = process.wait()
        print(f"[{_stamp()}] EXIT {code} {' '.join(command)}", flush=True)
        success &= code == 0
    return success


def _wait_for_sessions(names: tuple[str, ...], poll_seconds: int) -> None:
    previous = None
    while True:
        active = tuple(name for name in names if _session_alive(name))
        if active != previous:
            print(f"[{_stamp()}] waiting_for={list(active)}", flush=True)
            previous = active
        if not active:
            return
        time.sleep(max(10, poll_seconds))


def _deliver_report(env: dict[str, str]) -> bool:
    report_builder = str(
        Path.home() / ".codex/plugins/cache/openai-curated-remote/data-analytics" /
        "0.2.9-13ceeea1f599/skills/build-report/scripts/deliver_portable_artifact.mjs"
    )
    report_verifier = str(
        Path.home() / ".codex/plugins/cache/openai-curated-remote/data-analytics" /
        "0.2.9-13ceeea1f599/skills/build-report/scripts/verify_portable_artifact.mjs"
    )
    artifact = "docs/reports/expert_forecast_requirements/artifact.json"
    report_html = "docs/reports/expert_forecast_requirements/report.html"
    if not _run(["python3", "scripts/build_expert_harness_report.py"], env):
        return False
    if not (Path(report_builder).is_file() and Path(report_verifier).is_file()):
        # The portable HTML packager is an optional external plugin.  Its
        # absence must not abort the certified data/probe/training chain: the
        # source-backed artifact.json and appendix.md above are the auditable
        # deliverables; the HTML can be repackaged later from artifact.json.
        print(f"[{_stamp()}] WARN report packager missing ({report_builder}); "
              "artifact.json/appendix.md written, report.html left unchanged", flush=True)
        return True
    # The bundled reader has a known classic-scrollbar 100vw edge case.
    # Package structurally with the official deliverer, apply the one-rule
    # deterministic fix, then run the official two-viewport browser verifier.
    packaging_env = dict(env)
    packaging_env["CHROMIUM_EXECUTABLE_PATH"] = "/tmp/sitian-no-browser"
    return (
        _run(["node", report_builder, "--input", artifact,
              "--output", report_html], packaging_env)
        and _run(["python3", "scripts/fix_portable_report_layout.py", report_html], env)
        and _run(["node", report_verifier, "--html", report_html,
                  "--artifact", artifact], env)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--download-attempts", type=int, default=3)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--skip-tabular", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--skip-training-smoke", action="store_true")
    parser.add_argument(
        "--reuse-probe", action="store_true",
        help=("with --resume-from-probe: do not rerun probe_model.py when the v079 probe "
              "artifact already exists; rerun only the comparisons/preflight/training tail"),
    )
    parser.add_argument(
        "--resume-from-probe", action="store_true",
        help=("skip download verification, derivation, attachment, audits and the "
              "tabular rebuild; only valid right after a complete pass of those steps "
              "under the same evidence contract (e.g. after a tool-view-only change)"),
    )
    args = parser.parse_args()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env.setdefault("SITIAN_GFS_ROOT", "https://storage.googleapis.com/global-forecast-system")
    env.setdefault("FH_BASE_URL", "http://127.0.0.1:8000/v1")
    env.setdefault("FH_MODEL", "Qwen/Qwen3-8B-AWQ")
    manifest_path = args.root / "manifests" / "open_evidence_v1.json"
    source_args = ["--manifest", str(manifest_path), "--root", str(args.root)]

    if not args.no_wait:
        _wait_for_sessions(DOWNLOAD_SESSIONS, args.poll_seconds)

    # Refuse to certify a large, expensive rebuild if the executable Harness,
    # schema, reward, or leakage contracts have regressed since launch.
    if not _run(["python3", "-m", "pytest", "-q"], env):
        return 1

    if args.resume_from_probe:
        required = [
            args.root / "manifests" / "open_evidence_v1_coverage.json",
            REPO_ROOT / "data/interim/open_evidence_coverage.json",
            REPO_ROOT / "data/interim/eval_tabular_open_evidence_v079.json",
            REPO_ROOT / "data/interim/verl_runtime_contract.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            print(f"[{_stamp()}] BLOCKED resume-from-probe requires {missing}", flush=True)
            return 1
        coverage = json.loads(required[0].read_text(encoding="utf-8"))
        if not coverage.get("ready_for_full_attachment"):
            print(f"[{_stamp()}] BLOCKED resume-from-probe: coverage gate is false", flush=True)
            return 1
        print(f"[{_stamp()}] resume-from-probe: data/tabular steps reused from the "
              "previous certified pass", flush=True)
        return _probe_and_train(args, env)

    downloads = {
        "nwp": ["python3", "scripts/fetch_open_nwp.py", "--source", "both",
                "--product", "both", "--workers", "12", "--verify", *source_args],
        "cams_aerosol": ["python3", "scripts/fetch_open_cams.py", "--kind", "aerosol",
                         "--aerosol-chunk-days", "32", *source_args],
        "cams_trace_gases": ["python3", "scripts/fetch_open_cams.py", "--kind",
                             "trace_gases", "--trace-gas-chunk-days", "8", *source_args],
        # Total-column and model-level-137 trajectories are separate required
        # channels. Neither is permitted to fill missing values in the other.
        "cams_columns": ["python3", "scripts/fetch_cams_earthengine.py", *source_args],
        "firms_standard": ["python3", "scripts/fetch_open_firms.py", "--processing",
                           "standard", "--sensor", "both", *source_args],
        "firms_nrt": ["python3", "scripts/fetch_open_firms.py", "--processing", "nrt",
                      "--sensor", "both", "--workers", "3", *source_args],
        "static": ["python3", "scripts/fetch_open_static.py", "--root", str(args.root),
                   "--kind", "both"],
    }
    pending = dict(downloads)
    for attempt in range(1, args.download_attempts + 1):
        if not pending:
            break
        print(f"[{_stamp()}] idempotent download verification attempt {attempt}; "
              f"pending={list(pending)}", flush=True)
        for name, command in list(pending.items()):
            if _run(command, env):
                pending.pop(name)
    if pending:
        print(f"[{_stamp()}] BLOCKED download verification failed: {list(pending)}", flush=True)
        _run(["python3", "scripts/audit_open_evidence_coverage.py", *source_args], env)
        return 1

    # The evidence extra includes pygrib in the project environment. A separate
    # geoscience interpreter remains an explicit override for existing setups.
    synoptic_python = env.get(
        "SITIAN_SYNOPTIC_PYTHON", sys.executable
    )
    if not Path(synoptic_python).is_file():
        print(f"[{_stamp()}] BLOCKED synoptic Python missing: {synoptic_python}",
              flush=True)
        return 1
    if not _run([synoptic_python, "-c", "import pygrib"], env):
        print(f"[{_stamp()}] BLOCKED pygrib unavailable in {synoptic_python}",
              flush=True)
        return 1
    synoptic_commands = [
        [synoptic_python, "scripts/derive_open_synoptic.py", "--num-shards", "4",
         "--shard-index", str(index), "--skip-existing", *source_args]
        for index in range(4)
    ]
    if not _run_parallel(synoptic_commands, env):
        return 1
    derivations = [
        ["python3", "scripts/derive_spatial_observations.py", *source_args],
        ["python3", "scripts/derive_open_composition.py", "--skip-existing", *source_args],
        ["python3", "scripts/derive_open_fires.py", "--skip-existing", *source_args],
        # cfgrib (GFS orography) lives in the same geoscience env as pygrib.
        [synoptic_python, "scripts/derive_static_context.py", "--root", str(args.root)],
        ["python3", "scripts/audit_open_evidence_coverage.py", *source_args],
    ]
    if not all(_run(command, env) for command in derivations):
        return 1
    coverage_path = args.root / "manifests" / "open_evidence_v1_coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if not coverage.get("ready_for_full_attachment"):
        print(f"[{_stamp()}] BLOCKED derived coverage gate is false", flush=True)
        return 1

    if not _run(["python3", "scripts/attach_open_evidence.py", "--derived-root",
                 str(args.root / "derived" / "by_issue")], env):
        return 1
    if not _run(["python3", "scripts/audit_open_evidence_coverage.py", *source_args], env):
        return 1
    if not _run(["python3", "scripts/audit_open_evidence_coverage.py", *source_args,
                 "--out", "data/interim/open_evidence_coverage.json"], env):
        return 1
    if not _run(["python3", "scripts/audit_national_data.py"], env):
        return 1
    if not _run(["python3", "scripts/build_winter_challenge.py"], env):
        return 1
    # Freeze one full city per pre-defined pollution-regime cluster before any
    # train-derived calibration or baseline is rebuilt.  This produces both
    # the joint train manifest and genuinely unseen-city evaluation slices.
    if not _run(["python3", "scripts/build_spatial_ood.py"], env):
        return 1
    # The main offline comparison freezes historical guidance-error signals to
    # the purged training pool.  Allowing earlier val/test truth to update later
    # evaluation episodes would be operationally time-safe but would give the
    # agent an online-learning advantage absent from the frozen tabular baseline.
    if not _run(["python3", "scripts/build_guidance_bias.py"], env):
        return 1
    if not _run(["python3", "scripts/audit_expert_evidence_contract.py"], env):
        return 1
    if not _run(["python3", "scripts/audit_reward_contract.py"], env):
        return 1
    # Rebuild and attest the exact native-tool surface before any formal probe
    # or preflight consumes it.  Otherwise a stale runtime artifact can block a
    # healthy stack—or, worse, certify code/config that no longer matches.
    if not _run(["python3", "scripts/build_verl_tool_config.py"], env):
        return 1
    resolved_config = REPO_ROOT / "data/interim/verl_resolved_smoke_config.yaml"
    config_env = dict(env)
    config_env["SITIAN_CONFIG_ONLY"] = "1"
    if not _run_capture(
        ["bash", "scripts/run_verl_smoke.sh"], resolved_config, config_env
    ):
        return 1
    if not _run([
        str(Path(env.get("SITIAN_VERL_ROOT", str(REPO_ROOT / ".local/verl-upstream"))) / ".venv/bin/python"),
        "scripts/audit_verl_runtime_contract.py",
    ], env):
        return 1
    if not args.skip_tabular and not _run(
            ["python3", "scripts/eval_tabular_baseline.py",
             "--out", "data/interim/eval_tabular_open_evidence_v079.json"], env):
        return 1
    return _probe_and_train(args, env)


def _probe_and_train(args, env: dict[str, str]) -> int:
    if not args.skip_probe:
        probe_path = "data/interim/probe_open_evidence_v079_n50_g8.json"
        if args.reuse_probe and (REPO_ROOT / probe_path).is_file():
            print(f"[{_stamp()}] reuse-probe: {probe_path} kept, probe_model.py not rerun",
                  flush=True)
        elif not _run(
                ["python3", "scripts/probe_model.py", "--split", "val", "--n", "50",
                 "--stratified", "--group-size", "8", "--temperature", "1.0",
                 "--thinking", "--out", probe_path],
                env):
            return 1
        comparison_commands = [
            ["python3", "scripts/analyze_group_signal.py", "--probe", probe_path,
             "--out", "data/interim/group_signal_diagnostic_v079.json"],
            ["python3", "scripts/compare_probe_clustered.py", "--probe", probe_path,
             "--baseline", "guidance", "--out",
             "data/interim/compare_v079_frozen_vs_guidance.json"],
            ["python3", "scripts/compare_probe_clustered.py", "--probe", probe_path,
             "--baseline", "tabular", "--tabular",
             "data/interim/eval_tabular_open_evidence_v079.json", "--tabular-split", "val",
             "--out", "data/interim/compare_v079_frozen_vs_tabular.json"],
            ["python3", "scripts/compare_probe_hard_metrics.py", "--probe", probe_path,
             "--baseline", "guidance", "--out",
             "data/interim/compare_v079_frozen_vs_guidance_hard.json"],
            ["python3", "scripts/compare_probe_hard_metrics.py", "--probe", probe_path,
             "--baseline", "tabular", "--tabular",
             "data/interim/eval_tabular_open_evidence_v079.json", "--tabular-split", "val",
             "--out", "data/interim/compare_v079_frozen_vs_tabular_hard.json"],
            ["python3", "scripts/audit_reward_weight_sensitivity.py",
             "--probe", probe_path,
             "--tabular", "data/interim/eval_tabular_open_evidence_v079.json",
             "--out", "data/interim/reward_weight_sensitivity_v079.json"],
        ]
        if not all(_run(command, env) for command in comparison_commands):
            return 1
        # A frozen model cannot graduate and the local trainable stack may still
        # be absent, so rl_preflight intentionally returns non-zero here.  The
        # artifact is nevertheless required: it states the remaining blockers.
        _run(
            ["python3", "scripts/rl_preflight.py", "--probe", probe_path,
             "--min-smoke-steps", env.get("SITIAN_SMOKE_STEPS", "50"),
             "--guidance-comparison", "data/interim/compare_v079_frozen_vs_guidance.json",
             "--tabular-comparison", "data/interim/compare_v079_frozen_vs_tabular.json",
             "--tabular-hard-comparison",
             "data/interim/compare_v079_frozen_vs_tabular_hard.json",
             "--out", "data/interim/rl_preflight.json"],
            env,
        )
        if not _run(
                ["python3", "scripts/analyze_expert_forecast_requirements.py",
                 "--probe", probe_path, "--out", "data/interim/efr.json"], env):
            return 1
        if not _deliver_report(env):
            return 1

        # A trainable smoke is only meaningful after all frozen-policy
        # preflight gates except the smoke itself pass.  This prevents spending
        # GPU time on an all-zero/invalid policy and keeps SFT conditional.
        readiness = json.loads(
            (REPO_ROOT / "data/interim/rl_preflight.json").read_text(encoding="utf-8")
        )
        preflight = readiness.get("preflight") or {}
        prerequisites = {
            key: bool(value.get("pass")) for key, value in preflight.items()
            if key != "trainable_stack_and_hardware_smoke"
        }
        train_ready = bool(prerequisites) and all(prerequisites.values())
        print(f"[{_stamp()}] training_smoke_prerequisites={prerequisites} ready={train_ready}",
              flush=True)
        if train_ready and not args.skip_training_smoke:
            _wait_for_sessions(
                ("sitian_verl_env", "sitian_train_model_download"), args.poll_seconds
            )
            # The AWQ server has completed its only authorized role (frozen
            # evaluation).  Release the single GPU before starting current-policy RL.
            if _session_alive("sitian_vllm_probe_v04"):
                subprocess.run(
                    ["tmux", "kill-session", "-t", "sitian_vllm_probe_v04"], check=True
                )
                print(f"[{_stamp()}] stopped frozen AWQ rollout service", flush=True)
            training_commands = [
                ["python3", "scripts/prepare_verl_dataset.py"],
                ["bash", "scripts/run_verl_smoke.sh"],
            ]
            if not all(_run(command, env) for command in training_commands):
                return 1
            # Recompute the same preflight/report with the real smoke artifact.
            _run(
                ["python3", "scripts/rl_preflight.py", "--probe", probe_path,
             "--min-smoke-steps", env.get("SITIAN_SMOKE_STEPS", "50"),
                 "--guidance-comparison", "data/interim/compare_v079_frozen_vs_guidance.json",
                 "--tabular-comparison", "data/interim/compare_v079_frozen_vs_tabular.json",
                 "--tabular-hard-comparison",
                 "data/interim/compare_v079_frozen_vs_tabular_hard.json",
                 "--out", "data/interim/rl_preflight.json"],
                env,
            )
            if not _run(
                    ["python3", "scripts/analyze_expert_forecast_requirements.py",
                     "--probe", probe_path, "--out", "data/interim/efr.json"], env):
                return 1
            if not _deliver_report(env):
                return 1
    print(f"[{_stamp()}] COMPLETE open evidence attached and evaluated", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
