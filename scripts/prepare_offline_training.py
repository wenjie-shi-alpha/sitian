#!/usr/bin/env python3
"""Build fresh, versioned RL assets; certification and launch are separate checks."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sitian.case import CaseBundle
from sitian.env import ForecastEnv
from sitian.forecast_methods import BUILTIN_CARDS, METHODS_VERSION
from sitian.provenance import file_identity
from sitian.scoring import reward_spec
from build_historical_case_index import build_index
from build_verl_tool_config import _shallow_parameters
from prepare_verl_dataset import _row, _write
from rebuild_native_snapshot import write


def run(snapshot, out):
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=False)
    manifests = snapshot / "manifests"
    history = build_index(manifests / "train.json", [p for p in sorted(manifests.glob("*.json")) if p.stem not in {"all", "train"}])
    write(assets / "historical_cases.json", history)
    print(f"History index: {len(history['records'])}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/build_guidance_bias.py"), "--manifests", str(manifests / "train.json"),
                    "--out", str(assets / "guidance_bias.json")], cwd=ROOT, check=True)
    write(assets / "forecast_methods.json", {"contract_version": METHODS_VERSION, "cards": BUILTIN_CARDS})
    environment = {"FH_HISTORICAL_CASE_INDEX": str(assets / "historical_cases.json"),
                   "FH_GUIDANCE_BIAS_INDEX": str(assets / "guidance_bias.json"),
                   "FH_FORECAST_METHODS": str(assets / "forecast_methods.json"), "FH_METHOD_RETRIEVAL": "1",
                   "PYTHONPATH": str(ROOT / "src") + ":" + str(ROOT),
                   "UV_NO_SYNC": "1", "TOKENIZERS_PARALLELISM": "false"}
    os.environ.update(environment)
    write(out / "runtime_environment.json", environment)
    val = json.loads((manifests / "val.json").read_text())
    env = ForecastEnv(CaseBundle.load(val[0]))
    env.reset()
    tools = []
    for spec in env.tool_specs("openai"):
        f = spec["function"]
        tools.append({"class_name": "sitian.integrations.verl_runtime.SitianForecastTool",
            "config": {"type": "native", "tool_name": f["name"], "harness_resources": env.resource_identity},
            "tool_schema": {"type": "function", "function": {"name": f["name"], "description": f["description"],
                "parameters": _shallow_parameters(f["name"], f["parameters"])}}})
    (assets / "tools.yaml").write_text(yaml.safe_dump({"contract": {"harness_resources": env.resource_identity}, "tools": tools}, allow_unicode=True, sort_keys=False))
    counts = {}
    for manifest in sorted(manifests.glob("*.json")):
        if manifest.stem == "all": continue
        paths = json.loads(manifest.read_text())
        rows = []
        for i, path in enumerate(paths):
            row = _row(Path(path), manifest.stem, i, True)
            if row["extra_info"]["harness_resources"] != env.resource_identity:
                raise ValueError("row resource mismatch")
            if not set(row["extra_info"]["tool_selection"]) <= {t["config"]["tool_name"] for t in tools}:
                raise ValueError("tool union incomplete")
            rows.append(row)
            if (i+1)%500 == 0: print(f"Parquet {manifest.stem}: {i+1}/{len(paths)}", flush=True)
        _write(rows, assets / f"{manifest.stem}.parquet")
        counts[manifest.stem] = len(rows)
    plan = {
        "protocol": "offline-native-training-v1", "offline_publication_assumptions_accepted": True,
        "acceptance": "User explicitly approved offline training preparation under recorded latency assumptions on 2026-09-12.",
        "snapshot": str(snapshot), "resources": env.resource_identity, "reward": reward_spec(), "splits": counts,
        "latency_hours": {"cams": 10, "gfs": 5, "ifs": 8, "city_hourly_observations": 1, "regional_cmaq_naqp": 10, "firms_nrt": 6},
        "boundary": "available_at < issue_time at 08:00 Asia/Shanghai; equality is excluded",
        "historical_outcome_latency": "last target day + 1 day at 12:00 BJT; full horizon before query",
        "actual_publication_verified": False,
        "limitations": [
            "CAMS 3/6h native sampling; D1-D3 full native daily grids, D4 partial, D5 absent; no interpolation/extrapolation.",
            "CAMS O3 uses explicit ML137 and fixed 1.2 kg/m3 density proxy; instantaneous maximum is not O3_8h, no same-target O3 error is computed.",
            "tp accumulation semantics unresolved: native values retained, daily precipitation unavailable.",
            "GFS/IFS are existing derived city snapshots, not newly extracted native GRIB; sparse pressure levels cannot identify shallow inversion thickness/base.",
            "Mixed retrospective/NRT fire summaries and aggregate windows on the strict latency boundary are masked; raw fire detections are needed to enable this channel.",
            "CMAQ/NAQP PM2.5 retains original city/station averaging and project unit contract; source README does not independently establish units.",
            "Static maps and pre-2025-04 climatology remain explicitly labelled legacy derived background.",
            "Expert transcripts are archived inputs only; six common-sense hypothesis cards, zero reviewed JJJ_ATMO cards.",
            "No forecast skill, optimizer stability or operational publication certification is implied by CPU readiness checks."],
        "training_started": False, "requires_fresh_base_model": True, "resume_mode": "disable",
        "base_model": str(ROOT / "models/Qwen3-8B"), "n_gpus": 4,
        "provenance": {"snapshot_summary": file_identity(snapshot / "rebuild_summary.json"), "builder": file_identity(Path(__file__))},
    }
    write(out / "protocol.json", plan)
    print(json.dumps({"splits": counts, "tools": len(tools), "assets_built": True, "training_started": False}), flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--snapshot",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    run(a.snapshot.resolve(),a.out.resolve())
