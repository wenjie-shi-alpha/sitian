#!/usr/bin/env python3
"""Build the veRL-native tool registry from the executable ForecastEnv.

The current veRL schema model intentionally accepts only shallow parameter
properties.  We therefore preserve names, primitive types, enums and public
descriptions here; the exact per-case forecast object and dates remain in the
task brief/example and are enforced by ForecastEnv at submission time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.case import CaseBundle  # noqa: E402
from sitian.env import EnvConfig, ForecastEnv, HARNESS_VERSION  # noqa: E402


def _shallow_parameters(name: str, parameters: dict) -> dict:
    properties = {}
    for field, value in parameters.get("properties", {}).items():
        prop = {"type": value.get("type", "string")}
        if value.get("enum") is not None:
            prop["enum"] = value["enum"]
        description = value.get("description")
        if name == "submit_forecast" and field == "forecast":
            description = (
                "结构化预报对象；严格按任务简报 forecast_format 组织。forecast 顶层必须包含"
                "当前任务的 issue_date、region，以及 pm25_lo、pm25_hi 数字列表；"
                "多污染物任务还须包含 pm10_lo、pm10_hi、o3_lo、o3_hi。每列长度等于 horizon，"
                "第 k 项对应起报日后第 k 天；不要把列再包进 daily。"
                "提交 80% 浓度区间，lo <= hi；AQI等级、首要污染物、process 由环境派生，无需提交。"
                "evidence 引用工具返回的"
                "evidence_ref、JSON Pointer field 与精确 value。"
            )
        if description:
            prop["description"] = description
        properties[field] = prop
    return {
        "type": "object",
        "properties": properties,
        "required": list(parameters.get("required", [])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=None)
    parser.add_argument("--out", type=Path,
                        default=Path("configs/verl/sitian_tools.yaml"))
    parser.add_argument("--historical-index", type=str,
                        help="same FH_HISTORICAL_CASE_INDEX artifact used by rollout workers")
    parser.add_argument("--methods", type=str, help="same FH_FORECAST_METHODS library used by rollout workers")
    args = parser.parse_args()
    if args.case is None:
        candidates = sorted((REPO_ROOT / "cases" / "national" / "val").glob("*"))
        args.case = next((p for p in candidates if (p / "evidence.json").is_file()), None)
    if args.case is None:
        parser.error("no evidence-enabled national case is available; pass --case")

    cfg = EnvConfig()
    if args.historical_index:
        cfg.historical_cases_path = args.historical_index
    if args.methods:
        cfg.forecast_methods_path = args.methods
    env = ForecastEnv(CaseBundle.load(args.case), cfg)
    env.reset()
    tools = []
    for spec in env.tool_specs("openai"):
        function = spec["function"]
        name = function["name"]
        tools.append({
            "class_name": "sitian.integrations.verl_runtime.SitianForecastTool",
            "config": {"type": "native", "tool_name": name,
                       "harness_resources": env.resource_identity},
            "tool_schema": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": function["description"],
                    "parameters": _shallow_parameters(name, function["parameters"]),
                },
            },
        })
    document = {
        "contract": {
            "generator": "scripts/build_verl_tool_config.py",
            "source": "ForecastEnv.tool_specs(openai)",
            "harness_version": HARNESS_VERSION,
            "harness_resources": env.resource_identity,
            "note": "case-specific nested submit schema remains in the task brief and executable validator",
        },
        "tools": tools,
    }
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"tools={len(tools)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
