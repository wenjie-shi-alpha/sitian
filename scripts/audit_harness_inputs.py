#!/usr/bin/env python3
"""Exercise visible-input tools on an explicit case population, without training.

This is an input/tool audit, not a training-readiness certificate. Raw provenance,
retrieval, runtime replay, split isolation and forecast skill require other checks.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sitian.asset_quality import build_asset_quality  # noqa: E402
from sitian.case import CaseBundle  # noqa: E402
from sitian.data_contract import valid_concentration  # noqa: E402
from sitian.env import HARNESS_VERSION, EnvConfig, ForecastEnv  # noqa: E402
from sitian.paths import resolve_case_dir  # noqa: E402
from sitian.provenance import file_identity  # noqa: E402


def audit(manifests):
    paths = sorted({str(resolve_case_dir(row)) for manifest in manifests
                    for row in json.loads(Path(manifest).read_text())})
    warnings, tool_calls, failures, publication_unknown = Counter(), Counter(), [], Counter()
    seen_ids = set()
    for n, path in enumerate(paths, 1):
        try:
            bundle = CaseBundle.load(path)
            if bundle.case_id in seen_ids:
                raise ValueError("duplicate case_id across distinct paths")
            seen_ids.add(bundle.case_id)
            quality = build_asset_quality(bundle)
            warnings.update({code: 1 for code in {w["code"] for w in quality["warnings"]}})
            for kind in ("observations", "guidance"):
                if any(row["publication_status"] != "declared" for row in quality[kind].values()):
                    publication_unknown[kind] += 1
            required = list(bundle._TRUTH_KEYS) if bundle.meta.get("multi_pollutant") else ["pm25_avg"]
            if not all(valid_concentration((bundle.truth or {}).get("daily", {}).get(day, {}).get(key))
                       for day in bundle.forecast_dates() for key in required):
                failures.append({"case": path, "check": "truth", "reason": "missing_or_invalid_required_truth"})
            env = ForecastEnv(bundle, EnvConfig(max_steps=30, guidance_bias_path=None,
                                                historical_cases_path=None, forecast_methods_path=None))
            env.reset()
            actions = [("list_data_assets", {}), ("get_assessment", {}), ("get_diagnostics", {}),
                       ("get_model_guidance", {}), ("get_process_evidence", {}),
                       ("get_synoptic_evidence", {"detail": "full"}),
                       ("get_pollution_evidence", {"detail": "full"}),
                       ("retrieve_forecast_methods", {"query": "污染持续 扩散条件 逆温 冷空气"})]
            actions.extend(("get_observations", {"pollutant": p, "last_hours": 72, "stride": 1})
                           for p in bundle.observations)
            registered = {s["function"]["name"] for s in env.tool_specs("openai")}
            for name, args in actions:
                if name not in registered:
                    continue
                result, _, _, _ = env.step({"name": name, "args": args})
                tool_calls[name] += 1
                json.dumps(result, allow_nan=False)
                if not result["ok"]:
                    failures.append({"case": path, "check": name, "reason": result["content"].get("error")})
        except (ValueError, TypeError, KeyError, OSError) as exc:
            failures.append({"case": path, "check": "case_input", "reason": str(exc)})
        if n % 500 == 0:
            print(json.dumps({"processed": n, "total": len(paths), "failures": len(failures)}), flush=True)
    return {"audit": "visible-input-tools-v1", "harness_version": HARNESS_VERSION,
            "cases": len(paths), "tool_calls": dict(tool_calls), "failures": failures,
            "cases_by_warning": dict(warnings), "cases_with_unknown_publication": dict(publication_unknown),
            "execution_passed": not failures, "training_started": False,
            "scope_limits": ["not a training-readiness certificate", "raw data not revalidated",
                             "historical retrieval, split isolation and native runtime not tested here",
                             "tool execution does not validate forecast skill or scientific adequacy"],
            "provenance": {"manifests": [file_identity(Path(p)) for p in manifests],
                           "script": file_identity(Path(__file__)),
                           "implementation": [file_identity(p) for p in sorted((ROOT / "src/sitian").glob("*.py"))]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists; choose a new audit path")
    result = audit(args.manifests)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=1, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k not in {"provenance", "failures"}}))
    return 0 if result["execution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
