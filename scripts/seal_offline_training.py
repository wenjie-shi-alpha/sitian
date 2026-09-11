#!/usr/bin/env python3
"""Certify checked offline artifacts, preserving publication and capability limits."""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from sitian.provenance import case_bundle_snapshot, case_bundle_snapshot_matches, file_identity
from sitian.scoring import reward_spec
from sitian.training_ready import verify_bundle
from rebuild_native_snapshot import write


def seal(out):
    if (out/"certificate.json").exists(): raise ValueError("already sealed; use a new bundle")
    protocol=json.loads((out/"protocol.json").read_text())
    reports={n:json.loads((out/"audits"/f"{n}.json").read_text()) for n in ("native_snapshot","input_tools","runtime","reward","reward_population")}
    snapshot=Path(protocol["snapshot"])
    paths=json.loads((snapshot/"manifests/all.json").read_text())
    current_snapshot=case_bundle_snapshot(paths,relative_to=ROOT)
    for name in ("native_snapshot", "input_tools", "reward_population"):
        if reports[name].get("case_snapshot") != current_snapshot:
            raise ValueError(f"stale or unbound {name} audit")
    identities = [reports["native_snapshot"]["audit_implementation"], *reports["native_snapshot"]["source_inputs"],
                  *reports["runtime"]["bound_files"], *reports["input_tools"]["provenance"]["implementation"],
                  reports["input_tools"]["provenance"]["script"], reports["reward"]["scoring_implementation"],
                  reports["reward"]["schema_implementation"], *reports["reward_population"]["implementation"]]
    for expected in identities:
        path=Path(expected["path"])
        actual=file_identity(path if path.is_absolute() else ROOT/path)
        if (expected["sha256"], expected["bytes"]) != (actual["sha256"], actual["bytes"]):
            raise ValueError(f"stale audit input/implementation: {path}")
    for name in ("historical_cases.json", "guidance_bias.json"):
        history=json.loads((out/"assets"/name).read_text())
        if not case_bundle_snapshot_matches(history["provenance"]["input_snapshot"], ROOT):
            raise ValueError(f"history index no longer matches its cases: {name}")
    if (reports["native_snapshot"]["counts"]["cases"] != len(paths)
            or reports["input_tools"]["cases"] != len(paths)
            or reports["native_snapshot"]["split_counts"] != protocol["splits"]
            or reports["runtime"]["rows"] != protocol["splits"]):
        raise ValueError("audit populations differ")
    result=subprocess.run([str(ROOT/".venv/bin/python"),"-m","pytest","-q","--disable-warnings"],cwd=ROOT,text=True,capture_output=True)
    (out/"audits/tests.log").write_text(result.stdout+result.stderr)
    write(out/"audits/tests.json",{"passed":result.returncode==0,"exit_code":result.returncode,"output":result.stdout.strip().splitlines()[-1:]})
    checks={"native_values_truth_time_and_splits":reports["native_snapshot"]["passed"],
            "all_truths_expressible_with_full_outcome_credit":reports["reward_population"]["passed"],
            "all_case_input_tools":reports["input_tools"]["execution_passed"],
            "real_verl_runtime_and_tokens":reports["runtime"]["passed"],
            "reward_contract":reports["reward"]["passed"],"regression_tests":result.returncode==0,
            "explicit_offline_assumptions_accepted":protocol["offline_publication_assumptions_accepted"] is True,
            "fresh_training_required":protocol["requires_fresh_base_model"] is True and protocol["resume_mode"]=="disable"}
    if not all(checks.values()): raise ValueError(f"readiness failed: {checks}")
    files=set((ROOT/"src/sitian").rglob("*.py"))
    files.update((ROOT/"tests").rglob("*.py"))
    files.update((ROOT/".local/verl-upstream/verl").rglob("*.py"))
    files.update((ROOT/"models/Qwen3-8B").glob("*"))
    files.update(out.rglob("*.json"));files.update(out.rglob("*.yaml"));files.update(out.rglob("*.parquet"))
    files.update(snapshot.glob("*.json"));files.update((snapshot/"manifests").glob("*.json"))
    files.update(ROOT/"scripts"/name for name in (
        "rebuild_native_snapshot.py","attach_open_evidence.py","audit_native_snapshot.py","audit_harness_inputs.py",
        "prepare_offline_training.py","prepare_verl_dataset.py","build_historical_case_index.py","build_guidance_bias.py",
        "build_verl_tool_config.py","audit_offline_runtime.py","audit_verl_runtime_contract.py","audit_trajectory_mask.py",
        "seal_offline_training.py","launch_offline_training.py","run_verl_smoke.sh","run_local_training_chain.py","local_env.sh","audit_reward_contract.py","audit_reward_population.py"))
    files.update((ROOT/"pyproject.toml",ROOT/".local/verl-upstream/uv.lock",ROOT/"configs/verl/sitian_agent_loop.yaml"))
    certificate={"status":"offline_training_ready","generated_utc":datetime.now(timezone.utc).isoformat(),
        "actual_publication_verified":False,"training_started":False,"checks":checks,"reward":reward_spec(),
        "resources":protocol["resources"],"splits":protocol["splits"],"limitations":protocol["limitations"],
        "case_snapshot":current_snapshot,
        "files":[file_identity(p,relative_to=ROOT) for p in sorted(files) if p.is_file()],
        "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "git_worktree_diff_recorded_by_file_hashes":True,
        "verl_commit":subprocess.check_output(["git","-C",str(ROOT/".local/verl-upstream"),"rev-parse","HEAD"],text=True).strip(),
        "launch":"scripts/launch_offline_training.py --bundle "+str(out)+" --start"}
    write(out/"certificate.json",certificate)
    verify_bundle(out)
    print(json.dumps({"status":certificate["status"],"checks":checks,"splits":protocol["splits"],"training_started":False}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--bundle",required=True,type=Path);a=p.parse_args();seal(a.bundle.resolve())
