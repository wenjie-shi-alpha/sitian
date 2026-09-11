#!/usr/bin/env python3
"""Fail-closed contract test against the pinned, real veRL runtime.

Run this with the Python interpreter from the dedicated veRL environment.  It
checks both the fully resolved PPO configuration and a native BaseTool
create/execute/release trajectory.  No model server or GPU is required.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf
from verl.experimental.agent_loop.tool_agent_loop import AgentData
from verl.tools.tool_registry import initialize_tools_from_config

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.agents.scripted import GuidanceFollowAgent  # noqa: E402
from sitian.case import CaseBundle  # noqa: E402
from sitian.env import ForecastEnv  # noqa: E402
from sitian.provenance import file_identity  # noqa: E402
from sitian.scoring import reward_spec  # noqa: E402


PINNED_VERL_COMMIT = "c2429f29a25d573f63d9bcc29e7ceb690817dce9"


def _assistant(action: dict, call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": action["name"],
                "arguments": json.dumps(action.get("args", {}), ensure_ascii=False),
            },
        }],
    }


def _first_case(manifest: Path) -> Path:
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError(f"empty case manifest: {manifest}")
    value = Path(rows[0])
    return value if value.is_absolute() else REPO_ROOT / value


async def _lifecycle(case_dir: Path, tool_config: Path, *, optional_gases: bool = False) -> dict:
    # Upstream BaseTool currently prints every schema during construction.
    # Suppress that noise while retaining the schemas in the hashed YAML input.
    with contextlib.redirect_stdout(io.StringIO()):
        tools = initialize_tools_from_config(str(tool_config))
    by_name = {tool.name: tool for tool in tools}
    if len(by_name) != len(tools):
        raise ValueError("duplicate veRL tool names")

    bundle = CaseBundle.load(case_dir)
    environment = ForecastEnv(bundle)
    reset = environment.reset()
    expected_names = {
        item["function"]["name"] for item in environment.tool_specs("openai")
    }
    scripted = GuidanceFollowAgent()
    scripted.begin(reset["brief"])
    agent_data = AgentData(
        messages=[{"role": "system", "content": "native runtime contract test"}],
        image_data=[],
        video_data=[],
        audio_data=None,
        mm_processor_kwargs=None,
        metrics={},
        request_id="sitian-verl-contract",
        tools_kwargs={},
    )

    guidance_action = scripted.act(reset)
    agent_data.messages.append(_assistant(guidance_action, "call-guidance"))
    guidance_tool = by_name[guidance_action["name"]]
    instance, _ = await guidance_tool.create(
        create_kwargs={"case_dir": str(case_dir), "harness_resources": environment.resource_identity}
    )
    try:
        response, reward, metrics = await guidance_tool.execute(
            instance, guidance_action["args"], agent_data=agent_data
        )
    finally:
        await guidance_tool.release(instance)
    guidance_content = json.loads(response.text or "{}")
    if reward != 0.0 or metrics.get("sitian_terminal") or not guidance_content.get("sources"):
        raise AssertionError("non-terminal guidance lifecycle contract failed")
    agent_data.messages.append({
        "role": "tool",
        "tool_call_id": "call-guidance",
        "content": response.text,
    })

    submit_action = scripted.act({"content": guidance_content})
    gas_probe = {"so2_range": [5.0, 15.0], "no2_range": [190.0, 210.0], "co_range": [0.5, 1.5]}
    if optional_gases:
        # Synthetic submission fields exercise parser/replay/serialization;
        # this probe asserts no forecast skill and consults no hidden truth.
        for day in submit_action["args"]["forecast"]["daily"]:
            day.update(gas_probe)
    agent_data.messages.append(_assistant(submit_action, "call-submit"))
    submit_tool = by_name[submit_action["name"]]
    instance, _ = await submit_tool.create(create_kwargs={"case_dir": str(case_dir),
                                                         "harness_resources": environment.resource_identity})
    try:
        response, reward, metrics = await submit_tool.execute(
            instance, submit_action["args"], agent_data=agent_data
        )
    finally:
        await submit_tool.release(instance)
    submitted = json.loads(response.text or "{}")
    terminal_reward = agent_data.extra_fields.get("sitian_terminal_reward")
    terminal_outcome = agent_data.extra_fields.get(
        "sitian_terminal_outcome_composite"
    )
    checks = {
        "all_tool_names_unique": len(by_name) == len(tools),
        "all_declared_tools_loaded": set(by_name) == expected_names,
        "guidance_then_submit_replayed": metrics.get("sitian_actions_replayed") == 2,
        "valid_forecast_accepted": submitted.get("accepted") is True,
        "submit_terminates_episode": (
            metrics.get("sitian_terminal") is True
            and agent_data.extra_fields.get("sitian_terminal") is True
            and agent_data.extra_fields.get("sitian_terminal_reason") == "submitted"
        ),
        "terminal_reward_propagated": (
            isinstance(reward, float)
            and reward == terminal_reward
            and 0.0 <= reward <= 1.0
        ),
        "terminal_outcome_available_for_group_filter": (
            isinstance(terminal_outcome, float)
            and 0.0 <= terminal_outcome <= 1.0
        ),
    }
    if optional_gases:
        normalized = submitted.get("normalized_forecast", {}).get("daily", [])
        checks["optional_gas_ranges_survive_native_tool_submission"] = (
            len(normalized) == bundle.horizon
            and all(day.get(field) == value for day in normalized for field, value in gas_probe.items())
        )
    return {
        "checks": checks,
        "tools_loaded": len(tools),
        "tool_names": sorted(by_name),
        "expected_tool_names": sorted(expected_names),
        "case": str(case_dir.relative_to(REPO_ROOT)),
        "actions_replayed": metrics.get("sitian_actions_replayed"),
        "reward": reward,
        "outcome_composite": terminal_outcome,
        "terminal_reason": agent_data.extra_fields.get("sitian_terminal_reason"),
    }


def _config_checks(config_path: Path) -> dict[str, bool]:
    cfg = OmegaConf.load(config_path)
    actor = cfg.actor_rollout_ref.actor
    rollout = cfg.actor_rollout_ref.rollout
    multi = rollout.multi_turn
    return {
        "grpo_without_group_std_normalization": (
            cfg.algorithm.adv_estimator == "grpo"
            and cfg.algorithm.norm_adv_by_std_in_grpo is False
        ),
        "zero_variance_group_filter_enabled": (
            cfg.algorithm.filter_groups.enable is True
            and cfg.algorithm.filter_groups.metric == "outcome_composite"
        ),
        "current_policy_async_rollout": (
            rollout.name == "vllm"
            and rollout.mode == "async"
            # layered_summon is the multi-GPU FSDP weight-sync optimisation; on a
            # single GPU FSDP is NO_SHARD and layered summon falls back to an
            # offload_to_cpu summon that torch rejects, so plain summon is the
            # correct (and still on-policy) path there.
            and (rollout.layered_summon is True
                 or int(cfg.trainer.n_gpus_per_node) * int(cfg.trainer.nnodes) == 1)
        ),
        "native_multiturn_single_call": (
            multi.enable is True
            and multi.format == "hermes"
            and multi.max_parallel_calls == 1
            and rollout.agent.default_agent_loop == "sitian_tool_agent"
        ),
        "lora_only_trainable_checkpoint": (
            cfg.actor_rollout_ref.model.lora_rank == 32
            and actor.checkpoint.save_lora_only is True
        ),
        "tool_tokens_excluded_by_token_mean_contract": (
            actor.loss_agg_mode == "token-mean"
            and cfg.data.return_raw_chat is True
            and cfg.data.truncation == "error"
        ),
        "rollout_and_optimizer_group_sizes_align": (
            rollout.n == 8
            and cfg.data.train_batch_size == int(os.environ.get("SITIAN_TRAIN_BATCH_SIZE", "2"))
            and cfg.data.train_batch_size > 0
            # V1 expands this prompt count by rollout.n before actor training.
            and actor.ppo_mini_batch_size == cfg.data.train_batch_size
            and (actor.ppo_mini_batch_size * rollout.n) % (
                int(cfg.trainer.n_gpus_per_node) * int(cfg.trainer.nnodes)
            ) == 0
            and float(rollout.temperature) == 1.0
        ),
        # The stack smoke length is set by SITIAN_SMOKE_STEPS (6 locally: the
        # four mechanism checks need a few real steps, not 50 at 32K context on
        # one 4090); the resolved config must match whatever was requested.
        "resolved_smoke_steps_match_request": (
            int(cfg.trainer.total_training_steps)
            == int(os.environ.get("SITIAN_SMOKE_STEPS", "50")) >= 1
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl-root", type=Path, default=Path(os.environ.get("SITIAN_VERL_ROOT", str(REPO_ROOT / ".local/verl-upstream"))))
    parser.add_argument("--resolved-config", type=Path,
                        default=Path("data/interim/verl_resolved_smoke_config.yaml"))
    parser.add_argument("--tool-config", type=Path,
                        default=Path("configs/verl/sitian_tools.yaml"))
    parser.add_argument("--agent-config", type=Path,
                        default=Path("configs/verl/sitian_agent_loop.yaml"))
    parser.add_argument("--case-manifest", type=Path,
                        default=Path("data/interim/valid_cases_val.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/interim/verl_runtime_contract.json"))
    args = parser.parse_args()

    def absolute(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    config_path = absolute(args.resolved_config)
    tool_config = absolute(args.tool_config)
    agent_config = absolute(args.agent_config)
    case_manifest = absolute(args.case_manifest)
    out = absolute(args.out)
    commit = subprocess.check_output(
        ["git", "-C", str(args.verl_root), "rev-parse", "HEAD"], text=True
    ).strip()
    resolved = OmegaConf.load(config_path)
    config_checks = _config_checks(config_path)
    lifecycle = asyncio.run(_lifecycle(_first_case(case_manifest), tool_config))
    checks = {
        "pinned_upstream_commit": commit == PINNED_VERL_COMMIT,
        **config_checks,
        **lifecycle["checks"],
        "current_reward_identity_available": bool(
            reward_spec().get("version") and reward_spec().get("config_sha256")
        ),
    }
    report = {
        "artifact_type": "verl_native_runtime_contract",
        "contract_version": "1.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "passed": all(checks.values()),
        "reward": reward_spec(),
        "verl": {
            "root": str(args.verl_root.resolve()),
            "commit": commit,
            "expected_commit": PINNED_VERL_COMMIT,
        },
        "inputs": {
            "resolved_config": file_identity(config_path),
            "tool_config": file_identity(tool_config),
            "agent_config": file_identity(agent_config),
            "case_manifest": file_identity(case_manifest),
            "environment_implementation": file_identity(REPO_ROOT / "src/sitian/env.py"),
            "bridge_implementation": file_identity(
                REPO_ROOT / "src/sitian/integrations/verl_bridge.py"
            ),
            "runtime_implementation": file_identity(
                REPO_ROOT / "src/sitian/integrations/verl_runtime.py"
            ),
            "tool_config_generator": file_identity(
                REPO_ROOT / "scripts/build_verl_tool_config.py"
            ),
            "runtime_auditor": file_identity(Path(__file__)),
        },
        "checks": checks,
        "training_sampling_contract": {
            "rollout_group_size": int(resolved.actor_rollout_ref.rollout.n),
            "temperature": float(resolved.actor_rollout_ref.rollout.temperature),
            "group_filter_metric": str(resolved.algorithm.filter_groups.metric),
            "train_batch_size": int(resolved.data.train_batch_size),
            "ppo_mini_batch_size": int(
                resolved.actor_rollout_ref.actor.ppo_mini_batch_size
            ),
            "max_model_len": int(resolved.actor_rollout_ref.rollout.max_model_len),
            "minimum_completion_reserve_tokens": 4096,
        },
        "lifecycle": {key: value for key, value in lifecycle.items() if key != "checks"},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"-> {out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
