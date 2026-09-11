"""veRL runtime adapters for on-policy multi-turn forecast RL.

This module is imported only inside the dedicated veRL environment.  Rollout is
performed by veRL's token-in/token-out ``ToolAgentLoop``; the subclass below
only adds the ForecastEnv terminal semantics and direct rule-based reward.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import (
    AgentData,
    AgentState,
    ToolAgentLoop,
)
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from sitian.scoring import reward_spec
from sitian.paths import resolve_case_dir

from .verl_bridge import replay_current_action


class SitianForecastTool(BaseTool):
    """One configured ForecastEnv action exposed as a native veRL tool.

    The same class is instantiated once for every schema in the YAML registry.
    A call receives the case path only through ``create_kwargs`` (never in the
    prompt), replays prior assistant actions, and returns only the normal public
    tool result.  Truth and expert files therefore stay scorer-only.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        configured = config.get("tool_name")
        if configured and configured != self.name:
            raise ValueError(f"tool_name mismatch: {configured!r} != {self.name!r}")
        self._instances: dict[str, Path] = {}
        self._expected_harness_resources = config.get("harness_resources")

    async def create(self, instance_id: Optional[str] = None, create_kwargs=None, **_kwargs):
        values = create_kwargs or {}
        case_dir = values.get("case_dir")
        if not case_dir:
            raise ValueError(f"{self.name}: create_kwargs.case_dir is required")
        dataset_resources = values.get("harness_resources")
        if self._expected_harness_resources is not None and dataset_resources != self._expected_harness_resources:
            raise ValueError("dataset/tool-config harness resource mismatch: regenerate both from the same inputs")
        identifier = instance_id or uuid4().hex
        self._instances[identifier] = resolve_case_dir(case_dir)
        return identifier, ToolResponse()

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        *,
        agent_data: AgentData,
        **_kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        case_dir = self._instances[instance_id]
        replayed = replay_current_action(
            case_dir=case_dir,
            messages=agent_data.messages,
            current_name=self.name,
            current_args=parameters,
            expected_harness_resources=self._expected_harness_resources,
        )
        content = replayed.observation.get("content", {})
        terminal = replayed.done
        if terminal:
            # Kept in non-tensor rollout metadata, never rendered into the next
            # model prompt.  SitianToolAgentLoop terminates immediately below.
            agent_data.extra_fields["sitian_terminal"] = True
            agent_data.extra_fields["sitian_terminal_reason"] = replayed.info.get("reason")
            agent_data.extra_fields["sitian_terminal_reward"] = replayed.reward
            terminal_score = replayed.info.get("score") or {}
            agent_data.extra_fields["sitian_terminal_outcome_composite"] = float(
                terminal_score.get("outcome_composite", 0.0)
            )
            agent_data.extra_fields["sitian_actions_replayed"] = replayed.actions_replayed
            agent_data.extra_fields["sitian_forecast"] = content.get("normalized_forecast")
            agent_data.extra_fields["sitian_components"] = terminal_score.get("components", {})
        metrics = {
            "sitian_terminal": terminal,
            "sitian_reason": replayed.info.get("reason"),
            "sitian_actions_replayed": replayed.actions_replayed,
        }
        return (
            ToolResponse(text=json.dumps(content, ensure_ascii=False, separators=(",", ":"))),
            replayed.reward,
            metrics,
        )

    async def release(self, instance_id: str, **_kwargs) -> None:
        self._instances.pop(instance_id, None)


class SitianToolAgentLoop(ToolAgentLoop):
    """Token-preserving tool loop with ForecastEnv terminal reward semantics."""

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        state = await super()._handle_processing_tools_state(agent_data)
        if agent_data.extra_fields.get("sitian_terminal"):
            return AgentState.TERMINATED
        return state

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra = kwargs.get("extra_info") or {}
        evaluation_seed = None
        if extra.get("evaluation_panel"):
            raw = f"{extra['evaluation_seed']}\0{extra['case_id']}\0{kwargs.get('session_id', 0)}"
            evaluation_seed = int.from_bytes(hashlib.sha256(raw.encode()).digest()[:4], "big")
            sampling_params = dict(sampling_params, seed=evaluation_seed)
        output = await super().run(sampling_params, **kwargs)
        score = float(output.extra_fields.get("sitian_terminal_reward", 0.0))
        outcome = float(
            output.extra_fields.get("sitian_terminal_outcome_composite", 0.0)
        )
        output.reward_score = score
        # Required for veRL V1 DAPO group filtering before optimizer sampling.
        output.extra_fields["reward_extra_info"] = {
            "score": score,
            # DAPO filters on this field. Grounding-only variation remains in
            # the retained group's reward, but cannot make numerically equal
            # forecasts look learnable.
            "outcome_composite": outcome,
            "sitian_submitted": bool(
                output.extra_fields.get("sitian_terminal_reason") == "submitted"
            ),
            # String telemetry is persisted by native validation but excluded
            # from numeric metric aggregation. Never appended to model messages.
            "sitian_record": json.dumps({
                "case_id": extra.get("case_id"),
                "evaluation_seed": evaluation_seed,
                "forecast": output.extra_fields.get("sitian_forecast"),
                "components": output.extra_fields.get("sitian_components", {}),
                "termination_reason": output.extra_fields.get("sitian_terminal_reason") or "no_submission",
                "actions_replayed": output.extra_fields.get("sitian_actions_replayed"),
                "response_tokens": len(output.response_ids),
                "assistant_tokens": sum(output.response_mask),
            }, ensure_ascii=False),
        }
        output.extra_fields["sitian_reward_identity"] = {
            key: reward_spec()[key] for key in ("version", "config_sha256")
        }
        return output
