"""Pure-Python bridge between a veRL tool trajectory and ``ForecastEnv``.

veRL's native ``BaseTool`` lifecycle is per call in the current agent loop.  The
forecast harness, however, deliberately keeps episode state: evidence refs and
the set of successfully queried tools must survive until ``submit_forecast``.
This module rebuilds that state by deterministically replaying the assistant
tool calls already present in the token-preserving trajectory.  It contains no
veRL import, so the replay and leakage contract remain unit-testable without the
GPU training stack.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..case import CaseBundle
from ..env import ForecastEnv


@dataclass(frozen=True)
class ReplayedStep:
    observation: dict
    reward: float
    done: bool
    info: dict
    actions_replayed: int


def _value(value: Any, key: str, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _arguments(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def trajectory_actions(messages: Iterable[Any]) -> list[dict]:
    """Extract assistant actions without trusting tool-result text.

    Only assistant ``tool_calls`` are replayed.  Environment outputs in the
    transcript are intentionally ignored: recomputing them from the hidden case
    bundle prevents a model-authored tool-result string from influencing reward.
    """
    actions: list[dict] = []
    for message in messages:
        if _value(message, "role") != "assistant":
            continue
        for call in _value(message, "tool_calls", []) or []:
            function = _value(call, "function", {}) or {}
            name = _value(function, "name")
            if not isinstance(name, str) or not name:
                continue
            actions.append({"name": name, "args": _arguments(
                _value(function, "arguments", {})
            )})
    return actions


def replay_current_action(
    *,
    case_dir: str | Path,
    messages: Iterable[Any],
    current_name: str,
    current_args: dict,
) -> ReplayedStep:
    """Replay a trajectory through the real environment and return this call.

    ``ToolAgentLoop`` has already appended the current assistant tool call when
    it invokes a tool.  If an upstream interface changes and that call is not
    present, we append it explicitly and fail closed on any earlier terminal
    episode.  Parallel tool calls are disallowed in the training config.
    """
    case_path = Path(case_dir).resolve()
    bundle = CaseBundle.load(case_path)
    env = ForecastEnv(bundle)
    env.reset()
    actions = trajectory_actions(messages)
    expected = {"name": current_name, "args": current_args or {}}
    if not actions or actions[-1] != expected:
        actions.append(expected)

    last = None
    for index, action in enumerate(actions):
        if env.done:
            return ReplayedStep(
                observation={
                    "type": "tool_result",
                    "name": action.get("name"),
                    "ok": False,
                    "content": {"error": "episode already terminated"},
                },
                reward=0.0,
                done=True,
                info={"reason": "action_after_terminal"},
                actions_replayed=index,
            )
        obs, reward, done, info = env.step(action)
        last = ReplayedStep(obs, float(reward), bool(done), dict(info), index + 1)
    if last is None:  # defensive; expected is always appended above
        raise RuntimeError("no tool action was available for replay")
    return last
