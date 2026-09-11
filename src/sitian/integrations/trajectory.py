"""Executable safety contracts for multi-turn agent RL trajectories.

The module is framework-neutral on purpose.  veRL/TRL adapters can consume the
returned ids/mask, but cannot silently fall back to training on environment or
tool-result tokens when a chat template lacks an assistant mask.
"""
from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Iterable


@dataclass(frozen=True)
class TrajectoryAudit:
    messages: int
    assistant_messages: int
    tool_result_messages: int
    void_assistant_turns: int
    has_submit_call: bool
    eligible_for_gradient: bool
    reasons: tuple[str, ...]


def audit_messages(messages: list[dict]) -> TrajectoryAudit:
    """Check role ordering and reject text-only assistant turns from RL updates."""
    reasons = []
    assistant = [message for message in messages if message.get("role") == "assistant"]
    tool_results = [message for message in messages if message.get("role") == "tool"]
    void = 0
    submit = False
    pending_tool_ids: set[str] = set()
    for index, message in enumerate(messages):
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            reasons.append(f"message[{index}] has unsupported role {role!r}")
            continue
        if role == "assistant":
            calls = message.get("tool_calls") or []
            content = (message.get("content") or "").strip()
            if not calls:
                # A free-text reasoning turn without an environment action is a
                # distribution-drift/void turn in this tool-only harness.
                void += 1
            for call in calls:
                call_id = call.get("id")
                if call_id:
                    pending_tool_ids.add(call_id)
                if (call.get("function") or {}).get("name") == "submit_forecast":
                    submit = True
            if not calls and not content:
                reasons.append(f"message[{index}] is an empty assistant turn")
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if call_id and call_id not in pending_tool_ids:
                reasons.append(f"message[{index}] has unmatched tool_call_id {call_id!r}")
    if void:
        reasons.append(f"trajectory contains {void} void assistant turn(s)")
    if not submit:
        reasons.append("trajectory has no submit_forecast tool call")
    return TrajectoryAudit(
        messages=len(messages), assistant_messages=len(assistant),
        tool_result_messages=len(tool_results), void_assistant_turns=void,
        has_submit_call=submit, eligible_for_gradient=not reasons,
        reasons=tuple(reasons),
    )


def tokenize_with_assistant_mask(tokenizer: Any, messages: list[dict],
                                 tools: list[dict] | None = None) -> dict:
    """Render a trajectory and require the tokenizer's native assistant mask.

    Tool results, system prompts and user messages must have loss mask zero;
    assistant reasoning and tool-call actions have mask one.  A tokenizer/chat
    template that cannot return this mask is rejected rather than guessed.
    """
    template_patch = _ensure_native_generation_markers(tokenizer)
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": False,
        "return_dict": True,
        "return_assistant_tokens_mask": True,
    }
    if tools is not None:
        kwargs["tools"] = tools
    try:
        encoded = tokenizer.apply_chat_template(messages, **kwargs)
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise RuntimeError(
            "chat template cannot produce a native assistant token mask; "
            "multi-turn RL is unsafe with this tokenizer/template"
        ) from exc
    input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    mask = None
    if isinstance(encoded, Mapping):
        mask = encoded.get("assistant_masks")
        if mask is None:
            mask = encoded.get("assistant_mask")
    if input_ids is None or mask is None or len(input_ids) != len(mask):
        raise RuntimeError("missing or malformed assistant token mask from chat template")
    if any(value not in (0, 1, False, True) for value in mask):
        raise RuntimeError("assistant token mask must be binary")
    if not any(mask):
        raise RuntimeError("assistant token mask contains no trainable tokens")
    if all(mask):
        raise RuntimeError("assistant token mask covers the whole trajectory, including tool inputs")
    return {"input_ids": list(input_ids), "loss_mask": [int(value) for value in mask],
            "template_patch": template_patch}


def _ensure_native_generation_markers(tokenizer: Any) -> dict | None:
    """Patch the recognized Qwen3 assistant branch with Jinja generation tags.

    This is not a token-text heuristic: Transformers' chat-template engine
    produces the mask from explicit generation spans.  Unknown templates are
    left untouched and will fail closed in ``tokenize_with_assistant_mask``.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str):
        return None
    if "generation %}" in template and "endgeneration %}" in template:
        return None
    assistant = '    {%- elif message.role == "assistant" %}\n'
    tool = '    {%- elif message.role == "tool" %}\n'
    # Restrict the patch to the single, recognizable Qwen3 role loop.  A
    # template drift must become a visible training blocker, not a guessed mask.
    if template.count(assistant) != 1 or template.count(tool) != 1:
        return None
    original_sha = hashlib.sha256(template.encode("utf-8")).hexdigest()
    patched = template.replace(
        assistant, assistant + "        {%- generation %}\n", 1
    ).replace(
        tool, "        {%- endgeneration %}\n" + tool, 1
    )
    if patched.count("generation %}") != 2 or patched.count("endgeneration %}") != 1:
        raise RuntimeError("assistant generation-marker patch produced an invalid template")
    tokenizer.chat_template = patched
    return {
        "kind": "qwen3_explicit_assistant_generation_span_v1",
        "original_sha256": original_sha,
        "patched_sha256": hashlib.sha256(patched.encode("utf-8")).hexdigest(),
    }


def dr_grpo_advantages(rewards: Iterable[float], *, tolerance: float = 1e-12
                       ) -> list[float] | None:
    """Mean-center a group without standard-deviation normalization.

    Returns ``None`` for zero-variance/non-finite groups so the sampler can
    resample them instead of emitting a useless or unstable optimizer batch.
    """
    values = [float(value) for value in rewards]
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        return None
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    if max(values) - min(values) <= tolerance:
        return None
    return centered


def filter_trainable_groups(groups: dict[str, Iterable[float]], *, tolerance: float = 1e-12
                            ) -> tuple[dict[str, list[float]], dict]:
    """Apply the dynamic zero-variance filter and report sampling efficiency."""
    kept = {}
    dropped = []
    total_rollouts = 0
    for group_id, rewards in groups.items():
        values = [float(value) for value in rewards]
        total_rollouts += len(values)
        advantages = dr_grpo_advantages(values, tolerance=tolerance)
        if advantages is None:
            dropped.append(group_id)
        else:
            kept[group_id] = advantages
    kept_rollouts = sum(len(values) for values in kept.values())
    report = {
        "groups": len(groups), "kept_groups": len(kept),
        "dropped_group_ids": sorted(dropped),
        "usable_group_rate": len(kept) / len(groups) if groups else 0.0,
        "effective_rollout_rate": kept_rollouts / total_rollouts if total_rollouts else 0.0,
        "advantage_method": "Dr.GRPO mean-centering without group-sigma normalization",
        "zero_variance_tolerance": tolerance,
    }
    return kept, report
