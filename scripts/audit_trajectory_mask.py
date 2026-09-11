#!/usr/bin/env python3
"""Reproducibly verify assistant-only loss masking with a real tokenizer."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sitian.integrations.trajectory import tokenize_with_assistant_mask  # noqa: E402
from sitian.provenance import file_identity  # noqa: E402


def _direct_verl_response_mask(tokenizer_path: Path) -> dict:
    """Exercise veRL's real Continuous Token mask alignment path."""
    from verl.utils.tokenizer.continuous_token_wiring import (
        create_continuous_token_builder,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    model_type = json.loads(
        (tokenizer_path / "config.json").read_text(encoding="utf-8")
    ).get("model_type")
    builder = create_continuous_token_builder(
        tokenizer, hf_model_type=model_type, processor=None
    )
    tools = [{"type": "function", "function": {
        "name": name,
        "description": name,
        "parameters": {"type": "object", "properties": {}},
    }} for name in ("get_process_evidence", "submit_forecast")]
    messages = [
        {"role": "system", "content": "SYSTEM_OUTSIDE_RESPONSE"},
        {"role": "user", "content": "USER_OUTSIDE_RESPONSE"},
    ]
    prompt_ids = builder.build_initial_tokens(messages, tools=tools)
    runtime_ids = list(prompt_ids)
    response_mask: list[int] = []

    first_text = (
        "<think>\nVERL_ASSISTANT_REASONING_TRAINABLE\n</think>\n\n"
        '<tool_call>\n{"name":"get_process_evidence","arguments":{}}\n'
        "</tool_call><|im_end|>"
    )
    first_ids = tokenizer.encode(first_text, add_special_tokens=False)
    merged = builder.merge_assistant_tokens(runtime_ids, first_ids)
    response_mask, _ = builder.align_response_metadata(merged, response_mask)
    runtime_ids = merged.token_ids
    assistant_message = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "VERL_ASSISTANT_REASONING_TRAINABLE",
        "tool_calls": [{"id": "call-a", "type": "function", "function": {
            "name": "get_process_evidence", "arguments": "{}",
        }}],
    }
    previous = messages + [assistant_message]
    tool_message = {
        "role": "tool",
        "tool_call_id": "call-a",
        "name": "get_process_evidence",
        "content": "VERL_TOOL_PAYLOAD_MUST_BE_MASKED",
    }
    updated = previous + [tool_message]
    merged = builder.merge_non_assistant_tokens(
        previous, updated, runtime_ids, tools=tools
    )
    response_mask, _ = builder.align_response_metadata(merged, response_mask)
    runtime_ids = merged.token_ids

    second_text = (
        "<think>\nVERL_ASSISTANT_SUBMISSION_TRAINABLE\n</think>\n\n"
        '<tool_call>\n{"name":"submit_forecast","arguments":{"forecast":{}}}\n'
        "</tool_call><|im_end|>"
    )
    second_ids = tokenizer.encode(second_text, add_special_tokens=False)
    merged = builder.merge_assistant_tokens(runtime_ids, second_ids)
    response_mask, _ = builder.align_response_metadata(merged, response_mask)
    runtime_ids = merged.token_ids

    response_ids = runtime_ids[-len(response_mask):]
    trainable = tokenizer.decode([
        token for token, mask in zip(response_ids, response_mask) if mask
    ])
    masked = tokenizer.decode([
        token for token, mask in zip(response_ids, response_mask) if not mask
    ])
    checks = {
        "response_ids_mask_aligned": len(response_ids) == len(response_mask),
        "only_generated_assistant_tokens_trainable": (
            sum(response_mask) == len(first_ids) + len(second_ids)
        ),
        "assistant_reasoning_trainable": (
            "VERL_ASSISTANT_REASONING_TRAINABLE" in trainable
        ),
        "assistant_submission_trainable": (
            "VERL_ASSISTANT_SUBMISSION_TRAINABLE" in trainable
        ),
        "tool_call_actions_trainable": (
            "get_process_evidence" in trainable and "submit_forecast" in trainable
        ),
        "tool_payload_masked": (
            "VERL_TOOL_PAYLOAD_MUST_BE_MASKED" not in trainable
            and "VERL_TOOL_PAYLOAD_MUST_BE_MASKED" in masked
        ),
        "prompt_not_in_response_tensor": (
            "SYSTEM_OUTSIDE_RESPONSE" not in trainable + masked
            and "USER_OUTSIDE_RESPONSE" not in trainable + masked
        ),
    }
    return {
        "builder": type(builder).__name__,
        "model_type": model_type,
        "tokens": {
            "prompt": len(prompt_ids),
            "response": len(response_ids),
            "trainable": sum(response_mask),
            "masked_non_assistant": len(response_mask) - sum(response_mask),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default=os.environ.get("FH_TRAINABLE_TOKENIZER"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/interim/qwen3_assistant_mask_audit.json"))
    args = parser.parse_args()
    if not args.tokenizer:
        parser.error("--tokenizer or FH_TRAINABLE_TOKENIZER is required")
    tokenizer_path = Path(args.tokenizer).resolve()
    direct_verl = _direct_verl_response_mask(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    messages = [
        {"role": "system", "content": "SYSTEM_MUST_BE_MASKED"},
        {"role": "user", "content": "USER_MUST_BE_MASKED"},
        {"role": "assistant", "content": "ASSISTANT_REASONING_IS_TRAINABLE",
         "tool_calls": [{"id": "a", "type": "function", "function": {
             "name": "get_process_evidence", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "name": "get_process_evidence",
         "content": "TOOL_PAYLOAD_MUST_BE_MASKED"},
        {"role": "assistant", "content": "ASSISTANT_SUBMISSION_IS_TRAINABLE",
         "tool_calls": [{"id": "b", "type": "function", "function": {
             "name": "submit_forecast", "arguments": "{\"forecast\":{}}"}}]},
    ]
    tools = [{"type": "function", "function": {
        "name": name, "description": name,
        "parameters": {"type": "object", "properties": {}},
    }} for name in ("get_process_evidence", "submit_forecast")]
    encoded = tokenize_with_assistant_mask(tokenizer, messages, tools)
    trainable = tokenizer.decode([
        token for token, mask in zip(encoded["input_ids"], encoded["loss_mask"]) if mask
    ])
    masked = tokenizer.decode([
        token for token, mask in zip(encoded["input_ids"], encoded["loss_mask"]) if not mask
    ])
    checks = {
        "assistant_reasoning_trainable": "ASSISTANT_REASONING_IS_TRAINABLE" in trainable,
        "assistant_submission_trainable": "ASSISTANT_SUBMISSION_IS_TRAINABLE" in trainable,
        "tool_call_action_trainable": all(name in trainable for name in
                                           ("get_process_evidence", "submit_forecast")),
        "system_masked": "SYSTEM_MUST_BE_MASKED" not in trainable and
                         "SYSTEM_MUST_BE_MASKED" in masked,
        "user_masked": "USER_MUST_BE_MASKED" not in trainable and
                       "USER_MUST_BE_MASKED" in masked,
        "tool_payload_masked": "TOOL_PAYLOAD_MUST_BE_MASKED" not in trainable and
                               "TOOL_PAYLOAD_MUST_BE_MASKED" in masked,
    }
    report = {
        "artifact_type": "assistant_loss_mask_audit",
        "audit_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "tokenizer": str(tokenizer_path),
        "tokenizer_config": file_identity(tokenizer_path / "tokenizer_config.json"),
        "template_patch": encoded.get("template_patch"),
        "direct_verl_continuous_token": direct_verl,
        "tokens": {"total": len(encoded["input_ids"]),
                   "trainable": sum(encoded["loss_mask"]),
                   "masked": len(encoded["loss_mask"]) - sum(encoded["loss_mask"])},
        "checks": checks,
        "pass": all(checks.values()) and direct_verl["pass"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"-> {args.out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
