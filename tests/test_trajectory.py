import pytest

from sitian.integrations.trajectory import (
    audit_messages,
    dr_grpo_advantages,
    filter_trainable_groups,
    tokenize_with_assistant_mask,
)


def _messages(void=False):
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "task"},
    ]
    if void:
        messages.append({"role": "assistant", "content": "I will think."})
        messages.append({"role": "user", "content": "use tools"})
    messages.extend([
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "a", "function": {"name": "get_process_evidence", "arguments": "{}"}
        }]},
        {"role": "tool", "tool_call_id": "a", "content": "{\"wind\":2}"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "b", "function": {"name": "submit_forecast", "arguments": "{}"}
        }]},
    ])
    return messages


def test_void_turn_filter_and_submit_contract():
    good = audit_messages(_messages())
    assert good.eligible_for_gradient and good.has_submit_call
    bad = audit_messages(_messages(void=True))
    assert not bad.eligible_for_gradient
    assert bad.void_assistant_turns == 1


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_assistant_tokens_mask"] is True
        return {"input_ids": [1, 2, 3, 4], "assistant_masks": [0, 1, 0, 1]}


def test_native_assistant_mask_is_required():
    encoded = tokenize_with_assistant_mask(_Tokenizer(), _messages())
    assert encoded["loss_mask"] == [0, 1, 0, 1]

    class Unsafe:
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": [1, 2], "assistant_masks": [1, 1]}

    with pytest.raises(RuntimeError, match="whole trajectory"):
        tokenize_with_assistant_mask(Unsafe(), _messages())


def test_real_qwen3_template_patch_masks_tool_payload_and_keeps_assistant_actions():
    from pathlib import Path
    import os
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    cache = Path(os.environ.get("HF_HUB_CACHE", str(Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub")))
    model = Path(os.environ.get("FH_TRAINABLE_MODEL", str(Path(__file__).resolve().parents[1] / "models/Qwen3-8B")))
    roots = ([model] if (model / "tokenizer_config.json").is_file() else
             sorted((cache / "models--Qwen--Qwen3-8B/snapshots").glob("*")))
    if not roots:
        pytest.skip("local Qwen3 tokenizer is not available")
    tokenizer = AutoTokenizer.from_pretrained(roots[0], local_files_only=True)
    messages = _messages()
    messages[3]["content"] = "TOOL_PAYLOAD_MUST_BE_MASKED"
    encoded = tokenize_with_assistant_mask(tokenizer, messages, tools=[{
        "type": "function", "function": {
            "name": "get_process_evidence", "description": "evidence",
            "parameters": {"type": "object", "properties": {}},
        },
    }])
    trainable_text = tokenizer.decode([
        token for token, mask in zip(encoded["input_ids"], encoded["loss_mask"]) if mask
    ])
    assert "TOOL_PAYLOAD_MUST_BE_MASKED" not in trainable_text
    assert "get_process_evidence" in trainable_text
    assert "submit_forecast" in trainable_text
    assert encoded["template_patch"]["kind"].startswith("qwen3_")


def test_dr_grpo_drops_zero_variance_without_sigma_normalization():
    assert dr_grpo_advantages([0.5, 0.5]) is None
    assert dr_grpo_advantages([0.2, 0.6]) == pytest.approx([-0.2, 0.2])
    kept, report = filter_trainable_groups({"flat": [1, 1], "useful": [0.2, 0.6]})
    assert set(kept) == {"useful"}
    assert report["usable_group_rate"] == 0.5
    assert "without group-sigma" in report["advantage_method"]
