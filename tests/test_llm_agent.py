import json

from sitian.agents.llm_openai import OpenAICompatAgent


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return b'{"choices": [{"message": {"role": "assistant", "content": "ok"}}]}'


def test_thinking_flag_is_explicit_in_request(monkeypatch):
    payloads = []

    def fake_urlopen(request, timeout):
        payloads.append(json.loads(request.data))
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    for enabled in (False, True):
        agent = OpenAICompatAgent(model="test-model", enable_thinking=enabled)
        agent._chat([], [])

    assert [p["chat_template_kwargs"]["enable_thinking"] for p in payloads] == [False, True]


def test_seed_is_explicit_only_when_configured(monkeypatch):
    payloads = []

    def fake_urlopen(request, timeout):
        payloads.append(json.loads(request.data))
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    OpenAICompatAgent(model="test-model")._chat([], [])
    OpenAICompatAgent(model="test-model", seed=123)._chat([], [])

    assert "seed" not in payloads[0]
    assert payloads[1]["seed"] == 123


def test_thinking_env_parsing(monkeypatch):
    monkeypatch.setenv("FH_ENABLE_THINKING", "OFF")
    assert OpenAICompatAgent(model="test-model").enable_thinking is False
    monkeypatch.setenv("FH_ENABLE_THINKING", "yes")
    assert OpenAICompatAgent(model="test-model").enable_thinking is True


def test_inline_qwen_thinking_is_counted(monkeypatch):
    class InlineThinkingAgent(OpenAICompatAgent):
        def _chat(self, messages, tools):
            return {
                "choices": [{"message": {
                    "role": "assistant",
                    "content": "<think>abc</think>",
                    "tool_calls": [],
                }}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            }

    class Env:
        steps_used = 0
        transcript = []

        def reset(self):
            return {"brief": {}}

        def tool_specs(self, _format):
            return []

    agent = InlineThinkingAgent(model="test-model", max_llm_calls=1, enable_thinking=True)
    result = agent.run(Env())
    assert result["thinking_activity"] == {"messages": 1, "characters": 3}
