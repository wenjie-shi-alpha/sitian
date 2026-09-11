"""OpenAI 兼容接口的 LLM agent 驱动器（纯标准库，无三方依赖）。

对接 vLLM / SGLang / 任何 OpenAI-compatible 服务：
    vllm serve Qwen/Qwen3-8B --port 8000
    FH_BASE_URL=http://127.0.0.1:8000/v1 FH_MODEL=Qwen/Qwen3-8B \
        python -m sitian.cli llm --case <case_dir>

该驱动器同时是 RL rollout 的参考实现：多轮 tool calling，
终止于 submit_forecast 或步数预算。
"""
from __future__ import annotations

import json
import re
import os
import urllib.error
import urllib.request
from typing import Optional

from ..env import ForecastEnv

SYSTEM_PROMPT = """你是中国城市空气质量预报员，在一个可验证的预报任务环境中工作（目标城市见任务简报）。
规则：
1. 只能通过提供的工具与环境交互；最终必须调用 submit_forecast 提交结构化预报。
2. 提交前至少查证两类证据。若任务简报显示 open_evidence_available=true，先调用
   get_process_evidence 读取六项初态、高频天气转折、逐日扩散、上游与污染指导的紧凑总览；
   只在总览显示系统位置/模式分歧、输送、成分/沙尘/火点等会改变决策时，按需调用
   get_synoptic_evidence 或 get_pollution_evidence(kind=composition|fires|source_context)。
   推荐紧凑流程：过程总览→必要的专题深挖→模式指导/历史偏差→提交；不要机械遍历全部工具，
   覆盖、缺测、统计口径或来源独立性有疑问时查询 list_data_assets 的 quality，
   previous_forecast 仅在明确可用时查询；调用
   get_model_guidance 时 source 缺省即可返回所有真实可用源，不要猜测 EC 等不存在的名称。
   当模式可信度或事件漏报风险会改变订正时，查询 get_guidance_bias；其 error=guidance-truth，
   负偏差表示历史指导偏低，但它只是历史聚合证据，不是当前真值。
   判断方法不明确时可用 retrieve_forecast_methods；hypothesis 是待验证的常识方法，不是专家确认。
   历史类比可用时按问题选择 find_similar_cases，再用 get_historical_case 核查关键差异及已验证结果。
   方法和类比均非必经流程；相似距离不是概率，历史结果不能当作本次真值。
3. 预报对象必须严格符合 schema：区间按列提交在 forecast 顶层——pm25_lo、pm25_hi（多污染物个例
   另加 pm10_lo、pm10_hi、o3_lo、o3_hi），每列是长度恰为 horizon 的数字列表，第 k 项对应起报后
   第 k 天（目标 80% 覆盖，lo <= hi，不写日期，不要包 daily）。
   AQI 等级、首要污染物和污染过程由环境按 HJ 633 从区间中点派生并评分，不要提交
   aqi_level、primary_pollutant、process。结构见任务简报 forecast_format 与 submit_forecast 的参数
   schema；所有数字必须来自你对已查证据的判断，简报中没有也不应有可照抄的数值。
   evidence 的 type 只能取以下八个值之一，不得新造：
   observation（实况）/ diagnostic（诊断量）/ synoptic（天气形势）/ model_guidance（模式指导）/
   analog（相似个例）/ previous_forecast（昨日预报）/ expert_prior（经验先验）/ other。
   get_assessment 的信号按其来源归类（趋势→observation，气象信号→diagnostic 或 synoptic，
   指导分歧→model_guidance），不要写成 assessment。
   新开放工具也不产生新 type：环流→synoptic，稳定/输送/源区→diagnostic，CAMS 成分→model_guidance，
   空间实况/火点→observation；严禁填写 pollution_evidence、source_context 等工具或字段名。
4. 结论要与已查证据一致，不要编造未查询过的数据。每个成功数据工具返回 evidence_ref。
   evidence 中至少两条应填 ref、field 和 value：field 是指向该次工具返回内某个标量事实的
   RFC 6901 JSON Pointer，value 必须保持原始 JSON 类型并精确等于该字段值（数字不能加引号）。
   工具返回的 citation_examples 是从真实对象自动生成的可复制索引；优先原样复制其中至少两条
   type/ref/field/value，再为每条写与该事实一致的 claim，不要自行猜路径。例如
   {"type":"diagnostic","claim":"该日近地风较弱","ref":"e1","field":"/daily_surface_dispersion/2026-01-02/wind_speed_ms","value":1.8}。
5. 每次工具返回 budget；预留至少一步提交，不要重复查询相同数据。"""


TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _recover_tool_calls_from_text(text: str) -> list[dict]:
    """Same extraction rule as veRL's hermes parser (regex over the full text)."""
    if not text or "<tool_call>" not in text or "</tool_call>" not in text:
        return []
    calls = []
    for index, match in enumerate(TOOL_CALL_RE.findall(text)):
        try:
            call = json.loads(match)
            name, arguments = call["name"], call["arguments"]
        except Exception:  # noqa: BLE001 - undecodable block remains a void turn
            continue
        calls.append({
            "id": f"recovered-{index}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        })
    return calls


class OpenAICompatAgent:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.6,
        max_llm_calls: int = 16,
        timeout: float = 180.0,
        enable_thinking: Optional[bool] = None,
        seed: Optional[int] = None,
    ):
        self.base_url = (base_url or os.environ.get("FH_BASE_URL", "http://127.0.0.1:8000/v1")).rstrip("/")
        self.model = model or os.environ.get("FH_MODEL", "")
        self.api_key = api_key or os.environ.get("FH_API_KEY", "EMPTY")
        self.temperature = temperature
        self.max_llm_calls = max_llm_calls
        self.timeout = timeout
        self.seed = seed
        # Qwen3 等混合思维模型：命令行显式值优先，环境变量保持向后兼容。
        if enable_thinking is None:
            enable_thinking = os.environ.get("FH_ENABLE_THINKING", "0").strip().lower() not in (
                "0", "", "false", "no", "off",
            )
        self.enable_thinking = enable_thinking
        if not self.model:
            raise ValueError("model is required (set FH_MODEL or pass model=)")

    def _chat(self, messages: list[dict], tools: list[dict]) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": self.temperature,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        # 显式传入 on/off，避免服务端默认 chat template 变化污染消融。
        payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            try:
                message = json.loads(body).get("error", {}).get("message") or body
            except json.JSONDecodeError:
                message = body
            raise RuntimeError(f"LLM endpoint HTTP {exc.code}: {message[:1000]}") from exc

    def run(self, env: ForecastEnv) -> dict:
        obs = env.reset()
        tools = env.tool_specs("openai")
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "任务简报：\n" + json.dumps(obs["brief"], ensure_ascii=False, indent=1)},
        ]
        usage = {"prompt_tokens": 0, "completion_tokens": 0,
                 "max_prompt_tokens": 0, "max_completion_tokens": 0, "llm_calls": 0}
        thinking_activity = {"messages": 0, "characters": 0}
        reward, info = 0.0, {"reason": "llm_no_submit"}

        for _ in range(self.max_llm_calls):
            resp = self._chat(messages, tools)
            call_usage = resp.get("usage", {})
            prompt_tokens = int(call_usage.get("prompt_tokens", 0))
            completion_tokens = int(call_usage.get("completion_tokens", 0))
            usage["prompt_tokens"] += prompt_tokens
            usage["completion_tokens"] += completion_tokens
            usage["max_prompt_tokens"] = max(usage["max_prompt_tokens"], prompt_tokens)
            usage["max_completion_tokens"] = max(usage["max_completion_tokens"], completion_tokens)
            usage["llm_calls"] += 1
            msg = resp["choices"][0]["message"]
            reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            # vLLM 0.10.x may leave Qwen3 reasoning inline in content instead of
            # exposing reasoning_content, especially with the Hermes tool parser.
            content = msg.get("content") or ""
            if not reasoning and "<think>" in content:
                reasoning = content.split("<think>", 1)[1].split("</think>", 1)[0]
            if reasoning:
                thinking_activity["messages"] += 1
                thinking_activity["characters"] += len(reasoning)
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                # Mirror veRL's HermesToolParser: it extracts <tool_call> blocks by
                # regex from the whole response text, including ones the model
                # wrote inside its <think> block, which vLLM's top-level hermes
                # parser leaves as plain content.  Undecodable blocks stay void.
                recovered = _recover_tool_calls_from_text(content)
                if recovered:
                    tool_calls = recovered
                    msg = dict(msg)
                    msg["tool_calls"] = recovered
                    msg["content"] = TOOL_CALL_RE.sub("", content)
                    usage["recovered_tool_calls"] = (
                        usage.get("recovered_tool_calls", 0) + len(recovered)
                    )
            messages.append(msg)
            if not tool_calls:
                messages.append({
                    "role": "user",
                    "content": "请通过工具调用继续：查询数据或调用 submit_forecast 提交预报。",
                })
                continue
            done = False
            for call in tool_calls:
                fn = call["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                obs, reward, done, info = env.step({"name": fn["name"], "args": args})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": json.dumps(obs.get("content", {}), ensure_ascii=False),
                })
                if done:
                    break
            if done:
                break

        return {
            "reward": reward,
            "info": info,
            "steps": env.steps_used,
            "usage": usage,
            "thinking_activity": thinking_activity,
            "messages": messages,
            "transcript": env.transcript,
        }
