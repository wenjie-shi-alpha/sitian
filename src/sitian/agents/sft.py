"""SFT 冷启动数据生成：脚本基线 episode → OpenAI messages 格式 tool-calling 轨迹。

目的：GRPO 前的格式冷启动——教会策略模型工具调用语法、submit schema 与证据引用，
使 GRPO 组内从"全格式失败（零方差零梯度）"变为"格式全过、质量有差"。
消息格式与 agents/llm_openai.py 的 rollout 驱动器完全一致（system + 任务简报 user +
assistant tool_calls + tool 结果），可直接喂 LLaMA-Factory / trl 的 tool SFT。

注意：SFT 只教"格式与流程"，不教"预报得多准"——数值质量交给 RL。
"""
from __future__ import annotations

import json

from ..case import CaseBundle
from ..env import EnvConfig, ForecastEnv
from .llm_openai import SYSTEM_PROMPT
from .scripted import BASELINES, run_episode


class _OpenEvidenceFormatTutor:
    """Prefix a legal open-evidence workflow without supplying forecast labels.

    The wrapped baseline still determines every forecast number.  This wrapper
    only teaches tool order and the mapping from new tool names to the existing
    controlled evidence vocabulary, which is exactly the frozen-policy failure
    observed in the open-evidence pilot.
    """

    def __init__(self, base):
        self.base = base
        self.prefix = [
            {"name": "get_synoptic_evidence", "args": {"detail": "summary"}},
            {"name": "get_pollution_evidence", "args": {"detail": "summary"}},
        ]

    def begin(self, brief: dict) -> None:
        self.stage = 0
        self.base.begin(brief)

    def act(self, obs: dict) -> dict:
        if self.stage < len(self.prefix):
            action = self.prefix[self.stage]
            self.stage += 1
            return action
        action = self.base.act(obs)
        if action.get("name") == "submit_forecast":
            forecast = action.get("args", {}).get("forecast", {})
            evidence = list(forecast.get("evidence", []))
            evidence.extend([
                {"type": "synoptic", "claim": "已查证开放数值模式的天气形势与模式分歧"},
                {"type": "diagnostic", "claim": "已查证污染组成、空间上游与静态源区诊断"},
            ])
            forecast["evidence"] = evidence
        return action


def transcript_to_messages(env: ForecastEnv) -> list[dict]:
    """env transcript → OpenAI messages（与 rollout 驱动器同构）。"""
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    call_i = 0
    for entry in env.transcript:
        if entry["event"] == "reset":
            msgs.append({"role": "user", "content": "任务简报：\n"
                         + json.dumps(entry["obs"]["brief"], ensure_ascii=False, indent=1)})
        elif entry["event"] == "step":
            cid = f"call_{call_i}"
            call_i += 1
            action = entry["action"]
            msgs.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": cid, "type": "function",
                 "function": {"name": action["name"],
                              "arguments": json.dumps(action.get("args") or {}, ensure_ascii=False)}}]})
            msgs.append({"role": "tool", "tool_call_id": cid,
                         "content": json.dumps(entry["obs"].get("content"), ensure_ascii=False)})
    return msgs


def generate_sft_records(bundle: CaseBundle, agents: list[str] | None = None) -> list[dict]:
    """一个个例 × 各基线 → SFT 记录列表。只保留成功提交（reward 有效）的轨迹。"""
    records = []
    for name in agents or ["persistence", "guidance"]:
        try:
            agent = BASELINES[name](bundle)
        except ValueError:
            continue
        env = ForecastEnv(bundle, EnvConfig())
        tutor_used = bool(bundle.evidence)
        out = run_episode(env, _OpenEvidenceFormatTutor(agent) if tutor_used else agent)
        if out["info"].get("reason") != "submitted":
            continue
        records.append({
            "messages": transcript_to_messages(env),
            "tools": env.tool_specs("openai"),
            "meta": {"case_id": bundle.case_id, "agent": name,
                     "reward": out["reward"], "steps": out["steps"],
                     "open_evidence_format_tutor": tutor_used},
        })
    return records
