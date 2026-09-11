"""RL 框架的 reward 入口。

两种用法：
1. 单轮 RL（起步，TRL GRPOTrainer / verl 自定义 reward）：
   模型一次性输出含预报 JSON 的文本 → `compute_score(solution_str, case_dir)`。
   prompt 由离线脚本从 case bundle 渲染（case包全文 + schema 说明）。
2. 多轮 agent RL（目标形态，verl multi-turn rollout / 自定义 rollout）：
   直接用 ForecastEnv 做 rollout，episode 末端 reward 即环境返回值，
   参考 agents/llm_openai.py 的循环实现。
"""
from __future__ import annotations

import json
import re
from typing import Optional

from ..case import CaseBundle
from ..scoring import RewardConfig, score_forecast

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_forecast_json(text: str) -> Optional[dict]:
    """从模型自由文本中提取预报 JSON：优先 ```json 代码块（取最后一个），
    否则从最后一个 '{' 起做括号配平扫描。"""
    blocks = _JSON_BLOCK.findall(text)
    for raw in reversed(blocks):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for s in reversed(starts):
        depth = 0
        for i in range(s, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[s : i + 1])
                        if isinstance(obj, dict) and "daily" in obj:
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
    return None


def compute_score(
    solution_str: str,
    case_dir: str,
    cfg: Optional[RewardConfig] = None,
    **_ignored,
) -> float:
    """verl 风格 reward 函数：模型输出文本 + case 目录 → [0,1] 分数。

    解析失败 / 校验失败 / 无真值 → 0.0（格式与可验证性是门禁）。
    """
    obj = extract_forecast_json(solution_str)
    if obj is None:
        return 0.0
    # 允许模型输出 {"forecast": {...}} 或直接输出预报对象
    forecast = obj.get("forecast") if isinstance(obj.get("forecast"), dict) else obj
    bundle = CaseBundle.load(case_dir)
    # 全国任务必须按六污染物 AQI 口径评分；旧实现只传 PM2.5，
    # 会让单轮 RL 与真实多轮环境优化不同的目标。
    truth = bundle.truth_daily_full()
    if truth is None:
        return 0.0
    result = score_forecast(
        forecast,
        truth,
        issue_date=bundle.issue_date,
        horizon=bundle.horizon,
        region=bundle.region,
        expert_evidence_types=bundle.expert_evidence_types(),
        aqi_standard=(bundle.meta or {}).get("aqi_standard"),
        cfg=cfg,
    )
    return float(result.composite)
