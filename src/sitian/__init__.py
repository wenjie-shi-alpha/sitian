"""sitian：京津冀空气质量预报 agent 的可验证任务环境。

核心对象：
    CaseBundle      个例数据包（输入/真值/专家参照，时间门禁）
    ForecastEnv     多步工具调用环境（gym 风格，稀疏可验证 reward）
    score_forecast  规则化评分（等级/事件CSI/区间/转折点/证据一致性）
"""
from .case import CaseBundle
from .env import EnvConfig, ForecastEnv
from .schema import (
    SCHEMA_VERSION,
    example_forecast,
    pm25_to_iaqi,
    pm25_to_level,
    validate_forecast,
)
from .scoring import REWARD_VERSION, RewardConfig, ScoreResult, extract_event, reward_spec, score_forecast
from .synth import make_case, make_default_suite

__version__ = "0.1.0"
__all__ = [
    "CaseBundle",
    "EnvConfig",
    "ForecastEnv",
    "RewardConfig",
    "REWARD_VERSION",
    "reward_spec",
    "ScoreResult",
    "SCHEMA_VERSION",
    "example_forecast",
    "extract_event",
    "make_case",
    "make_default_suite",
    "pm25_to_iaqi",
    "pm25_to_level",
    "score_forecast",
    "validate_forecast",
]
