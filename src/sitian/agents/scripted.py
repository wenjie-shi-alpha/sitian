"""脚本化基线 agent：冒烟测试环境，并为评测表提供参照下限/上限。

- PersistenceAgent      持续性基线：未来=当前水平（业务上最弱的合理基线）
- GuidanceFollowAgent   盲从 EC 指导：数值预报直译（misleading 剧本下会翻车）
- ExpertReplayAgent     专家复现：提交 expert.json 的预报（评测表的"专家基线"行）
"""
from __future__ import annotations

from typing import Optional

from ..case import CaseBundle
from ..env import ForecastEnv
from ..schema import aqi_standard_for_date, daily_aqi, pm25_to_level
from ..scoring import extract_event


def _pointer_segment(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _process_from_daily(dates: list[str], values: list[float], event_level: int,
                        levels: Optional[dict[str, int]] = None) -> dict:
    ev = extract_event(dict(zip(dates, values)), event_level, levels=levels)
    if ev is None:
        return {"has_event": False, "start": None, "peak": None, "end": None}
    return {"has_event": True, "start": ev[0], "peak": ev[1], "end": ev[2]}


def run_episode(env: ForecastEnv, agent) -> dict:
    """运行一个 episode。返回 {reward, info, steps, transcript}。"""
    obs = env.reset()
    agent.begin(obs["brief"])
    reward, info = 0.0, {}
    while True:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break
    return {"reward": reward, "info": info, "steps": env.steps_used, "transcript": env.transcript}


class _Base:
    def begin(self, brief: dict) -> None:
        self.brief = brief
        self.stage = 0

    @property
    def multi_pollutant(self) -> bool:
        if "multi_pollutant" in self.brief:
            return bool(self.brief["multi_pollutant"])
        fmt = (self.brief.get("forecast_format") or {}).get("daily", {})
        if isinstance(fmt, dict) and isinstance(fmt.get("columns"), dict):
            return "o3_range" in fmt["columns"]
        ex = self.brief.get("forecast_example", {}).get("daily", [])
        if isinstance(ex, dict):  # compact table form
            return "o3_range" in ex
        return bool(ex and "o3_range" in ex[0])

    def _forecast_dates(self) -> list[str]:
        """Forecast dates from the brief (issue+1..issue+horizon), independent of
        whether the example uses per-day objects or the compact table."""
        from datetime import date, timedelta
        fmt = (self.brief.get("forecast_format") or {}).get("daily")
        if isinstance(fmt, dict) and fmt.get("day_order"):
            return list(fmt["day_order"])
        example = (self.brief.get("forecast_example") or {}).get("daily")
        if isinstance(example, list) and example and isinstance(example[0], dict):
            return [row["date"] for row in example]
        issue = self.brief.get("issue_date") or (self.brief.get("forecast_example") or {}).get("issue_date")
        horizon = self.brief.get("horizon")
        if horizon is None and isinstance(example, dict):
            horizon = max((len(column) for column in example.values()
                           if isinstance(column, list)), default=0)
        base = date.fromisoformat(issue)
        return [(base + timedelta(days=k)).isoformat() for k in range(1, int(horizon) + 1)]

    def _forecast_from_daily(self, dates: list[str], values: list[float], evidence: list[dict],
                             o3_values: Optional[list[float]] = None,
                             pm10_values: Optional[list[float]] = None) -> dict:
        daily = []
        levels: dict[str, int] = {}
        magnitude: list[float] = []
        for i, (d, v) in enumerate(zip(dates, values)):
            configured = self.brief.get("aqi_standard")
            standard = configured.get(d) if isinstance(configured, dict) else configured
            standard = standard or aqi_standard_for_date(d)
            if o3_values is not None:
                o3 = o3_values[i]
                pm10 = (pm10_values[i] if pm10_values is not None else v * 1.3)
                r = daily_aqi({"PM2.5": v, "PM10": pm10, "O3": o3}, standard=standard)
                level, primary = r["level"], (r["primary"][0] if r["primary"] else None)
                magnitude.append(float(r["aqi"]))
                item = {"date": d, "aqi_level": level, "primary_pollutant": primary,
                        "pm25_range": [max(0, round(v * 0.8)), round(v * 1.2) + 10],
                        "pm10_range": [max(0, round(pm10 * 0.8)), round(pm10 * 1.2) + 15],
                        "o3_range": [max(0, round(o3 * 0.8)), round(o3 * 1.2) + 20]}
            else:
                level = pm25_to_level(v, standard=standard)
                magnitude.append(v)
                item = {"date": d, "aqi_level": level,
                        "primary_pollutant": (None if level == 1 else "PM2.5"),
                        "pm25_range": [max(0, round(v * 0.8)), round(v * 1.2) + 10]}
            levels[d] = level
            daily.append(item)
        return {
            "issue_date": self.brief["issue_date"],
            "region": self.brief["region"],
            "daily": daily,
            # process 表示轻度及以上（AQI>=3）的污染过程；
            # 中度及以上的高影响 event 由评分器另行评估。
            "process": _process_from_daily(dates, magnitude, 3, levels=levels),
            "evidence": evidence,
            "confidence": "medium",
        }


class PersistenceAgent(_Base):
    """查 24h 实况 → 全时段维持当前浓度水平（多污染物个例连 O3 一起持续）。"""

    def act(self, obs: dict) -> dict:
        region = self.brief["region"]
        if self.stage == 0:
            self.stage = 1
            return {"name": "get_observations",
                    "args": {"pollutant": "PM2.5", "region": region, "last_hours": 24, "stride": 3}}
        if self.stage == 1:
            content = obs.get("content", {})
            raw_values = content.get("series", {}).get(region, [])
            vals = [v for v in raw_values if v is not None] or [60.0]
            self._pm25 = sum(vals) / len(vals)
            observed_indexes = [
                i for i in range(len(raw_values) - 1, -1, -1)
                if raw_values[i] is not None
            ][:2]
            self._observation_evidence = []
            for rank, observed_index in enumerate(observed_indexes, 1):
                item = {"type": "observation",
                        "claim": f"近时段实况样本{rank}支持持续性外推"}
                if content.get("evidence_ref"):
                    item.update({
                        "ref": content["evidence_ref"],
                        "field": f"/series/{_pointer_segment(region)}/{observed_index}",
                        "value": raw_values[observed_index],
                    })
                self._observation_evidence.append(item)
            if not self._observation_evidence:
                self._observation_evidence = [{
                    "type": "observation", "claim": "观测缺测时采用持续性兜底"
                }]
            if self.multi_pollutant:
                self.stage = 2
                return {"name": "get_observations",
                        "args": {"pollutant": "O3", "region": region, "last_hours": 24, "stride": 3}}
            self.stage = 3
        o3 = None
        if self.stage == 2:
            ovals = [v for v in obs.get("content", {}).get("series", {}).get(region, [])
                     if v is not None]
            o3 = max(ovals) if ovals else 100.0  # 持续性 O3 用近 24h 峰值近似 8h 日最大
        dates = self._forecast_dates()
        forecast = self._forecast_from_daily(
            dates, [self._pm25] * len(dates),
            self._observation_evidence,
            o3_values=[o3] * len(dates) if o3 is not None else None,
        )
        return {"name": "submit_forecast", "args": {"forecast": forecast}}


class GuidanceFollowAgent(_Base):
    """查模式指导 → 直译为预报。

    全国池优先使用全时段都有定义的 CAMS，以免 train 中有 cmaq、val/test 中没有时，
    同名 baseline 实际代表不同模型。指导尾部缺日采用最后一个可用日的持平外推，
    不再注入与个例无关的 60/100 常数。
    """

    def __init__(self, prefer: tuple[str, ...] = ("cams", "ec", "cmaq", "naqp")):
        self.prefer = prefer

    @staticmethod
    def _carry_forward(daily: dict, dates: list[str], fallback: float) -> list[float]:
        known = [(d, float(daily[d])) for d in dates if d in daily]
        if not known:
            return [fallback] * len(dates)
        first = known[0][1]
        out, last = [], first
        for d in dates:
            if d in daily:
                last = float(daily[d])
            out.append(last)
        return out

    def act(self, obs: dict) -> dict:
        if self.stage == 0:
            self.stage = 1
            return {"name": "get_model_guidance", "args": {}}
        content = obs.get("content", {})
        sources = content.get("sources", {})
        picked_name = next((key for key in self.prefer if key in sources),
                           next(iter(sources), None))
        picked = sources.get(picked_name, {})
        daily_pm25 = picked.get("daily_pm25", {})
        daily_pm10 = picked.get("daily_pm10", {})
        daily_o3 = picked.get("daily_o3max", {})
        dates = self._forecast_dates()
        values = self._carry_forward(daily_pm25, dates, 60.0)
        o3_values = (self._carry_forward(daily_o3, dates, 100.0)
                     if self.multi_pollutant and daily_o3 else None)
        pm10_values = (self._carry_forward(daily_pm10, dates, values[0] * 1.3)
                       if self.multi_pollutant and daily_pm10 else None)
        evidence = []
        if (content.get("evidence_ref") and picked_name and daily_pm25
                and any(day in daily_pm25 for day in dates)):
            for day in [day for day in dates if day in daily_pm25][:2]:
                evidence.append({
                    "type": "model_guidance",
                    "claim": f"模式指导 {day} 的 PM2.5 数值锚点",
                    "ref": content["evidence_ref"],
                    "field": (f"/sources/{_pointer_segment(picked_name)}/daily_pm25/"
                              f"{_pointer_segment(day)}"),
                    "value": daily_pm25[day],
                })
        if not evidence:
            evidence = [{"type": "model_guidance", "claim": "直接采用模式指导逐日数值"}]
        forecast = self._forecast_from_daily(
            dates, values, evidence,
            o3_values=o3_values, pm10_values=pm10_values,
        )
        return {"name": "submit_forecast", "args": {"forecast": forecast}}


class ClimatologyAgent(_Base):
    """气候态基线：逐日报当月 p50（多污染物个例连 O3 p50）。防"抄气候态"被误认为技能。"""

    def __init__(self, bundle: CaseBundle):
        clim = (bundle.meta or {}).get("climatology")
        month = f"{int(bundle.issue_date[5:7]):02d}"
        if not clim or month not in clim or "pm25" not in clim[month]:
            raise ValueError(f"case {bundle.case_id} has no climatology in meta")
        self._pm25 = clim[month]["pm25"]["p50"]
        self._o3 = (clim[month].get("o3_8h") or {}).get("p50")

    def act(self, obs: dict) -> dict:
        dates = self._forecast_dates()
        o3 = [self._o3] * len(dates) if (self.multi_pollutant and self._o3 is not None) else None
        forecast = self._forecast_from_daily(
            dates, [self._pm25] * len(dates),
            [{"type": "expert_prior", "claim": "当月气候态中位数"}], o3_values=o3)
        return {"name": "submit_forecast", "args": {"forecast": forecast}}


class ExpertReplayAgent(_Base):
    """提交专家预报原文（需要 bundle.expert；用于生成'专家基线'行）。"""

    def __init__(self, bundle: CaseBundle):
        if not bundle.expert or "forecast" not in bundle.expert:
            raise ValueError(f"case {bundle.case_id} has no expert.json")
        self._forecast = bundle.expert["forecast"]

    # 先完成标准查证（专家会商本就看过全部材料），再提交——保证 grounding 分量语义一致
    _CONSULT = ("get_observations", "get_diagnostics", "get_model_guidance")

    def act(self, obs: dict) -> dict:
        if self.stage < len(self._CONSULT):
            name = self._CONSULT[self.stage]
            self.stage += 1
            return {"name": name, "args": {}}
        return {"name": "submit_forecast", "args": {"forecast": self._forecast}}


BASELINES = {
    "persistence": lambda bundle: PersistenceAgent(),
    "guidance": lambda bundle: GuidanceFollowAgent(),
    "climatology": lambda bundle: ClimatologyAgent(bundle),
    "expert": lambda bundle: ExpertReplayAgent(bundle),
}


def run_baselines(bundle: CaseBundle, names: Optional[list[str]] = None) -> dict[str, dict]:
    """Run baselines using the forecast-outcome score for fair comparisons.

    ``composite`` is intentionally the evidence-neutral outcome composite.
    ``training_composite`` retains the interactive auxiliary incentives for
    diagnostics only; it must not be used against a non-interactive baseline.
    """
    from ..env import EnvConfig
    results = {}
    for name in names or list(BASELINES):
        try:
            agent = BASELINES[name](bundle)
        except ValueError:
            continue
        env = ForecastEnv(bundle, EnvConfig())
        out = run_episode(env, agent)
        score = out["info"].get("score", {})
        submitted_forecast = next((
            event.get("action", {}).get("args", {}).get("forecast")
            for event in reversed(out["transcript"])
            if event.get("event") == "step"
            and event.get("action", {}).get("name") == "submit_forecast"
            and event.get("obs", {}).get("content", {}).get("accepted")
        ), None)
        results[name] = {
            "composite": score.get("outcome_composite"),
            "outcome_composite": score.get("outcome_composite"),
            "training_composite": score.get("composite"),
            "components": score.get("components", {}),
            "forecast": submitted_forecast,
            "steps": out["steps"],
        }
    return results
