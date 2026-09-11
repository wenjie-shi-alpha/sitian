"""预报对象 schema：agent、评分器、RL 集成之间唯一的数据契约。

设计原则：
- 预报是结构化对象，不是自由文本。评分完全规则化、可复算。
- 字段和受控词表以专家会商语料的"论据覆盖率审计"为准演进，版本号显式管理。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Optional

SCHEMA_VERSION = "0.6.5"

# The process head and the daily head describe the same forecast at different
# resolutions.  Keeping the threshold in the schema contract prevents an agent
# from reporting polluted days while setting ``has_event=false`` to evade the
# process/turning responsibility.
PROCESS_EVENT_LEVEL = 3

AQI_STANDARD_2012 = "HJ633-2012"
AQI_STANDARD_2026 = "HJ633-2026"
AQI_STANDARD_EFFECTIVE_DATE = date(2026, 3, 1)
VALID_AQI_STANDARDS = {AQI_STANDARD_2012, AQI_STANDARD_2026}

# HJ 633 表：六项污染物日尺度 IAQI 分段浓度限值。2026 版自 2026-03-01
# 实施，调整了 PM2.5/PM10 限值；其余日报项目限值保持不变。把版本显式放进
# 数据契约，避免跨标准日期的数据被静默按同一口径打标签。
# 指标口径：PM2.5/PM10/SO2/NO2 为 24h 均值(µg/m³)，CO 为 24h 均值(mg/m³)，
# O3 为 8h 滑动均值日最大值(µg/m³)——标准只定义到 IAQI 300，超 800 按 300 封顶。
_IAQI_INDEX = (0, 50, 100, 150, 200, 300, 400, 500)
_IAQI_BREAKPOINTS_2012: dict[str, tuple] = {
    "PM2.5": (0, 35, 75, 115, 150, 250, 350, 500),
    "PM10": (0, 50, 150, 250, 350, 420, 500, 600),
    "SO2": (0, 50, 150, 475, 800, 1600, 2100, 2620),
    "NO2": (0, 40, 80, 180, 280, 565, 750, 940),
    "CO": (0, 2, 4, 14, 24, 36, 48, 60),
    "O3": (0, 100, 160, 215, 265, 800),
}
_IAQI_BREAKPOINTS_2026 = {
    **_IAQI_BREAKPOINTS_2012,
    "PM2.5": (0, 35, 60, 115, 150, 250, 350, 500),
    "PM10": (0, 50, 120, 250, 350, 420, 500, 600),
}
IAQI_BREAKPOINTS_BY_STANDARD = {
    AQI_STANDARD_2012: _IAQI_BREAKPOINTS_2012,
    AQI_STANDARD_2026: _IAQI_BREAKPOINTS_2026,
}
# 向后兼容：未指定版本的底层换算函数仍沿用 2012 版；面向 episode 的评分器
# 会依据有效日期自动选择标准，或读取 case.meta.aqi_standard 显式覆盖。
IAQI_BREAKPOINTS = _IAQI_BREAKPOINTS_2012


def aqi_standard_for_date(day: str | date) -> str:
    """返回某个有效日期应使用的 HJ 633 版本。"""
    d = parse_date(day) if isinstance(day, str) else day
    return AQI_STANDARD_2026 if d >= AQI_STANDARD_EFFECTIVE_DATE else AQI_STANDARD_2012


def _breakpoints(standard: str) -> dict[str, tuple]:
    try:
        return IAQI_BREAKPOINTS_BY_STANDARD[standard]
    except KeyError:
        raise ValueError(f"unknown AQI standard {standard!r}; expected one of {sorted(VALID_AQI_STANDARDS)}")


def iaqi(pollutant: str, conc: float, *, standard: str = AQI_STANDARD_2012) -> float:
    """污染物日尺度浓度 → IAQI（HJ 633 线性内插；超表封顶）。"""
    bps = _breakpoints(standard)[pollutant]
    if conc < 0:
        raise ValueError(f"{pollutant} concentration must be >= 0, got {conc}")
    for i in range(1, len(bps)):
        if conc <= bps[i]:
            lo, hi = bps[i - 1], bps[i]
            ilo, ihi = _IAQI_INDEX[i - 1], _IAQI_INDEX[i]
            return ilo + (ihi - ilo) * (conc - lo) / (hi - lo)
    return float(_IAQI_INDEX[len(bps) - 1])


def aqi_to_level(aqi: float) -> int:
    """AQI → 等级 1..6（50/100/150/200/300 分界）。"""
    for level, upper in enumerate((50, 100, 150, 200, 300), start=1):
        if aqi <= upper:
            return level
    return 6


def daily_aqi(concs: dict[str, float], *, standard: str = AQI_STANDARD_2012) -> dict:
    """六项（可缺项）日浓度 → {"aqi", "level", "primary"}。

    AQI = max(IAQI)（按 HJ 633 向上取整）；首要污染物 = IAQI 最大者（AQI≤50 无首要污染物，
    可并列）。concs 键 ∈ IAQI_BREAKPOINTS，CO 单位 mg/m³，其余 µg/m³。
    """
    if not concs:
        raise ValueError("concs must not be empty")
    import math
    iaqis = {p: iaqi(p, c, standard=standard) for p, c in concs.items()}
    aqi = math.ceil(max(iaqis.values()))
    primary = sorted(p for p, v in iaqis.items() if math.ceil(v) >= aqi) if aqi > 50 else []
    return {"aqi": aqi, "level": aqi_to_level(aqi), "primary": primary}
AQI_LEVEL_NAMES = {1: "优", 2: "良", 3: "轻度污染", 4: "中度污染", 5: "重度污染", 6: "严重污染"}

VALID_POLLUTANTS = {"PM2.5", "PM10", "O3", "NO2", "SO2", "CO"}
VALID_CONFIDENCE = {"low", "medium", "high"}
# 证据类型受控词表：过程监督/证据引用一致性都按这个词表对齐
EVIDENCE_TYPES = {
    "observation",        # 实况（浓度、能见度等观测）
    "synoptic",           # 天气形势（环流、系统）
    "diagnostic",         # 诊断量（逆温、边界层、湿度、输送）
    "model_guidance",     # 模式指导（EC/CMA/CMAQ 等）
    "analog",             # 相似历史个例
    "previous_forecast",  # 昨日预报（订正参照）
    "expert_prior",       # 经验先验（气候背景、模式系统性偏差）
    "other",
}
# 模型常把工具/返回字段名自然地写进 evidence.type。它们是 harness 命名而非
# 预报技能；validator 接受后立即归一到上面的规范语义，后续评分只看规范类型。
EVIDENCE_TYPE_ALIASES = {
    "pollution_evidence": "diagnostic",
    "source_context": "diagnostic",
    "assessment": "diagnostic",
    "composition": "model_guidance",
}


def pm25_to_level(pm25: float, *, standard: str = AQI_STANDARD_2012) -> int:
    """PM2.5 日均浓度(µg/m³) → AQI 等级 1..6。"""
    if pm25 < 0:
        raise ValueError(f"pm25 must be >= 0, got {pm25}")
    return aqi_to_level(iaqi("PM2.5", pm25, standard=standard))


def pm25_to_iaqi(pm25: float, *, standard: str = AQI_STANDARD_2012) -> float:
    """PM2.5 日均浓度 → IAQI（线性内插，>500 封顶 500）。"""
    return iaqi("PM2.5", pm25, standard=standard)


def parse_date(s: str, field_name: str = "date") -> date:
    try:
        return date.fromisoformat(s)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name}: invalid ISO date {s!r}")


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


@dataclass
class DailyForecast:
    date: str
    aqi_level: int
    pm25_range: tuple[float, float]
    primary_pollutant: Optional[str] = None
    o3_range: Optional[tuple[float, float]] = None  # O3_8h 日最大值区间（全国多污染物个例用）
    pm10_range: Optional[tuple[float, float]] = None  # PM10 日均区间（沙尘/PM10 主导过程必需）
    so2_range: Optional[tuple[float, float]] = None
    no2_range: Optional[tuple[float, float]] = None
    co_range: Optional[tuple[float, float]] = None  # CO is mg/m³, other gases µg/m³

    def to_dict(self) -> dict:
        out = {
            "date": self.date,
            "aqi_level": self.aqi_level,
            "pm25_range": list(self.pm25_range),
            "primary_pollutant": self.primary_pollutant,
        }
        if self.o3_range is not None:
            out["o3_range"] = list(self.o3_range)
        if self.pm10_range is not None:
            out["pm10_range"] = list(self.pm10_range)
        for name in ("so2_range", "no2_range", "co_range"):
            value = getattr(self, name)
            if value is not None:
                out[name] = list(value)
        return out


@dataclass
class ProcessForecast:
    has_event: bool
    start: Optional[str] = None
    peak: Optional[str] = None
    end: Optional[str] = None

    def to_dict(self) -> dict:
        return {"has_event": self.has_event, "start": self.start, "peak": self.peak, "end": self.end}


@dataclass
class Forecast:
    issue_date: str
    region: str
    daily: list[DailyForecast]
    process: Optional[ProcessForecast] = None
    evidence: list[dict] = field(default_factory=list)
    confidence: str = "medium"

    def to_dict(self) -> dict:
        return {
            "issue_date": self.issue_date,
            "region": self.region,
            "daily": [d.to_dict() for d in self.daily],
            "process": self.process.to_dict() if self.process else None,
            "evidence": self.evidence,
            "confidence": self.confidence,
        }

    def evidence_types(self) -> set[str]:
        return {e.get("type") for e in self.evidence if isinstance(e, dict) and e.get("type")}


DERIVED_PRIMARY_PRIORITY = ("PM2.5", "PM10", "O3", "NO2", "SO2", "CO")
OPTIONAL_GAS_FIELDS = {"so2_range": "SO2", "no2_range": "NO2", "co_range": "CO"}
OPTIONAL_GAS_MAX = {"so2_range": 6000, "no2_range": 2000, "co_range": 200}
DAILY_TABLE_FIELDS = ("pm25_range", "pm10_range", "o3_range", *OPTIONAL_GAS_FIELDS)
FLAT_COLUMN_PAIRS = {
    "pm25_range": ("pm25_lo", "pm25_hi"),
    "pm10_range": ("pm10_lo", "pm10_hi"),
    "o3_range": ("o3_lo", "o3_hi"),
    "so2_range": ("so2_lo", "so2_hi"),
    "no2_range": ("no2_lo", "no2_hi"),
    "co_range": ("co_lo", "co_hi"),
}
FLAT_COLUMN_KEYS = tuple(key for pair in FLAT_COLUMN_PAIRS.values() for key in pair)
OPTIONAL_GAS_COLUMNS = {
    f"{pollutant.lower()}_{side}": f"可选 {pollutant} 日均浓度区间{label} ({'mg/m³' if pollutant == 'CO' else 'µg/m³'})"
    for pollutant in OPTIONAL_GAS_FIELDS.values()
    for side, label in (("lo", "下限"), ("hi", "上限"))
}


EVIDENCE_VALUE_MAX_ITEMS = 8


def _is_evidence_value(value: Any) -> bool:
    """A cited fact is a scalar or a short tuple of scalars (schema-v0.6.4).

    The compact evidence views encode many facts as tuples such as
    ``[speed, from_deg, spread]``; citing the whole tuple is as verifiable as
    citing one element, so it is accepted.  Objects and nested lists are not.
    """
    if isinstance(value, dict):
        return False
    if isinstance(value, list):
        return (len(value) <= EVIDENCE_VALUE_MAX_ITEMS
                and all(not isinstance(item, (dict, list)) for item in value))
    return True


def _ordered_bounds(pair) -> tuple[float, float]:
    """An interval is an unordered pair of bounds: [hi, lo] denotes the same
    80% interval as [lo, hi].  schema-v0.6.3 accepts either order instead of
    rejecting the submission (column-wise entry made swapped bounds the most
    frequent residual format error of the frozen policy)."""
    lo, hi = float(pair[0]), float(pair[1])
    return (lo, hi) if lo <= hi else (hi, lo)


def expand_daily_table(table: dict, base: date, horizon: int) -> list[dict]:
    """Expand the compact ``daily`` table into per-day objects.

    Unknown keys are carried into every day so that field-level validation
    still reports them; a column with the wrong length is passed through as
    an empty list so the ``exactly horizon entries`` error fires.
    """
    columns = {key: table.get(key) for key in DAILY_TABLE_FIELDS if key in table}
    if not columns or any(not isinstance(value, list) or len(value) != horizon
                          for value in columns.values()):
        return []
    rows = []
    for index in range(horizon):
        row = {"date": (base + timedelta(days=index + 1)).isoformat()}
        for key, column in columns.items():
            row[key] = column[index]
        rows.append(row)
    return rows


def derive_process(daily: list["DailyForecast"], aqi_values: list[float]) -> "ProcessForecast":
    """Derive the AQI>=PROCESS_EVENT_LEVEL process head from daily levels.

    Select the run containing the highest numeric AQI, breaking equal AQI
    ties by earliest day, exactly as the truth-side event extraction does.
    """
    polluted = [index for index, item in enumerate(daily)
                if item.aqi_level >= PROCESS_EVENT_LEVEL]
    if not polluted:
        return ProcessForecast(False)
    runs: list[tuple[int, int]] = []
    run_start = previous = polluted[0]
    for index in polluted[1:]:
        if index != previous + 1:
            runs.append((run_start, previous))
            run_start = index
        previous = index
    runs.append((run_start, previous))
    peak = max(polluted, key=lambda index: aqi_values[index])
    start, end = next(run for run in runs if run[0] <= peak <= run[1])
    return ProcessForecast(True, daily[start].date, daily[peak].date, daily[end].date)


def validate_forecast(
    obj: Any,
    *,
    issue_date: str,
    horizon: int,
    region: Optional[str] = None,
    require_pm10_range: bool = False,
    require_o3_range: bool = False,
    aqi_standard: Optional[str | dict[str, str]] = None,
) -> tuple[Optional[Forecast], list[str]]:
    """校验预报对象。返回 (Forecast 或 None, 错误列表)。

    要求 daily 恰好覆盖 issue_date+1 .. issue_date+horizon 且顺序连续——
    这是评分和防泄漏审计的基础，宁严勿宽。
    """
    errors: list[str] = []
    if not isinstance(obj, dict):
        return None, ["forecast must be a JSON object"]

    base = parse_date(issue_date, "issue_date")

    if obj.get("issue_date") != issue_date:
        errors.append(f"issue_date must be {issue_date!r}, got {obj.get('issue_date')!r}")
    if region is not None and obj.get("region") != region:
        errors.append(f"region must be {region!r}, got {obj.get('region')!r}")

    daily_out: list[DailyForecast] = []
    daily_aqi_values: list[float] = []
    daily = obj.get("daily")
    if daily is None and any(key in obj for key in FLAT_COLUMN_KEYS):
        # schema-v0.6.2 flat-column form: pm25_lo/pm25_hi/... lists at the top
        # level (nesting depth 2).  The frozen 8B policy mis-closed brackets in
        # every deeper layout we tried; this is the shallowest faithful encoding.
        daily = {}
        for field, (lo_key, hi_key) in FLAT_COLUMN_PAIRS.items():
            lo, hi = obj.get(lo_key), obj.get(hi_key)
            if isinstance(lo, list) and isinstance(hi, list) and len(lo) == len(hi):
                daily[field] = [[a, b] for a, b in zip(lo, hi)]
            elif lo is not None or hi is not None:
                daily[field] = []  # length mismatch -> "exactly horizon" error
    if isinstance(daily, dict):
        # schema-v0.6.1 compact table form: {"pm25_range": [[lo,hi]*H], ...}.
        # Dates are implied by position (issue+1..issue+H).  The nested
        # per-day-object form stays accepted; the table needs far fewer
        # brackets, which the frozen 8B policy mangled systematically.
        daily = expand_daily_table(daily, base, horizon)
    if not isinstance(daily, list) or len(daily) != horizon:
        errors.append(f"daily must be a list of exactly {horizon} entries (issue+1..issue+{horizon})")
    else:
        for i, item in enumerate(daily):
            expected = (base + timedelta(days=i + 1)).isoformat()
            if not isinstance(item, dict):
                errors.append(f"daily[{i}] must be an object")
                continue
            if item.get("date") != expected:
                errors.append(f"daily[{i}].date must be {expected!r}, got {item.get('date')!r}")
            # schema-v0.6.0: aqi_level / primary_pollutant / process are derived
            # by the harness from the submitted concentration intervals (HJ 633
            # on interval midpoints).  Supplied categorical heads are accepted
            # for backward compatibility but never used; a policy therefore
            # cannot submit a self-contradictory forecast, and the reward sees
            # exactly one coherent object.
            level = item.get("aqi_level")
            if level is not None and (
                not isinstance(level, int) or isinstance(level, bool) or not 1 <= level <= 6
            ):
                errors.append(f"daily[{i}].aqi_level, if given, must be an integer in 1..6")
            level = 0
            rng = item.get("pm25_range")
            lo, hi = 0.0, 0.0
            if (
                not isinstance(rng, (list, tuple))
                or len(rng) != 2
                or not all(_is_number(v) for v in rng)
            ):
                errors.append(f"daily[{i}].pm25_range must be [lo, hi] numbers")
            else:
                lo, hi = _ordered_bounds(rng)
                if not (0 <= lo <= hi <= 1000):
                    errors.append(f"daily[{i}].pm25_range requires 0 <= lo <= hi <= 1000")
            pp = item.get("primary_pollutant")
            if pp is not None and pp not in VALID_POLLUTANTS:
                errors.append(f"daily[{i}].primary_pollutant, if given, must be one of {sorted(VALID_POLLUTANTS)} or null")
            pp = None
            o3rng = item.get("o3_range")
            o3_out: Optional[tuple[float, float]] = None
            if o3rng is None and require_o3_range:
                errors.append(f"daily[{i}].o3_range is required for national multi-pollutant cases")
            elif o3rng is not None:
                if (
                    not isinstance(o3rng, (list, tuple))
                    or len(o3rng) != 2
                    or not all(_is_number(v) for v in o3rng)
                ):
                    errors.append(f"daily[{i}].o3_range must be [lo, hi] numbers or omitted")
                else:
                    o3lo, o3hi = _ordered_bounds(o3rng)
                    if not (0 <= o3lo <= o3hi <= 1200):
                        errors.append(f"daily[{i}].o3_range requires 0 <= lo <= hi <= 1200")
                    else:
                        o3_out = (o3lo, o3hi)
            pm10rng = item.get("pm10_range")
            pm10_out: Optional[tuple[float, float]] = None
            if pm10rng is None and require_pm10_range:
                errors.append(f"daily[{i}].pm10_range is required for national multi-pollutant cases")
            elif pm10rng is not None:
                if (
                    not isinstance(pm10rng, (list, tuple))
                    or len(pm10rng) != 2
                    or not all(_is_number(v) for v in pm10rng)
                ):
                    errors.append(f"daily[{i}].pm10_range must be [lo, hi] numbers or omitted")
                else:
                    pm10lo, pm10hi = _ordered_bounds(pm10rng)
                    # The admitted native archive contains a 2113.8 µg/m³
                    # dust day; a 2000 cap made its correct forecast invalid.
                    if not (0 <= pm10lo <= pm10hi <= 10000):
                        errors.append(f"daily[{i}].pm10_range requires 0 <= lo <= hi <= 10000")
                    else:
                        pm10_out = (pm10lo, pm10hi)
            gas_ranges = {}
            for field_name in OPTIONAL_GAS_FIELDS:
                value = item.get(field_name)
                if value is None:
                    continue
                if (not isinstance(value, (list, tuple)) or len(value) != 2
                        or not all(_is_number(v) and 0 <= v <= OPTIONAL_GAS_MAX[field_name] for v in value)):
                    errors.append(f"daily[{i}].{field_name} requires two numbers in 0..{OPTIONAL_GAS_MAX[field_name]}")
                else:
                    gas_ranges[field_name] = _ordered_bounds(value)
            # Derive the categorical heads from the interval midpoints under the
            # AQI standard valid on that day.  Ties in IAQI resolve by a fixed
            # pollutant priority so the output is deterministic.
            standard = (
                aqi_standard.get(expected) if isinstance(aqi_standard, dict)
                else aqi_standard
            ) or aqi_standard_for_date(expected)
            midpoints = {"PM2.5": (lo + hi) / 2.0}
            if pm10_out is not None:
                midpoints["PM10"] = (pm10_out[0] + pm10_out[1]) / 2.0
            if o3_out is not None:
                midpoints["O3"] = (o3_out[0] + o3_out[1]) / 2.0
            for field_name, bounds in gas_ranges.items():
                midpoints[OPTIONAL_GAS_FIELDS[field_name]] = bounds[0] / 2.0 + bounds[1] / 2.0
            numeric_aqi = 0.0
            try:
                derived = daily_aqi(midpoints, standard=standard)
                numeric_aqi = float(derived["aqi"])
                level = int(derived["level"])
                pp = next((name for name in DERIVED_PRIMARY_PRIORITY
                           if name in derived["primary"]), None)
            except ValueError as exc:
                errors.append(f"daily[{i}]: cannot derive AQI from intervals ({exc})")
            daily_out.append(DailyForecast(
                item.get("date", expected), level, (lo, hi), pp, o3_out, pm10_out,
                **gas_ranges,
            ))
            daily_aqi_values.append(numeric_aqi)

    process_out: Optional[ProcessForecast] = None
    supplied_process = obj.get("process")
    if supplied_process is not None and not isinstance(supplied_process, dict):
        errors.append("process, if given, must be an object or null")
    if len(daily_out) == horizon and all(1 <= item.aqi_level <= 6 for item in daily_out):
        process_out = derive_process(daily_out, daily_aqi_values)

    evidence = obj.get("evidence", [])
    normalized_evidence: list[dict] = []
    if not isinstance(evidence, list) or len(evidence) > 12:
        errors.append("evidence must be a list with at most 12 entries")
    else:
        for i, ev in enumerate(evidence):
            supplied = ev.get("type") if isinstance(ev, dict) else None
            canonical = EVIDENCE_TYPE_ALIASES.get(supplied, supplied)
            if not isinstance(ev, dict) or canonical not in EVIDENCE_TYPES:
                errors.append(
                    f"evidence[{i}].type {supplied!r} must be one of {sorted(EVIDENCE_TYPES)}; "
                    f"accepted aliases are {sorted(EVIDENCE_TYPE_ALIASES)}"
                )
            elif not isinstance(ev.get("claim"), str) or not 0 < len(ev["claim"]) <= 300:
                errors.append(f"evidence[{i}].claim must be a non-empty string (<=300 chars)")
            else:
                normalized = dict(ev)
                normalized["type"] = canonical
                ref, pointer = ev.get("ref"), ev.get("field")
                if ref is not None and (not isinstance(ref, str) or not 0 < len(ref) <= 64):
                    errors.append(f"evidence[{i}].ref must be a non-empty string (<=64 chars) or omitted")
                if pointer is not None and (
                    not isinstance(pointer, str) or not pointer.startswith("/") or len(pointer) > 300
                ):
                    errors.append(f"evidence[{i}].field must be an RFC 6901 JSON Pointer (<=300 chars) or omitted")
                value = ev.get("value")
                if "value" in ev and not _is_evidence_value(value):
                    errors.append(
                        f"evidence[{i}].value must be a scalar, a short list of scalars, "
                        "or omitted"
                    )
                normalized_evidence.append(normalized)

    confidence = obj.get("confidence", "medium")
    if confidence not in VALID_CONFIDENCE:
        errors.append(f"confidence must be one of {sorted(VALID_CONFIDENCE)}")
        confidence = "medium"

    if errors:
        return None, errors
    return (
        Forecast(
            issue_date,
            obj.get("region", region or ""),
            daily_out,
            process_out,
            normalized_evidence,
            confidence,
        ),
        [],
    )


def example_forecast(issue_date: str, horizon: int, region: str, multi_pollutant: bool = False) -> dict:
    """合法示例（逐日对象形式；数值为占位）。测试与脚本基线沿用此形式。"""
    base = parse_date(issue_date)
    daily = []
    for i in range(horizon):
        d = (base + timedelta(days=i + 1)).isoformat()
        item = {"date": d, "pm25_range": [40, 70]}
        if multi_pollutant:
            item["pm10_range"] = [50, 100]
            item["o3_range"] = [80, 140]
        daily.append(item)
    return {
        "issue_date": issue_date,
        "region": region,
        "daily": daily,
        "evidence": [{"type": "model_guidance", "claim": "EC 指导未来一周扩散条件总体有利"}],
    }


def forecast_format_description(issue_date: str, horizon: int, region: str,
                                multi_pollutant: bool = False) -> dict:
    """Structure-only description of the submit payload for the task brief.

    Deliberately carries no numbers: a numeric example was copied verbatim by
    the frozen policy in most rollouts, which destroys the reward signal.
    """
    base = parse_date(issue_date)
    dates = [(base + timedelta(days=k)).isoformat() for k in range(1, horizon + 1)]
    columns = {"pm25_lo": "PM2.5 日均 80% 区间下限 µg/m³", "pm25_hi": "PM2.5 日均 80% 区间上限 µg/m³"}
    if multi_pollutant:
        columns["pm10_lo"] = "PM10 日均 80% 区间下限 µg/m³"
        columns["pm10_hi"] = "PM10 日均 80% 区间上限 µg/m³"
        columns["o3_lo"] = "O3_8h 日最大 80% 区间下限 µg/m³"
        columns["o3_hi"] = "O3_8h 日最大 80% 区间上限 µg/m³"
    return {
        "issue_date": issue_date,
        "region": region,
        "shape": (f"每个列键是长度恰为 {horizon} 的数字列表，第 k 项对应第 k 个预报日；"
                  "列键直接放在 forecast 顶层，不要再包一层 daily"),
        "day_order": dates,
        "columns": columns,
        **({"optional_columns": OPTIONAL_GAS_COLUMNS,
            "optional_gases": "有SO2/NO2/CO主导风险时可成对提交对应lo/hi整列；CO单位mg/m³。"
                "省略表示未预报该项，不当作已知为零；已提交中点参与AQI、首污、过程，"
                "真值仍为六污染物AQI；当前区间分量只评PM2.5/PM10/O3。"}
           if multi_pollutant else {}),
        "rule": "数字必须来自你对已查证据的判断；不要抄任何示例或占位值；区间目标 80% 覆盖，lo <= hi",
        "evidence": "列表，每项 {type, claim, ref, field, value}；type 取八类词表之一",
        "not_submitted": ["daily", "aqi_level", "primary_pollutant", "process"],
    }


def example_forecast_compact(issue_date: str, horizon: int, region: str,
                             multi_pollutant: bool = False) -> dict:
    """策略提示用的紧凑示例：daily 为按日顺序的区间表，日期由位置隐含。

    占位区间逐日略有不同，避免策略机械复制同一组数字。
    """
    pm25 = [[30 + 5 * i, 60 + 5 * i] for i in range(horizon)]
    daily = {"pm25_range": pm25}
    if multi_pollutant:
        daily["pm10_range"] = [[50 + 10 * i, 100 + 10 * i] for i in range(horizon)]
        daily["o3_range"] = [[90 + 10 * i, 150 + 10 * i] for i in range(horizon)]
    return {
        "issue_date": issue_date,
        "region": region,
        "daily": daily,
        "evidence": [{"type": "model_guidance", "claim": "CAMS 指导未来三天 PM2.5 缓升"}],
    }


def forecast_tool_schema(
    issue_date: str, horizon: int, region: str, *, multi_pollutant: bool = False
) -> dict:
    """submit_forecast 工具的参数 JSON Schema（供 OpenAI function calling）。"""
    return {
        "type": "object",
        "properties": {
            "forecast": {
                "type": "object",
                "description": (
                    f"结构化预报对象。issue_date 必须为 {issue_date}，region 必须为 {region}，"
                    f"区间按列提交：pm25_lo/pm25_hi（多污染物个例另加 pm10_lo/pm10_hi/o3_lo/o3_hi），"
                    f"每列恰好 {horizon} 个数字，第 k 项对应第 k 个预报日。只需提交浓度区间；"
                    "AQI 等级、首要污染物与 AQI>=3 污染过程由环境按 HJ 633 从区间中点派生，"
                    "峰值按数值AQI选日；可成对加so2_lo/hi、no2_lo/hi、co_lo/hi表达其他首污风险（CO为mg/m³）。"
                    "不接受也不需要单独提交。"
                ),
                "properties": {
                    "issue_date": {"type": "string"},
                    "region": {"type": "string"},
                    **{
                        key: {
                            "type": "array",
                            "minItems": horizon, "maxItems": horizon,
                            "items": {"type": "number"},
                            "description": (
                                f"{label}，长度恰为 {horizon}，第 k 项对应 issue_date 后第 k 天"
                            ),
                        }
                        for key, label in (
                            [("pm25_lo", "PM2.5 日均 80% 区间下限(µg/m³)"),
                             ("pm25_hi", "PM2.5 日均 80% 区间上限(µg/m³)")]
                            + ([("pm10_lo", "PM10 日均 80% 区间下限(µg/m³)"),
                                ("pm10_hi", "PM10 日均 80% 区间上限(µg/m³)"),
                                ("o3_lo", "O3_8h 日最大 80% 区间下限(µg/m³)"),
                                ("o3_hi", "O3_8h 日最大 80% 区间上限(µg/m³)")]
                               if multi_pollutant else [])
                            + (list(OPTIONAL_GAS_COLUMNS.items()) if multi_pollutant else [])
                        )
                    },
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": sorted(EVIDENCE_TYPES),
                                    "description": (
                                        "必须使用本枚举中的证据类别，不得填写工具名。尤其禁止 "
                                        "pollution_evidence、assessment；污染组成用 model_guidance，"
                                        "天气形势用 synoptic，客观诊断用 diagnostic。"
                                    ),
                                },
                                "claim": {"type": "string"},
                                "ref": {
                                    "type": "string",
                                    "description": "已查工具返回的 evidence_ref；用于可复算 grounding。",
                                },
                                "field": {
                                    "type": "string",
                                    "description": "指向该工具返回中一个标量事实的 RFC 6901 JSON Pointer。",
                                },
                                "value": {
                                    "type": ["string", "number", "integer", "boolean", "null"],
                                    "description": "field 指向的精确标量值。",
                                },
                            },
                            "required": ["type", "claim"],
                        },
                    },
                    # Categorical confidence remains validator-compatible for
                    # legacy artifacts but is deliberately absent from the
                    # policy-facing schema. Required 80% pollutant intervals
                    # carry the calibrated uncertainty claim.
                },
                "required": (["issue_date", "region", "pm25_lo", "pm25_hi"]
                             + (["pm10_lo", "pm10_hi", "o3_lo", "o3_hi"] if multi_pollutant else [])
                             + ["evidence"]),
            }
        },
        "required": ["forecast"],
    }
