"""预报评分：完全规则化、可复算，不依赖 LLM judge。

组成（可配权重，缺失分量自动去权归一）：
    level     逐日 AQI 等级得分（事件日加权；完全命中 1，偏一级 offby1_credit）
    event     污染事件日稠密 CSI 塑形（硬 CSI 单独记录；无真实事件正确否定时弃权）
    interval  PM2.5/O3 区间得分（覆盖与未覆盖均连续，区间过宽衰减）
    turning   污染过程（AQI>=3）转折点得分（起/峰/末各按日偏差衰减）
    evidence  证据引用一致性（与专家证据类型的 Jaccard；无专家参照时弃权）
    primary   首要污染物分类（六污染物真值时激活）
    grounding 证据 ref/工具/语义分区/字段/标量值与实际工具返回一致（环境内激活）；
              不声称自动理解或核验自由文本 claim

格式是门禁：校验不通过 composite 直接 0。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from dataclasses import dataclass, field
from typing import Optional

from .schema import (
    Forecast,
    SCHEMA_VERSION,
    aqi_standard_for_date,
    daily_aqi,
    parse_date,
    pm25_to_level,
    validate_forecast,
)


# 证据类型 → 可支撑它的工具（grounding 分量：引用必须对应本 episode 真实调用过的工具）
# expert_prior / other 无法由环境工具核验，因此不计作可验证证据；它们可以出现在
# evidence 中，但不能单独换取 grounding 奖励。
EVIDENCE_TOOL_MAP = {
    "observation": {"get_observations", "get_assessment", "get_pollution_evidence",
                    "get_process_evidence"},
    "diagnostic": {"get_diagnostics", "get_assessment", "get_synoptic_evidence",
                   "get_pollution_evidence", "get_process_evidence", "get_native_meteorology", "compute_diffusion_conditions"},
    "synoptic": {"get_diagnostics", "get_assessment", "get_synoptic_evidence",
                 "analyze_chart", "describe_image", "get_process_evidence"},
    "model_guidance": {"get_model_guidance", "get_assessment", "get_pollution_evidence",
                       "get_guidance_bias", "get_process_evidence"},
    "previous_forecast": {"get_previous_forecast"},
    "analog": {"find_similar_cases", "get_historical_case"},
    # validator 会把这些别名归一为规范类型；保留显式映射，供历史/预校验
    # Forecast 对象和 grounding 审计直接查询时使用。
    "pollution_evidence": {"get_pollution_evidence"},
    "assessment": {"get_assessment"},
    "composition": {"get_pollution_evidence"},
}

GROUNDING_SECTION_RULES = {
    "get_native_meteorology": {"diagnostic": ("/series/",)},
    "compute_diffusion_conditions": {"diagnostic": ("/samples/",)},
    "find_similar_cases": {"analog": ("/analogs/",)},
    "get_historical_case": {"analog": (
        "/historical_case/issue_inputs/feature_summary/",
        "/historical_case/issue_inputs/guidance/",
        "/historical_case/observed_outcomes/",
        "/historical_case/guidance_errors/",
    )},
    "get_process_evidence": {
        "observation": ("/initial_pollution_state/",),
        "synoptic": ("/weather_trajectory_",),
        "diagnostic": ("/daily_surface_dispersion/", "/transport_context/"),
        "model_guidance": ("/pollution_guidance_anchor/",),
    },
    "get_assessment": {
        "observation": ("/obs_trend/",),
        "diagnostic": ("/daily_signals/",),
        "synoptic": ("/daily_signals/",),
        "model_guidance": ("/guidance_meta/",),
    },
    "get_pollution_evidence": {
        "observation": ("/fires/",),
        "diagnostic": ("/source_context/",),
        "model_guidance": ("/composition/",),
    },
}

TRIVIAL_GROUNDING_LEAVES = {
    "selected_path", "surface_day_complete", "rain_evaluable", "aggregation_note",
    "available", "evidence_ref", "citation_examples", "grounding_instruction",
    "contract_version", "detail", "note", "query_note", "reason", "unit",
    "cycle", "source", "models", "n_models", "role", "valid_time",
    "valid_utc", "valid_bjt", "lead_hour", "issue_relative_hour", "lead_days", "step_hour",
    "submission_evidence_type", "submission_evidence_type_mapping", "n", "count",
}

# 任何改变 composite 语义、分量激活规则或默认参数的改动都必须升版。
# v0.7.9：同一科学事实的正确重复仍不重复得分，但同一 ref/pointer 后追加
# 矛盾或伪造值会被计为无效结构化引用并进入分母，不能利用去重隐藏错误断言。
# v0.7.8：schema v0.5.6 不再向策略索要无经验校准的 categorical confidence；
# interval 塑形改为随标准 80% interval score 连续指数衰减，移除超宽区间仍可
# 固定领取 0.2 的平台；CAMS issue_relative_hour 被明确视为元数据，地形自适应
# 低层事实可作为 diagnostic grounding。
# v0.7.7：schema v0.5.5 把 process start/peak/end 绑定到逐日 AQI>=3 的
# 连续污染段及其最高等级日，堵住逐日头与转折头分别猜、分别取分的漏洞。
# v0.7.6：schema v0.5.4 要求 process 头始终存在，并在 IAQI 尺度验证
# 声明首污确实可能压过其他区间的最低值，堵住“各输出头单独可得分、联合不可能”的漏洞。
# v0.7.5：schema v0.5.3 强制 AQI 等级、首要污染物与三类浓度区间形成
# 至少一组物理/标准口径上可能的联合预报；并支持逐日显式 AQI 标准映射。
# v0.7.4：结构化引用只要提供了 ref/field/value 中任一字段，就必须完整且
# 可语义核验；错误/伪造的结构化尝试不再回退领取 type-only 信用。
# v0.7.3：训练用 composite 与公平比较用 outcome_composite 显式分离；
# grounding 满分同时要求两条不重复事实和两个证据类别，并拒绝 null、
# 空字符串及非有限数值冒充科学事实；事实去重按工具返回内容哈希与
# JSON Pointer，不能靠 42/42.0 或重复调用换取多条信用；schema v0.5.2
# 要求全国多污染物输出同时提交 PM10/O3 区间。
# v0.8.0：schema v0.6.0 把 aqi_level / primary_pollutant / process 改为由 harness
# 从提交的浓度区间中点按 HJ 633 派生；策略只提交区间，联合一致性由构造保证而不再
# 作为格式门禁。分量定义与权重不变，但输入合同变化，旧版本分数不可直接比较。
# v0.8.1：evidence.value 允许为标量短列表（元组事实），逐元素精确核验；
# 部分匹配或含缺测的元组不算事实。分量与权重不变。
# v0.8.2：可验证 analog 扩展到门禁后的历史详情，限制引用到历史事实；方法卡不增加奖励。
# v0.8.3：原生气象与通风代理可引用；仅实际样本数值计分，参数/阈值/元数据不计分。
REWARD_VERSION = "0.8.3"
OUTCOME_COMPONENTS = ("level", "event", "interval", "turning", "primary")


@dataclass
class RewardConfig:
    weights: dict = field(default_factory=lambda: {
        "level": 0.35,
        "event": 0.25,
        "interval": 0.15,
        "turning": 0.15,
        "evidence": 0.10,
        "primary": 0.10,   # 首要污染物；仅多污染物真值时激活（旧个例自动弃权，分数不变）
        "grounding": 0.05, # 证据-工具一致性；仅环境内评分（有工具调用记录）时激活
    })
    event_level: int = 4          # "事件日"定义：AQI 等级 >= 4（中度及以上）
    process_level: int = 3        # "污染过程"定义：AQI 等级 >= 3（轻度及以上）
    offby1_credit: float = 0.5    # 等级偏一级的部分分
    event_day_weight: float = 2.0 # 事件日在 level 分量中的权重
    width_free: float = 30.0      # 区间宽度免罚额度（µg/m³）
    width_scale: float = 120.0    # 超额宽度的衰减尺度
    pm10_width_free: float = 50.0
    pm10_width_scale: float = 200.0
    o3_width_free: float = 40.0
    o3_width_scale: float = 160.0
    interval_alpha: float = 0.2     # 诊断用 80% 区间分数（不直接进入 composite）
    event_near_miss_credit: float = 0.25  # 真事件日仅报到 event_level-1 的塑形分
    grounding_type_only_credit: float = 0.2  # 调用对类工具但未结构化引用
    minimum_grounding_assertions: int = 2     # 满分所需的不重复、可语义核验断言数
    minimum_grounding_types: int = 2          # 满分所需的不同可核验证据类别数


def reward_spec(cfg: Optional[RewardConfig] = None) -> dict:
    """可写入训练/评测产物的 reward 身份；版本和配置 hash 共同防止误比较。"""
    cfg = cfg or RewardConfig()
    config = asdict(cfg)
    identity_payload = {
        "schema_version": SCHEMA_VERSION,
        "config": config,
        "outcome_components": OUTCOME_COMPONENTS,
    }
    encoded = json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "version": REWARD_VERSION,
        "schema_version": SCHEMA_VERSION,
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "config": config,
        "outcome_components": list(OUTCOME_COMPONENTS),
        "comparison_score": "outcome_composite",
    }


@dataclass
class ScoreResult:
    composite: float
    valid: bool
    outcome_composite: float = 0.0
    components: dict = field(default_factory=dict)   # 分量得分（弃权为 None）
    weights_used: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "composite": self.composite,
            "outcome_composite": self.outcome_composite,
            "valid": self.valid,
            "components": self.components,
            "weights_used": self.weights_used,
            "details": self.details,
            "errors": self.errors,
        }


def extract_event(
    truth_daily: dict[str, float],
    event_level: int,
    levels: Optional[dict[str, int]] = None,
) -> Optional[tuple[str, str, str]]:
    """从真值提取污染过程 (start, peak, end)：取包含最高浓度事件日的连续段。

    truth_daily 为逐日数值（PM2.5 或 AQI，仅用于定峰值）；levels 给出逐日等级则
    直接使用（多污染物 AQI 等级），否则按 PM2.5 换算。
    """
    days = sorted(truth_daily)
    if levels is not None:
        flags = [levels[d] >= event_level for d in days]
    else:
        flags = [pm25_to_level(truth_daily[d]) >= event_level for d in days]
    if not any(flags):
        return None
    runs: list[tuple[int, int]] = []
    start = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        if not f and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(days) - 1))
    peak_i = max((i for i, f in enumerate(flags) if f), key=lambda i: truth_daily[days[i]])
    run = next(r for r in runs if r[0] <= peak_i <= r[1])
    return days[run[0]], days[peak_i], days[run[1]]


def _point_closeness(day_diff: int) -> float:
    return max(0.0, 1.0 - 0.4 * day_diff)


def _json_pointer(document, pointer: str):
    """Resolve a strict RFC 6901 pointer; raise on any missing/invalid segment."""
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise KeyError(pointer)
    current = document
    for raw in pointer[1:].split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if not key.isdigit():
                raise KeyError(pointer)
            current = current[int(key)]
        elif isinstance(current, dict):
            current = current[key]
        else:
            raise KeyError(pointer)
    return current


def _scalar_equal(asserted, actual) -> bool:
    if isinstance(actual, dict) or isinstance(asserted, dict):
        return False
    if isinstance(actual, list) or isinstance(asserted, list):
        # v0.8.1: a short tuple fact (e.g. [speed, from_deg, spread]) is one
        # verifiable assertion when every element matches; a partially
        # matching or partially missing tuple is not evidence.
        if not (isinstance(actual, list) and isinstance(asserted, list)):
            return False
        if not actual or len(actual) != len(asserted) or len(actual) > 8:
            return False
        if any(isinstance(item, (dict, list)) for item in (*actual, *asserted)):
            return False
        return all(_scalar_equal(a, b) for a, b in zip(asserted, actual))
    # Missing/non-finite/empty payloads are not scientific evidence.  Without
    # this guard a policy could cite two ``null`` placeholders from different
    # tool sections and collect full grounding credit while learning nothing.
    if actual is None or asserted is None:
        return False
    if isinstance(actual, str) and not actual.strip():
        return False
    if isinstance(asserted, str) and not asserted.strip():
        return False
    if isinstance(actual, bool) or isinstance(asserted, bool):
        return actual is asserted
    if isinstance(actual, (int, float)) and isinstance(asserted, (int, float)):
        if not math.isfinite(float(actual)) or not math.isfinite(float(asserted)):
            return False
        tolerance = max(1e-6, abs(float(actual)) * 1e-6)
        return abs(float(asserted) - float(actual)) <= tolerance
    return asserted == actual


def _field_type_allowed(tool: str, pointer: str, evidence_type: str) -> bool:
    """Constrain broad multi-view tools to the semantic section actually cited."""
    if tool == "get_native_meteorology":
        parts = pointer.strip("/").split("/")
        return (evidence_type == "diagnostic" and len(parts) == 5 and parts[0] == "series"
                and parts[2] == "samples" and parts[3].isdigit() and parts[4] == "value")
    if tool == "compute_diffusion_conditions":
        parts = pointer.strip("/").split("/")
        return (evidence_type == "diagnostic" and len(parts) == 3 and parts[0] == "samples"
                and parts[1].isdigit() and parts[2] in {"wind_speed_ms", "blh_m", "ventilation_proxy_m2_s"})
    if tool == "find_similar_cases":
        parts = pointer.strip("/").split("/")
        return (evidence_type == "analog" and len(parts) == 5 and parts[0] == "analogs"
                and parts[1].isdigit() and parts[2] in {"similar_dimensions", "different_dimensions"}
                and parts[3].isdigit() and parts[4] in {"current", "historical"})
    if tool == "get_historical_case" and (
        pointer.startswith("/historical_case/issue_inputs/guidance/")
        or pointer.startswith("/historical_case/guidance_errors/")
    ):
        return evidence_type == "analog" and "/daily/" in pointer
    if (
        tool == "get_process_evidence"
        and evidence_type == "diagnostic"
        and pointer.startswith("/weather_trajectory_6h_to_72h_then_12h/")
        and ("/terrain_adaptive_low_level/" in pointer or pointer.endswith("/ventilation"))
    ):
        return True
    by_type = GROUNDING_SECTION_RULES.get(tool)
    return not by_type or any(pointer.startswith(prefix)
                              for prefix in by_type.get(evidence_type, ()))


def _semantic_evidence_match(evidence: dict, registry: dict, allowed_tools: set[str]) -> bool:
    """Verify the structured citation, never the free-text natural-language claim."""
    ref, pointer = evidence.get("ref"), evidence.get("field")
    if not isinstance(ref, str) or not isinstance(pointer, str) or "value" not in evidence:
        return False
    record = registry.get(ref)
    if not isinstance(record, dict) or record.get("tool") not in allowed_tools:
        return False
    if not _field_type_allowed(record["tool"], pointer, evidence.get("type")):
        return False
    # Harness metadata itself is not scientific evidence.
    leaf = pointer.rsplit("/", 1)[-1].replace("~1", "/").replace("~0", "~")
    if leaf in TRIVIAL_GROUNDING_LEAVES:
        return False
    try:
        actual = _json_pointer(record.get("content"), pointer)
    except (KeyError, IndexError, TypeError, ValueError):
        return False
    return _scalar_equal(evidence.get("value"), actual)


def _scientific_assertion_key(evidence: dict, registry: dict) -> tuple | None:
    """Canonical identity for one cited fact, independent of ref/value spelling.

    Repeating an identical tool query creates a new episode-local ref.  That
    must not turn one scientific scalar into two assertions.  Hash the public
    tool payload (which excludes the generated evidence_ref) and pair it with
    the JSON Pointer; integer/float spelling of the asserted value is therefore
    irrelevant as well.
    """
    ref, pointer = evidence.get("ref"), evidence.get("field")
    if not isinstance(ref, str) or not isinstance(pointer, str):
        return None
    record = registry.get(ref) if isinstance(registry, dict) else None
    if not isinstance(record, dict):
        return ("missing-ref", ref, pointer)
    payload = json.dumps(
        record.get("content"), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=True,
    ).encode("utf-8")
    return (record.get("tool"), hashlib.sha256(payload).hexdigest(), pointer)


def score_forecast(
    forecast_obj: dict,
    truth_daily: dict[str, float],
    *,
    issue_date: str,
    horizon: int,
    region: Optional[str] = None,
    expert_evidence_types: Optional[set[str]] = None,
    tools_called: Optional[set] = None,
    evidence_registry: Optional[dict] = None,
    aqi_standard: Optional[str | dict[str, str]] = None,
    cfg: Optional[RewardConfig] = None,
) -> ScoreResult:
    cfg = cfg or RewardConfig()
    spec = reward_spec(cfg)

    require_pm10_range = any(
        isinstance(value, dict) and "PM10" in value for value in truth_daily.values()
    )
    require_o3_range = any(
        isinstance(value, dict) and "O3" in value for value in truth_daily.values()
    )
    fc, errors = validate_forecast(
        forecast_obj, issue_date=issue_date, horizon=horizon, region=region,
        require_pm10_range=require_pm10_range,
        require_o3_range=require_o3_range,
        aqi_standard=aqi_standard,
    )
    if fc is None:
        return ScoreResult(composite=0.0, valid=False, errors=errors,
                           details={"reward_spec": spec})

    days = [d.date for d in fc.daily]
    missing = [d for d in days if d not in truth_daily]
    if missing:
        raise ValueError(f"truth_daily missing days: {missing}")

    # 真值归一化：值可为 float（旧：PM2.5）或 {"PM2.5":…, "O3":…} 多污染物字典。
    # 多污染物时等级/事件/峰值按全 AQI 口径；单 PM2.5 时保持旧行为（分数逐位不变）。
    truth_concs: dict[str, dict[str, float]] = {}
    for d in days:
        v = truth_daily[d]
        truth_concs[d] = dict(v) if isinstance(v, dict) else {"PM2.5": float(v)}
    multi = any(len(c) > 1 for c in truth_concs.values())
    standards = {
        d: ((aqi_standard.get(d) if isinstance(aqi_standard, dict) else aqi_standard)
            or aqi_standard_for_date(d))
        for d in days
    }
    if multi:
        truth_aqi = {d: daily_aqi(truth_concs[d], standard=standards[d]) for d in days}
        truth_levels = {d: truth_aqi[d]["level"] for d in days}
        event_magnitude = {d: float(truth_aqi[d]["aqi"]) for d in days}
    else:
        truth_aqi = None
        truth_levels = {
            d: pm25_to_level(truth_concs[d]["PM2.5"], standard=standards[d]) for d in days
        }
        event_magnitude = {d: truth_concs[d]["PM2.5"] for d in days}
    components: dict[str, Optional[float]] = {}
    details: dict = {"reward_spec": spec}
    details["aqi_standard_by_day"] = standards
    details["normalized_forecast"] = fc.to_dict()

    # --- level ---
    num = den = 0.0
    per_day = []
    for d in fc.daily:
        w = cfg.event_day_weight if truth_levels[d.date] >= cfg.event_level else 1.0
        diff = abs(d.aqi_level - truth_levels[d.date])
        pt = 1.0 if diff == 0 else (cfg.offby1_credit if diff == 1 else 0.0)
        num += w * pt
        den += w
        per_day.append({"date": d.date, "forecast": d.aqi_level, "truth": truth_levels[d.date], "point": pt})
    components["level"] = num / den
    details["level_per_day"] = per_day

    # --- event (CSI) ---
    pred_days = {d.date for d in fc.daily if d.aqi_level >= cfg.event_level}
    true_days = {d for d in days if truth_levels[d] >= cfg.event_level}
    hits = len(pred_days & true_days)
    misses = len(true_days - pred_days)
    false_alarms = len(pred_days - true_days)
    hard_denominator = hits + misses + false_alarms
    hard_csi = hits / hard_denominator if hard_denominator else None
    near_misses = sum(
        d.date in true_days and d.aqi_level == cfg.event_level - 1 for d in fc.daily
    )
    if not true_days and not pred_days:
        # clean 正确否定已在 level 里评分；CSI 在无观测/预报事件时也无定义。
        components["event"] = None
    else:
        components["event"] = (
            hits + cfg.event_near_miss_credit * near_misses
        ) / hard_denominator
    details["event"] = {"hits": hits, "misses": misses, "false_alarms": false_alarms,
                        "near_misses_at_level": cfg.event_level - 1,
                        "near_miss_count": near_misses, "hard_csi": hard_csi,
                        "event_level": cfg.event_level}

    # --- interval（PM2.5 必评；全国多污染物同时评 PM10 与 O3）---
    def _interval_score(
        rng: Optional[tuple], obs: float,
        width_free: Optional[float] = None, width_scale: Optional[float] = None,
    ) -> float:
        if rng is None:
            return 0.0
        lo, hi = rng
        width = hi - lo
        free = cfg.width_free if width_free is None else width_free
        scale = cfg.width_scale if width_scale is None else width_scale
        miss = max(lo - obs, obs - hi, 0.0)
        # Dense, bounded surrogate monotonically follows the standard central
        # interval score: width + 2/alpha * miss. A small free-width allowance
        # avoids rewarding degenerate point intervals, while exponential decay
        # preserves non-zero GRPO variation without the old 0.2 wide-interval
        # floor. Graduation still uses the untransformed proper interval score.
        adjusted_interval_score = max(0.0, width - free) + 2.0 / cfg.interval_alpha * miss
        return math.exp(-adjusted_interval_score / scale)

    def _interval_diag(rng: Optional[tuple], obs: float) -> dict:
        if rng is None:
            return {"covered": False, "width": None, "miss": None, "interval_score": None}
        lo, hi = rng
        miss = max(lo - obs, obs - hi, 0.0)
        raw = (hi - lo) + (2.0 / cfg.interval_alpha) * miss
        return {"covered": lo <= obs <= hi, "width": hi - lo,
                "miss": round(miss, 4), "interval_score": round(raw, 4)}

    scores = []
    pm10_scores = []
    o3_scores = []
    pm25_diag = []
    pm10_diag = []
    o3_diag = []
    for d in fc.daily:
        pm_obs = truth_concs[d.date]["PM2.5"]
        scores.append(_interval_score(d.pm25_range, pm_obs))
        pm25_diag.append(_interval_diag(d.pm25_range, pm_obs))
        if "PM10" in truth_concs[d.date]:
            pm10_obs = truth_concs[d.date]["PM10"]
            pm10_scores.append(_interval_score(
                d.pm10_range, pm10_obs,
                cfg.pm10_width_free, cfg.pm10_width_scale,
            ))
            pm10_diag.append(_interval_diag(d.pm10_range, pm10_obs))
        if "O3" in truth_concs[d.date]:
            o3_obs = truth_concs[d.date]["O3"]
            o3_scores.append(_interval_score(
                d.o3_range, o3_obs, cfg.o3_width_free, cfg.o3_width_scale
            ))
            o3_diag.append(_interval_diag(d.o3_range, o3_obs))
    all_scores = scores + pm10_scores + o3_scores
    components["interval"] = sum(all_scores) / len(all_scores)
    details["interval_per_day"] = scores
    if o3_scores:
        details["interval_o3_per_day"] = o3_scores
    if pm10_scores:
        details["interval_pm10_per_day"] = pm10_scores
    details["interval_diagnostics"] = {"nominal_coverage": 1.0 - cfg.interval_alpha,
                                       "PM2.5": pm25_diag}
    if o3_diag:
        details["interval_diagnostics"]["O3"] = o3_diag
    if pm10_diag:
        details["interval_diagnostics"]["PM10"] = pm10_diag

    # --- turning ---
    # event 评估中度+高影响日，process 评估轻度+污染过程的起/峰/末。
    # 两者分开才能让 turning 分层与奖励口径一致。
    truth_event = extract_event(event_magnitude, cfg.process_level, levels=truth_levels)
    pred_event = None
    if fc.process is not None and fc.process.has_event:
        pred_event = (fc.process.start, fc.process.peak, fc.process.end)
    if truth_event is None:
        # 转折点是“有过程条件下”的日期定位任务；无过程正确性已由 event 分量评分，
        # 这里弃权可避免 clean episode 因同一事实重复获得 event+turning 两份满分。
        components["turning"] = None
    elif pred_event is None:
        components["turning"] = 0.0
    else:
        diffs = [
            abs((parse_date(a) - parse_date(b)).days)
            for a, b in zip(pred_event, truth_event)
        ]
        components["turning"] = sum(_point_closeness(x) for x in diffs) / 3
        details["turning"] = {"pred": pred_event, "truth": truth_event, "day_diffs": diffs,
                              "process_level": cfg.process_level}
    if truth_event is not None:
        details.setdefault("turning", {})["truth"] = truth_event

    # --- primary（首要污染物；仅多污染物真值时激活，AQI≤50 的优日不计入）---
    if truth_aqi is not None:
        pts = []
        for d in fc.daily:
            true_primary = truth_aqi[d.date]["primary"]
            if not true_primary:
                continue  # 优日无首要污染物
            pts.append(1.0 if d.primary_pollutant in true_primary else 0.0)
        components["primary"] = (sum(pts) / len(pts)) if pts else None
        details["primary_truth"] = {d: truth_aqi[d]["primary"] for d in days}
    else:
        components["primary"] = None

    # --- evidence（无专家参照则弃权）---
    if expert_evidence_types:
        pred_types = fc.evidence_types()
        union = pred_types | expert_evidence_types
        components["evidence"] = (len(pred_types & expert_evidence_types) / len(union)) if union else 1.0
        details["evidence"] = {"pred": sorted(pred_types), "expert": sorted(expert_evidence_types)}
    else:
        components["evidence"] = None

    # --- grounding（证据-工具语义一致性；无调用记录则弃权）---
    if tools_called is not None:
        items = fc.evidence
        verifiable = [e for e in items if EVIDENCE_TOOL_MAP.get(e.get("type")) is not None]
        if not verifiable:
            components["grounding"] = 0.0  # 无任何证据引用：不合格
        else:
            scores = []
            seen_assertions = set()
            semantic_verified = 0
            semantic_types = set()
            structured_attempts = 0
            structured_invalid = 0
            type_only_items = 0
            for e in verifiable:
                need = EVIDENCE_TOOL_MAP.get(e.get("type"))
                type_supported = bool(need & tools_called)
                assertion_key = None
                structured_fields = tuple(key in e for key in ("ref", "field", "value"))
                if any(structured_fields):
                    structured_attempts += 1
                if all(structured_fields):
                    assertion_key = _scientific_assertion_key(e, evidence_registry or {})
                    semantic_match = bool(
                        type_supported and evidence_registry
                        and _semantic_evidence_match(e, evidence_registry, need)
                    )
                    if assertion_key in seen_assertions:
                        # A correct replay is harmless but earns no extra
                        # credit. A contradictory replay must not disappear
                        # behind deduplication: retain a zero in the denominator
                        # and expose it to the structured-invalid audit.
                        if not semantic_match:
                            scores.append(0.0)
                            structured_invalid += 1
                        continue
                    seen_assertions.add(assertion_key)
                    point = 0.0
                    if semantic_match:
                        point = 1.0
                        semantic_verified += 1
                        semantic_types.add(e.get("type"))
                    else:
                        structured_invalid += 1
                elif any(structured_fields):
                    # Incomplete structured claims are malformed evidence, not
                    # an unstructured citation eligible for type-only credit.
                    point = 0.0
                    structured_invalid += 1
                else:
                    type_only_items += 1
                    point = cfg.grounding_type_only_credit if type_supported else 0.0
                scores.append(point)
            denominator = max(cfg.minimum_grounding_assertions, len(scores))
            assertion_score = sum(scores) / denominator
            # Preserve partial credit while the policy has fewer than the
            # required number of verified assertions.  Once it has enough
            # facts to seek full credit, repeated facts from only one evidence
            # category are capped by the semantic breadth ratio.
            type_breadth = (
                min(1.0, len(semantic_types) / max(1, cfg.minimum_grounding_types))
                if semantic_verified >= cfg.minimum_grounding_assertions else 1.0
            )
            components["grounding"] = assertion_score * type_breadth
        details["grounding"] = {"tools_called": sorted(tools_called),
                                "verifiable_evidence_count": len(verifiable),
                                "unique_assertion_count": (
                                    len(seen_assertions) if verifiable else 0
                                ),
                                "scored_evidence_item_count": (
                                    len(scores) if verifiable else 0
                                ),
                                "semantic_verified_count": semantic_verified if verifiable else 0,
                                "semantic_verified_types": sorted(semantic_types) if verifiable else [],
                                "semantic_verified_type_count": len(semantic_types) if verifiable else 0,
                                "structured_attempt_count": structured_attempts if verifiable else 0,
                                "structured_invalid_count": structured_invalid if verifiable else 0,
                                "type_only_item_count": type_only_items if verifiable else 0,
                                "type_only_credit": cfg.grounding_type_only_credit,
                                "minimum_assertions_for_full_credit":
                                    cfg.minimum_grounding_assertions,
                                "minimum_types_for_full_credit":
                                    cfg.minimum_grounding_types,
                                "verification_scope":
                                    "tool+semantic_section+json_pointer+scalar_value",
                                "natural_language_claim_verified": False}
    else:
        components["grounding"] = None

    # --- composite（弃权分量去权归一）---
    active = {k: v for k, v in components.items() if v is not None}
    total_w = sum(cfg.weights[k] for k in active)
    weights_used = {k: cfg.weights[k] / total_w for k in active}
    composite = sum(active[k] * weights_used[k] for k in active)

    # Graduation and model-vs-baseline comparisons must use a score whose
    # estimand is identical for interactive agents and non-interactive
    # numerical baselines.  Evidence/grounding remain low-weight training
    # incentives, but are excluded from this forecast-outcome score.
    outcome_active = {
        key: components[key] for key in OUTCOME_COMPONENTS
        if components.get(key) is not None
    }
    outcome_total_w = sum(cfg.weights[key] for key in outcome_active)
    outcome_weights = {
        key: cfg.weights[key] / outcome_total_w for key in outcome_active
    }
    outcome_composite = sum(
        outcome_active[key] * outcome_weights[key] for key in outcome_active
    )
    details["outcome_weights_used"] = outcome_weights

    return ScoreResult(
        composite=round(composite, 4),
        outcome_composite=round(outcome_composite, 4),
        valid=True,
        components={k: (round(v, 4) if v is not None else None) for k, v in components.items()},
        weights_used=weights_used,
        details=details,
    )
