"""ForecastEnv：预报任务环境（gym 风格，多步工具调用 + 结构化提交）。

约定：
- 一个 episode = 一个起报时次（case bundle）。
- 所有交互都是工具调用，包括提交（submit_forecast）。
- 稀疏 reward：仅在合法提交时给出综合得分；预算耗尽未提交记 0。
- 提交不合法不终止（返回校验错误，允许重试），但消耗步数——这给 RL
  留出学习"格式自纠"的空间，同时格式最终仍是门禁。
- 当前 episode 的 truth/expert 永不通过工具暴露；独立历史索引只开放已验证的过去结果。
"""
from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .case import CaseBundle
from .asset_quality import build_asset_quality
from .analogs import load_historical_case_index
from .forecast_methods import ForecastMethodLibrary
from .data_contract import issue_time, observation_time, valid_concentration
from .guidance_bias import GuidanceBiasIndex, load_guidance_bias_index
from .process_evidence import build_process_evidence, compact_process_view
from .schema import aqi_standard_for_date, forecast_tool_schema, forecast_format_description
from .scoring import (
    GROUNDING_SECTION_RULES,
    TRIVIAL_GROUNDING_LEAVES,
    RewardConfig,
    ScoreResult,
    _field_type_allowed,
    score_forecast,
)

HARNESS_VERSION = "forecast-harness-v2.1"


_DEFAULT_CITATION_TYPE = {
    "get_observations": "observation",
    "get_diagnostics": "diagnostic",
    "get_model_guidance": "model_guidance",
    "get_previous_forecast": "previous_forecast",
    "find_similar_cases": "analog",
    "get_historical_case": "analog",
    "get_synoptic_evidence": "synoptic",
    "get_guidance_bias": "model_guidance",
    "analyze_chart": "synoptic",
}


def _pointer_escape(value: str) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _resolve_prefix(document: dict, prefix: str):
    current = document
    for segment in prefix.strip("/").split("/"):
        if not segment:
            continue
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]
    return current


def _scientific_scalars(value, pointer: str):
    """Yield reproducible non-metadata scalar facts in document order."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _scientific_scalars(child, pointer + "/" + _pointer_escape(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _scientific_scalars(child, pointer + f"/{index}")
    elif value is not None and not isinstance(value, (dict, list)):
        leaf = pointer.rsplit("/", 1)[-1].replace("~1", "/").replace("~0", "~")
        if leaf in TRIVIAL_GROUNDING_LEAVES:
            return
        if isinstance(value, str) and (not value.strip() or len(value) > 100):
            return
        if isinstance(value, (int, float)) and not isinstance(value, bool) \
                and not math.isfinite(float(value)):
            return
        yield pointer, value


def _citation_examples(tool: str, content: dict, ref: str, limit: int = 6) -> list[dict]:
    examples = []
    sections = GROUNDING_SECTION_RULES.get(tool)
    if sections:
        specifications = [
            (evidence_type, prefix)
            for evidence_type, prefixes in sections.items()
            for prefix in prefixes
        ]
    else:
        evidence_type = _DEFAULT_CITATION_TYPE.get(tool)
        specifications = [(evidence_type, "/")] if evidence_type else []
    for evidence_type, prefix in specifications:
        subtree = content if prefix == "/" else _resolve_prefix(content, prefix)
        if subtree is None:
            continue
        candidates = list(_scientific_scalars(subtree, prefix.rstrip("/")))
        candidates = [(pointer, value) for pointer, value in candidates
                      if _field_type_allowed(tool, pointer, evidence_type)]
        # Numeric measurements are easier to verify and less ambiguous than a
        # long categorical description. Keep at most two per semantic section.
        candidates.sort(key=lambda item: (not isinstance(item[1], (int, float)), len(item[0])))
        for pointer, value in candidates[:2]:
            examples.append({"type": evidence_type, "ref": ref,
                             "field": pointer, "value": value})
            if len(examples) >= limit:
                return examples
    return examples


def _derived_heads(normalized: Optional[dict]) -> Optional[dict]:
    """Compact echo of the harness-derived categorical heads for the policy."""
    if not isinstance(normalized, dict):
        return None
    return {
        "note": "aqi_level/primary_pollutant/process derived from interval midpoints (HJ 633)",
        "daily": [
            [item.get("date"), item.get("aqi_level"), item.get("primary_pollutant")]
            for item in normalized.get("daily", [])
        ],
        "process": normalized.get("process"),
    }


@dataclass
class EnvConfig:
    max_steps: int = 12
    obs_default_hours: int = 48
    obs_default_stride: int = 3
    reward: RewardConfig = field(default_factory=RewardConfig)
    vlm_client: Optional[Any] = None  # describe_image 用；None 时从环境变量构造
    enable_method_retrieval: bool = field(default_factory=lambda: (
        os.environ.get("FH_METHOD_RETRIEVAL", "1").lower() not in {"0", "false"}
    ))
    forecast_methods_path: Optional[str] = field(default_factory=lambda: os.environ.get("FH_FORECAST_METHODS"))
    historical_cases_path: Optional[str] = field(default_factory=lambda: os.environ.get("FH_HISTORICAL_CASE_INDEX"))
    guidance_bias_path: Optional[str] = field(default_factory=lambda: (
        os.environ.get("FH_GUIDANCE_BIAS_INDEX") or str(
            Path(__file__).resolve().parents[2] /
            "data/interim/guidance_bias_history_v1.json"
        )
    ))


class ForecastEnv:
    def __init__(self, bundle: CaseBundle, cfg: Optional[EnvConfig] = None):
        violations = bundle.audit_time_gate()
        if violations:
            raise ValueError(f"input time gate failed: {violations[:3]}")
        self.bundle = bundle
        self.cfg = cfg or EnvConfig()
        self.steps_used = 0
        self.done = False
        self.transcript: list[dict] = []
        self.methods = (ForecastMethodLibrary.load(self.cfg.forecast_methods_path)
                        if self.cfg.enable_method_retrieval else None)
        self.historical_cases = (load_historical_case_index(self.cfg.historical_cases_path)
                                 if self.cfg.historical_cases_path else None)
        self.resource_identity = {
            "harness_version": HARNESS_VERSION,
            "forecast_methods_sha256": self.methods.identity if self.methods else None,
            "historical_cases_sha256": self.historical_cases.identity if self.historical_cases else None,
        }
        self.guidance_bias: Optional[GuidanceBiasIndex] = None
        if self.cfg.guidance_bias_path:
            path = Path(self.cfg.guidance_bias_path)
            if path.is_file():
                self.guidance_bias = load_guidance_bias_index(str(path))
        self.resource_identity.update({
            "guidance_bias_sha256": self.guidance_bias.identity if self.guidance_bias else None,
            "max_steps": self.cfg.max_steps,
            "obs_default_hours": self.cfg.obs_default_hours,
            "obs_default_stride": self.cfg.obs_default_stride,
        })
        self._tools: dict[str, tuple[str, dict, Callable[..., Any]]] = {}
        self._register_tools()

    # ------------------------------------------------------------------ setup
    def _register_tools(self) -> None:
        b = self.bundle
        self._tools = {
            "list_data_assets": (
                "列出数据资产及质量：逐日指导缺口、实况缺测、统计口径、发布时间是否记录、同源依赖。",
                {"type": "object", "properties": {}},
                self._tool_list_data_assets,
            ),
            "get_observations": (
                "查询起报前实况小时浓度序列（按区域）。",
                {
                    "type": "object",
                    "properties": {
                        "pollutant": {"type": "string",
                                      "enum": ["PM2.5", "PM10", "O3", "SO2", "NO2", "CO"],
                                      "description": "默认 PM2.5"},
                        "region": {"type": ["string", "null"], "description": "缺省返回全部区域"},
                        "last_hours": {"type": "integer", "minimum": 1, "maximum": 168, "description": f"起报前实际小时窗口；默认 {self.cfg.obs_default_hours}"},
                        "stride": {"type": "integer", "description": f"抽稀步长小时，默认 {self.cfg.obs_default_stride}"},
                    },
                },
                self._tool_get_observations,
            ),
            "get_diagnostics": (
                "查询逐日污染气象诊断特征（未来日为数值模式派生，属合法指导信息）。",
                {
                    "type": "object",
                    "properties": {"date": {"type": ["string", "null"], "description": "ISO 日期；缺省返回全部"}},
                },
                self._tool_get_diagnostics,
            ),
            "get_model_guidance": (
                "查询个例中真实可用的逐日模式指导。source 缺省返回全部；不要猜测未列出的 EC 等名称。"
                "提交引用时 evidence.type 使用 model_guidance，不得写工具名。",
                {
                    "type": "object",
                    "properties": {"source": {"type": ["string", "null"], "description": "缺省返回全部源"}},
                },
                self._tool_get_model_guidance,
            ),
            "get_assessment": (
                "获取确定性推导的客观信号汇总：实况 24h 趋势、逐日气象信号旗标"
                "（冷空气/静稳/降水/O3 潜力，判据固定）、模式指导多源分歧、气候态分位。"
                "只含事实信号，不含预报结论。引用本工具结论时按信号来源选择 evidence type"
                "（趋势→observation，气象信号→diagnostic/synoptic，指导分歧→model_guidance）。",
                {"type": "object", "properties": {}},
                self._tool_get_assessment,
            ),
            "submit_forecast": (
                "提交结构化预报，结束本次任务。必须在预算内提交且通过校验。evidence.type 必须使用"
                " schema 枚举；pollution_evidence、assessment 等工具名不是合法 type。",
                forecast_tool_schema(
                    b.issue_date, b.horizon, b.region,
                    multi_pollutant=bool((b.meta or {}).get("multi_pollutant")),
                ),
                self._tool_submit_forecast,
            ),
        }
        # Do not advertise dead actions.  A missing optional source is already
        # visible in the task brief; exposing an unavailable tool only spends
        # policy entropy, a step, and context without adding information.
        if b.previous_forecast is not None:
            self._tools["get_previous_forecast"] = (
                "查询起报前已发布的上一版预报（用于订正对比）。",
                {"type": "object", "properties": {}},
                self._tool_get_previous_forecast,
            )
        if self.methods is not None and self.methods.eligible(b.issue_date):
            self._tools["retrieve_forecast_methods"] = (
                "按判断问题检索预报方法卡，返回适用条件、查证工具、支持/反对信号及失效情形。"
                "来源与审核状态显式标注；方法不是当前预报答案，不能代替数据证据。",
                {"type": "object", "properties": {
                    "query": {"type": "string", "description": "需要解决的判断问题，1..500字"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
                    "pollutant": {"type": ["string", "null"], "enum": ["PM2.5", "PM10", "O3", None]},
                }, "required": ["query"]},
                self._tool_retrieve_forecast_methods,
            )
        if self.historical_cases is not None and self.historical_cases.eligible(b):
            self._tools["find_similar_cases"] = (
                "使用污染初态、气象和指导演变检索已验证的训练期历史过程；返回相似维度、差异与覆盖。"
                "距离不是概率，不能直接复制历史浓度。",
                {"type": "object", "properties": {
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
                    "pollutant": {"type": ["string", "null"], "enum": ["PM2.5", "PM10", "O3", None]},
                    "same_region": {"type": "boolean", "description": "是否只检索目标城市，默认false"},
                }},
                self._tool_find_similar_cases,
            )
            self._tools["get_historical_case"] = (
                "查看已验证历史个例的起报特征、指导、结果及同口径误差；按case_id查询，也执行时间门禁。",
                {"type": "object", "properties": {
                    "case_id": {"type": "string"},
                    "detail": {"type": "string", "enum": ["summary", "full"], "description": "默认summary；full包含全部历史形势特征"},
                }, "required": ["case_id"]},
                self._tool_get_historical_case,
            )
        if b.evidence:
            self._tools["get_process_evidence"] = (
                "查询起报时可见的紧凑过程总览：六项污染初态、6/12h 低层风与"
                "稳定度轨迹、逐日边界层/湿度/降水/云/温度、上游候选和污染指导锚点。"
                "先用此工具判断需要哪个专题工具；返回是可复算信号，不是预报结论。",
                {"type": "object", "properties": {}},
                self._tool_get_process_evidence,
            )
            self._tools["get_synoptic_evidence"] = (
                "查询过去—当前—未来天气形势证据：环流、输送、稳定度、扩散条件及模式分歧。"
                "提交引用时 evidence.type 使用 synoptic 或 diagnostic，不得写工具名。",
                {"type": "object", "properties": {
                    "source": {"type": ["string", "null"], "description": "gfs/ifs；缺省返回全部"},
                    "valid_time": {"type": ["string", "null"], "description": "ISO UTC；缺省返回全部"},
                    "detail": {"type": "string", "enum": ["summary", "full"],
                               "description": "默认 summary；full 用于审计或定点深挖"},
                }},
                self._tool_get_synoptic_evidence,
            )
            self._tools["get_pollution_evidence"] = (
                "查询污染机理证据：气溶胶类型/AOD、气态污染物、沙尘与上风向火点。提交引用时"
                " composition→model_guidance、fires→observation、source_context→diagnostic；"
                "pollution_evidence 不是合法 evidence.type。",
                {"type": "object", "properties": {
                    "kind": {"type": ["string", "null"],
                             "description": "composition/fires/source_context；缺省返回全部"},
                    "detail": {"type": "string", "enum": ["summary", "full"],
                               "description": "默认 summary；full 返回全部数值与来源 hash"},
                }},
                self._tool_get_pollution_evidence,
            )
        if self.guidance_bias is not None:
            self._tools["get_guidance_bias"] = (
                "查询起报前已完成验证的历史模式指导偏差、误差分位和事件漏报/空报率。"
                "返回聚合信号，不返回单条真值，也不直接给订正结论。提交引用使用 "
                "model_guidance，不得把历史偏差当作当前真值。",
                {"type": "object", "properties": {
                    "source": {"type": ["string", "null"],
                               "description": "可选指导源；缺省返回全部"},
                    "pollutant": {"type": ["string", "null"],
                                  "enum": ["PM2.5", "PM10", "O3", None],
                                  "description": "可选污染物；缺省返回全部"},
                    "window_days": {"type": "integer", "minimum": 30, "maximum": 1095,
                                    "description": "历史窗口，默认365天"},
                    "detail": {"type": "string", "enum": ["summary", "full"],
                               "description": "默认summary；full用于审计回退链和时间窗口"},
                }},
                self._tool_get_guidance_bias,
            )
        # 图像工具：个例带 image_cycle_dir 时注册（确定性反演永远可用，VLM 描述按配置）
        if b.meta.get("image_cycle_dir"):
            self._tools["analyze_chart"] = (
                "对单张产品图做确定性色标反演，返回类别覆盖/象限分布等统计（不经过模型）。",
                {"type": "object", "properties": {
                    "product": {"type": "string",
                                "description": "如 temp_inversion/boundary_layer/rain/surface/850hpa"},
                    "lead_hour": {"type": "integer"},
                    "domain": {"type": "string", "description": "默认 jingjinji"},
                }, "required": ["product", "lead_hour"]},
                self._tool_analyze_chart,
            )
            self._tools["describe_image"] = (
                "把同产品多时效图拼成趋势板，请 VLM 给出文字描述（未配置 VLM 时返回 available=false）。",
                {"type": "object", "properties": {
                    "product": {"type": "string"},
                    "lead_hours": {"type": "array", "items": {"type": "integer"}, "maxItems": 6},
                    "domain": {"type": "string"},
                    "question": {"type": "string", "description": "可选的追加问题"},
                }, "required": ["product", "lead_hours"]},
                self._tool_describe_image,
            )

    # ------------------------------------------------------------------ 图像工具
    def _tool_analyze_chart(self, product: str, lead_hour: int, domain: str = "jingjinji") -> dict:
        from .imaging import analyze_chart, index_cycle_images
        cycle_dir = self.bundle.meta["image_cycle_dir"]
        idx = index_cycle_images(cycle_dir)
        doms = idx.get(product)
        if not doms:
            return {"error": f"无产品 {product!r}", "available_products": sorted(idx)}
        leadmap = doms.get(domain) or doms.get(next(iter(doms)))
        if lead_hour not in leadmap:
            near = min(leadmap, key=lambda k: abs(k - lead_hour), default=None)
            if near is None:
                return {"error": f"{product} 无可用时效"}
            lead_hour = near
        result = analyze_chart(leadmap[lead_hour], product=product)
        return {"product": product, "lead_hour": lead_hour, "domain": domain,
                **result.to_dict()}

    def _tool_describe_image(self, product: str, lead_hours: list, domain: str = "jingjinji",
                             question: str = "") -> dict:
        vlm = getattr(self.cfg, "vlm_client", None)
        if vlm is None:
            from .vlm import VLMClient
            vlm = VLMClient()
        if not vlm.available():
            return {"available": False,
                    "note": "VLM 未配置（设 FH_VLM_BASE_URL/FH_VLM_MODEL 或注入 cfg.vlm_client）；"
                            "可改用确定性工具 analyze_chart"}
        from .imaging import build_mosaic
        from .vlm import DESCRIBE_PROMPT
        board = build_mosaic(self.bundle.meta["image_cycle_dir"], product,
                             [int(h) for h in lead_hours], domain)
        text = vlm.describe(board, DESCRIBE_PROMPT.format(
            product=product, question=(" " + question) if question else ""))
        return {"available": True, "product": product, "domain": domain,
                "lead_hours": lead_hours, "description": text}

    # ------------------------------------------------------------------ API
    def reset(self) -> dict:
        self.steps_used = 0
        self.done = False
        self.transcript = []
        self.tools_called: set = set()  # grounding 分量的依据：本 episode 实际调用过的工具
        self.evidence_registry: dict[str, dict] = {}
        b = self.bundle
        brief = {
            "harness_version": HARNESS_VERSION,
            "harness_resources": dict(self.resource_identity),
            "case_id": b.case_id,
            "issue_date": b.issue_date,
            "region": b.region,
            "horizon": b.horizon,
            "regions_with_observations": b.regions,
            "max_steps": self.cfg.max_steps,
            "open_evidence_available": bool(b.evidence),
            "previous_forecast_available": b.previous_forecast is not None,
            "historical_analogs_available": "find_similar_cases" in self._tools,
            "forecast_methods_available": "retrieve_forecast_methods" in self._tools,
            "aqi_standard": (
                (b.meta or {}).get("aqi_standard")
                or {d: aqi_standard_for_date(d) for d in b.forecast_dates()}
            ),
            "instructions": (
                f"你是{b.region}空气质量预报员。今天是 {b.issue_date}（起报日）。"
                f"请先调用工具查证实况、诊断与模式指导，再通过 submit_forecast 提交"
                f"未来 {b.horizon} 天的结构化预报。所有结论必须有已查证据支撑。"
                + ("本个例有开放形势与污染机理证据；先调用 get_process_evidence "
                   "查过程总览，再根据转折、输送、稳定度或污染机制风险按需深挖"
                   " get_synoptic_evidence / get_pollution_evidence(kind=...)，不要机械遍历。"
                   if b.evidence else "")
                + "成功数据工具会返回 evidence_ref；证据引用应同时给出 ref、"
                  "指向标量事实的 JSON Pointer field 和精确 value，以便语义核验。"
                + "资料覆盖或口径有疑问时查询 list_data_assets；按当前判断问题自主选择方法检索和历史类比，"
                  "不要求调用全部工具。方法卡只提供查证建议，历史结果不是当前真值。预留一步提交。"
                + ("逐日只需给出 PM2.5/PM10 日均 80% 区间与 O3_8h 日最大 80% 区间；"
                   "AQI 等级、首要污染物和 AQI>=3 污染过程由环境按 HJ 633 从区间中点派生并评分，"
                   "不需要提交。"
                   if (b.meta or {}).get("multi_pollutant") else
                   "逐日只需给出 PM2.5 日均 80% 区间；AQI 等级与污染过程由环境从区间中点派生。")
            ),
            "multi_pollutant": bool((b.meta or {}).get("multi_pollutant")),
            # No numeric example: the frozen policy copied placeholder intervals
            # verbatim in >90% of submissions, so a worked example would be an
            # answer-shaped template rather than a format hint.  The structure
            # is described here and enforced by the submit_forecast JSON schema.
            "forecast_format": forecast_format_description(
                b.issue_date, b.horizon, b.region,
                multi_pollutant=bool((b.meta or {}).get("multi_pollutant"))),
        }
        obs = {"type": "task", "brief": brief}
        self.transcript.append({"event": "reset", "obs": obs})
        return obs

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        if self.done:
            raise RuntimeError("episode is done; call reset()")
        self.steps_used += 1
        name = action.get("name")
        args = action.get("args") or {}
        info: dict = {"steps_used": self.steps_used}

        if name not in self._tools:
            obs = {"type": "tool_result", "name": name, "ok": False,
                   "content": {"error": f"unknown tool {name!r}", "available": sorted(self._tools)}}
        else:
            _, _, fn = self._tools[name]
            try:
                result = fn(**args)
            except TypeError as exc:
                result = {"error": f"bad arguments: {exc}"}
            except Exception as exc:  # 工具内部错误不炸 episode
                result = {"error": f"tool failed: {exc}"}
            if name == "submit_forecast" and isinstance(result, dict) and result.get("accepted"):
                score: Optional[ScoreResult] = result.pop("_score", None)
                self.done = True
                if score is not None:
                    info["score"] = score.to_dict()
                    info["reason"] = "submitted"
                    reward = score.composite
                else:
                    info["reason"] = "submitted_unscoreable"
                    info["scoreable"] = False
                    reward = 0.0
                obs = {"type": "tool_result", "name": name, "ok": True, "content": result}
                result["budget"] = self._budget_status()
                self.transcript.append({"event": "step", "action": action, "obs": obs, "info": info})
                return obs, reward, True, info
            obs = {"type": "tool_result", "name": name, "ok": "error" not in (result if isinstance(result, dict) else {}),
                   "content": result}
            # grounding 依据：只有"成功且有内容"的调用才算查证过
            # （出错或 available=false 的调用不算，否则引用不存在的数据也能通过 grounding）
            if obs["ok"] and not (isinstance(result, dict) and result.get("available") is False):
                self.tools_called.add(name)
                if name not in {"list_data_assets", "submit_forecast"} and isinstance(result, dict):
                    ref = f"e{len(self.evidence_registry) + 1}"
                    self.evidence_registry[ref] = {
                        "tool": name,
                        "content": deepcopy(result),
                    }
                    examples = _citation_examples(
                        name, self.evidence_registry[ref]["content"], ref
                    )
                    # veRL truncates overlong tool responses on the right. Keep
                    # the verifiable handles first so truncation cannot remove
                    # the refs while leaving only an unauditable payload.
                    result = {
                        "evidence_ref": ref,
                        **({"citation_examples": examples} if examples else {}),
                        **result,
                    }
                    obs["content"] = result

        reward = 0.0
        if self.steps_used >= self.cfg.max_steps:
            self.done = True
            info["reason"] = "budget_exhausted"
        if isinstance(obs.get("content"), dict):
            obs["content"]["budget"] = self._budget_status()
        self.transcript.append({"event": "step", "action": action, "obs": obs, "info": dict(info)})
        return obs, reward, self.done, info

    def tool_specs(self, style: str = "plain") -> list[dict]:
        specs = []
        for name, (desc, params, _) in self._tools.items():
            if style == "openai":
                specs.append({"type": "function",
                              "function": {"name": name, "description": desc, "parameters": params}})
            else:
                specs.append({"name": name, "description": desc, "parameters": params})
        return specs

    # ------------------------------------------------------------------ tools
    def _budget_status(self) -> dict:
        remaining = max(0, self.cfg.max_steps - self.steps_used)
        return {"steps_used": self.steps_used, "steps_remaining": remaining,
                "queries_remaining_before_submission": max(0, remaining - 1)}

    def _tool_list_data_assets(self) -> dict:
        b = self.bundle
        obs_summary = {}
        for pollutant, block in b.observations.items():
            times = block.get("times", [])
            obs_summary[pollutant] = {
                "time_range": [times[0], times[-1]] if times else None,
                "hours": len(times),
                "regions": sorted(block.get("series", {})),
            }
        return {
            "quality": build_asset_quality(b),
            "observations": obs_summary,
            "diagnostics_dates": sorted(b.diagnostics.get("daily", {})),
            "guidance_sources": sorted(b.guidance.get("sources", {})),
            "previous_forecast_available": b.previous_forecast is not None,
            "open_evidence": {
                "available": bool(b.evidence),
                "contract_version": b.evidence.get("contract_version") if b.evidence else None,
                "sections": sorted(k for k in b.evidence if k not in {"contract_version", "provenance"}),
            },
            "guidance_bias": {
                "available": self.guidance_bias is not None,
                "aggregation_only": True,
            },
            "forecast_methods": {"available": "retrieve_forecast_methods" in self._tools},
            "historical_cases": {
                "available": "find_similar_cases" in self._tools,
                "index_configured": self.historical_cases is not None,
                "legacy_prefilled_analogs_ignored": bool(b.meta.get("analogs")),
            },
        }

    def _tool_get_observations(self, pollutant: str = "PM2.5", region: Optional[str] = None,
                               last_hours: Optional[int] = None, stride: Optional[int] = None) -> dict:
        b = self.bundle
        block = b.observations.get(pollutant)
        if block is None:
            return {"error": f"no observations for {pollutant!r}", "available": sorted(b.observations)}
        from datetime import timedelta
        last_hours = self.cfg.obs_default_hours if last_hours is None else last_hours
        stride = self.cfg.obs_default_stride if stride is None else stride
        if type(last_hours) is not int or not 1 <= last_hours <= 168:
            return {"error": "last_hours must be an integer in 1..168"}
        if type(stride) is not int or not 1 <= stride <= 24:
            return {"error": "stride must be an integer in 1..24"}
        times = block.get("times", [])
        series = block.get("series", {})
        if region is not None and region not in series:
            return {"error": f"unknown region {region!r}", "available": sorted(series)}
        regions = [region] if region else sorted(series)
        if any(len(series[r]) != len(times) for r in regions):
            return {"error": "observation time/value axes are not aligned"}
        parsed = [observation_time(t) for t in times]
        if len(set(parsed)) != len(parsed) or parsed != sorted(parsed):
            return {"error": "observation timestamps must be unique and ordered"}
        cutoff = issue_time(b.issue_date)
        selected = []
        last_selected = None
        for i, stamp in enumerate(parsed):
            if cutoff - timedelta(hours=last_hours) <= stamp < cutoff:
                if last_selected is None or stamp - last_selected >= timedelta(hours=stride):
                    selected.append(i)
                    last_selected = stamp
        out_times = [times[i] for i in selected]
        out_series = {r: [series[r][i] if valid_concentration(series[r][i]) else None for i in selected]
                      for r in regions}
        unit = block.get("unit") or ("mg/m³" if pollutant == "CO" else "µg/m³")
        return {"available": bool(selected), "pollutant": pollutant, "unit": unit,
                "times": out_times, "series": out_series,
                "window_start": (cutoff - timedelta(hours=last_hours)).isoformat(),
                "window_end_exclusive": cutoff.isoformat(),
                "sampling": f"actual timestamps; at least {stride}h between returned samples"}

    def _tool_get_diagnostics(self, date: Optional[str] = None) -> dict:
        daily = self.bundle.diagnostics.get("daily", {})
        if date is None:
            return {"daily": daily}
        if date not in daily:
            return {"error": f"no diagnostics for {date}", "available": sorted(daily)}
        return {"date": date, "features": daily[date]}

    def _tool_get_model_guidance(self, source: Optional[str] = None) -> dict:
        sources = self.bundle.guidance.get("sources", {})
        if source is not None:
            if source not in sources:
                return {"error": f"unknown source {source!r}", "available": sorted(sources)}
            return {"submission_evidence_type": "model_guidance",
                    "source": source, **sources[source]}
        return {"submission_evidence_type": "model_guidance", "sources": sources}

    def _tool_get_previous_forecast(self) -> dict:
        if self.bundle.previous_forecast is None:
            return {"available": False}
        return {"available": True, "forecast": self.bundle.previous_forecast}

    def _tool_get_guidance_bias(self, source: Optional[str] = None,
                                pollutant: Optional[str] = None,
                                window_days: int = 365,
                                detail: str = "summary") -> dict:
        if self.guidance_bias is None:
            return {"available": False, "reason": "guidance_bias_index_unavailable"}
        if not 30 <= int(window_days) <= 1095:
            return {"error": "window_days must be in 30..1095"}
        if detail not in {"summary", "full"}:
            return {"error": "detail must be summary or full"}
        result = self.guidance_bias.query(
            region=self.bundle.region,
            issue_date=self.bundle.issue_date,
            horizon=self.bundle.horizon,
            source=source,
            pollutant=pollutant,
            window_days=int(window_days),
        )
        result["submission_evidence_type"] = "model_guidance"
        result["detail"] = detail
        if detail == "summary":
            # Budgeted view: shared definitions are hoisted, quantiles and event
            # rates become fixed-order lists.  detail=full keeps the long form.
            rows = result["series"]
            definitions = sorted({row.get("error_definition") for row in rows
                                  if row.get("error_definition")})
            seasons = sorted({row.get("target_season") for row in rows
                              if row.get("target_season")})
            compact_rows = []
            for row in rows:
                item = {key: row.get(key) for key in
                        ("source", "pollutant", "lead_days", "scope", "n")
                        if row.get(key) is not None}
                if row.get("available") is False:
                    item["available"] = False
                    if row.get("reason"):
                        item["reason"] = row.get("reason")
                for key in ("mean_error", "median_error", "mae"):
                    if row.get(key) is not None:
                        item[key] = row.get(key)
                quantiles = row.get("error_quantiles") or {}
                if quantiles:
                    item["error_p10_p25_p50_p75_p90"] = [
                        quantiles.get(key) for key in ("p10", "p25", "p50", "p75", "p90")
                    ]
                event = row.get("event") or {}
                if event:
                    item["event_miss_rate_false_alarm_rate_truth_events"] = [
                        event.get("miss_rate_given_truth_event"),
                        event.get("false_alarm_rate_given_truth_non_event"),
                        event.get("truth_event_count"),
                    ]
                compact_rows.append(item)
            result["series"] = compact_rows
            result["series_legend"] = {
                "error_definition": definitions[0] if len(definitions) == 1 else definitions,
                "target_season": seasons[0] if len(seasons) == 1 else seasons,
                "error_p10_p25_p50_p75_p90": "error quantiles in the pollutant unit",
                "event_miss_rate_false_alarm_rate_truth_events": (
                    "[miss rate given a truth AQI>=4 day, false-alarm rate given a truth "
                    "non-event day, truth event count in the window]"
                ),
            }
            result.pop("index_provenance", None)
        return result

    def _tool_get_assessment(self) -> dict:
        from .assess import compute_assessment
        b = self.bundle
        result = compute_assessment(b.observations, b.diagnostics, b.guidance,
                                    region=b.region, issue_date=b.issue_date,
                                    climatology=(b.meta or {}).get("climatology"))
        result["submission_evidence_type_mapping"] = {
            "observation_trend": "observation",
            "meteorological_signal": "diagnostic",
            "synoptic_signal": "synoptic",
            "guidance_disagreement": "model_guidance",
            "climatology": "expert_prior",
        }
        return result

    def _tool_get_process_evidence(self) -> dict:
        result = compact_process_view(build_process_evidence(self.bundle))
        result["submission_evidence_type_mapping"] = {
            "initial_pollution_state": "observation",
            "weather_trajectory": "synoptic",
            "daily_surface_dispersion": "diagnostic",
            "transport_context": "diagnostic",
            "pollution_guidance_anchor": "model_guidance",
        }
        return result

    def _tool_get_synoptic_evidence(self, source: Optional[str] = None,
                                    valid_time: Optional[str] = None,
                                    detail: str = "summary") -> dict:
        synoptic = self.bundle.evidence.get("synoptic", {})
        sources = synoptic.get("sources", {})
        if source is not None:
            if source not in sources:
                return {"error": f"unknown synoptic source {source!r}",
                        "available": sorted(sources)}
            sources = {source: sources[source]}
        if valid_time is not None:
            filtered = {}
            for name, records in sources.items():
                matches = [row for row in records if row.get("valid_time") == valid_time]
                if matches:
                    filtered[name] = matches
            sources = filtered
        cross_model = synoptic.get("cross_model") or {}
        if valid_time is not None:
            cross_model = ({valid_time: cross_model[valid_time]}
                           if valid_time in cross_model else {})
        if detail == "full":
            return {"contract_version": self.bundle.evidence.get("contract_version"),
                    "detail": detail,
                    "temporal_profile": synoptic.get("temporal_profile"),
                    "submission_evidence_type_mapping": {
                        "circulation_or_weather_system": "synoptic",
                        "stability_or_transport": "diagnostic",
                    },
                    "sources": sources, "cross_model": cross_model}
        if detail != "summary":
            return {"error": "detail must be summary or full"}

        target = self.bundle.region
        # With no explicit source, a compact response uses the GFS trajectory
        # plus the registered GFS/IFS disagreement.  Returning two nearly
        # duplicate high-frequency trajectories needlessly consume the 32k rollout context;
        # the alternate IFS trajectory remains one explicit source query away.
        summary_sources = sources
        omitted_sources = []
        if source is None and "gfs" in sources and len(sources) > 1:
            summary_sources = {"gfs": sources["gfs"]}
            omitted_sources = sorted(name for name in sources if name != "gfs")
        compact_sources = {}
        # The v2 contract carries 20 snapshots per source; a 32k rollout cannot
        # hold the full 6-hourly trajectory of every tool.  The summary keeps
        # the 12-hourly skeleton (08/20 BJT); 6-hourly points stay one
        # valid_time query or detail=full away.
        def _keep(record: dict) -> bool:
            if valid_time is not None:
                return True
            if str(record.get("role") or "") == "analysis-24h":
                return False
            stamp = str(record.get("valid_time") or "")
            try:
                return int(stamp[11:13]) % 12 == 0
            except ValueError:
                return True
        if valid_time is None:
            cross_model = {stamp: values for stamp, values in cross_model.items()
                           if _keep({"valid_time": stamp})}
        compact_cross = {}
        for stamp, cities in cross_model.items():
            row = (cities or {}).get(target) if isinstance(cities, dict) else None
            if isinstance(row, dict):
                compact_cross[stamp] = [
                    row.get("mslp_abs_diff_hpa"), row.get("z500_abs_diff_gpm"),
                    row.get("transport_wind_vector_diff_ms"), row.get("shortwave_abs_diff_w_m2"),
                ]
            else:
                compact_cross[stamp] = cities
        cross_model = compact_cross
        for name, records in summary_sources.items():
            compact_records = []
            for record in records:
                if not _keep(record):
                    continue
                base = {"valid_time": record.get("valid_time")}
                if record.get("available") is not True:
                    base["available"] = record.get("available")
                city = record.get("cities", {}).get(target, {})
                wind = {level: [values.get("speed_ms"), values.get("from_deg")]
                        for level, values in (city.get("wind") or {}).items()
                        if level in {"925", "850", "700", "500"}
                        and isinstance(values, dict)}
                stability = city.get("stability") or {}
                base["target_city"] = {
                    "mslp_hpa": city.get("mslp_hpa"), "z500_gpm": city.get("z500_gpm"),
                    "wind_ms_from": wind,
                    "t925_850_700_c": [city.get("temperature_c", {}).get(key)
                                       for key in ("925", "850", "700")],
                    "rh925_700_pct": [city.get("relative_humidity_pct", {}).get(key)
                                      for key in ("925", "700")],
                    "omega700_pa_s": city.get("omega_pa_s", {}).get("700"),
                    "sw_w_m2": city.get("surface_downward_shortwave_w_m2"),
                    "low_level_inversion": stability.get("low_level_inversion_signal"),
                }
                systems = record.get("systems", {})

                def _leading(values):
                    row = values[0] if isinstance(values, list) and values else None
                    if not isinstance(row, dict):
                        return None
                    magnitude = next((row[key] for key in
                                      ("mslp_hpa", "z500_zonal_anomaly_gpm") if key in row), None)
                    return [row.get("lat"), row.get("lon"), magnitude]
                base["leading_system_signals"] = {
                    key: _leading(values) for key, values in systems.items()
                }
                compact_records.append(base)
            compact_sources[name] = compact_records
        return {"contract_version": self.bundle.evidence.get("contract_version"),
                "detail": detail, "target_city": target,
                "temporal_profile": synoptic.get("temporal_profile"),
                "submission_evidence_type_mapping": {
                    "circulation_or_weather_system": "synoptic",
                    "stability_or_transport": "diagnostic",
                },
                "field_legend": {
                    "wind_ms_from": "per pressure level [speed m/s, from-deg]",
                    "cross_model": "GFS-IFS absolute disagreement at the target "
                                   "[MSLP hPa, Z500 gpm, terrain-adaptive wind vector m/s, shortwave W/m2]",
                    "leading_system_signals": "[lat, lon, MSLP hPa or Z500 zonal anomaly gpm] "
                                              "of the leading deterministic extremum",
                },
                "sources": compact_sources, "cross_model": cross_model,
                "omitted_source_trajectories": omitted_sources,
                "note": ("summary keeps 12-hourly snapshots"
                         + (" of GFS plus GFS/IFS disagreement; query source=ifs for its trajectory"
                            if omitted_sources else "")
                         + "; pass valid_time or detail=full for 6-hourly points")}

    def _tool_get_pollution_evidence(self, kind: Optional[str] = None,
                                     detail: str = "summary") -> dict:
        pollution = self.bundle.evidence.get("pollution", {})
        overview_only = kind is None
        if kind is not None:
            if kind not in pollution:
                return {"error": f"unknown pollution evidence kind {kind!r}",
                        "available": sorted(pollution)}
            selected = {kind: pollution[kind]}
        else:
            selected = dict(pollution)
        if detail == "full":
            return {"contract_version": self.bundle.evidence.get("contract_version"),
                    "detail": detail,
                    "submission_evidence_type_mapping": {
                        "composition": "model_guidance",
                        "fires": "observation",
                        "source_context": "diagnostic",
                    },
                    **selected}
        if detail != "summary":
            return {"error": "detail must be summary or full"}
        compact = {}
        if "composition" in selected:
            composition = selected["composition"]
            if not any(name in composition for name in
                       ("aerosol", "trace_gases", "column_gases")):
                compact["composition"] = deepcopy(composition)
            else:
                target = self.bundle.region
                aerosol_full = composition.get("aerosol", {})
                aerosol_records = []
                for index, record in enumerate(aerosol_full.get("records", [])):
                    if overview_only and index not in {0, 2, 4}:
                        continue
                    target_values = record.get("cities", {}).get(target, {})
                    if overview_only:
                        aod = target_values.get("aod550", {})
                        fractions = target_values.get("fraction_of_total", {})
                        target_values = {
                            "aod550": {name: aod.get(name) for name in
                                       ("total", "fine", "dust", "sulphate", "nitrate")
                                       if aod.get(name) is not None},
                            "fine_fraction": target_values.get("fine_fraction"),
                            "dominant_component": target_values.get("dominant_component"),
                            "fraction_of_total": {
                                name: fractions.get(name) for name in
                                ("dust", "sulphate", "nitrate")
                                if fractions.get(name) is not None
                            },
                        }
                    item = {"valid_time": record.get("valid_time"), "target": target_values}
                    # Source centers evolve more slowly than city speciation;
                    # retain current/mid/end snapshots in the default summary.
                    if index in {0, 2, 4}:
                        item["leading_regional_sources"] = {
                            name: rows[:1] for name, rows in
                            record.get("regional_source_signals", {}).items()
                        }
                    aerosol_records.append(item)
                aerosol = {"available": aerosol_full.get("available"),
                           "cycle": aerosol_full.get("cycle"), "daily": aerosol_records}
                gases_full = composition.get("trace_gases", {})
                gas_records = gases_full.get("records", [])
                if overview_only:
                    gas_records = [record for record in gas_records
                                   if int(record.get("lead_hour", -1)) in {12, 60, 108}]
                gases = {"available": gases_full.get("available"), "cycle": gases_full.get("cycle"),
                         "trajectory_6h": [{"lead_hour": record.get("lead_hour"),
                                            "issue_relative_hour": record.get("issue_relative_hour"),
                                            "valid_time": record.get("valid_time"),
                                            "target": record.get("cities", {}).get(target, {})}
                                           for record in gas_records]}
                columns_full = composition.get("column_gases", {})
                column_records = columns_full.get("records", [])
                if overview_only:
                    column_records = [record for record in column_records
                                      if int(record.get("lead_hour", -1)) in {12, 60, 108}]
                columns = {
                    "available": columns_full.get("available"),
                    "cycle": columns_full.get("cycle"),
                    "semantics": columns_full.get("semantics"),
                    "unit": columns_full.get("unit"),
                    "interpretation": (
                        "column burden/transport evidence; not a surface concentration "
                        "and not a fill value for model-level-137 gases"
                    ),
                    "trajectory_6h": [
                        {"lead_hour": record.get("lead_hour"),
                         "issue_relative_hour": record.get("issue_relative_hour"),
                         "valid_time": record.get("valid_time"),
                         "target": record.get("cities", {}).get(target, {})}
                        for record in column_records
                    ],
                }
                compact["composition"] = {
                    "aerosol": aerosol,
                    "trace_gases": gases,
                    "column_gases": columns,
                }
        if "fires" in selected:
            fires = deepcopy(selected["fires"])
            fires.pop("raw", None)
            fires["regional_clusters"] = fires.get("regional_clusters", [])[:3 if overview_only else 6]
            if overview_only and isinstance(fires.get("cities"), dict):
                fires["cities"] = {self.bundle.region:
                                   fires["cities"].get(self.bundle.region, {})}
            compact["fires"] = fires
        if "source_context" in selected:
            context = deepcopy(selected["source_context"])

            def compact_city(row: dict) -> dict:
                pollutants = (("PM2.5", "PM10", "O3") if overview_only
                              else ("PM2.5", "PM10", "O3", "SO2", "NO2", "CO"))
                return {key: row.get(key) for key in
                        ("city", "distance_km", "bearing_from_target_deg", "angle_to_inflow_deg")} | {
                    name: [row.get(name, {}).get(key) for key in
                           ("latest", "mean_24h", "change_6h")]
                    for name in pollutants if row.get(name)
                }

            count = 2 if overview_only else 4
            pool = context.get("transport_candidate_pool")
            if isinstance(pool, list):
                # The direction-balanced pool (up to 64 cities x six pollutants x
                # four statistics) is a tabular-side projection; the policy view
                # keeps the geometry and the three key pollutants as
                # [latest, change_6h] for the nearest members.
                def compact_pool_row(row: dict) -> dict:
                    out = {key: row.get(key) for key in
                           ("city", "distance_km", "bearing_from_target_deg", "angle_to_inflow_deg")}
                    for pollutant in ("PM2.5", "PM10", "O3"):
                        values = row.get(pollutant) or {}
                        if values.get("latest") is not None:
                            out[pollutant] = [values.get("latest"), values.get("change_6h")]
                    return out
                context["transport_candidate_pool"] = [
                    compact_pool_row(row) for row in pool[:8 if overview_only else 16]
                ]
                context["candidate_value_order"] = (
                    "upwind/hotspot rows: [latest, mean_24h, change_6h]; pool rows: [latest, change_6h]"
                )
                context["transport_candidate_pool_note"] = (
                    f"{len(pool)} direction-balanced issue-time cities in the full pool; "
                    "showing the first "
                    f"{min(len(pool), 8 if overview_only else 16)} with [latest, change_6h]"
                )
            context["upwind_candidates"] = [compact_city(row)
                                             for row in context.get("upwind_candidates", [])[:count]]
            context["regional_hotspots"] = [compact_city(row)
                                             for row in context.get("regional_hotspots", [])[:count]]
            static = context.get("static", {})
            static.pop("raw", None)
            if overview_only:
                aligned = static.get("inflow_aligned", {})
                emissions = aligned.get("emissions", {})
                context["static"] = {"inflow_aligned": {
                    "direction": aligned.get("direction"),
                    "terrain": aligned.get("terrain"),
                    "dominant_emission_sectors": {
                        name: (values.get("dominant_sectors_within_300km") or [])[:2]
                        for name, values in emissions.items()
                        if name in {"PM2.5", "NOx", "SO2", "NMVOC"}
                    },
                }}
                context.pop("target", None)
                static = context["static"]
            target_static = static.get("target", {})
            terrain = target_static.get("terrain", {})
            target_static["terrain"] = {
                key: terrain.get(key) for key in ("elevation_m", "basin_index_m", "radius")
            }
            emissions = target_static.get("anthropogenic_emissions", {})
            target_static["anthropogenic_emissions"] = {
                name: {key: values.get(key) for key in
                       ("total_tonnes_per_year_by_radius_km", "dominant_sectors")}
                for name, values in emissions.items()
                if name in {"PM2.5", "NOx", "SO2", "NMVOC"}
            }
            compact["source_context"] = context
        return {"contract_version": self.bundle.evidence.get("contract_version"),
                "detail": detail,
                "submission_evidence_type_mapping": {
                    "composition": "model_guidance",
                    "fires": "observation",
                    "source_context": "diagnostic",
                },
                "query_note": ("overview is compact; query kind=composition, fires, or source_context "
                               "for the complete topic summary") if overview_only else None,
                **compact}

    def _tool_retrieve_forecast_methods(self, query: str, top_k: int = 3,
                                        pollutant: Optional[str] = None) -> dict:
        if self.methods is None:
            return {"available": False}
        return self.methods.query(query, issue_date=self.bundle.issue_date,
                                  region=self.bundle.region, available_tools=set(self._tools),
                                  top_k=top_k, pollutant=pollutant)

    def _tool_find_similar_cases(self, top_k: int = 3, pollutant: Optional[str] = None,
                                 same_region: bool = False) -> dict:
        if self.historical_cases is None:
            return {"available": False, "reason": "historical_index_not_configured"}
        return self.historical_cases.query(self.bundle, top_k=top_k, pollutant=pollutant,
                                           same_region=same_region)

    def _tool_get_historical_case(self, case_id: str, detail: str = "summary") -> dict:
        if self.historical_cases is None:
            return {"available": False, "reason": "historical_index_not_configured"}
        return self.historical_cases.get_case(self.bundle, case_id, detail=detail)

    def _tool_submit_forecast(self, forecast: Any = None) -> dict:
        b = self.bundle
        truth = b.truth_daily_full()  # 多污染物真值直达评分；纯 PM2.5 个例行为不变
        if truth is None:
            # 无真值个例（如尚未回填的真实个例）：接受合法提交但不打分
            from .schema import validate_forecast
            fc, errors = validate_forecast(
                forecast, issue_date=b.issue_date, horizon=b.horizon, region=b.region,
                require_pm10_range=bool((b.meta or {}).get("multi_pollutant")),
                require_o3_range=bool((b.meta or {}).get("multi_pollutant")),
                aqi_standard=(b.meta or {}).get("aqi_standard"),
            )
            if fc is None:
                return {"accepted": False, "errors": errors}
            normalized = fc.to_dict()
            return {"accepted": True, "scored": False,
                    "derived_heads": _derived_heads(normalized),
                    "normalized_forecast": normalized, "_score": None}
        result = score_forecast(
            forecast, truth,
            issue_date=b.issue_date, horizon=b.horizon, region=b.region,
            expert_evidence_types=b.expert_evidence_types(),
            tools_called=self.tools_called,
            evidence_registry=self.evidence_registry,
            aqi_standard=(b.meta or {}).get("aqi_standard"),
            cfg=self.cfg.reward,
        )
        if not result.valid:
            return {"accepted": False, "errors": result.errors}
        normalized = result.details.get("normalized_forecast")
        return {"accepted": True, "scored": True,
                "derived_heads": _derived_heads(normalized),
                "normalized_forecast": normalized,
                "_score": result}

    # ------------------------------------------------------------------ misc
    def dump_transcript(self) -> str:
        return json.dumps(self.transcript, ensure_ascii=False, indent=1)
