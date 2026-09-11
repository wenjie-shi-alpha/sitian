"""Small, inspectable method-card retrieval; no forecast answers or rewards.

Bundled cards are design hypotheses from common sense, not JJJ_ATMO extracts.
Corpus-derived cards require explicit review and historical source availability.
"""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from .guidance_bias import issue_time

METHODS_VERSION = "forecast-methods-v1"


def _card(identifier, title, keywords, pollutants, question, checks, supporting, refuting, failures):
    return {
        "method_id": identifier, "version": "1.0", "title": title,
        "keywords": keywords, "pollutants": pollutants, "regions": ["all"],
        "seasons": ["all"], "decision_question": question,
        "preconditions": ["先核实资料时间、覆盖、指标口径及目标城市的代表性"],
        "evidence_checks": checks, "supporting_signals": supporting,
        "refuting_signals": refuting, "failure_modes": failures,
        "provenance": {"source_kind": "common_sense", "source_refs": [],
                       "review_status": "hypothesis", "case_answers_removed": True,
                       "temporal_scope": "generic_method"},
    }


BUILTIN_CARDS = [
    _card("stability_persistence", "稳定层结和扩散不良会持续多久", ["逆温", "静稳", "扩散", "持续", "边界层", "混合层", "inversion", "stability"],
          ["PM2.5", "PM10", "O3"], "近地通风与垂直混合何时改善，是否足以结束污染积累？",
          [{"tool": "get_synoptic_evidence", "check": "对照多个时次的层间温差、不同高度风场和垂直运动；避开地面以下气压层"},
           {"tool": "get_diagnostics", "check": "核对边界层高度、近地风和可用降水信息；留意原生覆盖与缺测"},
           {"tool": "get_native_meteorology", "check": "若已注册，检查夜间BLH最低值及配对近地风，不用日最大代表夜间"},
           {"tool": "compute_diffusion_conditions", "check": "若已注册，比较阈值下的通风代理采样跨度；不能直接当污染持续时间"},
           {"tool": "get_observations", "check": "检查污染是否仍在积累，实况转折是否支持混合改善"}],
          ["多个时次持续存在稳定层结和弱近地风", "边界层发展受限与污染积累同步"],
          ["层结与近地通风均已改善", "有效降水或气团更替改变积累条件"],
          ["少数气压层温差不能确定浅薄逆温的底高和厚度", "日最大边界层高度不能代表夜间混合层", "高空风增强不保证地面污染清除"]),
    _card("cold_air_clearance", "冷空气是否足以清除污染", ["冷空气", "清除", "转折", "cold", "clearance"],
          ["PM2.5", "PM10"], "指导中的清除是否有足够证据，还是到达时间或作用高度存在疑问？",
          [{"tool": "get_synoptic_evidence", "check": "检查冷空气到达时间、不同高度风场及稳定度"},
           {"tool": "get_process_evidence", "check": "核对边界层、降水、上游候选与污染初态"}],
          ["近地通风与垂直混合改善的时间一致", "实况转折与预报清除时段相互支持"],
          ["高空风增强而近地仍静稳", "来流方向存在污染且到达时间不确定"],
          ["日均风掩盖短时逆转", "地形阻挡或降水落区偏差"]),
    _card("upstream_transport", "上游污染是否可能影响目标城市", ["输送", "上游", "风向", "transport"],
          ["PM2.5", "PM10", "O3"], "上游浓度与来流变化是否支持输送假设？",
          [{"tool": "get_pollution_evidence", "check": "kind=source_context，核查上游实况、地形与来源"},
           {"tool": "get_synoptic_evidence", "check": "检查风向持续性、多层一致性和到达时段"}],
          ["上游污染与持续来流方向一致"],
          ["风向迅速转变", "路径上存在清除或阻挡条件"],
          ["方向和距离筛选不等于空气团轨迹", "上游观测与目标尺度不一致"]),
    _card("guidance_conflict", "模式与实况冲突时如何查证", ["模式", "偏差", "订正", "初值", "guidance", "bias"],
          ["PM2.5", "PM10", "O3"], "冲突来自初态、统计口径、时效还是过程判断？",
          [{"tool": "list_data_assets", "check": "对齐污染物口径、日期、发布时间与来源依赖"},
           {"tool": "get_observations", "check": "检查近时段浓度及变化"},
           {"tool": "get_model_guidance", "check": "比较同一目标日的各指导源"},
           {"tool": "get_guidance_bias", "check": "样本范围与偏差符号是否适用于本次问题"}],
          ["同口径差异持续存在且有独立实况支持"],
          ["差异可由指标口径、缺测或模式循环不同解释"],
          ["历史总体偏差不能直接当作本次订正量", "同源摘要不能作为独立证据重复计数"]),
    _card("ozone_conditions", "臭氧生成与输送条件如何核对", ["臭氧", "O3", "辐射", "高温", "ozone"],
          ["O3"], "臭氧指导变化是否与辐射、温度、云雨和输送演变一致？",
          [{"tool": "list_data_assets", "check": "区分瞬时日最大与八小时滑动均值日最大"},
           {"tool": "get_synoptic_evidence", "check": "检查辐射、云雨、温度及风场时序"},
           {"tool": "get_pollution_evidence", "check": "核查气体证据的层次、单位及代表性"}],
          ["辐射与温度变化和指导转折时间一致"],
          ["云雨或气团改变与原假设冲突"],
          ["高温本身不能确定臭氧浓度", "柱总量不能代替地面浓度", "前体物敏感性未经验证"]),
    _card("analog_transfer", "如何判断历史相似过程是否适用", ["相似", "历史", "类比", "analog", "区间", "不确定性"],
          ["PM2.5", "PM10", "O3"], "相似维度是否与本次决策相关，哪些差异可能改变结果？",
          [{"tool": "find_similar_cases", "check": "查看相似维度、差异、覆盖及去重后的候选"},
           {"tool": "get_historical_case", "check": "比较历史起报资料、验证结果与指导误差"}],
          ["初态及关键气象演变同时相似"],
          ["地形、季节、指标口径或关键演变不同", "关键特征缺失"],
          ["相似分数不是发生概率", "少量类比不能校准80%区间", "不能直接复制历史浓度"]),
]


def _terms(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9.]+", text.lower()))
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        words.update(run[i:i + 2] for i in range(max(1, len(run) - 1)))
    return words


class ForecastMethodLibrary:
    def __init__(self, cards: list[dict]):
        self.cards = deepcopy(cards)
        seen = set()
        for card in self.cards:
            allowed = set(BUILTIN_CARDS[0])
            if card.keys() - allowed:
                raise ValueError("unsupported fields in method card; raw case content is not allowed")
            for key in ("method_id", "version", "title", "decision_question", "keywords",
                        "pollutants", "regions", "seasons", "preconditions", "evidence_checks",
                        "supporting_signals", "refuting_signals", "failure_modes", "provenance"):
                if not card.get(key):
                    raise ValueError(f"method card requires {key}")
            if card["method_id"] in seen:
                raise ValueError("duplicate method_id")
            seen.add(card["method_id"])
            provenance = card["provenance"]
            if provenance.get("case_answers_removed") is not True:
                raise ValueError("method cards must exclude case answers")
            kind = provenance.get("source_kind")
            if kind == "JJJ_ATMO":
                if provenance.get("review_status") != "reviewed" or not provenance.get("source_refs"):
                    raise ValueError("JJJ_ATMO cards require reviewed, traceable sources")
                if not provenance.get("reviewer") or not provenance.get("reviewed_at"):
                    raise ValueError("JJJ_ATMO cards require a reviewer and review timestamp")
                stamp = datetime.fromisoformat(provenance.get("source_available_at", ""))
                if stamp.tzinfo is None:
                    raise ValueError("method source availability must include timezone")
            elif kind != "common_sense" or provenance.get("review_status") != "hypothesis":
                raise ValueError("common-sense cards must remain labelled hypothesis")
        self.identity = hashlib.sha256(json.dumps(self.cards, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    @classmethod
    def load(cls, path: str | None) -> ForecastMethodLibrary:
        if path is None:
            return cls(BUILTIN_CARDS)
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        if artifact.get("contract_version") != METHODS_VERSION:
            raise ValueError("unsupported forecast method library")
        return cls(artifact["cards"])

    def eligible(self, issue_date: str) -> list[dict]:
        cutoff = issue_time(issue_date)
        return [card for card in self.cards
                if card["provenance"]["source_kind"] == "common_sense"
                or datetime.fromisoformat(card["provenance"]["source_available_at"]) < cutoff]

    def query(self, query: str, *, issue_date: str, region: str, available_tools: set[str],
              top_k: int = 3, pollutant: str | None = None) -> dict:
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("query must contain 1..500 characters")
        if type(top_k) is not int or not 1 <= top_k <= 5:
            raise ValueError("top_k must be an integer in 1..5")
        if pollutant is not None and pollutant not in {"PM2.5", "PM10", "O3"}:
            raise ValueError("unsupported pollutant")
        from .guidance_bias import season
        terms = _terms(query)
        matches = []
        for card in self.eligible(issue_date):
            if pollutant and pollutant not in card["pollutants"]:
                continue
            if "all" not in card["regions"] and region not in card["regions"]:
                continue
            if "all" not in card["seasons"] and season(issue_date) not in card["seasons"]:
                continue
            indexed = " ".join([card["title"], card["decision_question"], *card["keywords"]])
            matched = sorted(terms & _terms(indexed))
            if not matched:
                continue
            result = deepcopy(card)
            result["matched_terms"] = matched
            result["unavailable_tools"] = sorted({check["tool"] for check in card["evidence_checks"]}
                                                 - available_tools)
            matches.append((len(matched), result))
        matches.sort(key=lambda item: (-item[0], item[1]["method_id"]))
        return {
            "available": bool(matches), "contract_version": METHODS_VERSION,
            "retrieval": "deterministic lexical overlap; not validated relevance or probability",
            "methods": [card for _, card in matches[:top_k]],
            "submission_evidence_type": "expert_prior",
            "usage": "方法用于选择查证步骤，不提供当前浓度答案；hypothesis未获专家确认，不能代替实况证据或换取grounding奖励。",
        }
