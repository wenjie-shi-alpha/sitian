"""个例数据包（case bundle）：一个起报时次的全部输入、真值与专家参照。

目录契约（一个 case 一个目录）：
    case.json               元信息：case_id / issue_date / region / horizon / regions / meta
    observations.json       起报前实况（agent 可见）
    diagnostics.json        逐日诊断特征，未来日来自 NWP 派生（agent 可见）
    guidance.json           模式指导（agent 可见）
    previous_forecast.json  昨日预报，可选（agent 可见）
    evidence.json           全国共享原始场派生的形势/成分/来源证据（agent 可见）
    truth.json              未来逐日观测真值（仅评分器可见，工具永不暴露）
    expert.json             专家预报与证据类型，可选（仅评分器/基线可见）

时间门禁是构建方责任：observations 的时刻必须早于起报时刻。
`audit_time_gate()` 提供机检。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

from .schema import parse_date

# agent 可见文件（truth/expert 永不进入这个集合）
VISIBLE_FILES = ("observations.json", "diagnostics.json", "guidance.json",
                 "previous_forecast.json", "evidence.json")
HIDDEN_FILES = ("truth.json", "expert.json")


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")


@dataclass
class CaseBundle:
    case_id: str
    issue_date: str          # ISO date，会商/起报日
    region: str              # 预报目标区域（v0 单目标，默认 beijing）
    horizon: int             # 预报天数（issue+1 .. issue+horizon）
    regions: list[str] = field(default_factory=list)   # 实况覆盖的区域列表
    meta: dict = field(default_factory=dict)
    observations: dict = field(default_factory=dict)   # {pollutant: {"times": [...], "series": {region: [...]}}}
    diagnostics: dict = field(default_factory=dict)    # {"daily": {date: {feature: value}}}
    guidance: dict = field(default_factory=dict)       # {"sources": {name: {"daily_pm25": {date: value}}}}
    previous_forecast: Optional[dict] = None           # 昨日预报对象，可选
    evidence: dict = field(default_factory=dict)        # 开放数据派生证据；不得含真值/专家内容
    truth: Optional[dict] = None                       # {"daily": {date: {"pm25_avg": float}}}
    expert: Optional[dict] = None                      # {"forecast": {...}, "evidence_types": [...], "notes": str}

    # ---------- IO ----------
    @classmethod
    def load(cls, case_dir: str | Path) -> "CaseBundle":
        d = Path(case_dir)
        meta = _read_json(d / "case.json")
        bundle = cls(
            case_id=meta["case_id"],
            issue_date=meta["issue_date"],
            region=meta.get("region", "beijing"),
            horizon=int(meta.get("horizon", 6)),
            regions=list(meta.get("regions", [])),
            meta=meta.get("meta", {}),
        )
        for name, attr in [
            ("observations.json", "observations"),
            ("diagnostics.json", "diagnostics"),
            ("guidance.json", "guidance"),
            ("previous_forecast.json", "previous_forecast"),
            ("evidence.json", "evidence"),
            ("truth.json", "truth"),
            ("expert.json", "expert"),
        ]:
            p = d / name
            if p.exists():
                setattr(bundle, attr, _read_json(p))
        return bundle

    def save(self, case_dir: str | Path) -> Path:
        d = Path(case_dir)
        _write_json(d / "case.json", {
            "case_id": self.case_id,
            "issue_date": self.issue_date,
            "region": self.region,
            "horizon": self.horizon,
            "regions": self.regions,
            "meta": self.meta,
        })
        _write_json(d / "observations.json", self.observations)
        _write_json(d / "diagnostics.json", self.diagnostics)
        _write_json(d / "guidance.json", self.guidance)
        if self.previous_forecast is not None:
            _write_json(d / "previous_forecast.json", self.previous_forecast)
        if self.evidence:
            _write_json(d / "evidence.json", self.evidence)
        if self.truth is not None:
            _write_json(d / "truth.json", self.truth)
        if self.expert is not None:
            _write_json(d / "expert.json", self.expert)
        return d

    # ---------- 视图 ----------
    def forecast_dates(self) -> list[str]:
        base = parse_date(self.issue_date, "issue_date")
        return [(base + timedelta(days=i + 1)).isoformat() for i in range(self.horizon)]

    def truth_daily(self) -> Optional[dict[str, float]]:
        """目标区域逐日 PM2.5 真值 {date: pm25_avg}；truth 缺失或不完整返回 None。"""
        if not self.truth:
            return None
        daily = self.truth.get("daily", {})
        out: dict[str, float] = {}
        for day in self.forecast_dates():
            rec = daily.get(day)
            if rec is None or "pm25_avg" not in rec:
                return None
            out[day] = float(rec["pm25_avg"])
        return out

    # truth.json 字段名 → 评分用污染物键（口径见 schema.IAQI_BREAKPOINTS）
    _TRUTH_KEYS = {"pm25_avg": "PM2.5", "pm10_avg": "PM10", "o3_8h": "O3",
                   "so2_avg": "SO2", "no2_avg": "NO2", "co_avg": "CO"}

    def truth_daily_full(self) -> Optional[dict[str, dict[str, float]]]:
        """逐日多污染物真值 {date: {"PM2.5":…, "O3":…}}；至少需 PM2.5，缺失返回 None。"""
        if not self.truth:
            return None
        daily = self.truth.get("daily", {})
        out: dict[str, dict[str, float]] = {}
        for day in self.forecast_dates():
            rec = daily.get(day)
            if rec is None or "pm25_avg" not in rec:
                return None
            out[day] = {pol: float(rec[k]) for k, pol in self._TRUTH_KEYS.items() if k in rec}
        return out

    def expert_evidence_types(self) -> Optional[set[str]]:
        if not self.expert:
            return None
        types = self.expert.get("evidence_types")
        if not types:
            return None
        return set(types)

    # ---------- 审计 ----------
    def audit_time_gate(self) -> list[str]:
        """机检时间门禁：实况时刻必须严格早于起报日 08:00。返回违规清单。"""
        violations: list[str] = []
        cutoff = f"{self.issue_date}T08:00"
        for pollutant, block in self.observations.items():
            for t in block.get("times", []):
                if t >= cutoff:
                    violations.append(
                        f"observations[{pollutant}] time {t} not before cutoff {cutoff}"
                    )
        # Evidence records must carry explicit availability, independent of
        # their valid time.  A +120 h model field is legal if the forecast was
        # published before issue; an observation acquired later is not.
        if self.evidence:
            from datetime import datetime, timezone

            cutoff_utc = datetime.fromisoformat(self.issue_date).replace(tzinfo=timezone.utc)

            def visit(value: Any, path: str) -> None:
                if isinstance(value, dict):
                    stamp = value.get("available_at")
                    if stamp:
                        text = str(stamp)
                        parsed = datetime.fromisoformat(
                            text[:-1] + "+00:00" if text.endswith("Z") else text)
                        if parsed.tzinfo is None or parsed.astimezone(timezone.utc) > cutoff_utc:
                            violations.append(f"{path}.available_at {stamp} after cutoff {cutoff_utc.isoformat()}")
                    for key, child in value.items():
                        visit(child, f"{path}.{key}")
                elif isinstance(value, list):
                    for index, child in enumerate(value):
                        visit(child, f"{path}[{index}]")

            visit(self.evidence, "evidence")
        return violations
