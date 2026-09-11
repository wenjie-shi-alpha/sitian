"""Receiver-side reconstruction from source-bound native city exports."""
from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from .data_contract import CHINA_TZ, finite_number, issue_time, timestamp, valid_concentration
from .meteorology import wind_direction_label

INPUT_POLLUTANTS = ("PM2.5", "PM10", "O3", "SO2", "NO2", "CO")
TRUTH_FIELDS = {"PM2.5": "pm25_avg", "PM10": "pm10_avg", "O3_8h": "o3_8h",
                "SO2": "so2_avg", "NO2": "no2_avg", "CO": "co_avg"}
# Preserves the existing experiment's aggregation and high-O3 exception.
MIN_HOURS = {p: 14 if p == "O3_8h" else 20 for p in TRUTH_FIELDS}
VARIABLES = {
    "pm2p5": ("kg m**-3", "pm25_ug_m3", "µg/m³", 1e9, 0),
    "pm10": ("kg m**-3", "pm10_ug_m3", "µg/m³", 1e9, 0),
    "go3": ("kg kg**-1", "o3_mass_mixing_ratio", "kg/kg", 1, 0),
    "u10": ("m s**-1", "u10_ms", "m/s", 1, 0),
    "v10": ("m s**-1", "v10_ms", "m/s", 1, 0),
    "t2m": ("K", "temperature_2m_c", "°C", 1, -273.15),
    "d2m": ("K", "dewpoint_2m_c", "°C", 1, -273.15),
    "blh": ("m", "blh_m", "m", 1, 0),
    "tp": ("m", "precipitation_depth_m", "m", 1, 0),
    "tcc": ("(0 - 1)", "cloud_pct", "%", 100, 0),
}


def jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def parse_concentration(raw):
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if valid_concentration(value) else None


class NativeObservations:
    def __init__(self, path, sources):
        self.rows = {}
        self.sources = sources
        for row in jsonl(path):
            day = datetime.strptime(row["date"], "%Y%m%d").date().isoformat()
            hour = int(row["hour"])
            if not 0 <= hour <= 23 or str(hour) != str(row["hour"]):
                raise ValueError("invalid observation hour")
            if row["type"] not in {*INPUT_POLLUTANTS, "O3_8h"}:
                raise ValueError("unexpected observation type")
            key = day, hour, row["type"]
            if key in self.rows:
                raise ValueError(f"duplicate observation hour/type: {key}")
            sha = row["source_sha256"]
            if sha not in sources or Path(sources[sha]["path"]).stem != f"china_cities_{row['date']}":
                raise ValueError("observation source/date mismatch")
            self.rows[key] = row

    def inputs(self, city, issue_date):
        cutoff = issue_time(issue_date)
        out = {}
        for pol in INPUT_POLLUTANTS:
            times, values, available_times, refs = [], [], [], set()
            for offset in range(72, 0, -1):
                stamp = cutoff - timedelta(hours=offset)
                available = stamp + timedelta(hours=1)
                if available >= cutoff:
                    continue
                row = self.rows.get((stamp.date().isoformat(), stamp.hour, pol))
                if row is None:
                    continue
                times.append(stamp.isoformat())
                available_times.append(available.isoformat())
                values.append(parse_concentration(row["city_values"].get(city)))
                refs.add(row["source_sha256"])
            out[pol] = {"unit": "mg/m³" if pol == "CO" else "µg/m³", "times": times,
                        "series": {city: values}, "sample_available_at": available_times,
                        "available_at": max(available_times) if available_times else None,
                        "availability_basis": "configured_latency", "latency_hours": 1,
                        "actual_publication_verified": False, "source_sha256": sorted(refs),
                        "statistic": "hourly_concentration"}
        if len(out["PM2.5"]["times"]) < 48:
            raise ValueError(f"insufficient actual input timestamps: {city} {issue_date}")
        return out

    def truth(self, city, days):
        daily = {}
        for day in days:
            item, quality, sources = {}, {}, set()
            for pol, field in TRUTH_FIELDS.items():
                values = []
                for hour in range(24):
                    row = self.rows.get((day, hour, pol))
                    if row:
                        sources.add(row["source_sha256"])
                        value = parse_concentration(row["city_values"].get(city))
                        if value is not None:
                            values.append(value)
                exception = pol == "O3_8h" and bool(values) and max(values) > 160
                if len(values) < MIN_HOURS[pol] and not exception:
                    raise ValueError(f"invalid truth {city} {day} {pol}: {len(values)} hours")
                item[field] = round(max(values) if pol == "O3_8h" else sum(values) / len(values), 1)
                quality[pol] = {"valid_hours": len(values), "minimum": MIN_HOURS[pol],
                                "exceedance_exception": exception}
            item["quality"] = quality
            item["source_sha256"] = sorted(sources)
            daily[day] = item
        return {"daily": daily, "aggregation_contract": "legacy-six-pollutant-hourly-validity-v1",
                "source": "cnemc_native_city_hour_export", "actual_publication_verified": False}

    def spatial(self, cities, issue_date):
        """Regional context on an actual 24h clock, never the last 24 rows."""
        import numpy as np
        cutoff = issue_time(issue_date)
        output, refs = {}, set()
        for city in cities:
            output[city] = {}
            for pol in INPUT_POLLUTANTS:
                samples = {}
                for offset in range(24, 1, -1):
                    t = cutoff - timedelta(hours=offset)
                    row = self.rows.get((t.date().isoformat(), t.hour, pol))
                    if row:
                        refs.add(row["source_sha256"])
                        value = parse_concentration(row["city_values"].get(city))
                        if value is not None:
                            samples[t] = value
                latest = max(samples) if samples else None
                values = list(samples.values())
                change = (samples[latest] - samples[latest-timedelta(hours=6)]
                          if latest and latest-timedelta(hours=6) in samples else None)
                slope = (float(np.polyfit([(t-cutoff).total_seconds()/3600 for t in samples], values, 1)[0])
                         if len(values) >= 3 else None)
                output[city][pol] = {
                    "unit": "mg/m³" if pol == "CO" else "µg/m³", "n_24h": len(values),
                    "latest": samples[latest] if latest else None,
                    "latest_local_time": latest.isoformat() if latest else None,
                    "mean_24h": round(sum(values)/len(values), 1) if values else None,
                    "max_24h": max(values) if values else None,
                    "change_6h": round(change, 1) if change is not None else None,
                    "slope_24h_per_hour": round(slope, 2) if slope is not None else None,
                }
        return {"pollution": {"spatial_observations": {
            "available": True, "cutoff": (cutoff-timedelta(hours=2)).isoformat(),
            "cutoff_note": "Actual [issue-24h, issue) window; sample time + 1h must be strictly before issue.",
            "cities": output, "source_sha256": sorted(refs),
            "available_at": (cutoff-timedelta(hours=1)).isoformat(),
            "availability_basis": "configured_latency", "actual_publication_verified": False,
        }}}


def normalize_series(row):
    import numpy as np
    var = row["canonical_variable"]
    expected_unit, field, unit, scale, offset = VARIABLES[var]
    if row["unit"] != expected_unit:
        raise ValueError(f"unsupported {var} unit: {row['unit']}")
    cycle = timestamp(row["cycle"])
    if cycle != issue_time(row["issue_date"]) - timedelta(hours=12):
        raise ValueError("incorrect CAMS cycle")
    leads = row["lead_hours"]
    times = row["valid_times"]
    if not leads or leads != sorted(set(leads)) or len(times) != len(leads):
        raise ValueError("invalid native time axis")
    if any(not finite_number(h) or h < 0 for h in leads):
        raise ValueError("invalid forecast lead")
    if any(timestamp(t) != cycle + timedelta(hours=h) for t, h in zip(times, leads)):
        raise ValueError("forecast valid time/lead mismatch")
    dims = row["dimensions"]
    array = np.asarray(row["values"], dtype=object)
    if len(dims) != array.ndim or len(set(dims)) != len(dims):
        raise ValueError("invalid native dimensions")
    axis = dims.index(row["lead_dimension"])
    if array.shape[axis] != len(times):
        raise ValueError("native array/time mismatch")
    for i, dim in enumerate(dims):
        coord = row["coordinates"].get(dim)
        if coord is None or len(coord["values"]) != array.shape[i]:
            raise ValueError("missing or inconsistent native coordinate")
    remaining = [dim for dim in dims if dim != row["lead_dimension"]]
    if remaining and not (var == "go3" and remaining == ["model_level"]
                          and row["coordinates"]["model_level"] == {"values": [137.0], "unit": "1"}):
        raise ValueError("unsupported layer: never select the first level implicitly")
    raw = np.moveaxis(array, axis, 0).reshape(len(times)).tolist()
    if sum(v is None for v in raw) != row["missing_values"]:
        raise ValueError("native mask mismatch")
    if any(v is not None and (not finite_number(v) or
           (var not in {"u10", "v10"} and v < 0) or (var == "tcc" and v > 1)) for v in raw):
        raise ValueError(f"invalid {var} values")
    return field, {"unit": unit, "times": times,
                   "values": [None if v is None else v * scale + offset for v in raw],
                   "cycle": cycle.isoformat(), "available_at": (cycle + timedelta(hours=10)).isoformat(),
                   "availability_basis": "configured_latency", "actual_publication_verified": False,
                   "native_variable": var, "native_unit": expected_unit,
                   "conversion": {"scale": scale, "offset": offset},
                   "native_dimensions": dims, "native_coordinates": row["coordinates"],
                   "grib_step_type": row.get("grib_step_type"),
                   "selected_grid": row["selected_grid"], "source_sha256": row["source_sha256"]}


class NativeCams:
    def __init__(self, path, sources):
        self.rows = defaultdict(dict)
        self.overlaps = []
        for row in jsonl(path):
            if row["source_sha256"] not in sources:
                raise ValueError("unknown CAMS source")
            key = row["city"], row["issue_date"]
            name, normalized = normalize_series(row)
            previous = self.rows[key].get(name)
            if previous:
                a = dict(zip(previous["times"], previous["values"]))
                b = dict(zip(normalized["times"], normalized["values"]))
                shared = a.keys() & b.keys()
                if not shared or previous["unit"] != normalized["unit"] or any(a[t] != b[t] for t in shared):
                    raise ValueError("conflicting overlapping CAMS sources")
                if any(abs(previous["selected_grid"][k] - normalized["selected_grid"][k]) > 1e-6 for k in ("lat", "lon")):
                    raise ValueError("overlapping sources use different grids")
                # Select one source; never splice samples from different files.
                selected = sorted((previous, normalized), key=lambda r: (-len(r["times"]), r["source_sha256"]))[0]
                self.overlaps.append({"city": key[0], "issue_date": key[1], "field": name,
                                      "sources": sorted({previous["source_sha256"], normalized["source_sha256"]}),
                                      "selected_source": selected["source_sha256"], "common_values_identical": True})
                normalized = selected
            self.rows[key][name] = normalized

    def build(self, city, issue_date, days):
        fields = self.rows[(city, issue_date)]
        if set(fields) != {v[1] for v in VARIABLES.values()}:
            raise ValueError("missing native CAMS variables")
        source = {"cycle": fields["blh_m"]["cycle"], "available_at": fields["blh_m"]["available_at"],
                  "availability_basis": "configured_latency", "actual_publication_verified": False,
                  "dependency_group": "cams", "measurement_contracts": {}, "day_coverage": {},
                  "note": "CAMS原生采样聚合；完整日仅在循环覆盖整日且原生网格无缺测时给出。O3为3h瞬时最大代理，以固定空气密度1.2 kg/m³换算，非8小时指标。"}
        for native, target, statistic, factor, agg in [
            ("pm25_ug_m3", "daily_pm25", "daily_mean", 1, "mean"),
            ("pm10_ug_m3", "daily_pm10", "daily_mean", 1, "mean"),
            ("o3_mass_mixing_ratio", "daily_o3max", "daily_max_of_3hourly_instantaneous", 1.2e9, "max"),
        ]:
            block = fields[native]
            source[target], source["partial_" + target], source["day_coverage"][target] = {}, {}, {}
            source["measurement_contracts"][target] = {"statistic": statistic, "unit": "µg/m³",
                "temporal_representation": "native_instantaneous_samples", "native_source_sha256": block["source_sha256"],
                "density_assumption_kg_m3": 1.2 if native == "o3_mass_mixing_ratio" else None}
            for day in days:
                coverage, values = day_samples(block, day)
                source["day_coverage"][target][day] = coverage
                if values:
                    value = (max(values) if agg == "max" else sum(values) / len(values)) * factor
                    destination = target if coverage["complete"] else "partial_" + target
                    source[destination][day] = round(value, 1)
        daily = {}
        for day in days:
            d = {}
            sampled = {name: day_samples(block, day) for name, block in fields.items()}
            if not sampled["blh_m"][1]:
                continue
            pairs = paired(fields, ["u10_ms", "v10_ms", "temperature_2m_c", "dewpoint_2m_c"], day)
            speeds = [math.hypot(v[0], v[1]) for _, v in pairs]
            if speeds:
                u, v = (sum(values[i] for _, values in pairs) / len(pairs) for i in (0, 1))
                direction = (math.degrees(math.atan2(-u, -v)) + 360) % 360 if math.hypot(u, v) > 1e-8 else None
                d.update(wind_speed_ms=round(sum(speeds) / len(speeds), 2), wind_dir_deg=direction,
                         wind_dir=wind_direction_label(direction) if direction is not None else None)
                rh = [min(100, 100 * math.exp(17.625 * td / (243.04 + td) - 17.625 * t / (243.04 + t)))
                      for _, (_, _, t, td) in pairs]
                d["rh_pct"] = round(sum(rh) / len(rh), 1)
            for name, target, fn in [("blh_m", "blh_max_m", max), ("blh_m", "blh_min_m", min),
                                      ("temperature_2m_c", "tmax_c", max), ("cloud_pct", "cloud_pct", lambda v: sum(v)/len(v))]:
                values = sampled[name][1]
                d[target] = round(fn(values), 2) if values else None
            nights = [v for t, v in zip(fields["blh_m"]["times"], fields["blh_m"]["values"])
                      if timestamp(t).astimezone(CHINA_TZ).date().isoformat() == day
                      and (timestamp(t).astimezone(CHINA_TZ).hour < 8 or timestamp(t).astimezone(CHINA_TZ).hour >= 20)
                      and v is not None]
            d.update(blh_night_min_m=min(nights) if nights else None, rain_mm=None,
                     rain_status="unresolved_native_accumulation_semantics", source="cams_native",
                     native_coverage={name: meta for name, (meta, _) in sampled.items()})
            daily[day] = d
        return source, {"daily": daily, "native": {"cams": {"fields": fields}},
                        "native_contract": "city-native-v1", "precipitation_note":
                        "tp原值保留；来源标为instant且无累计区间说明，不求伪精确日降水，也不将未知当0。"}


def day_samples(block, day):
    start = datetime.fromisoformat(day).replace(tzinfo=CHINA_TZ)
    end = start + timedelta(days=1)
    stamps = [timestamp(t) for t in block["times"]]
    step = min((b-a).total_seconds()/3600 for a, b in zip(stamps, stamps[1:])) if len(stamps) > 1 else 24
    values = [v for t, v in zip(stamps, block["values"]) if start <= t < end and v is not None]
    expected = sum(start <= stamps[0] + timedelta(hours=step*i) < end
                   for i in range(int((end-stamps[0]).total_seconds()/3600/step)+1)) if end > stamps[0] else 0
    covered = stamps[0] <= start and stamps[-1] >= end
    return {"samples": len(values), "expected_native_samples": expected, "native_step_hours": step,
            "cycle_covers_full_day": covered, "complete": covered and len(values) == expected and expected > 0}, values


def paired(fields, names, day=None):
    maps = [{t: v for t, v in zip(fields[name]["times"], fields[name]["values"]) if v is not None} for name in names]
    shared = set.intersection(*(set(m) for m in maps))
    return [(t, [m[t] for m in maps]) for t in sorted(shared)
            if day is None or timestamp(t).astimezone(CHINA_TZ).date().isoformat() == day]
