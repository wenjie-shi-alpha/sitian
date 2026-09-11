#!/usr/bin/env python3
"""Collect official human/consultation air-quality forecasts as a same-layer reference.

Two sources, both scorer-only (never enter the agent prompt):

* ``cnemc``: 全国空气质量预报信息发布系统 (air.cnemc.cn:18014) city forecasts. Live only,
  no history endpoint, so it must be collected daily from the day it is switched on.
  Output: data/raw/official_forecasts/cnemc_city/<issue_date>.json
* ``nmc``: 中央气象台 空气污染气象条件预报图 (jpg) 与 大气环境气象公报 (pdf)。
  Output: data/raw/official_forecasts/nmc/
* ``mee``: 生态环境部半月全国空气质量预报会商结果 (regional + Beijing day-segments).
  Fully archived back to 2023-12; retroactive. Output: data/raw/official_forecasts/mee_bimonthly/<date>.json

Usage: python scripts/fetch_official_forecasts.py cnemc|mee|all [--root DIR]
"""
from __future__ import annotations

import argparse
import datetime as dt
import html as htmllib
import json
import re
import ssl
import sys
import urllib.parse
import urllib.request
from pathlib import Path

CNEMC = "https://air.cnemc.cn:18014"
MEE_INDEX = "https://www.mee.gov.cn/hjzl/dqhj/kqzlyb/"
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE
_UA = "Mozilla/5.0 (sitian official-forecast collector)"


def _get(url: str, data: dict | None = None, timeout: int = 60) -> str:
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _text(s: str) -> str:
    s = re.sub(r"<(script|style).*?</\1>", "", s, flags=re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    return htmllib.unescape(re.sub(r"\s+", " ", s)).strip()


# ---------------------------------------------------------------- CNEMC ----
_CITY_RE = re.compile(r'data-id="(\d+)"[^>]*>\s*([^<\s]+)')
_BLOCK_RE = re.compile(r'<div class="hourAqiDiv">(.*?)</dl>', re.S)
_DATE_RE = re.compile(r"(\d{4})年(\d{2})月(\d{2})日.*?（(\d+)小时预报）", re.S)
_RANGE_RE = re.compile(r"aqi_value_number.*?>(\d+)<.*?>(\d+)<", re.S)
_LEVEL_RE = re.compile(r'aqi_value_level">([^<]*)<')
_PRIM_RE = re.compile(r"首要污染物:</span><span>([^<]*)<")


def _parse_city(page: str) -> list[dict]:
    rows = []
    for block in _BLOCK_RE.findall(page):
        d = _DATE_RE.search(block)
        if not d:
            continue
        r = _RANGE_RE.search(block)
        lv = _LEVEL_RE.search(block)
        pr = _PRIM_RE.search(block)
        rows.append(
            {
                "target_date": f"{d.group(1)}-{d.group(2)}-{d.group(3)}",
                "lead_hours": int(d.group(4)),
                "aqi_lo": int(r.group(1)) if r else None,
                "aqi_hi": int(r.group(2)) if r else None,
                "level_text": lv.group(1).strip() if lv else None,
                "primary_pollutant": (pr.group(1).strip() or None) if pr else None,
            }
        )
    return rows


def fetch_cnemc(root: Path) -> Path:
    index = _get(f"{CNEMC}/CityForecast")
    cities = sorted({(code, name) for code, name in _CITY_RE.findall(index)})
    fetched_at = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    issue_date = fetched_at.date().isoformat()
    out = {
        "source": "air.cnemc.cn:18014/CityForecast",
        "fetched_at_bjt": fetched_at.isoformat(timespec="minutes"),
        "issue_date": issue_date,
        "cities": {},
        "errors": {},
    }
    for code, name in cities:
        try:
            rows = _parse_city(_get(f"{CNEMC}/CityForecast", {"CityCode": code}))
            if not rows:
                raise ValueError("no forecast blocks parsed")
            out["cities"][name] = {"city_code": code, "daily": rows}
        except Exception as exc:  # noqa: BLE001
            out["errors"][name] = f"{type(exc).__name__}: {exc}"
    for extra in ("ProvinceForecast", "AreaForecast"):
        try:
            out[extra.lower()] = _text(_get(f"{CNEMC}/{extra}"))[:20000]
        except Exception as exc:  # noqa: BLE001
            out["errors"][extra] = f"{type(exc).__name__}: {exc}"
    path = root / "cnemc_city" / f"{issue_date}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"cnemc: {len(out['cities'])} cities, {len(out['errors'])} errors -> {path}")
    return path


# ------------------------------------------------------------------ MEE ----
_MEE_LINK_RE = re.compile(r'href="([^"]*ywdt/xwfb/(\d{6})/t(\d{8})_\d+\.shtml)"')


def fetch_mee(root: Path, max_pages: int = 8) -> int:
    outdir = root / "mee_bimonthly"
    outdir.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for page in range(max_pages):
        url = MEE_INDEX + ("index.shtml" if page == 0 else f"index_{page}.shtml")
        try:
            idx = _get(url)
        except Exception as exc:  # noqa: BLE001
            print(f"mee: stop at page {page}: {exc}")
            break
        links = _MEE_LINK_RE.findall(idx)
        if not links:
            break
        for href, _ym, ymd in links:
            date = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
            path = outdir / f"{date}.json"
            if path.exists():
                continue
            full = urllib.parse.urljoin(url, href)
            try:
                body = _text(_get(full))
            except Exception as exc:  # noqa: BLE001
                print(f"mee: {date} failed: {exc}")
                continue
            m = re.search(r"\d{4}年\d{1,2}月\d{1,2}日，中国环境监测总站", body)
            body = body[m.start():] if m else body
            path.write_text(
                json.dumps({"source": full, "issue_date": date, "text": body}, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            n_new += 1
    print(f"mee: {n_new} new bulletins -> {outdir} ({len(list(outdir.glob('*.json')))} total)")
    return n_new


# ------------------------------------------------------------------ NMC ----
NMC_PAGES = {
    "apwf_air_pollution_24h": "http://www.nmc.cn/publish/environment/air_pollution-24.html",
    "atmos_env_bulletin": "http://www.nmc.cn/publish/environment/National-Bulletin-atmospheric-environment.htm",
}


def fetch_nmc(root: Path) -> int:
    """中央气象台 空气污染气象条件预报图 (daily jpg) + 大气环境气象公报 (pdf when it changes)."""
    outdir = root / "nmc"
    outdir.mkdir(parents=True, exist_ok=True)
    today = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date().isoformat()
    n = 0
    for key, url in NMC_PAGES.items():
        try:
            page = _get(url)
        except Exception as exc:  # noqa: BLE001
            print(f"nmc: {key} failed: {exc}")
            continue
        assets = re.findall(r'(?:src|href)="(https://image\.nmc\.cn/[^"]*(?:product/[^"]*\.jpg|\.pdf)[^"]*)"', page)
        for asset in sorted(set(assets)):
            clean = asset.split("?")[0]
            name = clean.rsplit("/", 1)[-1]
            target = outdir / f"{today}_{key}_{name}" if key.startswith("apwf") else outdir / name
            if target.exists():
                continue
            try:
                req = urllib.request.Request(clean, headers={"User-Agent": _UA, "Referer": url})
                with urllib.request.urlopen(req, timeout=120, context=_CTX) as resp:
                    target.write_bytes(resp.read())
                n += 1
            except Exception as exc:  # noqa: BLE001
                print(f"nmc: {name} failed: {exc}")
    print(f"nmc: {n} new files -> {outdir}")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["cnemc", "mee", "nmc", "all"])
    ap.add_argument("--root", default="data/raw/official_forecasts")
    args = ap.parse_args()
    root = Path(args.root)
    if args.what in ("cnemc", "all"):
        fetch_cnemc(root)
    if args.what in ("mee", "all"):
        fetch_mee(root)
    if args.what in ("nmc", "all"):
        fetch_nmc(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
