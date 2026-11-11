# -*- coding: utf-8 -*-
"""US macro snapshot from the FRED graph CSV endpoint.

The market review's news search needs a paid/keyed search provider; when
none is configured the review receives zero news items and the model then
asserts "no major macro data", missing prints like the October CPI
(+0.4% MoM). FRED's public graph CSV needs no key, so the two
highest-impact US prints (CPI, PPI) are injected as structured facts
instead. Everything is fail-open: any fetch/parse problem yields a
partial snapshot or None, and the review falls back to news only.
"""
from __future__ import annotations

import csv
import io
import logging
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib import request as urlrequest

logger = logging.getLogger(__name__)

_FRED_CSV_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id={series}&cosd={start}&coed={end}"
)
# NOTE (2026-11-11, measured): fredgraph.csv stalls reads indefinitely
# (read timeout, on this VM and on the LA VPS alike) for a descriptive or
# browser User-Agent, while the same request with curl's User-Agent answers
# in ~1s. The edge gates on UA, so this public keyless CSV is fetched with
# curl's UA; curl itself returns the identical payload.
_USER_AGENT = "curl/8.5.0"
_TIMEOUT_SECONDS = 10.0
# (snapshot key, FRED series id)
_SERIES: Tuple[Tuple[str, str], ...] = (
    ("cpi", "CPIAUCSL"),
    ("ppi", "PPIFIS"),
)


def _fetch_series_rows(
    series: str, start: str, end: str, _opener: Optional[Callable] = None
) -> List[Tuple[date, float]]:
    opener = _opener or urlrequest.urlopen
    url = _FRED_CSV_URL.format(series=series, start=start, end=end)
    req = urlrequest.Request(url, headers={"User-Agent": _USER_AGENT})
    with opener(req, timeout=_TIMEOUT_SECONDS) as response:
        raw = response.read()
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    rows: List[Tuple[date, float]] = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            value = float(row[series])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            observed = date.fromisoformat(str(row.get("observation_date", ""))[:10])
        except ValueError:
            continue
        rows.append((observed, value))
    rows.sort()
    return rows


def _summarize_series(rows: List[Tuple[date, float]]) -> Optional[Dict[str, Any]]:
    """Summarize monthly rows into latest index, MoM% and YoY% (or None)."""
    if len(rows) < 2:
        return None
    latest_date, latest = rows[-1]
    _prev_date, prev = rows[-2]
    if not prev:
        return None
    mom = round((latest - prev) / abs(prev) * 100.0, 2)
    yoy: Optional[float] = None
    for observed, value in reversed(rows[:-1]):
        if observed.year == latest_date.year - 1 and observed.month == latest_date.month:
            if value:
                yoy = round((latest - value) / abs(value) * 100.0, 2)
            break
    return {
        "month": latest_date.strftime("%Y-%m"),
        "index": round(latest, 3),
        "mom_pct": mom,
        "yoy_pct": yoy,
    }


def fetch_us_macro_snapshot(
    _opener: Optional[Callable] = None, _today: Optional[date] = None
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Fetch the latest US CPI/PPI snapshot, or None when both fail.

    Each series degrades independently, so a missing PPI release (e.g. the
    October PPI is not published until Nov 13) does not drop the CPI line.
    """
    today = _today or date.today()
    start = f"{today.year - 2}-01-01"
    end = today.isoformat()
    snapshot: Dict[str, Dict[str, Any]] = {}
    for key, series in _SERIES:
        try:
            summary = _summarize_series(_fetch_series_rows(series, start, end, _opener))
        except Exception as exc:  # noqa: BLE001 - fail-open by design
            logger.debug("[FRED] series=%s fetch failed: %s", series, exc)
            summary = None
        if summary is not None:
            snapshot[key] = summary
    return snapshot or None


def format_macro_block(snapshot: Optional[Dict[str, Dict[str, Any]]], language: str = "zh") -> str:
    """Render a snapshot as a prompt fact block, or "" when there is none."""
    if not snapshot:
        return ""
    lang = "en" if language in ("en", "ko") else "zh"
    if lang == "en":
        lines = ["## Macro Data (FRED, facts)"]
        cpi = snapshot.get("cpi")
        if cpi:
            yoy = f", YoY {cpi['yoy_pct']:+.2f}%" if cpi.get("yoy_pct") is not None else ""
            lines.append(
                f"- US CPI ({cpi['month']}): index {cpi['index']}, "
                f"MoM {cpi['mom_pct']:+.2f}%{yoy}. Cite this in the News Catalysts "
                "section; do not invent other CPI figures."
            )
        ppi = snapshot.get("ppi")
        if ppi:
            yoy = f", YoY {ppi['yoy_pct']:+.2f}%" if ppi.get("yoy_pct") is not None else ""
            lines.append(
                f"- US PPI final demand ({ppi['month']}): index {ppi['index']}, "
                f"MoM {ppi['mom_pct']:+.2f}%{yoy}. Cite this in the News Catalysts "
                "section; do not invent other PPI figures."
            )
        return "\n".join(lines)

    lines = ["## 宏观数据（FRED，事实数据）"]
    cpi = snapshot.get("cpi")
    if cpi:
        yoy = f"，同比 {cpi['yoy_pct']:+.2f}%" if cpi.get("yoy_pct") is not None else ""
        lines.append(
            f"- 美国 CPI（{cpi['month']}）：指数 {cpi['index']}，环比 {cpi['mom_pct']:+.2f}%{yoy}。"
            "请在第三节消息催化中引用该数据，不要臆断其他 CPI 数值。"
        )
    ppi = snapshot.get("ppi")
    if ppi:
        yoy = f"，同比 {ppi['yoy_pct']:+.2f}%" if ppi.get("yoy_pct") is not None else ""
        lines.append(
            f"- 美国 PPI（最终需求，{ppi['month']}）：指数 {ppi['index']}，环比 {ppi['mom_pct']:+.2f}%{yoy}。"
            "请在第三节消息催化中引用该数据，不要臆断其他 PPI 数值。"
        )
    return "\n".join(lines)
