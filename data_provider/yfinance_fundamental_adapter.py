# -*- coding: utf-8 -*-
"""
Yfinance fundamental adapter for HK/US markets (fail-open).

Mirrors the bundle shape of `AkshareFundamentalAdapter.get_fundamental_bundle`
so it can be plugged into `data_provider.base.get_fundamental_context()`
without changing downstream consumers. Adds HK/US-specific fields:

- ``earnings.financial_report.currency`` — financial statement currency
  (``USD`` / ``HKD`` / ``CNY``) from ``info.financialCurrency``. For HK ADRs
  yfinance commonly reports ``financialCurrency=CNY`` while trades settle in
  HKD, so this differs from the dividend currency below.
- ``earnings.dividend.currency`` — trading / dividend currency from
  ``info.currency`` (e.g. HKD for 0700.HK). Used to suffix 港元/美元/元 for
  per-share cash dividends and to scope the TTM yield denominator.
- ``earnings.dividend.ttm_dividend_yield_pct`` — computed as
  ``ttm_cash_dividend_per_share / latest_price * 100``, both sides in the
  trading currency (info.currentPrice/regularMarketPrice/previousClose).
  ``info.dividendYield`` is only used as a last-resort fallback and is
  passed through as-is (current yfinance reports it in percent units).
- ``belong_boards`` — derived from ``info.sector`` + ``info.industry``; the CN
  pipeline derives it from AkShare 板块名单, this is the HK/US analogue.

This adapter intentionally treats every yfinance call as best-effort and never
raises to caller. Partial data is allowed; downstream `_infer_block_status` will
mark the block as ``partial`` when only some fields are populated.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


_INCOME_REVENUE_KEYS = ("Total Revenue", "TotalRevenue", "Revenue")
_INCOME_NET_PROFIT_KEYS = (
    "Net Income Common Stockholders",
    "Net Income From Continuing Operation Net Minority Interest",
    "Net Income",
    "NetIncome",
)
_CASHFLOW_OP_KEYS = (
    "Operating Cash Flow",
    "Cash Flow From Continuing Operating Activities",
    "Total Cash From Operating Activities",
)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN guard
        return None
    return result


def _ratio_to_pct(value: Any) -> Optional[float]:
    """yfinance returns ratios as decimal (0.166 = 16.6%); convert to percent."""
    raw = _safe_float(value)
    if raw is None:
        return None
    return round(raw * 100.0, 4)


def _pick_row(df: pd.DataFrame, keys) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    for key in keys:
        if key in df.index:
            try:
                return df.loc[key]
            except KeyError:
                continue
    return None


def _latest_value(row: Optional[pd.Series]) -> Optional[float]:
    if row is None or row.empty:
        return None
    try:
        return _safe_float(row.iloc[0])
    except IndexError:
        return None


def _yoy_from_row(row: Optional[pd.Series]) -> Optional[float]:
    """Statement-derived YoY: requires the same quarter one year earlier.

    The prior-year quarter is matched by date (within 45 days), not by
    position, so a quarter missing from the statement cannot shift the base.
    yfinance ``quarterly_*_stmt`` returns 4 quarters by default, so this
    typically returns None and callers fall back to ``info.revenueGrowth`` /
    ``info.earningsGrowth`` (already TTM YoY ratios). Doing QoQ via ``iloc[1]``
    is wrong for seasonal businesses — explicitly refuse it.
    """
    if row is None or row.empty:
        return None
    try:
        dates = pd.to_datetime(row.index)
    except (TypeError, ValueError):
        return None
    series = pd.Series(row.values, index=dates).sort_index(ascending=False)
    latest_date = series.index[0]
    latest = _safe_float(series.iloc[0])
    target = latest_date - pd.DateOffset(years=1)
    candidates = [(abs((d - target).days), d) for d in series.index[1:] if abs((d - target).days) <= 45]
    if not candidates:
        return None
    prev_year = _safe_float(series[min(candidates)[1]])
    if latest is None or prev_year in (None, 0):
        return None
    return round((latest - prev_year) / abs(prev_year) * 100.0, 4)


def _epoch_to_date(value: Any) -> Optional[str]:
    raw = _safe_float(value)
    if raw is None:
        return None
    try:
        return datetime.fromtimestamp(raw, tz=timezone.utc).date().isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _convert_to_yf_symbol(stock_code: str) -> str:
    """Convert internal code to yfinance ticker. Lightweight inline reproduction
    of YFinanceFetcher._convert_stock_code to avoid pulling the full fetcher
    into the fundamental path.
    """
    code = (stock_code or "").strip().upper()
    if not code:
        return code
    if code.startswith("HK"):
        digits = code[2:].lstrip("0") or "0"
        return f"{digits.zfill(4)}.HK"
    if "." in code:
        # 美股点号类代码 Yahoo 要求连字符（BRK.B -> BRK-B）；港股/日韩台
        # 后缀（.HK/.T/.TW 等）保持原样。
        head, _, tail = code.partition(".")
        if head.isalpha() and len(head) <= 5 and len(tail) == 1 and tail.isalpha():
            return f"{head}-{tail}"
        return code
    # Assume US ticker by default for non-HK / non-CN callers
    return code


# ---------------- SEC EDGAR cross-check (US stocks) ----------------
# Yahoo's quarterly "Total Revenue" can disagree with the filed 10-Q
# (BRK.B 2026Q3: Yahoo $117.59B vs the filed $105.647B), while the filed
# NetIncomeLoss matches. SEC EDGAR companyfacts is a free, keyless,
# authoritative source for both figures, so prefer it when a matching
# 10-Q exists. Everything here is fail-open: any lookup/fetch failure
# returns None and callers keep the Yahoo values.
_SEC_EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SEC_EDGAR_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_SEC_EDGAR_USER_AGENT = "daily-stock-analysis/1.0 (fundamental cross-check)"
_SEC_EDGAR_TIMEOUT_SECONDS = 10.0
_SEC_EDGAR_TICKER_CACHE: Dict[str, Optional[str]] = {}
_SEC_EDGAR_FACTS_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}
# Public CIKs for the tickers this deployment actually holds, consulted only
# when the SEC ticker map (www.sec.gov) is unreachable — an egress proxy may
# 403 that host while data.sec.gov still answers. The map stays the authority;
# this is a fallback, not a second source of truth.
_SEC_EDGAR_KNOWN_CIKS = {"BRK/B": "0001067983"}


def _edgar_json(url: str, _opener=None) -> Any:
    opener = _opener or urllib.request.urlopen
    request = urllib.request.Request(url, headers={"User-Agent": _SEC_EDGAR_USER_AGENT})
    with opener(request, timeout=_SEC_EDGAR_TIMEOUT_SECONDS) as response:
        return json.load(response)


def _resolve_sec_cik(symbol: str, _opener=None) -> Optional[str]:
    """Resolve a Yahoo symbol to a zero-padded 10-digit SEC CIK, or None."""
    key = (symbol or "").strip().upper()
    if not key:
        return None
    if key in _SEC_EDGAR_TICKER_CACHE:
        return _SEC_EDGAR_TICKER_CACHE[key]
    cik: Optional[str] = None
    try:
        payload = _edgar_json(_SEC_EDGAR_TICKERS_URL, _opener)
        lookup: Dict[str, str] = {}
        for entry in payload.values():
            ticker = str(entry.get("ticker", "")).strip().upper()
            if ticker:
                lookup[ticker] = str(entry.get("cik_str", "")).zfill(10)
        # EDGAR writes share classes with a slash (BRK/B); Yahoo with a dash.
        for candidate in (key, key.replace("-", "/")):
            if candidate in lookup:
                cik = lookup[candidate]
                break
    except Exception:
        cik = None
    if cik is None:
        # Ticker-map fetch failed (or the ticker is absent): fall back to the
        # known-CIK table above so the cross-check still works.
        cik = _SEC_EDGAR_KNOWN_CIKS.get(key.replace("-", "/"))
    _SEC_EDGAR_TICKER_CACHE[key] = cik
    return cik


def _quarter_of_iso_date(iso_date: Any) -> Optional[tuple]:
    try:
        parsed = datetime.strptime(str(iso_date), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None
    return (parsed.year, (parsed.month - 1) // 3 + 1)


def fetch_sec_edgar_quarterly_figures(
    symbol: str, report_date: Any, _opener=None
) -> Optional[Dict[str, float]]:
    """Return filed quarterly figures for a US stock.

    Matches the 10-Q whose period end equals ``report_date`` (the quarter-end
    taken from Yahoo's income statement): ``{"revenue": ..., "net_income": ...}``.
    When the same filing also carries the prior-year quarter as a framed
    comparative row, ``prev_revenue`` / ``prev_net_income`` are included so
    callers can compute YoY from filed numbers instead of Yahoo's ``info``
    ratios (Yahoo's prior-year statement rows are often NaN, e.g. BRK.B
    2025-09-30). Frame-less rows are year-to-date cumulative figures and are
    skipped. Q4 has no standalone 10-Q (the 10-K reports the full year), so
    Q4 always returns None and keeps Yahoo data.
    """
    quarter = _quarter_of_iso_date(report_date)
    if quarter is None or quarter[1] == 4:
        return None
    year, q = quarter
    cik = _resolve_sec_cik(symbol, _opener)
    if not cik:
        return None
    if cik in _SEC_EDGAR_FACTS_CACHE:
        facts = _SEC_EDGAR_FACTS_CACHE[cik]
    else:
        try:
            facts = _edgar_json(_SEC_EDGAR_COMPANYFACTS_URL.format(cik=cik), _opener)
        except Exception:
            facts = None
        _SEC_EDGAR_FACTS_CACHE[cik] = facts
    if not isinstance(facts, dict):
        return None
    units = facts.get("facts", {}).get("us-gaap", {})
    try:
        prev_date = datetime.strptime(str(report_date), "%Y-%m-%d").date().replace(year=year - 1)
    except (TypeError, ValueError):
        prev_date = None
    prev_frame = f"CY{year - 1}Q{q}"

    def _match_quarter(entries, end, frame) -> Optional[float]:
        # The prior-year quarter is carried by the current filing as a
        # comparative row (same fy/fp, its own frame); the current quarter
        # matches any non-empty frame.
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("form") != "10-Q" or entry.get("end") != end:
                continue
            if entry.get("fy") != year or entry.get("fp") != f"Q{q}":
                continue
            if frame is None:
                if not entry.get("frame"):
                    continue
            elif entry.get("frame") != frame:
                continue
            value = _safe_float(entry.get("val"))
            if value is not None:
                return value
        return None

    out: Dict[str, float] = {}
    for concept, field in (("Revenues", "revenue"), ("NetIncomeLoss", "net_income")):
        entries = units.get(concept, {}).get("units", {}).get("USD", [])
        value = _match_quarter(entries, report_date, None)
        if value is not None:
            out[field] = value
        if prev_date is not None:
            prev_value = _match_quarter(entries, prev_date.isoformat(), prev_frame)
            if prev_value is not None:
                out[f"prev_{field}"] = prev_value
    return out or None


class YfinanceFundamentalAdapter:
    """HK/US fundamental adapter backed by yfinance.

    Returns the same bundle keys as :class:`AkshareFundamentalAdapter` so the
    aggregation in :func:`data_provider.base.get_fundamental_context` can stay
    market-agnostic.
    """

    def __init__(self, sec_edgar_fetcher=None):
        # Optional SEC EDGAR cross-check for the quarterly figures (injected
        # by the production wiring in data_provider.base; None keeps the
        # adapter fully offline, which the unit tests rely on).
        self._sec_edgar_fetcher = sec_edgar_fetcher

    def get_fundamental_bundle(self, stock_code: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "status": "not_supported",
            "growth": {},
            "earnings": {},
            "institution": {},
            "boards": {},
            "belong_boards": [],
            "source_chain": [],
            "errors": [],
        }

        try:
            import yfinance as yf
        except Exception as exc:
            result["errors"].append(f"import_yfinance:{type(exc).__name__}")
            return result

        symbol = _convert_to_yf_symbol(stock_code)
        if not symbol:
            result["errors"].append("empty_symbol")
            return result

        ticker = yf.Ticker(symbol)
        info: Dict[str, Any] = {}
        try:
            info = ticker.get_info() if hasattr(ticker, "get_info") else (ticker.info or {})
            if not isinstance(info, dict):
                info = {}
        except Exception as exc:
            result["errors"].append(f"info:{type(exc).__name__}:{exc}")
            info = {}

        # Financial statements (income/cashflow) are reported in `financialCurrency`;
        # for HK ADRs that is often CNY even when the stock trades in HKD. Dividends
        # and live price are paid/quoted in `currency` — keep them separate so the
        # renderer can suffix per-block currency tags correctly.
        financial_currency = str(info.get("financialCurrency") or info.get("currency") or "").upper() or None
        dividend_currency = str(info.get("currency") or info.get("financialCurrency") or "").upper() or None

        # ---------------- growth block ----------------
        growth_payload: Dict[str, Any] = {
            "revenue_yoy": _ratio_to_pct(info.get("revenueGrowth")),
            "net_profit_yoy": _ratio_to_pct(info.get("earningsGrowth")),
            "roe": _ratio_to_pct(info.get("returnOnEquity")),
            "gross_margin": _ratio_to_pct(info.get("grossMargins")),
        }
        if any(v is not None for v in growth_payload.values()):
            result["growth"] = growth_payload
            result["source_chain"].append("growth:yfinance.info")

        # ---------------- financial_report ----------------
        report_date: Optional[str] = None
        revenue_latest: Optional[float] = None
        net_profit_latest: Optional[float] = None
        operating_cash_flow_latest: Optional[float] = None
        revenue_row = None
        net_profit_row = None

        try:
            income_df = ticker.quarterly_income_stmt
        except Exception as exc:
            result["errors"].append(f"quarterly_income_stmt:{type(exc).__name__}")
            income_df = None
        if income_df is not None and not income_df.empty:
            try:
                if all(hasattr(col, "to_pydatetime") or isinstance(col, (datetime, pd.Timestamp)) for col in income_df.columns):
                    income_df = income_df.reindex(columns=sorted(income_df.columns, reverse=True))
                first_col = income_df.columns[0]
                ts = pd.to_datetime(first_col, errors="coerce")
                if pd.notna(ts):
                    report_date = ts.date().isoformat()
            except Exception:
                pass
            revenue_row = _pick_row(income_df, _INCOME_REVENUE_KEYS)
            net_profit_row = _pick_row(income_df, _INCOME_NET_PROFIT_KEYS)
            revenue_latest = _latest_value(revenue_row)
            net_profit_latest = _latest_value(net_profit_row)

        try:
            cashflow_df = ticker.quarterly_cashflow
        except Exception as exc:
            result["errors"].append(f"quarterly_cashflow:{type(exc).__name__}")
            cashflow_df = None
        if cashflow_df is not None and not cashflow_df.empty:
            operating_cash_flow_latest = _latest_value(_pick_row(cashflow_df, _CASHFLOW_OP_KEYS))

        # Fallback to TTM aggregates from .info when quarterly statements are
        # unavailable — still produces a non-empty row.
        if revenue_latest is None:
            revenue_latest = _safe_float(info.get("totalRevenue"))
        if operating_cash_flow_latest is None:
            operating_cash_flow_latest = _safe_float(info.get("operatingCashflow"))
        if net_profit_latest is None and revenue_latest is not None:
            margin = _safe_float(info.get("profitMargins"))
            if margin is not None:
                net_profit_latest = revenue_latest * margin

        # Statement-derived YoY (requires the same quarter a year earlier) is preferred
        # over .info ratios; otherwise keep the TTM growth values already set
        # from info.revenueGrowth / info.earningsGrowth above. Refuse QoQ
        # fallback — it produces misleading numbers for seasonal businesses.
        statement_revenue_yoy = _yoy_from_row(revenue_row)
        statement_net_profit_yoy = _yoy_from_row(net_profit_row)
        if statement_revenue_yoy is not None:
            growth_payload["revenue_yoy"] = statement_revenue_yoy
        if statement_net_profit_yoy is not None:
            growth_payload["net_profit_yoy"] = statement_net_profit_yoy
        if any(v is not None for v in growth_payload.values()):
            result["growth"] = growth_payload

        # SEC EDGAR cross-check (US stocks, Q1-Q3): prefer the filed 10-Q
        # quarterly figures over Yahoo's statement row when they exist.
        # Any lookup/fetch failure keeps the Yahoo values (fail-open).
        # The fetcher is injected by the production wiring; without it the
        # adapter stays fully offline.
        edgar_override = None
        if report_date is not None and self._sec_edgar_fetcher is not None:
            try:
                edgar_override = self._sec_edgar_fetcher(symbol, report_date)
            except Exception:
                edgar_override = None
        if edgar_override:
            if edgar_override.get("revenue") is not None:
                revenue_latest = edgar_override["revenue"]
            if edgar_override.get("net_income") is not None:
                net_profit_latest = edgar_override["net_income"]
            result["source_chain"].append("earnings.financial_report:sec-edgar")
            # Filed YoY: the same 10-Q carries the prior-year quarter as a
            # framed comparative row. It outranks both the statement rows
            # (often NaN for the prior year, e.g. BRK.B 2025-09-30) and the
            # .info ratios (BRK.B net YoY was -27.60% from info vs -28.20%
            # from the filed comparatives). Missing comparatives keep Yahoo.
            yoy_overridden = False
            for field, yoy_key, prev_key in (
                ("revenue", "revenue_yoy", "prev_revenue"),
                ("net_income", "net_profit_yoy", "prev_net_income"),
            ):
                current = edgar_override.get(field)
                prev = edgar_override.get(prev_key)
                if current is not None and prev:
                    growth_payload[yoy_key] = round((current - prev) / abs(prev) * 100.0, 4)
                    yoy_overridden = True
            if yoy_overridden:
                result["growth"] = growth_payload
                result["source_chain"].append("growth:sec-edgar")

        financial_report = {
            "report_date": report_date,
            "revenue": revenue_latest,
            "net_profit_parent": net_profit_latest,
            "operating_cash_flow": operating_cash_flow_latest,
            "roe": growth_payload.get("roe"),
            "currency": financial_currency,
        }
        if any(v is not None and v != "" for v in financial_report.values()):
            result.setdefault("earnings", {})["financial_report"] = financial_report
            result["source_chain"].append("earnings.financial_report:yfinance")

        # ---------------- dividend block ----------------
        events: List[Dict[str, Any]] = []
        as_of_date = datetime.now(timezone.utc).date()
        try:
            div_series = ticker.dividends
        except Exception as exc:
            result["errors"].append(f"dividends:{type(exc).__name__}")
            div_series = None
        if div_series is not None and not div_series.empty:
            # yfinance 1.2.x returns Ticker.dividends as a single-column DataFrame, not a
            # Series; coerce so `.items()` yields (timestamp, value) rather than
            # (column_name, Series). Otherwise every event is dropped (`_safe_float(Series)`
            # -> None) and TTM silently falls back to the annual-rate estimate.
            if hasattr(div_series, "columns"):
                div_series = div_series.iloc[:, 0]
            try:
                cutoff = pd.Timestamp(as_of_date).tz_localize(div_series.index.tz) - pd.Timedelta(days=365)
                for ts, value in div_series.items():
                    per_share = _safe_float(value)
                    if per_share is None or per_share <= 0:
                        continue
                    try:
                        event_date = pd.Timestamp(ts).date().isoformat()
                    except Exception:
                        continue
                    events.append({
                        "event_date": event_date,
                        "ex_dividend_date": event_date,
                        "record_date": None,
                        "announcement_date": None,
                        "cash_dividend_per_share": per_share,
                        "is_pre_tax": True,
                    })
                ttm_events = []
                for item in events:
                    try:
                        event_ts = pd.Timestamp(item["event_date"]).tz_localize(div_series.index.tz)
                    except Exception:
                        continue
                    if event_ts >= cutoff:
                        ttm_events.append(item)
            except Exception as exc:
                result["errors"].append(f"dividend_window:{type(exc).__name__}")
                ttm_events = []
        else:
            ttm_events = []

        ttm_cash = sum(item["cash_dividend_per_share"] for item in ttm_events) if ttm_events else None
        if ttm_cash is None:
            ttm_cash = _safe_float(info.get("trailingAnnualDividendRate"))

        if events or ttm_cash is not None:
            events.sort(key=lambda item: item.get("event_date") or "", reverse=True)
            dividend_payload: Dict[str, Any] = {
                "events": events[:5],
                "ttm_event_count": len(ttm_events),
                "ttm_cash_dividend_per_share": round(ttm_cash, 6) if ttm_cash is not None else None,
                "coverage": "cash_dividend_pre_tax",
                "currency": dividend_currency,
                "as_of": as_of_date.isoformat(),
            }

            # Yield: prefer recomputing from TTM cash / latest price so the
            # numerator and denominator are consistent (and both in the trading
            # currency). yfinance's `info.dividendYield` is now reported in
            # percent units, but past versions returned a ratio and some ADR
            # payloads still drift — keep it as a last-resort passthrough only.
            latest_price = (
                _safe_float(info.get("currentPrice"))
                or _safe_float(info.get("regularMarketPrice"))
                or _safe_float(info.get("previousClose"))
            )
            yield_pct: Optional[float] = None
            if ttm_cash is not None and latest_price not in (None, 0):
                yield_pct = round(float(ttm_cash) / float(latest_price) * 100.0, 4)
            elif _safe_float(info.get("trailingAnnualDividendYield")) is not None:
                yield_pct = _ratio_to_pct(info.get("trailingAnnualDividendYield"))
            else:
                raw_yield = _safe_float(info.get("dividendYield"))
                if raw_yield is not None:
                    # Pass through as-is; current yfinance already returns percent.
                    yield_pct = round(raw_yield, 4)
            if yield_pct is not None:
                dividend_payload["ttm_dividend_yield_pct"] = yield_pct
            result.setdefault("earnings", {})["dividend"] = dividend_payload
            result["source_chain"].append("earnings.dividend:yfinance")

        # ---------------- belong_boards (sector + industry) ----------------
        belong_boards: List[Dict[str, Any]] = []
        sector_name = str(info.get("sector") or info.get("sectorDisp") or "").strip()
        if sector_name:
            belong_boards.append({"name": sector_name, "type": "行业"})
        industry_name = str(info.get("industry") or info.get("industryDisp") or "").strip()
        if industry_name and industry_name != sector_name:
            belong_boards.append({"name": industry_name, "type": "概念"})
        if belong_boards:
            result["belong_boards"] = belong_boards
            result["source_chain"].append("belong_boards:yfinance.info")

        has_content = bool(
            result.get("growth")
            or result.get("earnings")
            or result.get("belong_boards")
        )
        result["status"] = "partial" if has_content else "not_supported"
        return result
