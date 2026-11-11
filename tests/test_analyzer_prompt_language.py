# -*- coding: utf-8 -*-
"""Regression tests for full-language stock analysis prompt templates (#2352).

With ``REPORT_LANGUAGE=en`` the per-stock Decision Dashboard prompt used to be a
Chinese template with only a trailing English output-language block, so smaller
local models (e.g. ``ollama/qwen3:14b``) kept answering in Chinese. English mode
must now use complete English system + user templates while keeping JSON keys,
enum values and the Chinese template unchanged.
"""

import copy
import json
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    from tests.litellm_stub import ensure_litellm_stub

    ensure_litellm_stub()

from src.agent.skills.defaults import CORE_TRADING_SKILL_POLICY_EN, CORE_TRADING_SKILL_POLICY_ZH
from src.analyzer import (
    AnalysisResult,
    GeminiAnalyzer,
    _legacy_audit_marker_specs,
    normalize_dashboard_price_units,
    normalize_stock_identity_terms,
)
from src.market_context import get_currency_unit

# CJK ideographs plus full-width punctuation/forms. Hangul is intentionally not
# matched so the Korean output directive does not count as Chinese template text.
_CJK_RE = re.compile(r"[　-〿㐀-䶿一-鿿＀-￯]")
_JSON_KEY_RE = re.compile(r'^\s*"([a-z_0-9]+)":', re.MULTILINE)


def _cjk_chars(text: str) -> str:
    return "".join(_CJK_RE.findall(text))


def _rich_context() -> dict:
    """A context that exercises every optional prompt block with ASCII-only data."""
    return {
        "code": "AAPL",
        "stock_name": "Apple",
        "date": "2026-03-16",
        "today": {
            "close": 10.5, "open": 10.1, "high": 10.8, "low": 10.0, "pct_chg": 1.2,
            "volume": 123456789, "amount": 987654321, "ma5": 10.2, "ma10": 10.0, "ma20": 9.8,
        },
        "ma_status": "bullish",
        "realtime": {
            "price": 10.5, "volume_ratio": 1.3, "volume_ratio_desc": "normal", "turnover_rate": 2.1,
            "pe_ratio": 20, "pb_ratio": 3, "total_mv": 1.2e11, "circ_mv": 1.0e11, "change_60d": 5.5,
        },
        "fundamental_context": {
            "earnings": {
                "data": {
                    "financial_report": {"report_date": "2025-12-31", "revenue": 1000, "roe": 12},
                    "dividend": {"ttm_cash_dividend_per_share": 1.2, "ttm_dividend_yield_pct": 2.4},
                }
            },
            "capital_flow": {
                "data": {
                    "stock_flow": {"main_net_inflow": 1000, "inflow_5d": 2000, "inflow_10d": 3000},
                    "sector_rankings": {"top": [{"name": "Semis"}, {"name": "Banks"}], "bottom": [{"name": "Coal"}]},
                }
            },
            "institution": {
                "status": "ok",
                "data": {"foreign_net": 1, "trust_net": 2, "dealer_net": 3, "total_net": 6, "date": "2026-03-16"},
            },
        },
        "chip": {"profit_ratio": 0.8, "avg_cost": 9.5, "concentration_90": 0.12, "concentration_70": 0.08},
        "trend_analysis": {
            "trend_status": "bullish", "ma_alignment": "MA5>MA10>MA20", "trend_strength": 70,
            "bias_ma5": 6.2, "bias_ma10": 3.1, "volume_status": "normal", "buy_signal": "buy",
            "signal_score": 72, "signal_reasons": ["reason a"], "risk_factors": ["risk b"],
        },
        "yesterday": {"close": 10.3},
        "volume_change_ratio": 12.5,
        "price_change_ratio": 1.9,
        "news_window_days": 5,
        "data_missing": True,
        "is_index_etf": True,
    }


def _make_analyzer(*, legacy: bool, skill_instructions: str = "", default_skill_policy: str = ""):
    with patch.object(GeminiAnalyzer, "_init_litellm", return_value=None):
        return GeminiAnalyzer(
            skill_instructions=skill_instructions,
            default_skill_policy=default_skill_policy,
            use_legacy_default_prompt=legacy,
        )


class EnglishStockPromptTemplateTestCase(unittest.TestCase):
    def test_english_system_prompt_has_no_chinese_template_text(self) -> None:
        for legacy in (True, False):
            analyzer = _make_analyzer(legacy=legacy, skill_instructions="### Skill 1: swing\n- watch support")
            for code in ("600519", "AAPL", "hk00700"):
                with self.subTest(legacy=legacy, code=code):
                    prompt = analyzer._get_analysis_system_prompt("en", stock_code=code)
                    self.assertEqual(_cjk_chars(prompt), "")
                    self.assertIn("Decision Dashboard", prompt)
                    self.assertIn('"decision_type": "buy/hold/sell"', prompt)
                    self.assertIn("## Output Language (highest priority)", prompt)

    def test_english_system_prompt_keeps_json_keys_of_chinese_template(self) -> None:
        for legacy in (True, False):
            analyzer = _make_analyzer(legacy=legacy)
            with self.subTest(legacy=legacy):
                zh_keys = _JSON_KEY_RE.findall(analyzer._get_analysis_system_prompt("zh", stock_code="600519"))
                en_keys = _JSON_KEY_RE.findall(analyzer._get_analysis_system_prompt("en", stock_code="600519"))
                self.assertGreater(len(zh_keys), 50)
                self.assertEqual(en_keys, zh_keys)

    def test_english_skill_prompt_uses_english_default_skill_baseline(self) -> None:
        analyzer = _make_analyzer(legacy=False, default_skill_policy=CORE_TRADING_SKILL_POLICY_ZH)

        en_prompt = analyzer._get_analysis_system_prompt("en", stock_code="AAPL")
        zh_prompt = analyzer._get_analysis_system_prompt("zh", stock_code="AAPL")

        self.assertIn(CORE_TRADING_SKILL_POLICY_EN, en_prompt)
        self.assertEqual(_cjk_chars(en_prompt), "")
        self.assertIn(CORE_TRADING_SKILL_POLICY_ZH, zh_prompt)

    def test_english_skill_prompt_keeps_builtin_skill_text_as_runtime_data(self) -> None:
        """Language boundary: English mode translates the analyzer template, not skill content.

        Built-in (and custom) skill instructions are loaded from YAML and rendered by
        ``SkillManager``; they are injected verbatim in their authored language. Everything
        outside that injected block must be English.
        """
        from src.agent.skills.base import SkillManager

        manager = SkillManager()
        self.assertGreater(manager.load_builtin_skills(), 0)
        manager.activate(["chan_theory", "volume_breakout"])
        skill_instructions = manager.get_skill_instructions()
        # The built-in skill YAML is authored in Chinese; this is what makes the boundary visible.
        self.assertNotEqual(_cjk_chars(skill_instructions), "")

        analyzer = _make_analyzer(legacy=False, skill_instructions=skill_instructions)
        prompt = analyzer._get_analysis_system_prompt("en", stock_code="600519")

        self.assertIn(f"## Active Trading Skills\n\n{skill_instructions}\n", prompt)
        self.assertEqual(_cjk_chars(prompt.replace(skill_instructions, "")), "")
        self.assertIn("## Output Language (highest priority)", prompt)

    def test_english_user_prompt_has_no_chinese_template_text(self) -> None:
        fake_cfg = SimpleNamespace(news_max_age_days=3, news_strategy_profile="short")
        for legacy in (True, False):
            analyzer = _make_analyzer(legacy=legacy)
            for news in ("AAPL headline", None):
                with self.subTest(legacy=legacy, news=bool(news)), \
                        patch("src.analyzer.get_config", return_value=fake_cfg):
                    prompt = analyzer._format_prompt(
                        _rich_context(), "Apple", news_context=news, report_language="en"
                    )
                    self.assertEqual(_cjk_chars(prompt), "")
                    self.assertIn("# Decision Dashboard Analysis Request", prompt)
                    self.assertIn("## 📈 Technical Data", prompt)
                    self.assertIn("## 📰 News Intelligence", prompt)
                    self.assertIn("### Output language requirements (highest priority)", prompt)
                    self.assertIn("123.46M shares", prompt)
                    if news:
                        self.assertIn("Ignore all news older than the last 5 days", prompt)

    def test_english_user_prompt_replaces_chinese_stock_name_placeholder(self) -> None:
        analyzer = _make_analyzer(legacy=True)
        context = {"code": "ZZZZ", "date": "2026-03-16", "today": {}}
        fake_cfg = SimpleNamespace(news_max_age_days=3, news_strategy_profile="short")
        with patch("src.analyzer.get_config", return_value=fake_cfg):
            prompt = analyzer._format_prompt(context, "", news_context=None, report_language="en")

        self.assertIn("| Stock Name | **Stock ZZZZ** |", prompt)
        self.assertEqual(_cjk_chars(prompt), "")

    def test_korean_reuses_english_scaffolding_with_korean_directive(self) -> None:
        analyzer = _make_analyzer(legacy=True)
        fake_cfg = SimpleNamespace(news_max_age_days=3, news_strategy_profile="short")
        with patch("src.analyzer.get_config", return_value=fake_cfg):
            ko_user = analyzer._format_prompt(copy.deepcopy(_rich_context()), "Apple", report_language="ko")
            en_user = analyzer._format_prompt(copy.deepcopy(_rich_context()), "Apple", report_language="en")
        ko_system = analyzer._get_analysis_system_prompt("ko", stock_code="AAPL")

        self.assertEqual(_cjk_chars(ko_user), "")
        self.assertEqual(_cjk_chars(ko_system), "")
        self.assertIn("# Decision Dashboard Analysis Request", ko_user)
        self.assertIn("All human-readable JSON values must be in Korean (한국어).", ko_user)
        self.assertIn("All human-readable JSON values must be written in Korean (한국어).", ko_system)
        en_headers = re.findall(r"^#{1,4} .+$", en_user, re.MULTILINE)
        ko_headers = re.findall(r"^#{1,4} .+$", ko_user, re.MULTILINE)
        self.assertEqual(ko_headers, en_headers)

    def test_chinese_prompt_keeps_chinese_template(self) -> None:
        analyzer = _make_analyzer(legacy=True)
        fake_cfg = SimpleNamespace(news_max_age_days=3, news_strategy_profile="short")
        with patch("src.analyzer.get_config", return_value=fake_cfg):
            prompt = analyzer._format_prompt(_rich_context(), "Apple", news_context="n", report_language="zh")
        system_prompt = analyzer._get_analysis_system_prompt("zh", stock_code="600519")

        self.assertIn("# 决策仪表盘分析请求", prompt)
        self.assertIn("## 📈 技术面数据", prompt)
        self.assertIn("1.23 亿股", prompt)
        self.assertIn("专注于趋势交易", system_prompt)
        self.assertNotIn("Decision Dashboard", system_prompt)

    def test_realtime_missing_ratios_render_as_na_not_none(self) -> None:
        """Missing volume_ratio / turnover_rate must render N/A, not the
        literal None (2026-11-11 run printed '换手率 None%' for SPYM/SMH)."""
        analyzer = _make_analyzer(legacy=True)
        fake_cfg = SimpleNamespace(news_max_age_days=3, news_strategy_profile="short")
        context = copy.deepcopy(_rich_context())
        context["realtime"]["volume_ratio"] = None
        context["realtime"]["turnover_rate"] = None
        with patch("src.analyzer.get_config", return_value=fake_cfg):
            zh_prompt = analyzer._format_prompt(context, "Apple", news_context="n", report_language="zh")
            en_prompt = analyzer._format_prompt(copy.deepcopy(context), "Apple", news_context="n", report_language="en")
            normal_prompt = analyzer._format_prompt(_rich_context(), "Apple", news_context="n", report_language="zh")

        self.assertIn("| **量比** | **N/A** |", zh_prompt)
        self.assertIn("| **换手率** | **N/A** |", zh_prompt)
        self.assertIn("| **Volume Ratio** | **N/A** |", en_prompt)
        self.assertIn("| **Turnover Rate** | **N/A** |", en_prompt)
        self.assertNotIn("None%", zh_prompt)
        self.assertNotIn("None%", en_prompt)
        # Normal values still render as before.
        self.assertIn("| **量比** | **1.3** |", normal_prompt)
        self.assertIn("| **换手率** | **2.1%** |", normal_prompt)

    def test_english_audit_markers_match_english_prompt_headers(self) -> None:
        markers = {
            marker["marker_name"]: marker["text"]
            for marker in _legacy_audit_marker_specs(
                {"date": "2026-03-16"},
                code="AAPL",
                stock_name="Apple",
                report_language="en",
                news_context="news",
                analysis_context_pack_summary=None,
            )
        }
        self.assertEqual(markers["quote"], "## 📈 Technical Data")
        self.assertEqual(markers["news_context"], "## 📰 News Intelligence")


class EnglishAnalyzeMessagesTestCase(unittest.TestCase):
    def test_analyze_sends_english_messages_for_ollama_model(self) -> None:
        analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
        analyzer._router = None
        analyzer._litellm_available = True
        analyzer._skill_instructions_override = ""
        analyzer._default_skill_policy_override = ""
        analyzer._use_legacy_default_prompt_override = True
        analyzer._config_override = SimpleNamespace(
            generation_backend="litellm",
            generation_fallback_backend="litellm",
            litellm_model="ollama/qwen3:14b",
            litellm_fallback_models=[],
            llm_model_list=[],
            report_language="en",
            gemini_request_delay=0,
            llm_temperature=0.7,
            report_integrity_enabled=False,
            report_integrity_retry=0,
            news_max_age_days=3,
            news_strategy_profile="short",
        )
        captured = {}

        def _capture(prompt, generation_config, **kwargs):
            captured["user"] = prompt
            captured["system"] = kwargs.get("system_prompt")
            raise RuntimeError("stop after capturing messages")

        with patch.object(analyzer, "_call_litellm", side_effect=_capture):
            analyzer.analyze(_rich_context(), news_context="AAPL headline")

        self.assertEqual(_cjk_chars(captured["system"]), "")
        self.assertEqual(_cjk_chars(captured["user"]), "")
        self.assertIn("US stock", captured["system"])
        self.assertIn("## Output Language (highest priority)", captured["system"])
        self.assertIn("### Output language requirements (highest priority)", captured["user"])


class TestSniperPointCurrencyUnit(unittest.TestCase):
    """Sniper-point price examples must use the market's own currency unit.

    The Chinese templates used to hard-code the A-share example ``XX元``,
    so US-stock reports wrote ``474.00元`` for SMH. The unit is now a
    per-market placeholder (美元 for US, 港元 for HK, 元 for A-shares).
    """

    def test_chinese_prompt_uses_market_currency_for_sniper_examples(self) -> None:
        cases = (
            ("BRK.B", "XX美元", "XX元"),
            ("SMH", "XX美元", "XX元"),
            ("00700.HK", "XX港元", "XX元"),
            ("600519", "XX元", None),
        )
        for legacy in (True, False):
            analyzer = _make_analyzer(legacy=legacy)
            for code, expected, forbidden in cases:
                with self.subTest(legacy=legacy, code=code):
                    prompt = analyzer._get_analysis_system_prompt("zh", stock_code=code)
                    self.assertIn(expected, prompt)
                    if forbidden is not None:
                        self.assertNotIn(forbidden, prompt)

    def test_english_prompt_uses_market_currency_for_sniper_examples(self) -> None:
        for legacy in (True, False):
            analyzer = _make_analyzer(legacy=legacy)
            with self.subTest(legacy=legacy):
                prompt = analyzer._get_analysis_system_prompt("en", stock_code="BRK.B")
                self.assertIn("XX USD", prompt)
            with self.subTest(legacy=legacy):
                prompt = analyzer._get_analysis_system_prompt("en", stock_code="600519")
                self.assertIn("XX CNY", prompt)

    def test_get_currency_unit(self) -> None:
        self.assertEqual(get_currency_unit("BRK.B", "zh"), "美元")
        self.assertEqual(get_currency_unit("BRK.B", "en"), "USD")
        self.assertEqual(get_currency_unit("00700.HK", "zh"), "港元")
        self.assertEqual(get_currency_unit("600519", "zh"), "元")
        self.assertEqual(get_currency_unit("", "zh"), "元")

    def test_normalize_dashboard_price_units_rewrites_bare_yuan(self) -> None:
        dashboard = {
            "battle_plan": {
                "sniper_points": {
                    "ideal_buy": "理想买入点：474.03元（在MA5附近）",
                    "secondary_buy": "次优买入点：470.00元（回踩确认）",
                    "stop_loss": "止损位：474.00元（跌破今日低点坚决止损离场）",
                    "take_profit": "目标位：506.57美元（反弹至MA10）",
                }
            }
        }
        normalize_dashboard_price_units(dashboard, "SMH")
        sniper = dashboard["battle_plan"]["sniper_points"]
        self.assertIn("474.03美元", sniper["ideal_buy"])
        self.assertIn("470.00美元", sniper["secondary_buy"])
        self.assertIn("474.00美元", sniper["stop_loss"])
        # Already-correct 美元 is left alone.
        self.assertEqual(sniper["take_profit"], "目标位：506.57美元（反弹至MA10）")

    def test_normalize_dashboard_price_units_keeps_a_share_yuan(self) -> None:
        dashboard = {
            "battle_plan": {
                "sniper_points": {
                    "stop_loss": "止损位：10.50元（跌破MA20）",
                    "take_profit": "目标位：12.00元（前高）",
                }
            }
        }
        normalize_dashboard_price_units(dashboard, "600519")
        sniper = dashboard["battle_plan"]["sniper_points"]
        self.assertIn("10.50元", sniper["stop_loss"])
        self.assertIn("12.00元", sniper["take_profit"])

    def test_user_prompt_price_rows_use_market_currency(self) -> None:
        # The user prompt hands the model the same price rows (close, realtime
        # price, chip average cost); a hard-coded 元 there feeds the same
        # unit error, so those rows get the market unit too.
        analyzer = _make_analyzer(legacy=False)
        fake_cfg = SimpleNamespace(news_max_age_days=3, news_strategy_profile="short")
        with patch("src.analyzer.get_config", return_value=fake_cfg):
            us_prompt = analyzer._format_prompt(
                _rich_context(), "Apple", news_context="n", report_language="zh"
            )
            cn_context = _rich_context()
            cn_context["code"] = "600519"
            cn_prompt = analyzer._format_prompt(
                cn_context, "贵州茅台", news_context="n", report_language="zh"
            )

        self.assertIn("10.5 美元", us_prompt)
        self.assertIn("9.5 美元", us_prompt)
        self.assertNotIn("10.5 元", us_prompt)
        self.assertNotIn("9.5 元", us_prompt)
        self.assertIn("10.5 元", cn_prompt)
        self.assertIn("9.5 元", cn_prompt)


class TestV32AmountUnitAndIdentityFixes(unittest.TestCase):
    """v3.2: amounts use the market currency; SPYM must never be mid-cap.

    2026-11-11 trial run: the prompt wrote BRK.B's market cap as
    "12393.17 亿元" (should be 亿美元), and the model, given only
    "股票SPYM" as the name, invented "SPDR标普中盘ETF" and reused
    "中盘宽基" / "中盘类资产" in the narrative.
    """

    def test_format_amount_uses_market_currency_unit(self) -> None:
        analyzer = _make_analyzer(legacy=False)
        # BRK.B market cap from the 2026-11-11 trial run (live Yahoo value).
        self.assertEqual(
            analyzer._format_amount(1239317086208, currency_unit="美元"),
            "12393.17 亿美元",
        )
        self.assertEqual(
            analyzer._format_amount(1239317086208, currency_unit="港元"),
            "12393.17 亿港元",
        )
        self.assertEqual(analyzer._format_amount(987654321, currency_unit="美元"), "9.88 亿美元")
        self.assertEqual(analyzer._format_amount(5000, currency_unit="美元"), "5000 美元")
        # Legacy callers that pass no unit keep the old 元 behaviour.
        self.assertEqual(analyzer._format_amount(1239317086208), "12393.17 亿元")
        self.assertEqual(analyzer._format_amount(None, currency_unit="美元"), "N/A")
        self.assertEqual(analyzer._format_amount(1239317086208, "en"), "1239.32B")

    def test_user_prompt_market_cap_uses_market_currency(self) -> None:
        analyzer = _make_analyzer(legacy=False)
        fake_cfg = SimpleNamespace(news_max_age_days=3, news_strategy_profile="short")
        context = _rich_context()
        context["code"] = "SPYM"
        context["stock_name"] = "标普500指数ETF"
        context["realtime"]["total_mv"] = 1239317086208
        context["realtime"]["circ_mv"] = 1239317086208
        with patch("src.analyzer.get_config", return_value=fake_cfg):
            prompt = analyzer._format_prompt(
                context, "标普500指数ETF", news_context="n", report_language="zh"
            )
        self.assertIn("12393.17 亿美元", prompt)
        self.assertNotIn("12393.17 亿元", prompt)
        # The turnover row follows the same rule (987654321 -> 9.88 亿美元).
        self.assertIn("9.88 亿美元", prompt)
        self.assertNotIn("9.88 亿元", prompt)

    def test_user_prompt_uses_mapped_chinese_name_for_spym(self) -> None:
        analyzer = _make_analyzer(legacy=False)
        fake_cfg = SimpleNamespace(news_max_age_days=3, news_strategy_profile="short")
        context = _rich_context()
        context["code"] = "SPYM"
        context.pop("stock_name", None)
        with patch("src.analyzer.get_config", return_value=fake_cfg):
            prompt = analyzer._format_prompt(
                context, "股票SPYM", news_context="n", report_language="zh"
            )
        self.assertIn("标普500指数ETF", prompt)
        # The stock-name row now carries the mapped name. The prompt's
        # "output the correct name" instruction still quotes the fallback
        # form ("股票SPYM") conditionally, so it may appear elsewhere in
        # the template; only the displayed name row must be correct.
        self.assertIn("| 股票名称 | **标普500指数ETF** |", prompt)

    def test_market_snapshot_amount_uses_market_currency(self) -> None:
        analyzer = _make_analyzer(legacy=False)
        snapshot = analyzer._build_market_snapshot(_rich_context(), currency_unit="美元")
        self.assertEqual(snapshot["amount"], "9.88 亿美元")
        snapshot_cn = analyzer._build_market_snapshot(_rich_context())
        self.assertEqual(snapshot_cn["amount"], "9.88 亿元")

    def test_normalize_stock_identity_terms_fixes_spym_mislabels(self) -> None:
        result = AnalysisResult(
            code="SPYM",
            name="SPDR标普中盘ETF（SPYM）",
            sentiment_score=44,
            trend_prediction="震荡",
            operation_advice="观望",
            dashboard={
                "intelligence": {
                    "sentiment_summary": "大盘风险偏好较低，市场对中盘宽基偏向防守中性观望",
                    "risk_alerts": ["中盘类资产波动承压", "中盘股跑输大盘指数"],
                    "hot_topics": "美股大盘防守轮动、标普中盘配置",
                }
            },
            sector_position="所属美股大盘及中盘板块在宏观谨慎预期下整体承压",
            raw_response='{"stock_name": "SPDR标普中盘ETF（SPYM）"}',
        )
        normalize_stock_identity_terms(result)
        self.assertEqual(result.name, "SPDR标普500指数ETF（SPYM）")
        intel = result.dashboard["intelligence"]
        self.assertEqual(
            intel["sentiment_summary"], "大盘风险偏好较低，市场对大盘宽基偏向防守中性观望"
        )
        self.assertEqual(intel["risk_alerts"][0], "大盘类资产波动承压")
        # Legitimate market-segment wording is untouched.
        self.assertEqual(intel["risk_alerts"][1], "中盘股跑输大盘指数")
        self.assertEqual(intel["hot_topics"], "美股大盘防守轮动、标普500配置")
        self.assertEqual(result.sector_position, "所属美股大盘及中盘板块在宏观谨慎预期下整体承压")
        # Raw model output is kept verbatim for audit.
        self.assertIn("SPDR标普中盘ETF", result.raw_response)

    def test_normalize_stock_identity_terms_skips_other_tickers(self) -> None:
        result = AnalysisResult(
            code="QQQM",
            name="纳斯达克100指数ETF",
            sentiment_score=50,
            trend_prediction="震荡",
            operation_advice="观望",
            sector_position="中盘宽基表现分化",
        )
        normalize_stock_identity_terms(result)
        self.assertEqual(result.sector_position, "中盘宽基表现分化")

    def test_analyze_corrects_spym_mislabels_end_to_end(self) -> None:
        # The guard hook lives in analyze(), after _parse_response: a canned
        # v3-style SPYM reply (model invents the 中盘ETF name because the
        # input name was the bare fallback "股票SPYM") must come out
        # corrected in the final result.
        analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
        analyzer._router = None
        analyzer._litellm_available = True
        analyzer._skill_instructions_override = ""
        analyzer._default_skill_policy_override = ""
        analyzer._use_legacy_default_prompt_override = False
        analyzer._config_override = SimpleNamespace(
            generation_backend="litellm",
            generation_fallback_backend="litellm",
            litellm_model="gemini/gemini-3.6-flash",
            litellm_fallback_models=[],
            llm_model_list=[],
            report_language="zh",
            gemini_request_delay=0,
            llm_temperature=0.7,
            report_integrity_enabled=False,
            report_integrity_retry=0,
            news_max_age_days=3,
            news_strategy_profile="short",
        )
        response_text = json.dumps(
            {
                "stock_name": "SPDR标普中盘ETF（SPYM）",
                "sentiment_score": 44,
                "trend_prediction": "震荡",
                "operation_advice": "观望",
                "decision_type": "hold",
                "confidence_level": "中",
                "dashboard": {
                    "intelligence": {
                        "sentiment_summary": "市场对中盘宽基偏向防守中性观望",
                        "risk_alerts": ["中盘类资产波动承压"],
                    }
                },
                "sector_position": "所属美股大盘及中盘板块承压",
            },
            ensure_ascii=False,
        )

        def _fake_call(prompt, generation_config, **kwargs):
            return response_text, "gemini/gemini-3.6-flash", {}

        context = _rich_context()
        context["code"] = "SPYM"
        context["stock_name"] = "股票SPYM"
        with patch.object(analyzer, "_call_litellm", side_effect=_fake_call):
            result = analyzer.analyze(context, news_context="n")
        self.assertTrue(result.success)
        # With the mapped name in place the model's invented name is not
        # adopted at all (the parse step only takes the AI name when the
        # input name is the bare fallback); the guard's job is the
        # narrative fields below.
        self.assertEqual(result.name, "标普500指数ETF")
        intel = result.dashboard["intelligence"]
        self.assertIn("大盘宽基", intel["sentiment_summary"])
        self.assertNotIn("中盘宽基", intel["sentiment_summary"])
        self.assertIn("大盘类资产", intel["risk_alerts"][0])


if __name__ == "__main__":
    unittest.main()
