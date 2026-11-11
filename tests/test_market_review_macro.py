# -*- coding: utf-8 -*-
"""Regression tests: US market review keeps CPI/PPI visible via FRED facts.

When no news-search provider is configured, the market review used to
receive zero news items and the model asserted "no major macro data",
missing the October CPI print. The FRED macro block is injected as
structured facts instead, and an empty news search must no longer be
reported as "no macro events".
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.market_analyzer import MarketAnalyzer, MarketOverview

_MACRO_SNAPSHOT = {
    "cpi": {"month": "2026-10", "index": 337.476, "mom_pct": 0.35, "yoy_pct": 3.52},
    "ppi": {"month": "2026-09", "index": 159.78, "mom_pct": 1.35, "yoy_pct": None},
}


def _zh_config():
    return patch(
        "src.market_analyzer.get_config",
        return_value=SimpleNamespace(report_language="zh"),
    )


class TestMarketReviewMacroFacts(unittest.TestCase):
    def test_us_prompt_includes_fred_cpi_when_news_empty(self) -> None:
        with _zh_config():
            analyzer = MarketAnalyzer(region="us")
            prompt = analyzer._build_review_prompt(
                MarketOverview(date="2026-11-11"), [], macro_snapshot=_MACRO_SNAPSHOT
            )

        self.assertIn("美国 CPI（2026-10）", prompt)
        self.assertIn("环比 +0.35%", prompt)
        self.assertIn("美国 PPI（最终需求，2026-09）", prompt)
        self.assertIn("不得据此断言无重大宏观事件", prompt)
        self.assertNotIn("暂无相关新闻", prompt)

    def test_us_prompt_is_honest_when_no_news_and_no_macro(self) -> None:
        with _zh_config():
            analyzer = MarketAnalyzer(region="us")
            prompt = analyzer._build_review_prompt(
                MarketOverview(date="2026-11-11"), [], macro_snapshot=None
            )

        self.assertIn("本次复盘不得断言宏观事件的有无", prompt)
        self.assertNotIn("暂无相关新闻", prompt)
        self.assertNotIn("美国 CPI", prompt)

    def test_non_us_region_ignores_macro_snapshot(self) -> None:
        with _zh_config():
            analyzer = MarketAnalyzer(region="cn")
            prompt = analyzer._build_review_prompt(
                MarketOverview(date="2026-11-11"), [], macro_snapshot=_MACRO_SNAPSHOT
            )

        self.assertNotIn("美国 CPI", prompt)
        self.assertNotIn("FRED", prompt)


if __name__ == "__main__":
    unittest.main()
