# -*- coding: utf-8 -*-
"""Offline tests for the FRED US macro snapshot (src/services/fred_macro.py).

The snapshot is built from FRED's public graph CSV (no key). Tests use a
fake opener returning fixture CSV text, so no live network is involved.
"""
from __future__ import annotations

import unittest
from datetime import date

from src.services.fred_macro import (
    fetch_us_macro_snapshot,
    format_macro_block,
)

_CPI_CSV = """observation_date,CPIAUCSL
2025-09-01,325.100
2025-10-01,326.000
2026-08-01,334.131
2026-09-01,336.298
2026-10-01,337.476
"""

_PPI_CSV = """observation_date,PPIFIS
2025-09-01,152.100
2025-10-01,153.000
2026-08-01,157.662
2026-09-01,159.780
2026-10-01,160.500
"""


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _opener_for(csv_by_series):
    def _opener(request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        for series, text in csv_by_series.items():
            if f"id={series}" in url:
                return _FakeResponse(text.encode())
        raise AssertionError(f"unexpected URL in test: {url}")

    return _opener


class TestFredMacroSnapshot(unittest.TestCase):
    def test_computes_mom_and_yoy(self) -> None:
        snapshot = fetch_us_macro_snapshot(
            _opener=_opener_for({"CPIAUCSL": _CPI_CSV, "PPIFIS": _PPI_CSV}),
            _today=date(2026, 11, 11),
        )
        self.assertIsNotNone(snapshot)
        cpi = snapshot["cpi"]
        self.assertEqual(cpi["month"], "2026-10")
        self.assertAlmostEqual(cpi["index"], 337.476, places=3)
        # (337.476 - 336.298) / 336.298 = +0.35%
        self.assertAlmostEqual(cpi["mom_pct"], 0.35, places=2)
        # (337.476 - 326.000) / 326.000 = +3.52%
        self.assertAlmostEqual(cpi["yoy_pct"], 3.52, places=2)
        ppi = snapshot["ppi"]
        self.assertEqual(ppi["month"], "2026-10")
        self.assertAlmostEqual(ppi["mom_pct"], 0.45, places=2)

    def test_missing_ppi_release_does_not_drop_cpi(self) -> None:
        # October PPI is not published until Nov 13: an empty PPI series
        # must not drop the CPI line.
        snapshot = fetch_us_macro_snapshot(
            _opener=_opener_for({"CPIAUCSL": _CPI_CSV, "PPIFIS": "observation_date,PPIFIS\n"}),
            _today=date(2026, 11, 11),
        )
        self.assertEqual(set(snapshot), {"cpi"})

    def test_both_series_failing_returns_none(self) -> None:
        def _failing_opener(request, timeout=None):
            raise OSError("network down")

        self.assertIsNone(fetch_us_macro_snapshot(_opener=_failing_opener))

    def test_single_observation_returns_none(self) -> None:
        snapshot = fetch_us_macro_snapshot(
            _opener=_opener_for(
                {
                    "CPIAUCSL": "observation_date,CPIAUCSL\n2026-10-01,337.476\n",
                    "PPIFIS": "observation_date,PPIFIS\n2026-10-01,160.500\n",
                }
            ),
            _today=date(2026, 11, 11),
        )
        self.assertIsNone(snapshot)


class TestFredMacroBlock(unittest.TestCase):
    def test_chinese_block_cites_cpi_mom(self) -> None:
        snapshot = {
            "cpi": {"month": "2026-10", "index": 337.476, "mom_pct": 0.35, "yoy_pct": 3.52},
            "ppi": {"month": "2026-09", "index": 159.78, "mom_pct": 1.35, "yoy_pct": None},
        }
        block = format_macro_block(snapshot, "zh")
        self.assertIn("美国 CPI（2026-10）", block)
        self.assertIn("环比 +0.35%", block)
        self.assertIn("同比 +3.52%", block)
        self.assertIn("美国 PPI（最终需求，2026-09）", block)
        self.assertIn("请在第三节消息催化中引用该数据", block)
        self.assertNotIn("None", block)

    def test_empty_snapshot_renders_empty(self) -> None:
        self.assertEqual(format_macro_block(None, "zh"), "")
        self.assertEqual(format_macro_block({}, "en"), "")


if __name__ == "__main__":
    unittest.main()
