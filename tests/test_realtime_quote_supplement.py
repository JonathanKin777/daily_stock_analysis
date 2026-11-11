# -*- coding: utf-8 -*-
"""Tests that _merge_quote_fields never accepts a zero turnover rate."""

import unittest
from types import SimpleNamespace

from data_provider.base import DataFetcherManager


def _quote(**overrides):
    fields = {
        "volume_ratio": None,
        "turnover_rate": None,
        "pe_ratio": None,
        "pb_ratio": None,
        "total_mv": None,
        "circ_mv": None,
        "amplitude": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class MergeQuoteFieldsTestCase(unittest.TestCase):
    def test_zero_turnover_from_secondary_is_rejected(self) -> None:
        # 2026-11-11 run: QQQM showed a false 0.0% turnover because a
        # secondary source returned 0 for a US ETF it could not compute.
        primary = _quote()
        secondary = _quote(turnover_rate=0, volume_ratio=0.7)
        filled = DataFetcherManager._merge_quote_fields(primary, secondary)
        self.assertNotIn("turnover_rate", filled)
        self.assertIsNone(primary.turnover_rate)
        self.assertIn("volume_ratio", filled)
        self.assertEqual(primary.volume_ratio, 0.7)

    def test_real_turnover_is_accepted(self) -> None:
        primary = _quote()
        secondary = _quote(turnover_rate=1.03, volume_ratio=0.7)
        filled = DataFetcherManager._merge_quote_fields(primary, secondary)
        self.assertIn("turnover_rate", filled)
        self.assertEqual(primary.turnover_rate, 1.03)


if __name__ == "__main__":
    unittest.main()
