"""Regression tests for US dotted tickers (BRK.B).

Yahoo Finance requires the hyphen form (BRK-B); passing BRK.B returns 404
("possibly delisted"). Stooq needs brk-b.us. These tests pin the conversion
in the fetcher, the fundamental adapter and the shared US-code pattern.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_provider.us_index_mapping import is_us_stock_code
from data_provider.yfinance_fetcher import YfinanceFetcher, _correct_brk_b_pb_ratio
from data_provider.yfinance_fundamental_adapter import _convert_to_yf_symbol


def test_convert_stock_code_maps_brk_b_to_hyphen() -> None:
    fetcher = YfinanceFetcher()
    assert fetcher._convert_stock_code("BRK.B") == "BRK-B"
    assert fetcher._convert_stock_code("BF.B") == "BF-B"


def test_convert_stock_code_leaves_plain_and_suffixed_codes_alone() -> None:
    fetcher = YfinanceFetcher()
    assert fetcher._convert_stock_code("AAPL") == "AAPL"
    assert fetcher._convert_stock_code("QQQM") == "QQQM"
    assert fetcher._convert_stock_code("0700.HK") == "0700.HK"
    assert fetcher._convert_stock_code("7203.T") == "7203.T"
    assert fetcher._convert_stock_code("600519") == "600519.SS"


def test_is_us_stock_code_accepts_hyphen_form() -> None:
    assert is_us_stock_code("BRK-B") is True
    assert is_us_stock_code("BRK.B") is True
    assert is_us_stock_code("AAPL") is True
    assert is_us_stock_code("SPX") is False
    assert is_us_stock_code("600519") is False


def test_fundamental_adapter_symbol_maps_brk_b_to_hyphen() -> None:
    assert _convert_to_yf_symbol("BRK.B") == "BRK-B"
    assert _convert_to_yf_symbol("AAPL") == "AAPL"
    assert _convert_to_yf_symbol("0700.HK") == "0700.HK"
    assert _convert_to_yf_symbol("7203.T") == "7203.T"
    assert _convert_to_yf_symbol("2330.TW") == "2330.TW"


def test_correct_brk_b_pb_ratio_scales_a_share_book_value() -> None:
    # Live Yahoo values verified 2026-11-11: priceToBook=0.001087483,
    # bookValue=536413.0 (A-share scale) -> true PB ~1.63.
    assert abs(_correct_brk_b_pb_ratio("BRK-B", 0.001087483) - 1.6312245) < 1e-6
    assert abs(_correct_brk_b_pb_ratio("BRK.B", 0.001087483) - 1.6312245) < 1e-6


def test_correct_brk_b_pb_ratio_leaves_sane_values_alone() -> None:
    # If Yahoo ever switches to the B-share book value (~1.6), no correction.
    assert _correct_brk_b_pb_ratio("BRK-B", 1.63) == 1.63
    assert _correct_brk_b_pb_ratio("BRK-B", None) is None
    # Other tickers are never rescaled, even for tiny values.
    assert _correct_brk_b_pb_ratio("AAPL", 0.001) == 0.001
