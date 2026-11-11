# -*- coding: utf-8 -*-
"""US watchlist names: STOCK_NAME_MAP / STOCK_ENGLISH_NAME_MAP stay in sync.

2026-11-11 v3.2: the four holdings were missing from STOCK_NAME_MAP, so a
failed name lookup left the prompt with only "股票SPYM" and the model
invented "SPDR标普中盘ETF". The English map must carry the same keys: the
module invariant requires every foreign ticker in STOCK_NAME_MAP to have an
English alias entry and vice versa.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.stock_mapping import STOCK_ENGLISH_NAME_MAP, STOCK_NAME_MAP


def test_watchlist_chinese_names() -> None:
    assert STOCK_NAME_MAP["QQQM"] == "纳斯达克100指数ETF"
    assert STOCK_NAME_MAP["SMH"] == "半导体ETF"
    assert STOCK_NAME_MAP["BRK.B"] == "伯克希尔·哈撒韦B类股"
    assert STOCK_NAME_MAP["SPYM"] == "标普500指数ETF"


def test_watchlist_english_aliases() -> None:
    assert STOCK_ENGLISH_NAME_MAP["QQQM"] == ("Invesco NASDAQ 100 ETF", "Nasdaq 100")
    assert STOCK_ENGLISH_NAME_MAP["SMH"] == ("VanEck Semiconductor ETF", "VanEck Semiconductor")
    assert STOCK_ENGLISH_NAME_MAP["BRK.B"] == ("Berkshire Hathaway Inc.", "Berkshire Hathaway")
    assert STOCK_ENGLISH_NAME_MAP["SPYM"] == ("SPDR Portfolio S&P 500 ETF", "SPDR S&P 500")


def test_foreign_name_and_english_maps_stay_in_sync() -> None:
    foreign_keys = {c for c in STOCK_NAME_MAP if not (c.isdigit() and len(c) == 6)}
    assert set(STOCK_ENGLISH_NAME_MAP) == foreign_keys
