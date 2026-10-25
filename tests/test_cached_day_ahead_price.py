"""Tests for CachedDayAheadPrice around the autumn DST change (2026-10-25), where
02:00-02:59 local time occurs twice."""

import pandas as pd

from interfaces.get_day_ahead_prices import CachedDayAheadPrice, DayAheadPrice


def test_get_prices_during_repeated_dst_hour(monkeypatch):
    """Prices must start at the current 15-min interval in both halves of the
    repeated hour (CEST and CET) instead of raising an AmbiguousTimeError."""

    idx = pd.date_range("2026-10-24 22:00", "2026-10-26 23:45", freq="15min",
        tz="Europe/Vienna")
    prices = pd.Series(range(len(idx)), index=idx, dtype=float)
    monkeypatch.setattr(DayAheadPrice, "get_prices",
        lambda price_type, start_date, epex_sources: (prices[start_date:], prices[start_date:]))

    cache = CachedDayAheadPrice()
    for utc_time in ["2026-10-25 00:50", "2026-10-25 01:00", "2026-10-25 01:50"]:
        current_time = pd.Timestamp(utc_time, tz="UTC").tz_convert("Europe/Berlin")
        price_sell, price_buy = cache.get_prices(current_time, epex_sources=["energycharts"])

        expected_start = pd.Timestamp(utc_time, tz="UTC").floor("15min")
        assert price_sell.index[0] == expected_start
        assert price_buy.index[0] == expected_start
