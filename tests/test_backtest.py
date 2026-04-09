"""Tests for tradingagents.backtest module."""

import math
from unittest.mock import patch

import pandas as pd
import pytest

from tradingagents.backtest import Portfolio, fetch_trading_data


class TestPortfolio:
    """Portfolio tracker tests."""

    def test_initial_state(self):
        p = Portfolio(cash=100_000)
        assert p.shares == 0
        assert p.total_value(price=50) == 100_000

    def test_buy_uses_fraction_of_cash(self):
        p = Portfolio(cash=100_000, shares=0)
        traded = p.execute("BUY", price=200)
        assert traded == 100  # floor(100_000 * 0.20 / 200)
        assert p.shares == 100
        assert p.cash == 80_000

    def test_overweight_uses_10pct_of_cash(self):
        p = Portfolio(cash=100_000, shares=0)
        traded = p.execute("OVERWEIGHT", price=100)
        assert traded == 100  # floor(100_000 * 0.10 / 100)
        assert p.shares == 100
        assert p.cash == 90_000

    def test_hold_does_nothing(self):
        p = Portfolio(cash=100_000, shares=50)
        traded = p.execute("HOLD", price=100)
        assert traded == 0
        assert p.shares == 50
        assert p.cash == 100_000

    def test_underweight_sells_half_position(self):
        p = Portfolio(cash=0, shares=100)
        traded = p.execute("UNDERWEIGHT", price=50)
        assert traded == -50
        assert p.shares == 50
        assert p.cash == 2_500

    def test_sell_clears_position(self):
        p = Portfolio(cash=0, shares=100)
        traded = p.execute("SELL", price=50)
        assert traded == -100
        assert p.shares == 0
        assert p.cash == 5_000

    def test_buy_with_no_cash_does_nothing(self):
        p = Portfolio(cash=0, shares=0)
        traded = p.execute("BUY", price=100)
        assert traded == 0
        assert p.shares == 0
        assert p.cash == 0

    def test_sell_with_no_position_does_nothing(self):
        p = Portfolio(cash=100_000, shares=0)
        traded = p.execute("SELL", price=100)
        assert traded == 0
        assert p.shares == 0
        assert p.cash == 100_000

    def test_total_value(self):
        p = Portfolio(cash=50_000, shares=100)
        assert p.total_value(price=200) == 70_000


class TestTradingData:
    """Tests for fetch_trading_data using mocked yfinance."""

    @patch("tradingagents.backtest.yf.download")
    def test_returns_only_dates_in_range(self, mock_download):
        # 4 dates, filter to middle 3
        dates = pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11"])
        df = pd.DataFrame(
            {"Open": [100, 101, 102, 103], "Close": [110, 111, 112, 113]},
            index=dates,
        )
        mock_download.return_value = df

        result = fetch_trading_data("AAPL", "2024-01-08", "2024-01-10")
        assert len(result) == 3
        assert result.index[0] == pd.Timestamp("2024-01-08")
        assert result.index[-1] == pd.Timestamp("2024-01-10")

    @patch("tradingagents.backtest.yf.download")
    def test_weekly_samples_first_day_of_week(self, mock_download):
        # 2 full weeks of business days (Mon-Fri)
        dates = pd.bdate_range("2024-01-08", periods=10)  # Mon Jan 8 - Fri Jan 19
        df = pd.DataFrame(
            {"Open": range(10), "Close": range(10, 20)},
            index=dates,
        )
        mock_download.return_value = df

        result = fetch_trading_data(
            "AAPL", "2024-01-08", "2024-01-19", frequency="weekly"
        )
        assert len(result) == 2
        # First trading day of each week
        assert result.index[0] == pd.Timestamp("2024-01-08")  # Mon week 1
        assert result.index[1] == pd.Timestamp("2024-01-15")  # Mon week 2
