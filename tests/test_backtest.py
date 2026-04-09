"""Tests for tradingagents.backtest module."""

import math
from unittest.mock import patch

import pandas as pd
import pytest

from tradingagents.backtest import Portfolio, compute_metrics, fetch_trading_data


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


class TestMetrics:
    """Tests for compute_metrics."""

    def test_total_return(self):
        equity = [100_000, 105_000, 110_000]
        result = compute_metrics(equity, [])
        assert result["total_return_pct"] == 10.0

    def test_max_drawdown(self):
        equity = [100_000, 110_000, 95_000, 105_000]
        result = compute_metrics(equity, [])
        # Drawdown from peak 110k to trough 95k = 15k / 110k = 13.636%
        assert result["max_drawdown_pct"] == 13.636

    def test_win_rate(self):
        trades = [100, -50, 200, -10]
        result = compute_metrics([100_000, 100_000], trades)
        assert result["win_rate_pct"] == 50.0

    def test_win_rate_no_trades(self):
        result = compute_metrics([100_000, 100_000], [])
        assert result["win_rate_pct"] == 0.0

    def test_sharpe_ratio(self):
        # Constant daily gains: equity goes up by 100 each day for 10 days
        equity = [100_000 + i * 100 for i in range(11)]
        result = compute_metrics(equity, [])
        assert result["sharpe_ratio"] > 0
