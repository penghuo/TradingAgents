"""Tests for tradingagents.backtest module."""

import math

import pytest

from tradingagents.backtest import Portfolio


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
