"""Backtest driver for TradingAgents."""

import math
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf


class Portfolio:
    """Tracks cash, shares, and executes trades based on 5-level signals."""

    RULES: Dict[str, Dict[str, float]] = {
        "BUY": {"side": "buy", "fraction": 0.20},
        "OVERWEIGHT": {"side": "buy", "fraction": 0.10},
        "HOLD": {"side": "hold", "fraction": 0.0},
        "UNDERWEIGHT": {"side": "sell", "fraction": 0.50},
        "SELL": {"side": "sell", "fraction": 1.0},
    }

    def __init__(self, cash: float = 100_000, shares: int = 0):
        self.cash = cash
        self.shares = shares

    def execute(self, signal: str, price: float) -> int:
        """Execute a trade signal at the given price.

        Returns the number of shares traded: positive for buys,
        negative for sells, 0 for hold or no-op.
        """
        rule = self.RULES[signal]
        side = rule["side"]
        fraction = rule["fraction"]

        if side == "hold":
            return 0

        if side == "buy":
            amount = self.cash * fraction
            qty = math.floor(amount / price)
            if qty <= 0:
                return 0
            cost = qty * price
            self.cash -= cost
            self.shares += qty
            return qty

        # side == "sell"
        qty = math.floor(self.shares * fraction)
        if qty <= 0:
            return 0
        proceeds = qty * price
        self.cash += proceeds
        self.shares -= qty
        return -qty

    def total_value(self, price: float) -> float:
        """Return total portfolio value at current price."""
        return self.cash + self.shares * price
