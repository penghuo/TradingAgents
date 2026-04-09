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


def fetch_trading_data(
    ticker: str, start_date: str, end_date: str, frequency: str = "daily"
) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance and return Open/Close columns.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    start_date, end_date : str
        Date range (inclusive) in YYYY-MM-DD format.
    frequency : str
        ``"daily"`` (default) or ``"weekly"`` (first trading day of each week).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with Open and Close columns.

    Raises
    ------
    ValueError
        If yfinance returns no data.
    """
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for {ticker} in [{start_date}, {end_date}]"
        )

    # Handle yfinance MultiIndex columns (ticker level)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "Close"]]

    # Filter to [start_date, end_date] inclusive
    df = df.loc[start_date:end_date]

    if frequency == "weekly":
        # Keep the first trading day of each ISO week
        week_numbers = df.index.isocalendar().week.values
        mask = np.concatenate(([True], week_numbers[1:] != week_numbers[:-1]))
        df = df[mask]

    return df


def compute_metrics(
    equity_curve: List[float], trades_pnl: List[float]
) -> Dict[str, float]:
    """Compute backtest performance metrics.

    Parameters
    ----------
    equity_curve : list of float
        Portfolio value at each time step (at least 2 entries).
    trades_pnl : list of float
        Profit/loss for each individual trade.

    Returns
    -------
    dict
        ``total_return_pct``, ``max_drawdown_pct``, ``sharpe_ratio``,
        ``win_rate_pct`` -- all rounded as specified.
    """
    eq = np.array(equity_curve, dtype=float)

    # Total return
    total_return_pct = round((eq[-1] / eq[0] - 1) * 100, 3)

    # Max drawdown from peak
    running_max = np.maximum.accumulate(eq)
    drawdowns = (running_max - eq) / running_max * 100
    max_drawdown_pct = round(float(np.max(drawdowns)), 3)

    # Sharpe ratio (annualised, rf=0)
    daily_returns = np.diff(eq) / eq[:-1]
    if len(daily_returns) > 1 and np.std(daily_returns, ddof=1) > 0:
        sharpe_ratio = round(
            float(np.mean(daily_returns) / np.std(daily_returns, ddof=1) * np.sqrt(252)),
            3,
        )
    else:
        sharpe_ratio = 0.0

    # Win rate
    if len(trades_pnl) > 0:
        wins = sum(1 for t in trades_pnl if t > 0)
        win_rate_pct = round(wins / len(trades_pnl) * 100, 1)
    else:
        win_rate_pct = 0.0

    return {
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio": sharpe_ratio,
        "win_rate_pct": win_rate_pct,
    }
