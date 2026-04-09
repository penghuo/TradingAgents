"""Backtest driver for TradingAgents."""

import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


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


class BacktestDriver:
    """Run a multi-day backtest of TradingAgentsGraph on a single ticker."""

    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        initial_cash: float = 100_000,
        frequency: str = "daily",
        reflect: bool = False,
        config: dict = None,
        selected_analysts: list = None,
    ):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.frequency = frequency
        self.reflect = reflect
        self.config = config or DEFAULT_CONFIG
        self.selected_analysts = selected_analysts or [
            "market",
            "social",
            "news",
            "fundamentals",
        ]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Execute the full backtest and return a metrics dictionary."""
        print(
            f"\n{'='*60}\n"
            f"  Backtest: {self.ticker}  {self.start_date} -> {self.end_date}\n"
            f"  Initial cash: ${self.initial_cash:,.2f}  "
            f"Frequency: {self.frequency}\n"
            f"{'='*60}\n"
        )

        # 1. Fetch trading data
        ohlcv = fetch_trading_data(
            self.ticker, self.start_date, self.end_date, self.frequency
        )
        dates = ohlcv.index.tolist()
        print(f"  Trading days loaded: {len(dates)}\n")

        # 2. Build the agent graph
        graph = TradingAgentsGraph(
            selected_analysts=self.selected_analysts,
            debug=False,
            config=self.config,
        )

        # 3. Portfolio
        portfolio = Portfolio(cash=self.initial_cash)

        # 4. Tracking containers
        equity_curve: List[float] = []
        trades_pnl: List[float] = []
        trade_log: List[dict] = []
        prev_value = self.initial_cash

        # 5. Main loop: signal on date[i], execute on date[i+1]
        for i in range(len(dates) - 1):
            signal_date = dates[i]
            exec_date = dates[i + 1]

            signal_date_str = (
                signal_date.strftime("%Y-%m-%d")
                if hasattr(signal_date, "strftime")
                else str(signal_date)[:10]
            )
            exec_date_str = (
                exec_date.strftime("%Y-%m-%d")
                if hasattr(exec_date, "strftime")
                else str(exec_date)[:10]
            )

            exec_price = float(ohlcv.loc[exec_date, "Open"])

            # Get signal from the agent graph
            print(
                f"  [{i + 1}/{len(dates) - 1}] Analyzing {signal_date_str} ...",
                end="",
                flush=True,
            )
            try:
                _state, signal = graph.propagate(self.ticker, signal_date_str)
                signal = signal.strip().upper()
                if signal not in Portfolio.RULES:
                    signal = "HOLD"
            except Exception as exc:
                print(f" ERROR: {exc}")
                signal = "HOLD"

            # Execute at next-day open
            shares_traded = portfolio.execute(signal, exec_price)

            # Track equity
            current_value = portfolio.total_value(exec_price)
            equity_curve.append(current_value)

            # Track trade P&L
            if shares_traded != 0:
                trades_pnl.append(current_value - prev_value)

            prev_value = current_value

            # Trade log entry
            position_value = portfolio.shares * exec_price
            trade_log.append(
                {
                    "signal_date": signal_date_str,
                    "exec_date": exec_date_str,
                    "signal": signal,
                    "exec_price": exec_price,
                    "shares_traded": shares_traded,
                    "shares_held": portfolio.shares,
                    "cash": portfolio.cash,
                    "position_value": position_value,
                    "total_value": current_value,
                }
            )

            print(
                f" {signal:12s} | exec {exec_date_str} @ ${exec_price:.2f}"
                f" | traded={shares_traded:+5d}"
                f" | value=${current_value:,.2f}"
            )

            # Reflect if requested
            if self.reflect and shares_traded != 0:
                print("    Reflecting on trade...", end="", flush=True)
                graph.reflect_and_remember(current_value - self.initial_cash)
                print(" done")

        # 6. Buy-and-hold benchmark
        first_open = float(ohlcv.iloc[0]["Open"])
        last_open = float(ohlcv.iloc[-1]["Open"])
        buy_hold_return_pct = round((last_open / first_open - 1) * 100, 3)

        # 7. Compute metrics
        metrics = compute_metrics(equity_curve, trades_pnl)
        metrics["buy_hold_return_pct"] = buy_hold_return_pct
        metrics["num_trades"] = len(trades_pnl)
        metrics["total_days"] = len(dates) - 1
        metrics["ticker"] = self.ticker
        metrics["start_date"] = self.start_date
        metrics["end_date"] = self.end_date

        # 8. Save outputs
        out_dir = Path(self.config["results_dir"]) / self.ticker / "backtest"
        out_dir.mkdir(parents=True, exist_ok=True)

        self._save_trade_log(trade_log, out_dir)
        self._save_equity_chart(equity_curve, ohlcv, out_dir)
        self._print_summary(metrics)

        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_trade_log(trade_log: List[dict], out_dir: Path) -> None:
        """Write the trade log to a CSV file."""
        path = out_dir / "trade_log.csv"
        fieldnames = [
            "signal_date",
            "exec_date",
            "signal",
            "exec_price",
            "shares_traded",
            "shares_held",
            "cash",
            "position_value",
            "total_value",
        ]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trade_log)
        print(f"\n  Trade log saved to {path}")

    def _save_equity_chart(
        self,
        equity_curve: List[float],
        ohlcv: pd.DataFrame,
        out_dir: Path,
    ) -> None:
        """Save an equity-curve PNG comparing strategy vs buy-and-hold."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        first_open = float(ohlcv.iloc[0]["Open"])
        bh_shares = math.floor(self.initial_cash / first_open)
        bh_leftover = self.initial_cash - bh_shares * first_open

        # Build buy-and-hold equity from the execution dates (index 1 onward)
        opens = ohlcv["Open"].values
        bh_equity = [bh_shares * float(opens[i]) + bh_leftover for i in range(1, len(opens))]

        x = list(range(len(equity_curve)))
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x, equity_curve, label="Strategy", linewidth=1.4)
        ax.plot(x[: len(bh_equity)], bh_equity[: len(x)], label="Buy & Hold", linewidth=1.4, linestyle="--")
        ax.set_title(f"{self.ticker} Backtest: {self.start_date} to {self.end_date}")
        ax.set_xlabel("Trading Day")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = out_dir / "equity_curve.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Equity chart saved to {path}")

    @staticmethod
    def _print_summary(metrics: dict) -> None:
        """Print a formatted summary box to the console."""
        w = 50
        print(f"\n  {'='*w}")
        print(f"  {'BACKTEST SUMMARY':^{w}}")
        print(f"  {'='*w}")
        print(f"  Ticker:             {metrics['ticker']}")
        print(f"  Period:             {metrics['start_date']} -> {metrics['end_date']}")
        print(f"  Trading days:       {metrics['total_days']}")
        print(f"  Trades executed:    {metrics['num_trades']}")
        print(f"  {'-'*w}")
        print(f"  Strategy return:    {metrics['total_return_pct']:+.3f} %")
        print(f"  Buy & Hold return:  {metrics['buy_hold_return_pct']:+.3f} %")
        print(f"  Sharpe ratio:       {metrics['sharpe_ratio']:.3f}")
        print(f"  Max drawdown:       {metrics['max_drawdown_pct']:.3f} %")
        print(f"  Win rate:           {metrics['win_rate_pct']:.1f} %")
        print(f"  {'='*w}\n")
