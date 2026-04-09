"""Example: backtest AMZN over a short period using Bedrock."""

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.backtest import BacktestDriver

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "bedrock"
config["aws_role_arn"] = "arn:aws:iam::YOUR_ACCOUNT:role/YOUR_ROLE"  # set your role ARN
config["quick_think_llm"] = "global.anthropic.claude-opus-4-6-v1"
config["deep_think_llm"] = "global.anthropic.claude-opus-4-6-v1"
config["backend_url"] = None
config["max_debate_rounds"] = 1
config["max_risk_discuss_rounds"] = 1

driver = BacktestDriver(
    ticker="AMZN",
    start_date="2025-01-06",
    end_date="2025-01-17",  # ~2 weeks, ~10 trading days
    initial_cash=100_000,
    frequency="daily",
    reflect=False,
    config=config,
)

results = driver.run()
