"""Example: backtest AMZN over a short period using Bedrock."""

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.backtest import BacktestDriver

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "bedrock"
config["aws_role_arn"] = "arn:aws:iam::070785328308:role/Bedrock-Access"  # set your role ARN
config["quick_think_llm"] = "global.anthropic.claude-sonnet-4-6"   # analysts, researchers, trader
config["deep_think_llm"] = "global.anthropic.claude-opus-4-6-v1"  # managers, portfolio decisions
config["aws_region"] = "us-west-2"
config["backend_url"] = None
config["max_debate_rounds"] = 1
config["max_risk_discuss_rounds"] = 1

driver = BacktestDriver(
    ticker="AMZN",
    start_date="2025-03-01",
    end_date="2025-03-31",  # ~2 weeks, ~10 trading days
    initial_cash=100_000,
    frequency="daily",
    reflect=False,
    config=config,
)

results = driver.run()
