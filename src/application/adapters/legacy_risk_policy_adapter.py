"""旧 DataFrame 风控到 RiskPolicy 的适配器。

旧 risk manager 仍然接收并返回 DataFrame；这里把它限制在 application 层，
对领域 pipeline 暴露 `RiskDecision`。
"""

from __future__ import annotations

import pandas as pd

from src.application.adapters.risk_decision_adapter import RiskDecisionAdapter
from src.domain.models import MarketSlice, PortfolioSnapshot, RiskDecision, StrategySignal


class LegacyRiskPolicyAdapter:
    """把旧 risk manager 包装成领域层 `RiskPolicy`。"""

    def __init__(self, risk_manager) -> None:
        self.risk_manager = risk_manager
        self.decision_adapter = RiskDecisionAdapter()

    async def evaluate(self, signal: StrategySignal, portfolio: PortfolioSnapshot, market: MarketSlice) -> RiskDecision:
        """调用旧风控，并把结果转换成 RiskDecision。"""
        dataframe = self._dataframe_from_signal(signal, market)
        validated = await self.risk_manager.validate_signals(dataframe)
        decisions = self.decision_adapter.prepare(validated).decisions
        if decisions:
            return decisions[0]

        return RiskDecision(
            accepted=False,
            reason="legacy_risk_returned_no_decision",
            target_notional=0,
            target_quantity=0,
            adjusted_signal=signal,
        )

    def _dataframe_from_signal(self, signal: StrategySignal, market: MarketSlice) -> pd.DataFrame:
        """把单个 StrategySignal 转成旧 risk manager 需要的一行 DataFrame。"""
        bar = market.bars_by_symbol[signal.symbol]
        quantity = float(signal.metadata.get("quantity", 0) or 0)
        price = float(signal.metadata.get("price", bar.close) or bar.close)
        timestamp = pd.Timestamp(signal.timestamp)
        return pd.DataFrame([{
            "datetime": timestamp,
            "timestamp": int(timestamp.timestamp() * 1000),
            "symbol": signal.symbol,
            "action": signal.side,
            "side": signal.side,
            "quantity": quantity,
            "price": price,
        }])
