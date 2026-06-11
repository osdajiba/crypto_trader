"""实盘交易用例。

LiveTradingUseCase 与 paper use case 一样只做编排：获取最新行情，调用
DomainTradingPipeline，并把 PortfolioBook 状态同步回 mode 供报告兼容读取。
"""

from __future__ import annotations

from typing import Any, Sequence


class LiveTradingUseCase:
    """实盘的一轮交易编排。"""

    def __init__(self, market_data_feed, domain_pipeline, mode: Any) -> None:
        self.market_data_feed = market_data_feed
        self.domain_pipeline = domain_pipeline
        self.mode = mode

    async def run_once(self, symbols: Sequence[str], timeframe: str):
        """读取最新行情，并交给领域流水线处理。"""
        market = await self.market_data_feed.latest(symbols, timeframe)
        self.mode.state["timestamp"] = market.timestamp
        result = await self.domain_pipeline.run_once(market)
        self.mode._sync_state_from_portfolio()
        return result
