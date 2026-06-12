from typing import Any, Dict, List

import pandas as pd

from src.domain.models import MarketSlice


class BacktestUseCase:
    """编排回测主循环。

    UseCase 只负责按时间推进、调用交易处理入口、记录绩效；具体策略、风控、
    执行和报告仍由 mode 组装的组件完成。这样可以逐步把回测循环从 mode 中拆出来。
    """

    def __init__(self, mode: Any):
        self.mode = mode

    async def run(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        try:
            return await self._run_domain_pipeline(symbols, timeframe)

        except Exception as e:
            self.mode.logger.error(f"Backtest execution error: {e}", exc_info=True)
            raise

    async def _run_domain_pipeline(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """使用领域流水线执行回测主循环。"""
        if vars(self.mode).get("domain_pipeline") is None:
            raise ValueError("Domain pipeline requires domain_pipeline")

        market_slices = await self._load_market_slices(symbols, timeframe)

        for i, market_slice in enumerate(market_slices):
            if i % 100 == 0 or i == 0:
                self.mode.logger.info(f"Backtest progress: {i}/{len(market_slices)}")

            self.mode.state["timestamp"] = market_slice.timestamp
            result = await self.mode.domain_pipeline.run_once(market_slice)
            self.mode._sync_state_from_portfolio()

            if self.mode.performance_monitor:
                for fill in result.fills:
                    self.mode.performance_monitor.record_trade(
                        timestamp=fill.timestamp,
                        symbol=fill.symbol,
                        direction=fill.side,
                        entry_price=fill.price,
                        exit_price=fill.price,
                        quantity=fill.quantity,
                        commission=fill.commission,
                    )
                self.mode.performance_monitor.update_equity_curve(market_slice.timestamp, result.snapshot.equity)

            if not self.mode._should_continue():
                self.mode.logger.info("Stopping backtest due to risk breach or user request")
                break

        self.mode.performance_monitor.calculate_performance_metrics()
        return self.mode.performance_monitor.generate_detailed_report()

    async def _load_market_slices(self, symbols: List[str], timeframe: str) -> List[MarketSlice]:
        """从 MarketDataFeed 读取领域行情切片。"""
        market_data_feed = vars(self.mode).get("market_data_feed")
        if market_data_feed is None:
            raise ValueError("Domain pipeline requires market_data_feed")

        return list(await market_data_feed.load_range(
            symbols=symbols,
            timeframe=timeframe,
            start=getattr(self.mode, "start_date", None),
            end=getattr(self.mode, "end_date", None),
        ))

    def _data_map_from_market_slice(self, market_slice: MarketSlice) -> Dict[str, pd.DataFrame]:
        """Legacy helper kept only for compatibility tests and old callers."""
        data_map = {}

        for symbol, bar in market_slice.bars_by_symbol.items():
            timestamp = pd.Timestamp(bar.timestamp)
            # 旧策略仍依赖 datetime 列和毫秒级 timestamp 列；唯一 DatetimeIndex
            # 用于避免策略内部增量 buffer concat 后出现重复索引。
            data_map[symbol] = pd.DataFrame(
                [{
                    "symbol": bar.symbol,
                    "timeframe": bar.timeframe,
                    "datetime": timestamp,
                    "timestamp": int(timestamp.timestamp() * 1000),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }],
                index=pd.DatetimeIndex([timestamp]),
            )

        return data_map
