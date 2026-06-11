"""旧 DataFrame 策略到 StrategyPort 的适配器。

旧策略仍然使用 pandas DataFrame 输入输出；这个适配器把 DataFrame 细节
限制在 application 层，让领域 pipeline 只看到 StrategySignal。
"""

from __future__ import annotations

import pandas as pd

from src.application.adapters.dataframe_domain_adapter import strategy_signals_from_dataframe
from src.domain.models import MarketSlice, PortfolioSnapshot, StrategySignal


class LegacyDataFrameStrategyAdapter:
    """把旧策略包装成领域层 `StrategyPort`。"""

    def __init__(self, strategy, strategy_id: str | None = None) -> None:
        self.strategy = strategy
        self.strategy_id = strategy_id or strategy.__class__.__name__
        self._processed_signal_ids: set[str] = set()

    async def generate(self, market: MarketSlice, portfolio: PortfolioSnapshot) -> list[StrategySignal]:
        """按 symbol 调用旧策略，并返回领域信号列表。"""
        signals: list[StrategySignal] = []

        for symbol, bar in market.bars_by_symbol.items():
            dataframe = self._dataframe_from_bar(bar)
            legacy_signals = await self.strategy.process_data(dataframe, symbol)
            if legacy_signals.empty:
                continue
            # 旧策略通常会返回整个 buffer 中的所有历史信号；domain pipeline
            # 每次只处理当前 MarketSlice 的新信号，避免重复下单。
            current_signals = self._current_unprocessed_signals(
                strategy_signals_from_dataframe(legacy_signals, strategy_id=self.strategy_id),
                market.timestamp,
            )
            signals.extend(current_signals)

        return signals

    def _dataframe_from_bar(self, bar) -> pd.DataFrame:
        """把单根 MarketBar 转成旧策略需要的一行 DataFrame。"""
        timestamp = pd.Timestamp(bar.timestamp)
        return pd.DataFrame(
            [{
                "datetime": timestamp,
                "timestamp": int(timestamp.timestamp() * 1000),
                "symbol": bar.symbol,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }],
            index=pd.DatetimeIndex([timestamp]),
        )

    def _current_unprocessed_signals(self, signals: list[StrategySignal], current_timestamp) -> list[StrategySignal]:
        """只保留当前时间点且未处理过的信号。"""
        current = pd.Timestamp(current_timestamp)
        if current.tzinfo is None:
            current = current.tz_localize("UTC")

        filtered: list[StrategySignal] = []
        for signal in signals:
            signal_timestamp = pd.Timestamp(signal.timestamp)
            if signal_timestamp.tzinfo is None:
                signal_timestamp = signal_timestamp.tz_localize("UTC")
            if signal_timestamp != current:
                continue
            if signal.signal_id in self._processed_signal_ids:
                continue
            self._processed_signal_ids.add(signal.signal_id)
            filtered.append(signal)

        return filtered
