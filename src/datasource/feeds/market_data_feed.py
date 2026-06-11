"""交易流水线使用的行情 Feed。

Feed 是数据源和用例之间的边界：数据源负责读文件或实时行情，feed 负责把
DataFrame 合并成领域层 MarketSlice。
"""

from __future__ import annotations

from typing import Iterable, Sequence

from src.application.adapters.dataframe_domain_adapter import market_slices_from_dataframes
from src.domain.models import MarketSlice


class HistoricalMarketDataFeed:
    def __init__(self, store):
        self.store = store

    async def load_range(self, symbols: Sequence[str], timeframe: str, start, end) -> Iterable[MarketSlice]:
        """按时间范围加载多个交易对，并输出有序行情切片。"""
        data_by_symbol = {}
        for symbol in symbols:
            data = await self.store.load(symbol=symbol, timeframe=timeframe, start=start, end=end)
            if not data.empty:
                data_by_symbol[symbol] = data
        return market_slices_from_dataframes(data_by_symbol, timeframe=timeframe)


class RealtimeMarketDataFeed:
    """实时/模拟盘行情 Feed，把 DataManager 输出转成最新 MarketSlice。"""

    def __init__(self, data_manager):
        self.data_manager = data_manager

    async def latest(self, symbols: Sequence[str], timeframe: str) -> MarketSlice:
        """读取最新行情，并返回最后一个 MarketSlice。"""
        data_by_symbol = await self.data_manager.fetch_all_data_for_symbols(list(symbols), timeframe)
        market_slices = market_slices_from_dataframes(data_by_symbol, timeframe=timeframe)
        if not market_slices:
            raise ValueError("No latest market data available")
        return market_slices[-1]
