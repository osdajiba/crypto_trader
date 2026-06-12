"""DataFrame 协议与领域模型之间的兼容适配器。

DataFrame 属于旧策略和基础设施边界，不应继续放在 domain 包里。
这里集中做转换，让领域核心保持纯领域对象协议。
"""

from __future__ import annotations

from typing import Dict, List, Mapping

import pandas as pd

from src.domain.models import MarketBar, MarketSlice, StrategySignal


def _to_timestamp(value) -> pd.Timestamp:
    """统一把 DataFrame 中的 datetime/timestamp 转为 UTC 感知时间。"""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp


def market_slices_from_dataframes(data_by_symbol: Mapping[str, pd.DataFrame], timeframe: str) -> List[MarketSlice]:
    """把多交易对历史 DataFrame 按时间合并成有序 MarketSlice 列表。"""
    bars_by_time: Dict[pd.Timestamp, Dict[str, MarketBar]] = {}

    for symbol, dataframe in data_by_symbol.items():
        if dataframe.empty:
            continue

        for _, row in dataframe.iterrows():
            # 兼容旧数据协议：优先使用 datetime，缺失时退回毫秒级 timestamp。
            timestamp = _to_timestamp(row["datetime"] if "datetime" in row else row["timestamp"])
            bar = MarketBar(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            bars_by_time.setdefault(timestamp, {})[symbol] = bar

    return [
        MarketSlice(timestamp=timestamp, bars_by_symbol=bars_by_time[timestamp])
        for timestamp in sorted(bars_by_time)
    ]


def strategy_signals_from_dataframe(signals: pd.DataFrame, strategy_id: str) -> List[StrategySignal]:
    """把旧策略输出的信号 DataFrame 转换成领域 StrategySignal。"""
    converted: List[StrategySignal] = []

    if signals.empty:
        return converted

    for _, row in signals.iterrows():
        timestamp = _to_timestamp(row["datetime"] if "datetime" in row else row["timestamp"])
        symbol = row["symbol"]
        side = str(row["action"]).lower()
        metadata = {
            key: row[key]
            for key in row.index
            if key not in {"datetime", "timestamp", "symbol", "action"}
        }
        # signal_id 需要稳定可复现，用于 pipeline 侧去重，避免同一信号重复下单。
        signal_id = f"{strategy_id}:{symbol}:{side}:{timestamp.isoformat()}"
        converted.append(
            StrategySignal(
                signal_id=signal_id,
                symbol=symbol,
                timestamp=timestamp,
                side=side,
                strength=float(row["strength"]) if "strength" in row else 1.0,
                reason=str(row["reason"]) if "reason" in row else "",
                metadata=metadata,
            )
        )

    return converted
