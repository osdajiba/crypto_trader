from dataclasses import dataclass
from typing import List

import pandas as pd

from src.domain.models import StrategySignal


@dataclass(frozen=True)
class AdaptedStrategySignals:
    """同时保留兼容 DataFrame 和领域信号，方便渐进迁移。"""
    dataframe: pd.DataFrame
    domain_signals: List[StrategySignal]


class StrategySignalAdapter:
    """把当前策略 DataFrame 信号转换成稳定的领域信号。"""

    def prepare(self, signals: pd.DataFrame, current_timestamp=None) -> AdaptedStrategySignals:
        """补齐 signal_id，并生成对应的 StrategySignal 列表。"""
        if signals.empty:
            return AdaptedStrategySignals(signals.copy(), [])

        prepared = signals.copy()
        # 旧策略没有 signal_id，因此在 adapter 层生成，避免修改策略实现。
        if "signal_id" not in prepared.columns:
            prepared["signal_id"] = prepared.apply(
                lambda row: self._build_signal_id(row, current_timestamp),
                axis=1,
            )

        domain_signals = [
            self._row_to_domain_signal(row, current_timestamp)
            for _, row in prepared.iterrows()
        ]
        return AdaptedStrategySignals(prepared, domain_signals)

    def _row_to_domain_signal(self, row, current_timestamp=None) -> StrategySignal:
        """把单行 DataFrame 信号映射成领域对象。"""
        timestamp = self._row_timestamp(row, current_timestamp)
        metadata = {
            key: row[key]
            for key in row.index
            if key not in {"signal_id", "symbol", "action", "side", "datetime"}
        }
        return StrategySignal(
            signal_id=str(row["signal_id"]),
            symbol=str(row.get("symbol", "")),
            timestamp=timestamp.to_pydatetime(),
            side=str(row.get("side", row.get("action", "hold"))),
            reason=str(row.get("reason", "")),
            metadata=metadata,
        )

    def _build_signal_id(self, row, current_timestamp=None) -> str:
        """构造稳定 signal_id，用于 pipeline 去重。"""
        timestamp_value = row.get("timestamp", None)
        if pd.notna(timestamp_value):
            timestamp_part = str(int(timestamp_value)) if isinstance(timestamp_value, float) else str(timestamp_value)
        else:
            timestamp_part = str(self._row_timestamp(row, current_timestamp))

        side = str(row.get("side", row.get("action", ""))).lower()
        symbol = str(row.get("symbol", ""))
        return f"{symbol}:{side}:{timestamp_part}"

    def _row_timestamp(self, row, current_timestamp=None) -> pd.Timestamp:
        """按 datetime、当前时间、毫秒 timestamp 的优先级解析信号时间。"""
        if "datetime" in row.index and pd.notna(row.get("datetime")):
            timestamp = pd.Timestamp(row["datetime"])
        elif current_timestamp is not None:
            timestamp = pd.Timestamp(current_timestamp)
        elif "timestamp" in row.index and pd.notna(row.get("timestamp")):
            timestamp = pd.to_datetime(row["timestamp"], unit="ms", utc=True)
        else:
            raise ValueError("Strategy signal row requires datetime, timestamp, or current_timestamp")

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp
