"""市场数据规范化工具。

本地历史数据可能来自不同下载脚本或目录格式。进入 domain/feed 前统一字段、
时间和排序，可以让后续 pipeline 不再关心原始文件细节。
"""

from __future__ import annotations

import pandas as pd


class MarketDataNormalizer:
    REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

    @staticmethod
    def normalize_ohlcv(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """规范化 OHLCV 数据，输出当前策略和 MarketDataFeed 都能消费的格式。"""
        if data.empty:
            return data.copy()

        normalized = data.copy()

        # 兼容两类历史数据：有 datetime 列，或只有毫秒级 timestamp 列。
        if "datetime" not in normalized.columns:
            if "timestamp" not in normalized.columns:
                raise ValueError("Market data must include datetime or timestamp")
            normalized["datetime"] = pd.to_datetime(normalized["timestamp"], unit="ms", utc=True)
        else:
            normalized["datetime"] = pd.to_datetime(normalized["datetime"], utc=True)

        missing = [column for column in MarketDataNormalizer.REQUIRED_COLUMNS if column not in normalized.columns]
        if missing:
            raise ValueError(f"Market data missing required columns: {missing}")

        for column in MarketDataNormalizer.REQUIRED_COLUMNS:
            normalized[column] = pd.to_numeric(normalized[column], errors="raise")

        # 去重保留最后一条，避免同一时间点重复 K 线导致 rolling 或 reindex 出错。
        normalized["symbol"] = symbol
        normalized = normalized.sort_values("datetime")
        normalized = normalized.drop_duplicates(subset=["datetime"], keep="last")
        normalized = normalized.reset_index(drop=True)

        return normalized
