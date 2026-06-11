"""基于 Parquet 文件的历史行情存储。

这个 store 只负责“从本地文件优雅地找出并读出历史数据”。它不参与策略、
风控或回测循环，读出的 DataFrame 会再交给 normalizer 和 feed 继续处理。
"""

from __future__ import annotations

import os
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import pandas as pd

from src.datasource.normalizer import MarketDataNormalizer


class ParquetHistoricalStore:
    def __init__(self, base_path: str):
        self.base_path = os.path.abspath(os.path.expanduser(base_path))

    def find_files(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> List[str]:
        """根据交易对、周期和时间范围查找可能相关的 parquet 文件。"""
        start = self._ensure_tz_aware(start)
        end = self._ensure_tz_aware(end)
        start_ts, end_ts = int(start.timestamp()), int(end.timestamp())
        symbol_name = symbol.replace("/", "_")

        # 兼容项目中出现过的几种本地目录布局，避免把文件结构写死在业务代码里。
        candidate_dirs = [
            os.path.join(self.base_path, timeframe, symbol_name),
            os.path.join(self.base_path, "binance", symbol_name, timeframe),
            os.path.join(self.base_path, symbol_name, timeframe),
        ]

        parent_dir = os.path.dirname(self.base_path)
        if os.path.basename(self.base_path).lower() == "historical":
            candidate_dirs.append(os.path.join(parent_dir, "binance", symbol_name, timeframe))

        # 部分数据按 year/month 分桶存储，所以需要把时间范围展开为月份列表。
        date_ranges = self._month_ranges(start, end)
        file_paths = []

        for symbol_dir in dict.fromkeys(candidate_dirs):
            if not os.path.exists(symbol_dir):
                continue

            search_dirs = [symbol_dir]
            search_dirs.extend(
                os.path.join(symbol_dir, str(year), f"{month:02d}")
                for year, month in date_ranges
                if os.path.exists(os.path.join(symbol_dir, str(year), f"{month:02d}"))
            )

            for search_dir in search_dirs:
                for file_name in os.listdir(search_dir):
                    if not file_name.endswith(".parquet"):
                        continue
                    file_range = self._parse_file_range(file_name, timeframe)
                    if file_range is None or self._overlaps(start_ts, end_ts, *file_range):
                        file_paths.append(os.path.join(search_dir, file_name))

        return sorted(dict.fromkeys(file_paths))

    async def load(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """加载并规范化指定时间范围内的本地历史数据。"""
        file_paths = self.find_files(symbol, timeframe, start, end)
        if not file_paths:
            return pd.DataFrame()

        data = await self.load_files(file_paths, date_filter=(start, end))
        if data.empty:
            return data
        return MarketDataNormalizer.normalize_ohlcv(data, symbol=symbol)

    async def load_files(
        self,
        file_paths: List[str],
        date_filter: Optional[Tuple[datetime, datetime]] = None,
    ) -> pd.DataFrame:
        """并发读取多个 parquet 文件，并在读取后做时间范围过滤。"""
        if not file_paths:
            return pd.DataFrame()

        frames = await asyncio.gather(*[asyncio.to_thread(pd.read_parquet, path) for path in file_paths])
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame()

        if date_filter:
            start, end = [pd.Timestamp(self._ensure_tz_aware(value)) for value in date_filter]
            filtered_frames = []
            for frame in frames:
                if "datetime" not in frame.columns:
                    filtered_frames.append(frame)
                    continue

                frame = frame.copy()
                frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
                filtered = frame[(frame["datetime"] >= start) & (frame["datetime"] <= end)]
                if not filtered.empty:
                    filtered_frames.append(filtered)
            frames = filtered_frames

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        if "datetime" in combined.columns:
            combined = combined.sort_values("datetime").reset_index(drop=True)
        return combined

    @staticmethod
    def _ensure_tz_aware(value: datetime) -> datetime:
        """统一转成 UTC 感知时间，避免 naive datetime 参与比较。"""
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(timezone.utc)
        return timestamp.to_pydatetime()

    @staticmethod
    def _overlaps(start_ts: int, end_ts: int, file_start_ts: int, file_end_ts: int) -> bool:
        """判断文件时间范围是否和查询范围有交集。"""
        return (
            start_ts <= file_start_ts <= end_ts or
            start_ts <= file_end_ts <= end_ts or
            file_start_ts <= start_ts and file_end_ts >= end_ts
        )

    @staticmethod
    def _month_ranges(start: datetime, end: datetime) -> List[Tuple[int, int]]:
        """生成查询范围覆盖到的 year/month，用于查找分桶目录。"""
        ranges = []
        current = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
        end_month = datetime(end.year, end.month, 1, tzinfo=timezone.utc)

        while current <= end_month:
            ranges.append((current.year, current.month))
            current = (
                datetime(current.year + 1, 1, 1, tzinfo=timezone.utc)
                if current.month == 12
                else datetime(current.year, current.month + 1, 1, tzinfo=timezone.utc)
            )

        return ranges

    @staticmethod
    def _parse_file_range(file_name: str, timeframe: str) -> Optional[Tuple[int, int]]:
        """从常见文件名格式中解析文件覆盖的时间范围。"""
        stem = os.path.splitext(file_name)[0]

        parts = stem.split("_")
        if len(parts) >= 2:
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                pass

        if stem.startswith(f"{timeframe}-"):
            try:
                date_dt = datetime.strptime(stem.replace(f"{timeframe}-", ""), "%Y-%m-%d")
                file_start = date_dt.replace(tzinfo=timezone.utc)
                file_end = (date_dt + timedelta(days=1)).replace(tzinfo=timezone.utc)
                return int(file_start.timestamp()), int(file_end.timestamp())
            except ValueError:
                pass

        if "to" in stem and "T" in stem:
            try:
                start_text, end_text = stem.split("to", 1)
                file_start = datetime.fromisoformat(start_text.replace("_", ":"))
                file_end = datetime.fromisoformat(end_text.replace("_", ":"))
                file_start = ParquetHistoricalStore._ensure_tz_aware(file_start)
                file_end = ParquetHistoricalStore._ensure_tz_aware(file_end)
                return int(file_start.timestamp()), int(file_end.timestamp())
            except ValueError:
                pass

        return None
