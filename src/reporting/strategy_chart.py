"""Strategy visualization charts for backtest reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


class StrategyChartWriter:
    """Write price, MA/EMA crossover, and spread charts."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    def write(
        self,
        historical_data: dict[str, pd.DataFrame],
        report: dict[str, Any],
        short_window: int,
        long_window: int,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Path:
        if not historical_data:
            raise ValueError("historical_data is required")

        symbol, frame = next(iter(historical_data.items()))
        data = self._price_frame(frame)
        if data.empty:
            raise ValueError("historical_data has no plottable close prices")

        short_ma = data["close"].rolling(window=int(short_window), min_periods=int(short_window)).mean()
        long_ma = data["close"].rolling(window=int(long_window), min_periods=int(long_window)).mean()
        short_ema = data["close"].ewm(span=int(short_window), adjust=False).mean()
        long_ema = data["close"].ewm(span=int(long_window), adjust=False).mean()
        ma_spread_pct = ((short_ma - long_ma) / long_ma) * 100

        ma_cross_up = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        ma_cross_down = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
        ema_cross_up = (short_ema > long_ema) & (short_ema.shift(1) <= long_ema.shift(1))
        ema_cross_down = (short_ema < long_ema) & (short_ema.shift(1) >= long_ema.shift(1))

        fig, (price_axis, spread_axis) = plt.subplots(
            2,
            1,
            figsize=(18, 11),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        price_axis.plot(data.index, data["close"], color="0.55", linewidth=1.0, label="Price")
        price_axis.plot(data.index, short_ma, color="blue", linewidth=1.0, label=f"Short MA ({short_window})")
        price_axis.plot(data.index, long_ma, color="red", linewidth=1.0, label=f"Long MA ({long_window})")
        price_axis.plot(data.index, short_ema, color="green", linestyle="--", linewidth=1.0, label=f"Short EMA ({short_window})")
        price_axis.plot(data.index, long_ema, color="purple", linestyle="--", linewidth=1.0, label=f"Long EMA ({long_window})")

        self._scatter_cross(price_axis, data, ma_cross_up, marker="^", color="green", label="MA Cross Up (Buy)")
        self._scatter_cross(price_axis, data, ma_cross_down, marker="v", color="red", label="MA Cross Down (Sell)")
        self._scatter_cross(price_axis, data, ema_cross_up, marker="^", color="lime", label="EMA Cross Up")
        self._scatter_cross(price_axis, data, ema_cross_down, marker="v", color="orange", label="EMA Cross Down")

        price_axis.set_title(f"{symbol} Price with MA/EMA Crossovers - {self._date_label(start_date, end_date, data)}")
        price_axis.set_ylabel("Price")
        price_axis.grid(True, alpha=0.25)
        price_axis.legend(loc="upper left")

        spread_axis.plot(data.index, ma_spread_pct, color="blue", linewidth=1.0, label="MA Spread %")
        spread_axis.fill_between(
            data.index,
            ma_spread_pct.to_numpy(dtype=float),
            0,
            where=(ma_spread_pct.to_numpy(dtype=float) >= 0),
            color="green",
            alpha=0.25,
        )
        spread_axis.fill_between(
            data.index,
            ma_spread_pct.to_numpy(dtype=float),
            0,
            where=(ma_spread_pct.to_numpy(dtype=float) < 0),
            color="red",
            alpha=0.25,
        )
        spread_axis.axhline(0, color="red", linewidth=0.8, alpha=0.6)
        spread_axis.set_title("MA Spread Percentage")
        spread_axis.set_ylabel("Spread %")
        spread_axis.set_xlabel("Date")
        spread_axis.grid(True, alpha=0.25)
        spread_axis.legend(loc="upper left")

        fig.autofmt_xdate()
        fig.tight_layout()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"ma_ema_crossovers_{self._date_slug(start_date, data, first=True)}_{self._date_slug(end_date, data, first=False)}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    def _price_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        if "datetime" in data.columns:
            data["datetime"] = pd.to_datetime(data["datetime"], utc=True)
            data = data.set_index("datetime")
        elif not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("historical_data requires a datetime column or DatetimeIndex")

        if "close" not in data.columns:
            raise ValueError("historical_data requires a close column")

        data = data.sort_index()
        data["close"] = pd.to_numeric(data["close"], errors="coerce")
        return data.dropna(subset=["close"])

    def _scatter_cross(self, axis, data, mask, marker: str, color: str, label: str) -> None:
        points = data[mask.fillna(False)]
        if points.empty:
            return
        axis.scatter(points.index, points["close"], marker=marker, color=color, s=70, label=label, zorder=5)

    def _date_label(self, start_date: str | None, end_date: str | None, data: pd.DataFrame) -> str:
        return f"{start_date or data.index.min().date()} to {end_date or data.index.max().date()}"

    def _date_slug(self, configured_date: str | None, data: pd.DataFrame, first: bool) -> str:
        if configured_date:
            return configured_date.replace("-", "")
        timestamp = data.index.min() if first else data.index.max()
        return timestamp.strftime("%Y%m%d")
