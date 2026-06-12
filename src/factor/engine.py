"""基础因子计算引擎。"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Mapping

from src.domain.models import MarketBar, MarketSlice
from src.factor.models import FactorView


class FactorEngine:
    """维护每个 symbol 的行情窗口，并生成基础因子视图。"""

    def __init__(
        self,
        ma_windows: list[int] | None = None,
        factor_config: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.ma_windows = sorted(set(ma_windows or [3]))
        self.factor_config = dict(factor_config or {})
        self._max_window = max(
            self.ma_windows + self._configured_windows(),
            default=1,
        )
        self._bars_by_symbol: dict[str, Deque[MarketBar]] = defaultdict(
            lambda: deque(maxlen=self._max_window)
        )

    @classmethod
    def from_config(cls, factor_config: Mapping[str, Mapping[str, Any]]) -> "FactorEngine":
        return cls(ma_windows=[], factor_config=factor_config)

    def update(self, market: MarketSlice) -> FactorView:
        values: dict[str, dict[str, float]] = {}

        for symbol, bar in market.bars_by_symbol.items():
            history = self._bars_by_symbol[symbol]
            history.append(bar)
            symbol_values = self._calculate_symbol_values(history)
            if symbol_values:
                values[symbol] = symbol_values

        return FactorView(values)

    def _calculate_symbol_values(self, history: Deque[MarketBar]) -> dict[str, float]:
        values: dict[str, float] = {}
        closes = [bar.close for bar in history]

        for window in self.ma_windows:
            if len(closes) < window:
                continue
            recent = closes[-window:]
            values[f"ma_{window}"] = sum(recent) / window

        configured_values = self._calculate_configured_values(list(history))
        values.update(configured_values)

        return values

    def _calculate_configured_values(self, history: list[MarketBar]) -> dict[str, float]:
        raw_values: dict[str, float] = {}
        signal_values: dict[str, float] = {}

        for name, config in self.factor_config.items():
            raw_series = self._calculate_factor_series(history, config)
            if not raw_series:
                continue

            raw_value = raw_series[-1]
            if raw_value is None:
                continue

            raw_values[name] = raw_value
            signal_values[name] = self._factor_signal(raw_series, config)

        if signal_values:
            raw_values["composite_signal"] = self._composite_signal(signal_values)

        return raw_values

    def _calculate_factor_series(
        self,
        history: list[MarketBar],
        config: Mapping[str, Any],
    ) -> list[float | None]:
        factor_type = str(config.get("type", "")).lower()
        params = config.get("params", {}) or {}

        if factor_type == "rsi":
            return self._rsi_series([bar.close for bar in history], int(params.get("period", 14)))
        if factor_type == "macd":
            return self._macd_histogram_series(
                [bar.close for bar in history],
                int(params.get("fast_period", 12)),
                int(params.get("slow_period", 26)),
                int(params.get("signal_period", 9)),
            )
        if factor_type == "bollinger":
            return self._bollinger_position_series(
                [bar.close for bar in history],
                int(params.get("period", 20)),
                float(params.get("std_dev", 2.0)),
            )
        if factor_type == "volume_osc":
            return self._volume_oscillator_series(
                [bar.volume for bar in history],
                int(params.get("fast_period", 5)),
                int(params.get("slow_period", 14)),
            )

        return []

    def _factor_signal(self, raw_series: list[float | None], config: Mapping[str, Any]) -> float:
        raw_value = raw_series[-1]
        if raw_value is None:
            return 0.0

        signal_type = str(config.get("signal_type", "standard")).lower()
        if signal_type == "threshold":
            upper = float(config.get("upper_threshold", 0.7))
            lower = float(config.get("lower_threshold", 0.3))
            if raw_value < lower:
                return 1.0
            if raw_value > upper:
                return -1.0
            return 0.0

        previous = self._previous_value(raw_series)
        if signal_type == "crossover":
            if previous is None:
                return 0.0
            if raw_value > 0 and previous <= 0:
                return 1.0
            if raw_value < 0 and previous >= 0:
                return -1.0
            return 0.0

        if signal_type == "momentum":
            if previous is None:
                return 0.0
            if raw_value > previous:
                return 1.0
            if raw_value < previous:
                return -1.0
            return 0.0

        return self._normalized_value(raw_series) if config.get("normalize", False) else float(raw_value)

    def _composite_signal(self, signal_values: Mapping[str, float]) -> float:
        weighted_sum = 0.0
        total_weight = 0.0
        for name, signal_value in signal_values.items():
            weight = float(self.factor_config.get(name, {}).get("weight", 1.0))
            weighted_sum += signal_value * weight
            total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _configured_windows(self) -> list[int]:
        windows = []
        for config in self.factor_config.values():
            params = config.get("params", {}) or {}
            factor_type = str(config.get("type", "")).lower()
            configured = int(config.get("window_size", 0) or 0)
            derived = 0
            if factor_type == "rsi":
                derived = int(params.get("period", 14)) + 1
            elif factor_type == "macd":
                derived = int(params.get("slow_period", 26)) + int(params.get("signal_period", 9))
            elif factor_type == "bollinger":
                derived = int(params.get("period", 20))
            elif factor_type == "volume_osc":
                derived = int(params.get("slow_period", 14))
            windows.append(max(configured, derived, 1))
        return windows

    def _rsi_series(self, closes: list[float], period: int) -> list[float | None]:
        values: list[float | None] = []
        for index in range(len(closes)):
            if index < period:
                values.append(None)
                continue
            deltas = [closes[i] - closes[i - 1] for i in range(index - period + 1, index + 1)]
            gains = [delta for delta in deltas if delta > 0]
            losses = [-delta for delta in deltas if delta < 0]
            avg_gain = sum(gains) / period
            avg_loss = sum(losses) / period
            if avg_loss == 0:
                values.append(100.0 if avg_gain > 0 else 50.0)
                continue
            rs = avg_gain / avg_loss
            values.append(100 - (100 / (1 + rs)))
        return values

    def _macd_histogram_series(
        self,
        closes: list[float],
        fast_period: int,
        slow_period: int,
        signal_period: int,
    ) -> list[float | None]:
        if len(closes) < slow_period + signal_period:
            return [None] * len(closes)
        fast = self._ema_series(closes, fast_period)
        slow = self._ema_series(closes, slow_period)
        macd = [
            None if fast_value is None or slow_value is None else fast_value - slow_value
            for fast_value, slow_value in zip(fast, slow)
        ]
        signal = self._ema_series([value or 0.0 for value in macd], signal_period)
        return [
            None if macd_value is None or signal_value is None else macd_value - signal_value
            for macd_value, signal_value in zip(macd, signal)
        ]

    def _bollinger_position_series(
        self,
        closes: list[float],
        period: int,
        std_dev: float,
    ) -> list[float | None]:
        values: list[float | None] = []
        for index in range(len(closes)):
            if index + 1 < period:
                values.append(None)
                continue
            window = closes[index - period + 1:index + 1]
            middle = sum(window) / period
            variance = sum((value - middle) ** 2 for value in window) / period
            band_width = (variance ** 0.5) * std_dev
            if band_width == 0:
                values.append(0.0)
            else:
                values.append((closes[index] - middle) / band_width)
        return values

    def _volume_oscillator_series(
        self,
        volumes: list[float],
        fast_period: int,
        slow_period: int,
    ) -> list[float | None]:
        values: list[float | None] = []
        for index in range(len(volumes)):
            if index + 1 < slow_period:
                values.append(None)
                continue
            fast_window = volumes[index - fast_period + 1:index + 1]
            slow_window = volumes[index - slow_period + 1:index + 1]
            fast = sum(fast_window) / fast_period
            slow = sum(slow_window) / slow_period
            values.append(0.0 if slow == 0 else 100 * ((fast - slow) / slow))
        return values

    def _ema_series(self, values: list[float], period: int) -> list[float | None]:
        if not values:
            return []
        alpha = 2 / (period + 1)
        ema_values: list[float | None] = []
        ema = values[0]
        for value in values:
            ema = (value * alpha) + (ema * (1 - alpha))
            ema_values.append(ema)
        return ema_values

    def _previous_value(self, values: list[float | None]) -> float | None:
        for value in reversed(values[:-1]):
            if value is not None:
                return value
        return None

    def _normalized_value(self, values: list[float | None]) -> float:
        valid = [value for value in values if value is not None]
        if not valid:
            return 0.0
        minimum = min(valid)
        maximum = max(valid)
        if maximum == minimum:
            return 0.0
        return (valid[-1] - minimum) / (maximum - minimum)
