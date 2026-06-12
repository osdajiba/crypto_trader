"""因子视图模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class FactorView:
    """策略读取的只读因子视图。"""

    values_by_symbol: Mapping[str, Mapping[str, float]]

    def has(self, symbol: str, factor_name: str) -> bool:
        return factor_name in self.values_by_symbol.get(symbol, {})

    def get(self, symbol: str, factor_name: str, default=None):
        return self.values_by_symbol.get(symbol, {}).get(factor_name, default)
