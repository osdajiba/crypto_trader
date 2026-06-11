"""订单层辅助模型。"""

from __future__ import annotations

from dataclasses import dataclass

from src.domain.models import StrategySignal


@dataclass(frozen=True)
class PositionSizingDecision:
    """仓位 sizing 的结果，负责把策略意图转换为目标金额和数量。"""

    signal: StrategySignal
    target_notional: float
    target_quantity: float
    reason: str = ""
