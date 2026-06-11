"""交易流水线的纯领域模型。

这些对象不依赖 pandas、交易所 SDK 或 mode 状态，目的是给后续策略、风控、
执行和报告之间的解耦提供稳定协议。当前仍保留 DataFrame 兼容层，所以这些
模型会先从 adapter 侧逐步进入主流程。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Mapping, Optional


def _require_timezone_aware(timestamp: datetime, field_name: str = "timestamp") -> None:
    """领域层统一要求时区感知时间，避免本地时间和 UTC 混用导致回测漂移。"""
    if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) is None:
        raise ValueError(f"{field_name} must be timezone-aware")


@dataclass(frozen=True)
class MarketBar:
    """单个交易对在某个时间点的一根 OHLCV K 线。"""
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        _require_timezone_aware(self.timestamp)


@dataclass(frozen=True)
class MarketSlice:
    """同一时间点的多交易对行情切片，是回测/实盘 pipeline 的行情输入单位。"""
    timestamp: datetime
    bars_by_symbol: Mapping[str, MarketBar]

    def __post_init__(self) -> None:
        _require_timezone_aware(self.timestamp)


@dataclass(frozen=True)
class StrategySignal:
    """策略产生的领域信号；当前由 DataFrame 信号兼容转换而来。"""
    signal_id: str
    symbol: str
    timestamp: datetime
    side: str
    strength: float = 1.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_timezone_aware(self.timestamp)
        normalized_side = self.side.lower()
        if normalized_side not in {"buy", "sell", "short", "cover", "hold"}:
            raise ValueError(f"Unsupported signal side: {self.side}")
        object.__setattr__(self, "side", normalized_side)


@dataclass(frozen=True)
class OrderIntent:
    """通过风控后的下单意图，还不是交易所订单。"""
    order_intent_id: str
    symbol: str
    side: str
    quantity: float
    order_type: str
    created_at: datetime
    limit_price: Optional[float] = None
    source_signal_id: Optional[str] = None

    def __post_init__(self) -> None:
        _require_timezone_aware(self.created_at, "created_at")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        object.__setattr__(self, "side", self.side.lower())
        object.__setattr__(self, "order_type", self.order_type.lower())


@dataclass(frozen=True)
class RiskDecision:
    """风控对策略信号的裁决结果，包含是否接受以及调整后的目标数量。"""
    accepted: bool
    reason: str
    target_notional: float
    target_quantity: float
    adjusted_signal: StrategySignal

    def to_order_intent(self, order_type: str = "market", limit_price: Optional[float] = None) -> OrderIntent:
        """只有被接受的风控结果才能继续生成下单意图。"""
        if not self.accepted:
            raise ValueError("Rejected risk decisions cannot create order intents")
        return OrderIntent(
            order_intent_id=f"intent:{self.adjusted_signal.signal_id}",
            symbol=self.adjusted_signal.symbol,
            side=self.adjusted_signal.side,
            quantity=self.target_quantity,
            order_type=order_type,
            created_at=self.adjusted_signal.timestamp,
            limit_price=limit_price,
            source_signal_id=self.adjusted_signal.signal_id,
        )


@dataclass(frozen=True)
class Fill:
    """执行模型返回的成交结果，用于后续更新组合账本。"""
    fill_id: str
    order_intent_id: str
    symbol: str
    side: str
    timestamp: datetime
    quantity: float
    price: float
    commission: float
    slippage: float = 0.0
    status: str = "filled"

    def __post_init__(self) -> None:
        _require_timezone_aware(self.timestamp)
        object.__setattr__(self, "side", self.side.lower())
        object.__setattr__(self, "status", self.status.lower())


@dataclass(frozen=True)
class PortfolioSnapshot:
    """组合在某一时刻的只读快照，供风控和报告使用。"""
    timestamp: datetime
    cash: float
    positions: Mapping[str, float]
    market_prices: Mapping[str, float]
    equity: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    def __post_init__(self) -> None:
        _require_timezone_aware(self.timestamp)
