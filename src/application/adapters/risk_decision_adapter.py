from dataclasses import dataclass
from typing import List

import pandas as pd

from src.application.adapters.strategy_signal_adapter import StrategySignalAdapter
from src.domain.models import RiskDecision


@dataclass(frozen=True)
class AdaptedRiskDecisions:
    """同时保留风控后的 DataFrame 和领域 RiskDecision。"""
    dataframe: pd.DataFrame
    decisions: List[RiskDecision]


class RiskDecisionAdapter:
    """把当前风控 DataFrame 结果转换成 accepted/rejected 决策。"""

    def prepare(self, signals: pd.DataFrame) -> AdaptedRiskDecisions:
        """补齐 risk_accepted/risk_reason，并生成领域 RiskDecision。"""
        if signals.empty:
            return AdaptedRiskDecisions(signals.copy(), [])

        prepared = signals.copy()
        # 旧风控没有显式拒绝列时，按兼容原则默认接受。
        if "risk_accepted" not in prepared.columns:
            prepared["risk_accepted"] = True
        if "risk_reason" not in prepared.columns:
            prepared["risk_reason"] = ""

        strategy_signals = StrategySignalAdapter().prepare(prepared).domain_signals
        decisions = []
        for (_, row), signal in zip(prepared.iterrows(), strategy_signals):
            quantity = float(row.get("quantity", 0) or 0)
            price = float(row.get("price", 0) or 0)
            decisions.append(
                RiskDecision(
                    accepted=bool(row.get("risk_accepted", True)),
                    reason=str(row.get("risk_reason", "")),
                    target_notional=quantity * price,
                    target_quantity=quantity,
                    adjusted_signal=signal,
                )
            )

        return AdaptedRiskDecisions(prepared, decisions)

    def filter_accepted(self, signals: pd.DataFrame) -> pd.DataFrame:
        """过滤掉风控显式拒绝的信号，保留旧 DataFrame 执行协议。"""
        if signals.empty:
            return signals.copy()

        prepared = signals.copy()
        if "risk_accepted" not in prepared.columns:
            prepared["risk_accepted"] = True

        accepted = prepared[prepared["risk_accepted"].astype(bool)]
        return accepted.reset_index(drop=True)
