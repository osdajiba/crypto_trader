"""Performance metrics from domain fills and portfolio snapshots."""

from __future__ import annotations

from collections import Counter

from src.domain.models import Fill, PortfolioSnapshot
from src.reporting.backtest_diagnostics import diagnose_backtest_report


class PerformanceAnalyzer:
    """Build report-compatible metrics from fills and snapshots."""

    def analyze(
        self,
        fills: list[Fill],
        snapshots: list[PortfolioSnapshot],
        initial_capital: float,
    ) -> dict:
        initial_capital = float(initial_capital)
        final_snapshot = snapshots[-1] if snapshots else None
        final_equity = float(final_snapshot.equity) if final_snapshot else initial_capital
        remaining_cash = float(final_snapshot.cash) if final_snapshot else initial_capital
        positions = dict(final_snapshot.positions) if final_snapshot else {}
        total_return = final_equity - initial_capital
        total_return_pct = (total_return / initial_capital * 100) if initial_capital > 0 else 0
        trades = [self._trade_from_fill(fill) for fill in fills]

        report = {
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": self._max_drawdown_pct(snapshots),
            "total_trades": len(trades),
            "buy_trades": len([trade for trade in trades if trade["action"] == "buy"]),
            "sell_trades": len([trade for trade in trades if trade["action"] == "sell"]),
            "current_positions": positions,
            "remaining_cash": remaining_cash,
            "trades": trades,
            "equity_curve": [
                {"timestamp": snapshot.timestamp, "equity": snapshot.equity}
                for snapshot in snapshots
            ],
            "costs": self._cost_summary(fills),
            "quality": self._quality_summary(fills, snapshots, positions),
        }
        report["diagnostics"] = diagnose_backtest_report(report)
        return report

    def _trade_from_fill(self, fill: Fill) -> dict:
        return {
            "timestamp": fill.timestamp,
            "symbol": fill.symbol,
            "action": fill.side,
            "quantity": fill.quantity,
            "price": fill.price,
            "commission": fill.commission,
        }

    def _max_drawdown_pct(self, snapshots: list[PortfolioSnapshot]) -> float:
        peak = None
        max_drawdown = 0.0
        for snapshot in snapshots:
            equity = float(snapshot.equity)
            peak = equity if peak is None else max(peak, equity)
            if peak and peak > 0:
                max_drawdown = max(max_drawdown, (peak - equity) / peak)
        return max_drawdown * 100

    def _quality_summary(
        self,
        fills: list[Fill],
        snapshots: list[PortfolioSnapshot],
        positions: dict,
    ) -> dict:
        timestamps = [snapshot.timestamp for snapshot in snapshots]
        positive_deltas = [
            (right - left).total_seconds()
            for left, right in zip(timestamps, timestamps[1:])
            if right > left
        ]
        expected_interval_seconds = self._expected_interval_seconds(positive_deltas)
        gap_deltas = [
            delta
            for delta in positive_deltas
            if expected_interval_seconds is not None and delta > expected_interval_seconds
        ]
        invalid_fill_counts = self._invalid_fill_counts(fills)
        return {
            "snapshot_count": len(snapshots),
            "trade_count": len(fills),
            "first_timestamp": timestamps[0] if timestamps else None,
            "last_timestamp": timestamps[-1] if timestamps else None,
            "time_order_valid": timestamps == sorted(timestamps),
            "has_negative_equity": any(float(snapshot.equity) < 0 for snapshot in snapshots),
            "has_open_positions": any(abs(float(quantity)) > 0 for quantity in positions.values()),
            "duplicate_timestamp_count": len(timestamps) - len(set(timestamps)),
            "expected_interval_seconds": expected_interval_seconds,
            "gap_count": len(gap_deltas),
            "max_gap_seconds": max(gap_deltas) if gap_deltas else 0,
            "fills_outside_snapshot_range": self._fills_outside_snapshot_range(fills, timestamps),
            **invalid_fill_counts,
        }

    def _expected_interval_seconds(self, positive_deltas: list[float]):
        if not positive_deltas:
            return None
        counts = Counter(positive_deltas)
        return min(counts, key=lambda delta: (-counts[delta], delta))

    def _fills_outside_snapshot_range(self, fills: list[Fill], timestamps: list) -> int:
        if not timestamps:
            return len(fills)
        first = min(timestamps)
        last = max(timestamps)
        return len([fill for fill in fills if fill.timestamp < first or fill.timestamp > last])

    def _invalid_fill_counts(self, fills: list[Fill]) -> dict[str, int]:
        invalid_fill_ids = set()
        quantity_count = 0
        price_count = 0
        commission_count = 0
        notional_count = 0

        for index, fill in enumerate(fills):
            fill_key = fill.fill_id or index
            quantity = float(fill.quantity)
            price = float(fill.price)
            commission = float(fill.commission)

            if quantity <= 0:
                quantity_count += 1
                invalid_fill_ids.add(fill_key)
            if price <= 0:
                price_count += 1
                invalid_fill_ids.add(fill_key)
            if commission < 0:
                commission_count += 1
                invalid_fill_ids.add(fill_key)
            if quantity * price <= 0:
                notional_count += 1
                invalid_fill_ids.add(fill_key)

        return {
            "invalid_fill_quantity_count": quantity_count,
            "invalid_fill_price_count": price_count,
            "invalid_fill_commission_count": commission_count,
            "invalid_fill_notional_count": notional_count,
            "invalid_fill_count": len(invalid_fill_ids),
        }

    def _cost_summary(self, fills: list[Fill]) -> dict[str, float]:
        positive_notionals = [
            max(float(fill.quantity) * float(fill.price), 0.0)
            for fill in fills
        ]
        total_notional = sum(positive_notionals)
        total_commission = sum(float(fill.commission) for fill in fills)
        slippage_costs = [
            notional * abs(float(fill.slippage))
            for fill, notional in zip(fills, positive_notionals)
        ]
        estimated_slippage_cost = sum(slippage_costs)

        return {
            "fill_count": len(fills),
            "total_notional": total_notional,
            "total_commission": total_commission,
            "average_commission_per_fill": total_commission / len(fills) if fills else 0,
            "commission_rate_bps": (total_commission / total_notional * 10000) if total_notional > 0 else 0,
            "estimated_slippage_cost": estimated_slippage_cost,
            "average_slippage_bps": (estimated_slippage_cost / total_notional * 10000) if total_notional > 0 else 0,
            "max_slippage_bps": max([abs(float(fill.slippage)) * 10000 for fill in fills], default=0),
            "total_transaction_cost": total_commission + estimated_slippage_cost,
        }
