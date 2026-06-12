"""Research diagnostics for backtest reports."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def diagnose_backtest_report(report: dict[str, Any]) -> dict[str, Any]:
    """Build research-oriented diagnostics without changing trading behavior."""
    initial_capital = float(report.get("initial_capital") or 0)
    total_return = float(report.get("total_return") or 0)
    total_return_pct = float(report.get("total_return_pct") or 0)
    max_drawdown_pct = float(report.get("max_drawdown_pct") or 0)
    total_trades = int(report.get("total_trades") or 0)
    costs = report.get("costs") or {}
    total_transaction_cost = float(costs.get("total_transaction_cost") or 0)
    total_notional = float(costs.get("total_notional") or 0)

    gross_return_estimate = total_return + total_transaction_cost
    gross_return_pct_estimate = (
        gross_return_estimate / initial_capital * 100
        if initial_capital > 0
        else 0
    )

    positions = report.get("current_positions") or {}
    behavior = _trade_behavior(report.get("trades") or [])
    diagnostics = {
        "net_return": total_return,
        "net_return_pct": total_return_pct,
        "gross_return_estimate": gross_return_estimate,
        "gross_return_pct_estimate": gross_return_pct_estimate,
        "total_transaction_cost": total_transaction_cost,
        "cost_drag_pct_of_initial": (
            total_transaction_cost / initial_capital * 100
            if initial_capital > 0
            else 0
        ),
        "cost_to_abs_net_return": (
            total_transaction_cost / abs(total_return)
            if total_return != 0
            else None
        ),
        "total_trades": total_trades,
        "average_trade_notional": (
            total_notional / total_trades
            if total_trades > 0
            else 0
        ),
        "turnover_pct_of_initial": (
            total_notional / initial_capital * 100
            if initial_capital > 0
            else 0
        ),
        "max_drawdown_pct": max_drawdown_pct,
        "return_to_drawdown": (
            total_return_pct / max_drawdown_pct
            if max_drawdown_pct > 0
            else None
        ),
        "has_open_positions": any(abs(float(quantity)) > 0 for quantity in positions.values()),
    }
    diagnostics.update(behavior)
    return diagnostics


def _trade_behavior(trades: list[dict[str, Any]]) -> dict[str, Any]:
    lots_by_symbol: dict[str, list[dict[str, Any]]] = {}
    round_trips: list[dict[str, float]] = []

    for trade in trades:
        action = _action_value(trade)
        symbol = str(trade.get("symbol") or "")
        quantity = float(trade.get("quantity") or 0)
        price = float(trade.get("price") or 0)
        commission = float(trade.get("commission") or 0)
        timestamp = _parse_timestamp(trade.get("timestamp"))
        if not symbol or quantity <= 0 or price <= 0 or timestamp is None:
            continue

        if action == "buy":
            lots_by_symbol.setdefault(symbol, []).append({
                "remaining": quantity,
                "price": price,
                "remaining_commission": commission,
                "timestamp": timestamp,
            })
            continue

        if action != "sell":
            continue

        remaining_sell = quantity
        sell_commission_remaining = commission
        lots = lots_by_symbol.setdefault(symbol, [])
        while remaining_sell > 0 and lots:
            lot = lots[0]
            matched_quantity = min(float(lot["remaining"]), remaining_sell)
            buy_commission = float(lot["remaining_commission"]) * (matched_quantity / float(lot["remaining"]))
            sell_commission = (
                sell_commission_remaining * (matched_quantity / remaining_sell)
                if remaining_sell > 0
                else 0
            )
            gross_cost = matched_quantity * float(lot["price"])
            proceeds = matched_quantity * price
            realized_return = proceeds - gross_cost - buy_commission - sell_commission

            round_trips.append({
                "holding_minutes": (timestamp - lot["timestamp"]).total_seconds() / 60,
                "return": realized_return,
                "return_pct": (realized_return / gross_cost * 100) if gross_cost > 0 else 0,
            })

            lot["remaining"] = float(lot["remaining"]) - matched_quantity
            lot["remaining_commission"] = float(lot["remaining_commission"]) - buy_commission
            remaining_sell -= matched_quantity
            sell_commission_remaining -= sell_commission
            if float(lot["remaining"]) <= 0:
                lots.pop(0)

    if not round_trips:
        return {
            "round_trip_count": 0,
            "round_trip_win_rate": None,
            "average_holding_minutes": 0,
            "average_round_trip_return": 0,
            "average_round_trip_return_pct": 0,
        }

    winning = [trip for trip in round_trips if trip["return"] > 0]
    return {
        "round_trip_count": len(round_trips),
        "round_trip_win_rate": len(winning) / len(round_trips) * 100,
        "average_holding_minutes": sum(trip["holding_minutes"] for trip in round_trips) / len(round_trips),
        "average_round_trip_return": sum(trip["return"] for trip in round_trips) / len(round_trips),
        "average_round_trip_return_pct": sum(trip["return_pct"] for trip in round_trips) / len(round_trips),
    }


def _action_value(trade: dict[str, Any]) -> str:
    action = trade.get("action", "")
    if hasattr(action, "value"):
        return str(action.value).lower()
    return str(action).lower()


def _parse_timestamp(value) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
