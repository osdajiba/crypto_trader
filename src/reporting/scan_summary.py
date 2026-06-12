"""Summaries for parameter scan and walk-forward backtest results."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.application.backtest_batch import BacktestRunResult


def summarize_scan_results(
    results: list[BacktestRunResult],
    metric: str = "final_equity",
    descending: bool = True,
) -> dict[str, Any]:
    success_rows = []
    failed_rows = []

    for result in results:
        research = result.report.get("research", {})
        row = {
            "run_id": result.run_id,
            "run_hash": research.get("run_hash") or _run_hash(result),
            "status": result.status,
            "parameters": result.parameters,
            "window": _window_value(result.window),
            "metric": _metric_value(result.report, metric),
            "diagnostics": result.report.get("diagnostics", {}),
            "config_hash": research.get("config_hash"),
            "config_snapshot": research.get("config_snapshot"),
            "artifacts": research.get("artifacts") or result.report.get("artifacts", {}),
            "error": result.error,
        }
        if result.status == "success" and row["metric"] is not None:
            success_rows.append(row)
        else:
            failed_rows.append(row)

    success_rows.sort(key=lambda row: row["metric"], reverse=descending)
    for index, row in enumerate(success_rows, start=1):
        row["rank"] = index
    for row in failed_rows:
        row["rank"] = None

    best = success_rows[0] if success_rows else None
    return {
        "metric": metric,
        "run_count": len(results),
        "success_count": len(success_rows),
        "failed_count": len(results) - len(success_rows),
        "best_run_id": best["run_id"] if best else None,
        "best_metric": best["metric"] if best else None,
        "results": success_rows + failed_rows,
    }


def _run_hash(result: BacktestRunResult) -> str:
    payload = {
        "run_id": result.run_id,
        "parameters": result.parameters,
        "window": _window_value(result.window),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _metric_value(report: dict[str, Any], metric: str):
    value: Any = report
    for part in metric.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _window_value(window):
    if window is None:
        return None
    return {
        "train_start": window.train_start.isoformat(),
        "train_end": window.train_end.isoformat(),
        "test_start": window.test_start.isoformat(),
        "test_end": window.test_end.isoformat(),
    }
