"""Application entrypoints for local backtest research runs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from src.application.backtest_batch import BacktestBatchRunner, BacktestRunSpec, build_backtest_run_specs
from src.application.backtest_planning import WalkForwardWindow
from src.reporting.research_writer import ResearchSummaryWriter
from src.reporting.scan_summary import summarize_scan_results


def run_backtest_research(
    *,
    parameter_grid: Mapping[str, list[Any]],
    run_backtest: Callable[[BacktestRunSpec], Mapping[str, Any]],
    windows: Iterable[WalkForwardWindow] | None = None,
    metric: str = "final_equity",
    descending: bool = True,
    summary_report_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build specs, run injected backtests, and return a scan summary."""
    specs = build_backtest_run_specs(
        parameter_grid=parameter_grid,
        windows=windows,
    )
    results = BacktestBatchRunner(run_backtest).run(specs)
    summary = summarize_scan_results(
        results,
        metric=metric,
        descending=descending,
    )
    if summary_report_dir is not None:
        summary_path = ResearchSummaryWriter(summary_report_dir).write(summary)
        summary["summary_path"] = str(summary_path)
    return summary
