"""Batch planning and execution helpers for backtest scans."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from src.application.backtest_planning import WalkForwardWindow, expand_parameter_grid


@dataclass(frozen=True)
class BacktestRunSpec:
    run_id: str
    parameters: dict[str, Any]
    window: WalkForwardWindow | None = None


@dataclass(frozen=True)
class BacktestRunResult:
    run_id: str
    parameters: dict[str, Any]
    window: WalkForwardWindow | None
    status: str
    report: dict[str, Any]
    error: str = ""


class BacktestBatchRunner:
    """Run backtest specs through an injected execution function."""

    def __init__(self, run_backtest: Callable[[BacktestRunSpec], Mapping[str, Any]]):
        self.run_backtest = run_backtest

    def run(self, specs: Iterable[BacktestRunSpec]) -> list[BacktestRunResult]:
        results = []
        for spec in specs:
            try:
                report = dict(self.run_backtest(spec))
                results.append(BacktestRunResult(
                    run_id=spec.run_id,
                    parameters=spec.parameters,
                    window=spec.window,
                    status="success",
                    report=report,
                ))
            except Exception as exc:
                results.append(BacktestRunResult(
                    run_id=spec.run_id,
                    parameters=spec.parameters,
                    window=spec.window,
                    status="failed",
                    report={},
                    error=str(exc),
                ))
        return results


def build_backtest_run_specs(
    *,
    parameter_grid: Mapping[str, list[Any]],
    windows: Iterable[WalkForwardWindow] | None = None,
) -> list[BacktestRunSpec]:
    parameter_sets = expand_parameter_grid(parameter_grid)
    window_sets = list(windows) if windows is not None else [None]

    specs = []
    index = 1
    for parameters in parameter_sets:
        for window in window_sets:
            specs.append(BacktestRunSpec(
                run_id=f"run-{index:03d}",
                parameters=dict(parameters),
                window=window,
            ))
            index += 1
    return specs
