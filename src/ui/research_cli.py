"""CLI adapter for local backtest research runs."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from src.application.backtest_planning import generate_walk_forward_windows
from src.application.backtest_research import run_backtest_research
from src.application.configured_backtest_runner import make_configured_backtest_runner


def run_research_cli_mode(args, config, logger) -> dict[str, Any]:
    """Parse research CLI arguments and delegate to the application layer."""
    _apply_common_overrides(args, config, logger)
    parameter_grid = _parse_parameter_grid(args.research_grid)
    runner = make_configured_backtest_runner(
        config,
        mode=args.mode or "backtest",
        backtest_engine=args.backtest_engine or "ohlcv",
    )
    summary = run_backtest_research(
        parameter_grid=parameter_grid,
        run_backtest=runner,
        windows=_build_walk_forward_windows(args),
        metric=args.research_metric,
        descending=not args.research_ascending,
        summary_report_dir=args.research_output_dir,
    )
    logger.info(f"Research summary saved to {summary.get('summary_path', '<not written>')}")
    print(json.dumps(summary, indent=2, default=str))
    return summary


def _parse_parameter_grid(raw_grid: str | None) -> dict[str, list[Any]]:
    if not raw_grid:
        return {}

    try:
        parsed = json.loads(raw_grid)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "--research-grid must be valid JSON. In PowerShell, escape inner "
            "double quotes, for example: "
            "'{\\\"short_window\\\":[10,20],\\\"long_window\\\":[30,50]}'"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("--research-grid must be a JSON object")

    grid: dict[str, list[Any]] = {}
    for key, values in parsed.items():
        if not isinstance(values, list):
            raise ValueError(f"--research-grid values must be lists: {key}")
        grid[str(key)] = values
    return grid


def _apply_common_overrides(args, config, logger) -> None:
    if getattr(args, "strategy", None):
        config.set("strategy", "active", args.strategy)
        logger.info(f"Overriding strategy: {args.strategy}")

    if getattr(args, "symbol", None):
        symbols = [symbol.strip() for symbol in args.symbol.split(",")]
        config.set("trading", "instruments", symbols)
        logger.info(f"Overriding trading symbols: {symbols}")

    if getattr(args, "timeframe", None):
        config.set("trading", "timeframe", args.timeframe)
        logger.info(f"Overriding timeframe: {args.timeframe}")

    if getattr(args, "start_date", None):
        config.set("backtest", "period", "start", args.start_date)
        logger.info(f"Overriding backtest start date: {args.start_date}")

    if getattr(args, "end_date", None):
        config.set("backtest", "period", "end", args.end_date)
        logger.info(f"Overriding backtest end date: {args.end_date}")


def _build_walk_forward_windows(args):
    start = getattr(args, "research_walk_forward_start", None)
    end = getattr(args, "research_walk_forward_end", None)
    train_days = getattr(args, "research_train_days", None)
    test_days = getattr(args, "research_test_days", None)
    step_days = getattr(args, "research_step_days", None)

    provided = [start, end, train_days, test_days]
    if not any(value is not None for value in provided):
        return None
    if any(value is None for value in provided):
        raise ValueError("Walk-forward research requires start, end, train days, and test days")

    return generate_walk_forward_windows(
        start=datetime.fromisoformat(start),
        end=datetime.fromisoformat(end),
        train_size=timedelta(days=train_days),
        test_size=timedelta(days=test_days),
        step_size=timedelta(days=step_days) if step_days is not None else None,
    )
