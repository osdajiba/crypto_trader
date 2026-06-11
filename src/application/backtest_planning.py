"""Planning helpers for parameter scans and walk-forward backtests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import product
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


def expand_parameter_grid(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Expand a parameter grid while preserving caller-provided key order."""
    if not grid:
        return [{}]

    keys = list(grid.keys())
    values_by_key = []
    for key in keys:
        values = list(grid[key])
        if not values:
            raise ValueError(f"Parameter grid values cannot be empty: {key}")
        values_by_key.append(values)

    return [
        dict(zip(keys, combination))
        for combination in product(*values_by_key)
    ]


def generate_walk_forward_windows(
    *,
    start: datetime,
    end: datetime,
    train_size: timedelta,
    test_size: timedelta,
    step_size: timedelta | None = None,
) -> list[WalkForwardWindow]:
    """Generate half-open walk-forward train/test windows."""
    _require_positive_duration("train_size", train_size)
    _require_positive_duration("test_size", test_size)
    step = step_size if step_size is not None else test_size
    _require_positive_duration("step_size", step)

    if end <= start:
        raise ValueError("end must be after start")

    windows = []
    current = start
    while current + train_size + test_size <= end:
        train_start = current
        train_end = train_start + train_size
        test_end = train_end + test_size
        windows.append(WalkForwardWindow(train_start, train_end, train_end, test_end))
        current += step
    return windows


def _require_positive_duration(name: str, value: timedelta) -> None:
    if value <= timedelta(0):
        raise ValueError(f"{name} must be positive")
