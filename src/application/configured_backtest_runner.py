"""Adapters that turn backtest run specs into real TradingCore runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any

from common.config import ConfigManager
from src.application.backtest_batch import BacktestRunSpec
from src.application.backtest_planning import WalkForwardWindow


def make_configured_backtest_runner(
    base_config: ConfigManager,
    *,
    mode: str = "backtest",
    backtest_engine: str | None = "ohlcv",
    core_factory: Callable[..., Any] | None = None,
    executor_factory: Callable[[], Any] | None = None,
) -> Callable[[BacktestRunSpec], Mapping[str, Any]]:
    """Create a run_backtest(spec) callable backed by TradingCore."""

    def run_backtest(spec: BacktestRunSpec) -> Mapping[str, Any]:
        run_config = _clone_config(base_config)
        _apply_spec_to_config(run_config, spec)

        core_cls = core_factory or _default_core_factory()
        executor = (executor_factory or _default_executor_factory())()
        result = executor.run(core_cls(run_config, mode, backtest_engine).run_pipeline())
        if result is None:
            raise RuntimeError("backtest returned no result")
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(str(result["error"]))

        report = dict(result)
        config_snapshot = _sanitized_config_snapshot(run_config.get_all())
        report["research"] = {
            "run_id": spec.run_id,
            "run_hash": _run_hash(spec),
            "parameters": dict(spec.parameters),
            "window": _window_to_dict(spec.window),
            "config_hash": _hash_payload(config_snapshot),
            "config_snapshot": config_snapshot,
            "artifacts": dict(report.get("artifacts", {})),
        }
        return report

    return run_backtest


def _clone_config(config: ConfigManager) -> ConfigManager:
    cloned = ConfigManager()
    cloned.update(config.get_all())
    return cloned


def _apply_spec_to_config(config: ConfigManager, spec: BacktestRunSpec) -> None:
    for name, value in spec.parameters.items():
        if "." in name:
            config.set(*name.split("."), value=value)
        else:
            active_strategy = config.get("strategy", "active", default="dual_ma")
            config.set("strategy", "parameters", name, value=value)
            config.set("strategy", active_strategy, name, value=value)

    if spec.window is not None:
        config.set("backtest", "period", "start", value=_date_string(spec.window.test_start))
        config.set("backtest", "period", "end", value=_date_string(spec.window.test_end))


def _date_string(value) -> str:
    return value.date().isoformat()


def _window_to_dict(window: WalkForwardWindow | None) -> dict[str, str] | None:
    if window is None:
        return None
    return {
        "train_start": window.train_start.isoformat(),
        "train_end": window.train_end.isoformat(),
        "test_start": window.test_start.isoformat(),
        "test_end": window.test_end.isoformat(),
    }


def _run_hash(spec: BacktestRunSpec) -> str:
    return _hash_payload({
        "run_id": spec.run_id,
        "parameters": spec.parameters,
        "window": _window_to_dict(spec.window),
    })


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _sanitized_config_snapshot(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: "<redacted>" if _is_sensitive_key(str(key)) else _sanitized_config_snapshot(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitized_config_snapshot(item) for item in value]
    return value


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in ("api_key", "secret", "password", "token"))


def _default_core_factory():
    from src.core.core import TradingCore

    return TradingCore


def _default_executor_factory():
    from src.common.async_executor import AsyncExecutor

    return AsyncExecutor
