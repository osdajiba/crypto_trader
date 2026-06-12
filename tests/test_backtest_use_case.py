import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pandas as pd

from src.application.backtest_use_case import BacktestUseCase


class BacktestUseCaseTest(unittest.IsolatedAsyncioTestCase):
    async def test_run_processes_market_slices_through_domain_pipeline(self):
        from src.domain.models import MarketBar, MarketSlice

        first_ts = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
        second_ts = pd.Timestamp("2025-01-01 00:01:00", tz="UTC")

        def market_slice(timestamp, close):
            return MarketSlice(
                timestamp=timestamp.to_pydatetime(),
                bars_by_symbol={
                    "BTC/USDT": MarketBar(
                        "BTC/USDT",
                        "1m",
                        timestamp.to_pydatetime(),
                        close,
                        close,
                        close,
                        close,
                        10,
                    )
                },
            )

        mode = Mock()
        mode.domain_pipeline = Mock()
        mode.domain_pipeline.run_once = AsyncMock(side_effect=[
            SimpleNamespace(fills=[], snapshot=SimpleNamespace(equity=100000.0)),
            SimpleNamespace(
                fills=[
                    SimpleNamespace(
                        timestamp=second_ts.to_pydatetime(),
                        symbol="BTC/USDT",
                        side="buy",
                        price=101.0,
                        quantity=0.01,
                        commission=0.00101,
                    )
                ],
                snapshot=SimpleNamespace(equity=99999.0),
            ),
        ])
        mode.market_data_feed = Mock()
        mode.market_data_feed.load_range = AsyncMock(return_value=[
            market_slice(first_ts, 100.0),
            market_slice(second_ts, 101.0),
        ])
        mode.start_date = "2025-01-01"
        mode.end_date = "2025-01-02"
        mode.state = {}
        mode.logger = Mock()
        mode.performance_monitor = Mock()
        mode.performance_monitor.generate_detailed_report.return_value = {"summary": "ok"}
        mode._get_data_at_timestamp = Mock(side_effect=AssertionError("legacy timestamp lookup should not be used"))
        mode._process_market_data = AsyncMock(side_effect=AssertionError("legacy DataFrame pipeline should not be used"))
        mode._sync_state_from_portfolio = Mock()
        mode._should_continue.return_value = True

        result = await BacktestUseCase(mode).run(["BTC/USDT"], "1m")

        self.assertEqual(result, {"summary": "ok"})
        self.assertEqual(mode.domain_pipeline.run_once.await_count, 2)
        mode._get_data_at_timestamp.assert_not_called()
        mode._process_market_data.assert_not_called()
        mode.performance_monitor.record_trade.assert_called_once()
        self.assertEqual(mode.performance_monitor.update_equity_curve.call_count, 2)
        mode.performance_monitor.calculate_performance_metrics.assert_called_once_with()
        mode.performance_monitor.generate_detailed_report.assert_called_once_with()
        self.assertEqual(mode.state["timestamp"], second_ts.to_pydatetime())
        mode.market_data_feed.load_range.assert_awaited_once_with(
            symbols=["BTC/USDT"],
            timeframe="1m",
            start="2025-01-01",
            end="2025-01-02",
        )

    async def test_run_uses_domain_pipeline_when_available(self):
        from src.domain.models import MarketBar, MarketSlice

        timestamp = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
        market_slice = MarketSlice(
            timestamp=timestamp.to_pydatetime(),
            bars_by_symbol={
                "BTC/USDT": MarketBar("BTC/USDT", "1m", timestamp.to_pydatetime(), 100.0, 101.0, 99.0, 100.0, 10.0)
            },
        )

        domain_pipeline = Mock()
        domain_pipeline.run_once = AsyncMock(return_value=Mock(fills=[], snapshot=Mock(equity=100000.0)))
        mode = Mock()
        mode.domain_pipeline = domain_pipeline
        mode.market_data_feed = Mock()
        mode.market_data_feed.load_range = AsyncMock(return_value=[market_slice])
        mode.start_date = "2025-01-01"
        mode.end_date = "2025-01-02"
        mode.state = {}
        mode.logger = Mock()
        mode.performance_monitor = Mock()
        mode.performance_monitor.generate_detailed_report.return_value = {"summary": "ok"}
        mode._process_market_data = AsyncMock(side_effect=AssertionError("legacy DataFrame pipeline should not be used"))
        mode._calculate_equity.return_value = 100000.0
        mode._sync_state_from_portfolio = Mock()
        mode._should_continue.return_value = True

        result = await BacktestUseCase(mode).run(["BTC/USDT"], "1m")

        self.assertEqual(result, {"summary": "ok"})
        domain_pipeline.run_once.assert_awaited_once_with(market_slice)
        mode._process_market_data.assert_not_called()
        mode.performance_monitor.update_equity_curve.assert_called_once_with(timestamp.to_pydatetime(), 100000.0)

    async def test_run_requires_domain_pipeline(self):
        mode = Mock()
        mode.domain_pipeline = None
        mode.state = {}
        mode.logger = Mock()

        with self.assertRaisesRegex(ValueError, "Domain pipeline requires domain_pipeline"):
            await BacktestUseCase(mode).run(["BTC/USDT"], "1m")


if __name__ == "__main__":
    unittest.main()
