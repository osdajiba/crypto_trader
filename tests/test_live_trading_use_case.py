import datetime
import unittest
from unittest.mock import AsyncMock, Mock

from src.domain.models import MarketBar, MarketSlice


class LiveTradingUseCaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_once_processes_latest_market_slice_through_domain_pipeline(self):
        from src.application.live_trading_use_case import LiveTradingUseCase

        timestamp = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
        market = MarketSlice(
            timestamp=timestamp,
            bars_by_symbol={
                "BTC/USDT": MarketBar("BTC/USDT", "1m", timestamp, 100.0, 101.0, 99.0, 100.0, 10.0)
            },
        )
        feed = Mock()
        feed.latest = AsyncMock(return_value=market)
        pipeline = Mock()
        pipeline.run_once = AsyncMock(return_value=Mock(fills=[], snapshot=Mock(equity=100000.0)))
        mode = Mock()
        mode.state = {}
        mode._sync_state_from_portfolio = Mock()

        result = await LiveTradingUseCase(feed, pipeline, mode).run_once(["BTC/USDT"], "1m")

        self.assertEqual(result.snapshot.equity, 100000.0)
        feed.latest.assert_awaited_once_with(["BTC/USDT"], "1m")
        pipeline.run_once.assert_awaited_once_with(market)
        mode._sync_state_from_portfolio.assert_called_once_with()
        self.assertEqual(mode.state["timestamp"], timestamp)


if __name__ == "__main__":
    unittest.main()
