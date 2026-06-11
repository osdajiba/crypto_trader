import unittest
from datetime import datetime, timezone

from src.domain.models import Fill, MarketBar, MarketSlice, RiskDecision, StrategySignal
from src.domain.portfolio import PortfolioBook
from src.domain.trading_pipeline import DomainTradingPipeline


class FakeStrategy:
    async def generate(self, market, portfolio):
        return [
            StrategySignal(
                signal_id="fake:BTC/USDT:buy:1",
                symbol="BTC/USDT",
                timestamp=market.timestamp,
                side="buy",
                reason="test signal",
            )
        ]


class FakeRiskPolicy:
    async def evaluate(self, signal, portfolio, market):
        return RiskDecision(
            accepted=True,
            reason="accepted",
            target_notional=1000,
            target_quantity=0.01,
            adjusted_signal=signal,
        )


class FakeExecutionModel:
    async def execute(self, order_intent, market, portfolio):
        return Fill(
            fill_id=f"fill:{order_intent.order_intent_id}",
            order_intent_id=order_intent.order_intent_id,
            symbol=order_intent.symbol,
            side=order_intent.side,
            timestamp=market.timestamp,
            quantity=order_intent.quantity,
            price=market.bars_by_symbol[order_intent.symbol].close,
            commission=1.0,
        )


class FakeReporter:
    def __init__(self):
        self.snapshots = []
        self.fills = []

    async def record_snapshot(self, snapshot):
        self.snapshots.append(snapshot)

    async def record_fill(self, fill):
        self.fills.append(fill)


class DomainTradingPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_once_applies_fill_and_records_report_events(self):
        timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        market = MarketSlice(
            timestamp=timestamp,
            bars_by_symbol={
                "BTC/USDT": MarketBar(
                    symbol="BTC/USDT",
                    timeframe="1m",
                    timestamp=timestamp,
                    open=100000,
                    high=100500,
                    low=99500,
                    close=100000,
                    volume=10,
                )
            },
        )
        portfolio = PortfolioBook(initial_cash=100000)
        reporter = FakeReporter()
        pipeline = DomainTradingPipeline(
            strategy=FakeStrategy(),
            risk_policy=FakeRiskPolicy(),
            execution_model=FakeExecutionModel(),
            portfolio=portfolio,
            reporter=reporter,
        )

        result = await pipeline.run_once(market)

        self.assertEqual(len(result.fills), 1)
        self.assertEqual(portfolio.positions["BTC/USDT"], 0.01)
        self.assertEqual(portfolio.cash, 98999.0)
        self.assertEqual(result.snapshot.equity, 99999.0)
        self.assertEqual(reporter.fills, result.fills)
        self.assertEqual(reporter.snapshots[-1], result.snapshot)

    async def test_rejected_risk_decision_does_not_execute(self):
        class RejectingRiskPolicy:
            async def evaluate(self, signal, portfolio, market):
                return RiskDecision(
                    accepted=False,
                    reason="blocked",
                    target_notional=0,
                    target_quantity=0,
                    adjusted_signal=signal,
                )

        class CountingExecutionModel(FakeExecutionModel):
            def __init__(self):
                self.calls = 0

            async def execute(self, order_intent, market, portfolio):
                self.calls += 1
                return await super().execute(order_intent, market, portfolio)

        timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        market = MarketSlice(
            timestamp=timestamp,
            bars_by_symbol={
                "BTC/USDT": MarketBar("BTC/USDT", "1m", timestamp, 1, 1, 1, 1, 1)
            },
        )
        execution = CountingExecutionModel()
        pipeline = DomainTradingPipeline(
            strategy=FakeStrategy(),
            risk_policy=RejectingRiskPolicy(),
            execution_model=execution,
            portfolio=PortfolioBook(initial_cash=100000),
            reporter=FakeReporter(),
        )

        result = await pipeline.run_once(market)

        self.assertEqual(result.fills, [])
        self.assertEqual(execution.calls, 0)

    async def test_ignores_signals_that_do_not_match_current_market_timestamp(self):
        class StaleAndCurrentStrategy:
            async def generate(self, market, portfolio):
                stale_timestamp = datetime(2024, 12, 31, tzinfo=timezone.utc)
                return [
                    StrategySignal(
                        signal_id="stale:BTC/USDT:buy:1",
                        symbol="BTC/USDT",
                        timestamp=stale_timestamp,
                        side="buy",
                    ),
                    StrategySignal(
                        signal_id="current:BTC/USDT:buy:1",
                        symbol="BTC/USDT",
                        timestamp=market.timestamp,
                        side="buy",
                    ),
                ]

        class CountingExecutionModel(FakeExecutionModel):
            def __init__(self):
                self.order_ids = []

            async def execute(self, order_intent, market, portfolio):
                self.order_ids.append(order_intent.source_signal_id)
                return await super().execute(order_intent, market, portfolio)

        timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        market = MarketSlice(
            timestamp=timestamp,
            bars_by_symbol={
                "BTC/USDT": MarketBar("BTC/USDT", "1m", timestamp, 100000, 100000, 100000, 100000, 10)
            },
        )
        execution = CountingExecutionModel()
        pipeline = DomainTradingPipeline(
            strategy=StaleAndCurrentStrategy(),
            risk_policy=FakeRiskPolicy(),
            execution_model=execution,
            portfolio=PortfolioBook(initial_cash=100000),
            reporter=FakeReporter(),
        )

        result = await pipeline.run_once(market)

        self.assertEqual(len(result.fills), 1)
        self.assertEqual(execution.order_ids, ["current:BTC/USDT:buy:1"])


if __name__ == "__main__":
    unittest.main()
