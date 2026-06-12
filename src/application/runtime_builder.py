"""运行时组件组装器。

RuntimeBuilder 是应用层的组装边界：mode 提供已经初始化好的配置、策略、
风控和账本，builder 负责把它们连接成 feed、domain pipeline、reporter 和
use case。这样策略层、风控层和执行层通过领域端口协作，而不是在 mode 中
互相耦合。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.application.adapters.legacy_risk_policy_adapter import LegacyRiskPolicyAdapter
from src.application.adapters.legacy_strategy_adapter import LegacyDataFrameStrategyAdapter
from src.application.backtest_use_case import BacktestUseCase
from src.application.live_trading_use_case import LiveTradingUseCase
from src.application.paper_trading_use_case import PaperTradingUseCase
from src.application.reporting import InMemoryReporter
from src.datasource.feeds.market_data_feed import HistoricalMarketDataFeed, RealtimeMarketDataFeed
from src.domain.risk_policies import (
    CompositeRiskPolicy,
    PositionAvailabilityPolicy,
    SellQuantityClampPolicy,
)
from src.domain.trading_pipeline import DomainTradingPipeline
from src.factor.engine import FactorEngine
from src.order.risk_policy import fixed_fraction_sizing_policy, fixed_notional_sizing_policy
from src.strategy.implementations.domain_dual_ma import DomainDualMAStrategy
from src.strategy.implementations.domain_multi_factors import DomainMultiFactorsStrategy
from src.trading.execution.backtest_model import BacktestExecutionModel
from src.trading.execution.live_model import LiveExecutionModel
from src.trading.execution.paper_model import PaperExecutionModel


@dataclass(frozen=True)
class TradingRuntime:
    """单个 mode 运行交易流水线所需的组件集合。"""

    market_data_feed: Any
    domain_pipeline: DomainTradingPipeline
    reporter: InMemoryReporter
    use_case: Any


class RuntimeBuilder:
    """按交易模式组装运行时依赖。"""

    def __init__(self, mode: Any) -> None:
        self.mode = mode
        self.config = mode.config

    def build_backtest_runtime(self, historical_data) -> Optional[TradingRuntime]:
        """组装回测运行时；没有本地历史 store 时返回 None 让旧路径兜底。"""
        market_data_feed = self.create_historical_market_data_feed()
        if market_data_feed is None:
            return None

        reporter = InMemoryReporter()
        pipeline = self._create_domain_pipeline(
            execution_model=BacktestExecutionModel(
                historical_data=historical_data,
                commission=self._default_commission(),
                slippage=self._default_slippage(),
            ),
            reporter=reporter,
        )
        use_case = BacktestUseCase(self.mode)
        return self._attach_runtime(
            TradingRuntime(
                market_data_feed=market_data_feed,
                domain_pipeline=pipeline,
                reporter=reporter,
                use_case=use_case,
            ),
            use_case_attr="backtest_use_case",
        )

    def build_paper_runtime(self) -> TradingRuntime:
        """组装模拟盘运行时；执行模型只产生模拟 Fill，不触达真实下单能力。"""
        market_data_feed = RealtimeMarketDataFeed(self.mode.data_manager)
        reporter = InMemoryReporter()
        pipeline = self._create_domain_pipeline(
            execution_model=PaperExecutionModel(
                commission=self._paper_commission(),
                slippage=self._paper_slippage(),
            ),
            reporter=reporter,
        )
        use_case = PaperTradingUseCase(market_data_feed, pipeline, self.mode)
        return self._attach_runtime(
            TradingRuntime(
                market_data_feed=market_data_feed,
                domain_pipeline=pipeline,
                reporter=reporter,
                use_case=use_case,
            ),
            use_case_attr="paper_use_case",
        )

    def build_live_runtime(self) -> TradingRuntime:
        """组装实盘运行时；调用方必须先完成 live 安全门和交易所初始化。"""
        exchange_client = self._exchange_client()
        if exchange_client is None:
            raise ValueError("Live execution requires an exchange client")

        market_data_feed = RealtimeMarketDataFeed(self.mode.data_manager)
        reporter = InMemoryReporter()
        pipeline = self._create_domain_pipeline(
            execution_model=LiveExecutionModel(exchange_client),
            reporter=reporter,
        )
        use_case = LiveTradingUseCase(market_data_feed, pipeline, self.mode)
        return self._attach_runtime(
            TradingRuntime(
                market_data_feed=market_data_feed,
                domain_pipeline=pipeline,
                reporter=reporter,
                use_case=use_case,
            ),
            use_case_attr="live_use_case",
        )

    def _exchange_client(self):
        direct_client = vars(self.mode).get("exchange_client")
        if direct_client is not None:
            return direct_client
        return None

    def create_historical_market_data_feed(self):
        """从本地数据源提取 historical_store 并创建领域行情 feed。"""
        local_source = getattr(self.mode.data_manager, "primary_source", None)
        historical_store = getattr(local_source, "historical_store", None)
        if historical_store is None:
            return None
        return HistoricalMarketDataFeed(historical_store)

    def _create_domain_pipeline(self, execution_model, reporter: InMemoryReporter) -> DomainTradingPipeline:
        """把策略、风控、执行、账本和 reporter 连接成统一领域流水线。"""
        return DomainTradingPipeline(
            strategy=self._create_strategy_port(),
            risk_policy=self._create_risk_policy(),
            execution_model=execution_model,
            portfolio=self.mode.portfolio_book,
            reporter=reporter,
        )

    def _create_strategy_port(self):
        if self._strategy_interface() != "domain":
            return LegacyDataFrameStrategyAdapter(
                self.mode.strategy,
                strategy_id=self.mode.strategy.__class__.__name__,
            )

        strategy_name = self._strategy_name()
        if strategy_name == "dual_ma":
            return DomainDualMAStrategy(
                short_window=self._strategy_parameter("short_window", 20),
                long_window=self._strategy_parameter("long_window", 50),
            )
        if strategy_name == "multi_factors":
            return DomainMultiFactorsStrategy(
                threshold=float(self._strategy_parameter("threshold", 0.5)),
                factor_engine=FactorEngine.from_config(self._factor_config()),
            )

        raise ValueError(f"Unsupported domain strategy: {strategy_name}")

    def _create_risk_policy(self) -> CompositeRiskPolicy:
        if self._strategy_interface() != "domain":
            return CompositeRiskPolicy([
                LegacyRiskPolicyAdapter(self.mode.risk_manager),
                PositionAvailabilityPolicy(),
                SellQuantityClampPolicy(),
            ])

        return CompositeRiskPolicy([
            PositionAvailabilityPolicy(),
            self._create_sizing_policy(),
        ])

    def _create_sizing_policy(self):
        commission_rate = self._default_commission()
        fixed_notional = self.config.get("trading", "position_sizing", "notional", default=None)
        if fixed_notional is not None:
            return fixed_notional_sizing_policy(
                notional=float(fixed_notional),
                commission_rate=commission_rate,
            )

        fraction = self.config.get("trading", "position_sizing", "fraction", default=0.01)
        return fixed_fraction_sizing_policy(
            fraction=float(fraction),
            commission_rate=commission_rate,
        )

    def _strategy_interface(self) -> str:
        return str(self.config.get("strategy", "interface", default="domain")).lower()

    def _strategy_name(self) -> str:
        return str(
            self.config.get(
                "strategy",
                "active",
                default=self.config.get("strategy", "default", default="dual_ma"),
            )
        )

    def _strategy_parameter(self, name: str, default):
        active = self._strategy_name()
        return self.config.get(
            "strategy",
            "parameters",
            name,
            default=self.config.get("strategy", active, name, default=default),
        )

    def _factor_config(self):
        active = self._strategy_name()
        configured = self.config.get("strategy", "factors", default=None)
        if configured:
            return configured
        return self.config.get("strategy", active, "factors", default={})

    def _attach_runtime(self, runtime: TradingRuntime, use_case_attr: str) -> TradingRuntime:
        """写回 mode 兼容属性，让旧报告、测试和外部调用仍能读取。"""
        self.mode.market_data_feed = runtime.market_data_feed
        self.mode.domain_pipeline = runtime.domain_pipeline
        self.mode.domain_reporter = runtime.reporter
        setattr(self.mode, use_case_attr, runtime.use_case)
        return runtime

    def _default_commission(self) -> float:
        return self.config.get("default_config", "user_config", "commission", default=0.001)

    def _default_slippage(self) -> float:
        return self.config.get("default_config", "user_config", "slippage", default=0.001)

    def _paper_commission(self) -> float:
        return self.config.get(
            "paper_trading",
            "commission_rate",
            default=self._default_commission(),
        )

    def _paper_slippage(self) -> float:
        return self.config.get(
            "paper_trading",
            "slippage",
            default=self._default_slippage(),
        )
