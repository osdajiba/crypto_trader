# src/mode/base_trading_mode.py

from abc import ABC, abstractmethod
import asyncio
from typing import Dict, Optional, Any, Type, List

from common.config import ConfigManager
from common.logging import LogManager
from src.common.abstract_factory import AbstractFactory, register_factory_class
from src.common.helpers import TradingMode
from datasource.manager import DataManager
from risk.manager import RiskManagerFactory
from backtest.performance import PerformanceMonitor
from strategy.base import StrategyFactory
from src.domain.portfolio import PortfolioBook
from src.application.report_use_case import TradingReportUseCase


class BaseTradingMode(ABC):
    """Trading mode base class, defines the common interface for all trading modes"""
    
    def __init__(
        self, 
        config: ConfigManager,
        data_manager: Optional[DataManager] = None,
        strategy_factory: Optional[StrategyFactory] = None,
        risk_manager: Optional[RiskManagerFactory] = None,
        performance_monitor: Optional[PerformanceMonitor] = None
    ):
        """
        Initialize the trading mode base class
        
        Args:
            config: Configuration manager
            data_manager: Data manager, will create new instance if None
            strategy_factory: Strategy factory, will create new instance if None
            risk_manager: Risk manager, will create new instance if None
            performance_monitor: Performance monitor, will create new instance if None
        """
        self.config = config
        # Initialize logger
        self.logger = LogManager.get_logger(f"mode.{self.__class__.__name__.lower()}")
        self._running = False
        
        # Create new component instances if not provided
        self.data_manager = data_manager or self._create_data_manager()
        self.strategy_factory = strategy_factory or self._create_strategy_factory()
        self.risk_manager = risk_manager or self._create_risk_manager()
        self.performance_monitor = performance_monitor or self._create_performance_monitor()
        
        # Common state tracking
        self.state = {
            'timestamp': None,
            'cash': 0,
            'positions': {},
            'trades': [],
            'equity_curve': [],
            'market_prices': {},
            'current_equity': 0,
            'peak_equity': 0,
            'max_drawdown': 0.0,
            'processed_signals': set()
        }
        
        self.strategy = None
        self.portfolio_book = PortfolioBook(initial_cash=0)
        
        self.logger.info(f"Initializing {self.mode_name} trading mode")
    
    def _create_data_manager(self) -> DataManager:
        """Create data manager"""
        source_type = self._get_source_type()
        return DataManager(
            source_type=source_type,
            config=self.config
        )
        
    def _create_strategy_factory(self) -> StrategyFactory:
        """Create strategy factory"""
        return StrategyFactory(config=self.config)
    
    def _create_risk_manager(self) -> RiskManagerFactory:
        """Create risk manager"""
        return RiskManagerFactory.create_risk_manager(self.mode_name, self.config)
        
    def _create_performance_monitor(self) -> PerformanceMonitor:
        """Create performance monitor"""
        return PerformanceMonitor(config=self.config)
    
    def _get_source_type(self) -> str:
        """Get data source type based on mode"""
        if self.mode_name == "live":
            return "exchange"
        return "local"
    
    @property
    def mode_name(self) -> str:
        """Get mode name derived from class name"""
        return self.__class__.__name__.replace('TradingMode', '').lower()
    
    @property
    def is_running(self) -> bool:
        """Check if the mode is currently running"""
        return self._running
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize mode specific components
        
        This method should be implemented by subclasses to perform mode-specific
        initialization. A subclass implementation should typically:
        1. Initialize the strategy
        2. Set up any mode-specific state
        3. Prepare runtime dependencies through the application layer
        """
        pass
    
    async def run(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """
        Run the trading mode - Template Method pattern implementation
        
        Args:
            symbols: List of trading symbols
            timeframe: Time period for the data
            
        Returns:
            Dict: Results of the run
        """
        self.logger.info(f"Starting {self.mode_name} mode | Symbols: {symbols} | Timeframe: {timeframe}")
        
        try:
            # 1. Pre-run preparation
            await self._pre_run_setup(symbols, timeframe)
            
            # 2. Main trading loop
            result = await self._execute_trading_loop(symbols, timeframe)
            
            # 3. Generate report
            report = self._generate_report()
            
            # 4. Save report
            self._save_report(report)
            
            return report
            
        except asyncio.CancelledError:
            self.logger.warning(f"{self.mode_name} mode cancelled")
            return {"status": "cancelled"}
            
        except Exception as e:
            self.logger.error(f"{self.mode_name} mode error: {e}", exc_info=True)
            return {"error": str(e)}
            
        finally:
            await self.shutdown()
    
    async def _pre_run_setup(self, symbols: List[str], timeframe: str) -> None:
        """
        Prepare for running (common pre-run logic)
        
        Args:
            symbols: List of trading symbols
            timeframe: Time period
        """
        # Initialize default state
        self._init_state()
        
        # Additional preparation can be done in subclasses
        await self._prepare_run(symbols, timeframe)
    
    def _init_state(self) -> None:
        """Initialize the trading state with default values"""
        initial_capital = self.config.get(
            self.mode_name, "initial_capital", 
            default=self.config.get("default_config", "user_config", "initial_cash", default=100000)
        )
        
        self.state.update({
            'timestamp': None,
            'cash': initial_capital,
            'positions': {},
            'trades': [],
            'equity_curve': [],
            'market_prices': {},
            'current_equity': initial_capital,
            'peak_equity': initial_capital,
            'max_drawdown': 0.0,
            'processed_signals': set()
        })
        self.portfolio_book = PortfolioBook(initial_capital)
        self._sync_state_from_portfolio()
        
        self.logger.info(f"State initialized with {initial_capital} initial capital")
    
    @abstractmethod
    async def _prepare_run(self, symbols: List[str], timeframe: str) -> None:
        """
        Mode-specific preparation steps
        
        Args:
            symbols: List of trading symbols
            timeframe: Time period
        """
        pass
    
    @abstractmethod
    async def _execute_trading_loop(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """
        Execute the main trading loop
        
        Args:
            symbols: List of trading symbols
            timeframe: Time period
            
        Returns:
            Dict: Trading results
        """
        pass
    
    def _update_performance_metrics(self) -> None:
        """
        Update equity curve and drawdown
        """
        self.portfolio_book.record_equity(self.state['timestamp'])
        self._sync_state_from_portfolio()
    
    def _calculate_equity(self) -> float:
        """
        Calculate current equity
        
        Returns:
            float: Current equity value
        """
        return self.portfolio_book.calculate_equity()

    def _sync_state_from_portfolio(self) -> None:
        # 兼容旧代码读取 self.state 的方式，同时把真实账本收敛到 PortfolioBook。
        self.state['cash'] = self.portfolio_book.cash
        self.state['positions'] = self.portfolio_book.positions
        self.state['trades'] = self.portfolio_book.trades
        self.state['equity_curve'] = self.portfolio_book.equity_curve
        self.state['market_prices'] = self.portfolio_book.market_prices
        self.state['current_equity'] = self.portfolio_book.current_equity
        self.state['peak_equity'] = self.portfolio_book.peak_equity
        self.state['max_drawdown'] = self.portfolio_book.max_drawdown
    
    def _generate_report(self) -> Dict[str, Any]:
        """
        Generate performance report
        
        Returns:
            Dict: Performance report
        """
        # 报告生成已经从 mode 抽到应用层；这里保留旧方法名，降低外部调用方迁移成本。
        return TradingReportUseCase(self).generate()

    def _trade_action_value(self, trade: Dict[str, Any]) -> str:
        # 旧测试和旧报告路径仍会调用该方法，暂时作为报告用例的薄代理保留。
        return TradingReportUseCase(self)._trade_action_value(trade)
    
    def _add_mode_specific_metrics(self, report: Dict[str, Any]) -> None:
        """
        Add mode-specific metrics to report
        
        Args:
            report: Performance report to update
        """
        # To be overridden by subclasses if needed
        pass
    
    def _save_report(self, report: Dict[str, Any]) -> None:
        """
        Save trading report to files
        
        Args:
            report: Trading performance report
        """
        # 报告写盘逻辑委托给 TradingReportUseCase，mode 不再关心文件命名和 JSON 清洗细节。
        TradingReportUseCase(self).save(report)
    
    def _prepare_report_for_serialization(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare report for JSON serialization
        
        Args:
            report: Original report
            
        Returns:
            Dict: Serializable report
        """
        # 兼容旧入口：实际序列化清洗规则集中在报告用例里维护。
        return TradingReportUseCase(self).prepare_for_serialization(report)
    
    def _log_report_summary(self, report: Dict[str, Any]) -> None:
        """
        Log report summary
        
        Args:
            report: Performance report
        """
        # 兼容旧入口：日志摘要由报告用例统一生成，避免字段解释散落在 mode 层。
        TradingReportUseCase(self).log_summary(report)
    
    def _should_continue(self) -> bool:
        """
        Check if the mode should continue running
        
        Returns:
            bool: True if should continue, False otherwise
        """
        return self._running and not self.risk_manager.is_risk_breached()
    
    async def _sleep_interval(self, interval: Optional[int] = None) -> None:
        """
        Wait for the specified time
        
        Args:
            interval: Wait time in seconds, if None uses config value
        """
        if interval is None:
            interval = self.config.get("data", "polling_interval", default=60)
        
        if interval > 0:
            await asyncio.sleep(interval)
    
    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shutdown mode specific components
        
        This method should be implemented by subclasses to perform mode-specific
        cleanup. A subclass implementation should typically:
        1. Close/shutdown the strategy
        2. Close/shutdown the execution engine
        3. Clean up any mode-specific resources
        """
        pass

class TradingModeFactory(AbstractFactory):
    """Optimized factory for trading modes"""
    
    def __init__(self, config):
        """
        Initialize trading mode factory
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        
        # Register built-in trading modes
        self._register_default_modes()
        
        # Auto-discover additional trading modes
        self._discover_trading_modes()
    
    def _register_default_modes(self):
        """Register default trading modes"""
        self.register(TradingMode.BACKTEST.value, "src.trading.modes.backtest.BacktestTradingMode", {
            "description": "Historical data backtesting",
            "features": ["historical_data", "performance_analysis"]
        })
        self.register(TradingMode.PAPER.value, "src.trading.modes.paper.PaperTradingMode", {
            "description": "Paper trading (uses real market data without real funds)",
            "features": ["real_time_data", "virtual_execution"]
        })
        self.register(TradingMode.LIVE.value, "src.trading.modes.live.LiveTradingMode", {
            "description": "Live trading (uses real funds on exchange)",
            "features": ["real_time_data", "real_execution", "risk_management"]
        })
    
    def _discover_trading_modes(self):
        """Auto-discover trading mode modules"""
        try:
            mode_dir = "src.trading.modes"
            self.discover_registrable_classes(BaseTradingMode, mode_dir, "trading_mode_factory")
        except Exception as e:
            self.logger.error(f"Error auto-discovering trading modes: {e}")
    
    async def _get_concrete_class(self, name: str) -> Type[BaseTradingMode]:
        """
        Get trading mode class
        
        Args:
            name: Trading mode name
            
        Returns:
            Type[BaseTradingMode]: Trading mode class
        """
        return await self._load_class_from_path(name, BaseTradingMode)
    
    async def _resolve_name(self, name: Optional[str]) -> str:
        """
        Validate and resolve trading mode name
        
        Args:
            name: Trading mode name
            
        Returns:
            str: Resolved trading mode name
        """
        # Validate mode type
        try:
            name_value = name.lower() if name else TradingMode.BACKTEST.value
            mode = TradingMode(name_value)
        except ValueError:
            valid_modes = [m.value for m in TradingMode]
            raise ValueError(f"Unsupported trading mode: {name}. Must be one of: {valid_modes}")
            
        # Force LIVE mode if real-time data is enabled
        use_live_data = self.config.get("data", "use_live_data", default=False)
        if use_live_data:
            mode = TradingMode.LIVE
            self.logger.info(f"Overriding mode to {mode.value} due to use_live_data=True")
            
        return mode.value
    
    def get_available_modes(self) -> Dict[str, str]:
        """
        Get available trading modes
        
        Returns:
            Dict[str, str]: Mode names with descriptions
        """
        result = {}
        for name, info in self.get_registered_items().items():
            metadata = info.get('metadata', {})
            description = metadata.get('description', '')
            result[name] = description
        
        # Ensure basic enum modes are included
        for mode in TradingMode:
            if mode.value not in result:
                result[mode.value] = self._get_default_description(mode)
                
        return result
    
    def _get_default_description(self, mode: TradingMode) -> str:
        """
        Get default description for trading mode
        
        Args:
            mode: Trading mode enum
            
        Returns:
            str: Default description
        """
        descriptions = {
            TradingMode.BACKTEST: "Historical data backtesting",
            TradingMode.PAPER: "Paper trading (uses real market data without real funds)",
            TradingMode.LIVE: "Live trading (uses real funds on exchange)"
        }
        return descriptions.get(mode, "Unknown trading mode")
    
    def get_mode_features(self, mode_name: str) -> List[str]:
        """
        Get features of a trading mode
        
        Args:
            mode_name: Trading mode name
            
        Returns:
            List[str]: List of features
        """
        metadata = self._metadata.get(mode_name.lower(), {})
        return metadata.get('features', [])


# Example of using decorator in trading mode implementation
@register_factory_class('trading_mode_factory', 'custom_mode', 
                       description="Custom trading mode",
                       features=["feature1", "feature2"])
class CustomTradingMode(BaseTradingMode):
    """
    Custom trading mode demonstrating automatic registration
    
    Note: This class should be defined in a separate file, shown here as an example
    """
    pass
