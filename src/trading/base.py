#!/usr/bin/env python3
# src/trading/modes/base.py

from abc import ABC, abstractmethod
import asyncio
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, Type, List
from pathlib import Path

from src.common.config import ConfigManager
from src.common.log_manager import LogManager
from src.common.abstract_factory import AbstractFactory, register_factory_class
from src.common.helpers import TradingMode
from src.datasource.manager import DataManager
from src.portfolio.risk import RiskManagerFactory
from src.portfolio.performance import PerformanceManager
from src.strategy.base import StrategyFactory
from src.portfolio.execution.order import Direction


class BaseTradingMode(ABC):
    """Trading mode base class, defines the common interface for all trading modes"""
    
    def __init__(
        self, 
        config: ConfigManager,
        data_manager: Optional[DataManager] = None,
        strategy_factory: Optional[StrategyFactory] = None,
        risk_manager: Optional[RiskManagerFactory] = None,
        performance_monitor: Optional[PerformanceManager] = None
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
            'max_drawdown': 0.0
        }
        
        self.execution_engine = None
        self.strategy = None
        
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
        
    def _create_performance_monitor(self) -> PerformanceManager:
        """Create performance monitor"""
        return PerformanceManager(config=self.config)
    
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
        1. Create an execution engine appropriate for the mode
        2. Initialize the strategy
        3. Set up any mode-specific state
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
            'max_drawdown': 0.0
        })
        
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
    
    async def _process_market_data(self, data_map: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Process market data and generate signals
        
        Args:
            data_map: Mapping of symbols to market data
            
        Returns:
            List: Executed trades
        """
        executed_trades = []
        
        # Update current time from data
        self._update_timestamp_from_data(data_map)
        
        # Update market prices
        self._update_market_prices(data_map)
        
        # Process each symbol
        for symbol, data in data_map.items():
            if data.empty:
                self.logger.warning(f"No data for {symbol}")
                continue
                
            # Generate signals
            signals = await self.strategy.process_data(data, symbol)
            
            if not signals.empty:
                # Validate signals
                valid_signals = await self.risk_manager.validate_signals(signals)
                
                # Execute signals
                if not valid_signals.empty:
                    trades = await self._execute_signals(valid_signals, data_map)
                    if trades:
                        executed_trades.extend(trades)
        
        # Update equity curve and drawdown
        self._update_performance_metrics()
        
        return executed_trades
    
    def _update_timestamp_from_data(self, data_map: Dict[str, pd.DataFrame]) -> None:
        """
        Update current timestamp from data
        
        Args:
            data_map: Mapping of symbols to market data
        """
        # Find the latest timestamp across all data
        for df in data_map.values():
            if not df.empty and 'datetime' in df.columns:
                latest_time = df['datetime'].iloc[-1]
                self.state['timestamp'] = latest_time
                break
    
    def _update_market_prices(self, data_map: Dict[str, pd.DataFrame]) -> None:
        """
        Update current market prices
        
        Args:
            data_map: Mapping of symbols to market data
        """
        for symbol, df in data_map.items():
            if not df.empty and 'close' in df.columns:
                self.state['market_prices'][symbol] = df['close'].iloc[-1]
    
    async def _execute_signals(self, signals: pd.DataFrame, data_map: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Execute trading signals
        
        Args:
            signals: Trading signals
            data_map: Current market data
            
        Returns:
            List: Executed trades
        """
        if not self.execution_engine:
            self.logger.error("Execution engine not initialized")
            return []
            
        # Execute signals
        executed_trades = []
        
        try:
            # Call execution engine
            executed_orders, _ = await self.execution_engine.execute(signals)
            
            # Process executed orders
            for _, order in executed_orders.iterrows():
                trade = self._process_executed_order(order)
                if trade:
                    executed_trades.append(trade)
                    
            return executed_trades
            
        except Exception as e:
            self.logger.error(f"Error executing signals: {e}")
            return []
    
    def _process_executed_order(self, order: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Process an executed order and update state
        
        Args:
            order: Executed order details
            
        Returns:
            Dict: Trade record or None if invalid
        """
        # Extract order details
        try:
            symbol = order.symbol
            direction = order.direction
            price = order.price
            quantity = order.filled_qty
            # timestamp = order.timestamp
            
            # Calculate commission
            commission_rate = self.config.get(self.mode_name, "commission_rate", 
                                            default=self.config.get("default_config", "user_config", "commission", default=0.001))
            commission = price * quantity * commission_rate
            
            # Update cash and positions
            if direction.value == 'buy':
                # Update cash (deduct cost and commission)
                total_cost = (price * quantity) + commission
                if total_cost > self.state['cash']:
                    self.logger.warning(f"Insufficient cash for {direction} {quantity} {symbol} @ {price}")
                    return None
                    
                self.state['cash'] -= total_cost
                
                # Update position
                if symbol not in self.state['positions']:
                    self.state['positions'][symbol] = 0
                self.state['positions'][symbol] += quantity
                
            elif direction.value == 'sell' or 'short':
                # Check position
                current_position = self.state['positions'].get(symbol, 0)
                
                if current_position < quantity and direction.value == 'sell':
                    self.logger.warning(f"Insufficient position for {direction} {quantity} {symbol}, CANNOT sell without a positive position: current position: {current_position}")
                    return None
                
                # Update position
                self.state['positions'][symbol] -= quantity
                if self.state['positions'][symbol] <= 0:
                    del self.state['positions'][symbol]
                
                # Update cash (add proceeds, subtract commission)
                self.state['cash'] += (price * quantity) - commission
            
            # Create trade record
            trade = {
                'timestamp': self.state['timestamp'],
                'symbol': symbol,
                'action': direction,
                'quantity': quantity,
                'price': price,
                'commission': commission,
                'cash_after': self.state['cash']
            }
            
            # Record the trade
            self.state['trades'].append(trade)
            self.logger.info(f"Executed {direction} {quantity} {symbol} @ {price}")
            
            return trade
            
        except KeyError as e:
            self.logger.error(f"Missing required field in order: {e}")
            return None
    
    def _update_performance_metrics(self) -> None:
        """
        Update equity curve and drawdown
        """
        # Calculate current equity
        equity = self._calculate_equity()
        self.state['current_equity'] = equity
        
        # Update equity curve
        equity_point = {
            'timestamp': self.state['timestamp'],
            'equity': equity
        }
        self.state['equity_curve'].append(equity_point)
        
        # Update peak and drawdown
        if equity > self.state['peak_equity']:
            self.state['peak_equity'] = equity
        
        if self.state['peak_equity'] > 0:
            drawdown = (self.state['peak_equity'] - equity) / self.state['peak_equity']
            self.state['max_drawdown'] = max(self.state['max_drawdown'], drawdown)
    
    def _calculate_equity(self) -> float:
        """
        Calculate current equity
        
        Returns:
            float: Current equity value
        """
        equity = self.state['cash']
        
        # Add position values
        for symbol, quantity in self.state['positions'].items():
            if symbol in self.state['market_prices']:
                price = self.state['market_prices'][symbol]
                equity += quantity * price
        
        return equity
    
    def _generate_report(self) -> Dict[str, Any]:
        """
        Generate performance report
        
        Returns:
            Dict: Performance report
        """
        # Get initial values
        initial_capital = self.config.get(
            self.mode_name, "initial_capital", 
            default=self.config.get("default_config", "user_config", "initial_cash", default=100000)
        )
        
        # Current values
        final_equity = self._calculate_equity()
        total_return = final_equity - initial_capital
        total_return_pct = (total_return / initial_capital * 100) if initial_capital > 0 else 0
        
        # Calculate trade statistics
        trades = self.state['trades']
        buy_trades = len([t for t in trades if t['action'] == 'buy'])
        sell_trades = len([t for t in trades if t['action'] == 'sell'])
        
        # Build report
        report = {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'max_drawdown_pct': self.state['max_drawdown'] * 100,
            'total_trades': len(trades),
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'current_positions': self.state['positions'],
            'remaining_cash': self.state['cash'],
            'trades': trades
        }
        
        # Calculate additional metrics if we have enough data
        equity_curve = self.state['equity_curve']
        if len(equity_curve) > 1:
            # Convert to DataFrame for calculations
            equity_df = pd.DataFrame(equity_curve)
            
            # Calculate Sharpe ratio or other metrics
            if 'equity' in equity_df.columns:
                equity_df['return'] = equity_df['equity'].pct_change()
                
                # Calculate cumulative return
                if not equity_df['return'].empty:
                    equity_df['cumulative_return'] = (1 + equity_df['return']).cumprod() - 1
                    
                    # Calculate Sharpe ratio (assume zero risk-free rate)
                    if equity_df['return'].std() > 0:
                        sharpe_ratio = (equity_df['return'].mean() / equity_df['return'].std()) * (252 ** 0.5)
                        report['sharpe_ratio'] = sharpe_ratio
        
        # Add mode-specific metrics
        self._add_mode_specific_metrics(report)
        
        return report
    
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
        # Create report directory
        report_dir = self.config.get("reporting", f"{self.mode_name}_reports_dir", 
                                    default=f"reports/{self.mode_name}/")
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Determine output formats
        output_formats = self.config.get("reporting", "output_formats", default=["json"])
        
        # Save trade records
        if 'csv' in output_formats and report.get('trades'):
            trades_df = pd.DataFrame(report['trades'])
            trades_csv = os.path.join(report_dir, f"{self.mode_name}_trades_{timestamp}.csv")
            trades_df.to_csv(trades_csv, index=False)
            self.logger.info(f"Trade records saved to {trades_csv}")
        
        # Save equity curve
        if 'csv' in output_formats and report.get('equity_curve'):
            equity_df = pd.DataFrame(report['equity_curve'])
            equity_csv = os.path.join(report_dir, f"{self.mode_name}_equity_curve_{timestamp}.csv")
            equity_df.to_csv(equity_csv, index=False)
            self.logger.info(f"Equity curve saved to {equity_csv}")
        
        # Save as JSON
        if 'json' in output_formats:
            # Prepare report for serialization
            clean_report = self._prepare_report_for_serialization(report)
            
            # Save JSON report
            report_json = os.path.join(report_dir, f"{self.mode_name}_report_{timestamp}.json")
            with open(report_json, 'w') as f:
                json.dump(clean_report, f, indent=4, default=str)
            
            self.logger.info(f"Performance report saved to {report_json}")
        
        # Log summary
        self._log_report_summary(report)
    
    def _prepare_report_for_serialization(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare report for JSON serialization
        
        Args:
            report: Original report
            
        Returns:
            Dict: Serializable report
        """
        # Create a copy to avoid modifying the original
        clean_report = report.copy()
        
        # Handle non-serializable objects
        for key in ['equity_curve', 'strategy']:
            if key in clean_report:
                if key == 'equity_curve':
                    clean_report[key] = str(f"{len(clean_report[key])} records")
                elif key == 'strategy':
                    clean_report[key] = clean_report[key].__class__.__name__
        
        return clean_report
    
    def _log_report_summary(self, report: Dict[str, Any]) -> None:
        """
        Log report summary
        
        Args:
            report: Performance report
        """
        self.logger.info(f"==== {self.mode_name.capitalize()} Performance Summary ====")
        self.logger.info(f"Initial capital: ${report.get('initial_capital', 0):,.2f}")
        self.logger.info(f"Final equity: ${report.get('final_equity', 0):,.2f}")
        self.logger.info(f"Total return: ${report.get('total_return', 0):,.2f} ({report.get('total_return_pct', 0):.2f}%)")
        self.logger.info(f"Max drawdown: {report.get('max_drawdown_pct', 0):.2f}%")
        
        if 'sharpe_ratio' in report:
            self.logger.info(f"Sharpe ratio: {report.get('sharpe_ratio', 0):.2f}")
            
        self.logger.info(f"Total trades: {report.get('total_trades', 0)}")
        self.logger.info("==========================================")
    
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
        self.register(TradingMode.BACKTEST.value, "src.backtest.mode.BacktestMode", {
            "description": "Historical data backtesting",
            "features": ["historical_data", "performance_analysis"]
        })
        self.register(TradingMode.PAPER.value, "src.trading.modes.paper.PaperMode", {
            "description": "Paper trading (uses real market data without real funds)",
            "features": ["real_time_data", "virtual_execution"]
        })
        self.register(TradingMode.LIVE.value, "src.trading.modes.live.LiveMode", {
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
# @register_factory_class('trading_mode_factory', 'custom_mode', 
#                        description="Custom trading mode",
#                        features=["feature1", "feature2"])
# class CustomTradingMode(BaseTradingMode):
#     """
#     Custom trading mode demonstrating automatic registration
    
#     Note: This class should be defined in a separate file, shown here as an example
#     """
#     pass
