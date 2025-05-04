#!/usr/bin/env python3
# src/trading/base.py

from abc import ABC, abstractmethod
import asyncio
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, List

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.portfolio.manager import PortfolioManager
from src.strategy.factory import get_strategy_factory
from src.datasource.sources.factory import get_datasource_factory
from src.portfolio.risk.factory import get_risk_factory
from src.portfolio.execution.factory import get_execution_factory
from src.strategy.performance.factory import get_analyzer_factory


class TradingModeError(Exception):
    """Base exception for trading mode errors"""
    pass


class TradingModeInitError(TradingModeError):
    """Error during trading mode initialization"""
    pass


class TradingModeExecutionError(TradingModeError):
    """Error during trading mode execution"""
    pass


class BaseTradingMode(ABC):
    """Base class for all trading modes"""
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """Initialize trading mode"""
        self.config = config
        self.params = params or {}
        
        self.logger = LogManager.get_logger(f"mode.{self.__class__.__name__.lower()}")
        self.portfolio = PortfolioManager(config)
        self._running = False
        
        # Component factories
        self.datasource_factory = get_datasource_factory(config)
        self.risk_factory = get_risk_factory(config)
        self.performance_factory = get_analyzer_factory(config)
        self.execution_factory = get_execution_factory(config)  # Use the new factory
        self.strategy_factory = get_strategy_factory(config)
          
        # Components will be initialized later
        self.data_source = None
        self.risk_manager = None
        self.performance_analyzer = None
        self.execution_engine = None
        self.strategy = None
        
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
        
        self.logger.info(f"Creating {self.mode_name} trading mode instance")
    
    async def initialize(self) -> None:
        """Initialize trading mode components"""
        try:
            # Create common component instances
            await self._initialize_components()
            
            # Mode-specific initialization
            await self._initialize_mode_specific()
            
            # Mark initialization complete
            self._running = True
            self.logger.info(f"{self.mode_name} mode initialization complete")
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.mode_name} mode: {e}")
            raise TradingModeInitError(f"Trading mode initialization failed: {str(e)}")
    
    async def _initialize_components(self) -> None:
        """Initialize common components"""
        resolved_name = self.config.get("system", "operational_mode", default="backtest")
                    
        # Create performance analyzer
        initial_capital = self._get_initial_capital()
        perf_params = {"initial_balance": initial_capital}
        
        # Create data source        
        source_type = self._get_source_type()
        self.data_source = await self.datasource_factory.create(source_type)
        if self.data_source:
            self.logger.info(f"Registered {resolved_name} data source")
                    
        # Create risk manager
        self.risk_manager = await self.risk_factory.create_with_config_params(self.portfolio)
        if self.risk_manager:
            self.logger.info(f"Registered {resolved_name} risk managers")
            
        # Create performance analyzer with appropriate mode
        self.performance_analyzer = await self.performance_factory.create(params=perf_params)
        if self.performance_analyzer:
            self.logger.info(f"Registered {resolved_name} performance analyzer")
        
    @abstractmethod
    async def _initialize_mode_specific(self) -> None:
        """Initialize mode-specific components"""
        pass
    
    def _get_source_type(self) -> str:
        """Get data source type based on mode"""
        if self.mode_name == "live":
            return "exchange"
        elif self.mode_name == "paper":
            return "hybrid"
        return "local"
    
    def _get_initial_capital(self) -> float:
        """Get initial capital from configuration"""
        return self.config.get(
            self.mode_name, "initial_capital", 
            default=self.config.get("trading", "capital", "initial", default=100000)
        )
    
    @property
    def mode_name(self) -> str:
        """Get mode name derived from class name"""
        return self.__class__.__name__.replace('Mode', '').lower()
    
    @property
    def is_running(self) -> bool:
        """Check if the mode is currently running"""
        return self._running
    
    async def run(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Run the trading mode - Template Method pattern implementation"""
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
            raise TradingModeExecutionError(f"Trading mode execution failed: {str(e)}")
            
        finally:
            await self.shutdown()
    
    async def _pre_run_setup(self, symbols: List[str], timeframe: str) -> None:
        """Prepare for running (common pre-run logic)"""
        self._init_state()
        await self._prepare_run(symbols, timeframe)
    
    def _init_state(self) -> None:
        """Initialize the trading state with default values"""
        initial_capital = self._get_initial_capital()
        
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
        """Mode-specific preparation steps"""
        pass
    
    @abstractmethod
    async def _execute_trading_loop(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Execute the main trading loop"""
        pass
    
    async def _process_market_data(self, data_map: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Process market data and generate signals"""
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
        """Update current timestamp from data"""
        for df in data_map.values():
            if not df.empty and 'timestamp' in df.columns:
                latest_time = df['timestamp'].iloc[-1]
                self.state['timestamp'] = latest_time
                break
    
    def _update_market_prices(self, data_map: Dict[str, pd.DataFrame]) -> None:
        """Update current market prices"""
        for symbol, df in data_map.items():
            if not df.empty and 'close' in df.columns:
                self.state['market_prices'][symbol] = df['close'].iloc[-1]
    
    async def _execute_signals(self, signals: pd.DataFrame, data_map: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Execute trading signals"""
        if not self.execution_engine:
            self.logger.error("Execution engine not initialized")
            return []
            
        executed_trades = []
        
        try:
            # Execute signals using the execution engine
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
        """Process an executed order and update state"""
        try:
            symbol = order.get('symbol')
            direction = order.get('direction')
            price = order.get('price', 0) or order.get('avg_price', 0)
            quantity = order.get('filled_qty', 0)
            
            # Skip if invalid data
            if not symbol or not direction or quantity <= 0:
                return None
            
            # Calculate commission
            commission = order.get('commission', 0)
            if commission == 0:
                commission_rate = self.config.get(self.mode_name, "commission_rate", 
                    default=self.config.get("trading", "commission", default=0.001))
                commission = price * quantity * commission_rate
            
            # Update cash and positions
            if direction == 'buy':
                # Update cash
                total_cost = (price * quantity) + commission
                if total_cost > self.state['cash']:
                    self.logger.warning(f"Insufficient cash for {direction} {quantity} {symbol} @ {price}")
                    return None
                    
                self.state['cash'] -= total_cost
                
                # Update position
                if symbol not in self.state['positions']:
                    self.state['positions'][symbol] = 0
                self.state['positions'][symbol] += quantity
                
            elif direction == 'sell' or direction == 'short':
                # Check position
                current_position = self.state['positions'].get(symbol, 0)
                
                if current_position < quantity and direction == 'sell':
                    self.logger.warning(f"Insufficient position for {direction} {quantity} {symbol}, current position: {current_position}")
                    return None
                
                # Update position
                if symbol not in self.state['positions']:
                    self.state['positions'][symbol] = 0
                    
                self.state['positions'][symbol] -= quantity
                if self.state['positions'][symbol] <= 0 and direction == 'sell':
                    del self.state['positions'][symbol]
                
                # Update cash
                self.state['cash'] += (price * quantity) - commission
            
            # Create trade record
            trade = {
                'timestamp': self.state['timestamp'] or order.get('timestamp'),
                'symbol': symbol,
                'action': direction,
                'quantity': quantity,
                'price': price,
                'commission': commission,
                'cash_after': self.state['cash'],
                'order_id': order.get('order_id'),
                'status': order.get('status', 'filled')
            }
            
            # Record the trade
            self.state['trades'].append(trade)
            self.logger.info(f"Executed {direction} {quantity} {symbol} @ {price}")
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Error processing order: {e}")
            return None
    
    def _update_performance_metrics(self) -> None:
        """Update equity curve and drawdown"""
        equity = self._calculate_equity()
        self.state['current_equity'] = equity
        
        equity_point = {
            'timestamp': self.state['timestamp'],
            'equity': equity
        }
        self.state['equity_curve'].append(equity_point)
        
        if equity > self.state['peak_equity']:
            self.state['peak_equity'] = equity
        
        if self.state['peak_equity'] > 0:
            drawdown = (self.state['peak_equity'] - equity) / self.state['peak_equity']
            self.state['max_drawdown'] = max(self.state['max_drawdown'], drawdown)
    
    def _calculate_equity(self) -> float:
        """Calculate current equity"""
        equity = self.state['cash']
        
        for symbol, quantity in self.state['positions'].items():
            if symbol in self.state['market_prices']:
                price = self.state['market_prices'][symbol]
                equity += quantity * price
        
        return equity
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        initial_capital = self._get_initial_capital()
        
        final_equity = self._calculate_equity()
        total_return = final_equity - initial_capital
        total_return_pct = (total_return / initial_capital * 100) if initial_capital > 0 else 0
        
        trades = self.state['trades']
        buy_trades = len([t for t in trades if t['action'] == 'buy'])
        sell_trades = len([t for t in trades if t['action'] == 'sell'])
        
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
            'trades': trades,
            'mode': self.mode_name
        }
        
        equity_curve = self.state['equity_curve']
        if len(equity_curve) > 1:
            equity_df = pd.DataFrame(equity_curve)
            
            if 'equity' in equity_df.columns:
                equity_df['return'] = equity_df['equity'].pct_change()
                
                if not equity_df['return'].empty:
                    equity_df['cumulative_return'] = (1 + equity_df['return']).cumprod() - 1
                    
                    if equity_df['return'].std() > 0:
                        sharpe_ratio = (equity_df['return'].mean() / equity_df['return'].std()) * (252 ** 0.5)
                        report['sharpe_ratio'] = sharpe_ratio
        
        self._add_mode_specific_metrics(report)
        
        return report
    
    def _add_mode_specific_metrics(self, report: Dict[str, Any]) -> None:
        """Add mode-specific metrics to report"""
        pass
    
    def _save_report(self, report: Dict[str, Any]) -> None:
        """Save trading report to files"""
        report_dir = self.config.get("reporting", f"{self.mode_name}_reports_dir", 
                                   default=f"reports/{self.mode_name}/")
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        output_formats = self.config.get("reporting", "output_formats", default=["json"])
        
        if 'csv' in output_formats and report.get('trades'):
            trades_df = pd.DataFrame(report['trades'])
            trades_csv = os.path.join(report_dir, f"{self.mode_name}_trades_{timestamp}.csv")
            trades_df.to_csv(trades_csv, index=False)
            self.logger.info(f"Trade records saved to {trades_csv}")
        
        if 'csv' in output_formats and report.get('equity_curve'):
            equity_df = pd.DataFrame(report['equity_curve'])
            equity_csv = os.path.join(report_dir, f"{self.mode_name}_equity_curve_{timestamp}.csv")
            equity_df.to_csv(equity_csv, index=False)
            self.logger.info(f"Equity curve saved to {equity_csv}")
        
        if 'json' in output_formats:
            clean_report = self._prepare_report_for_serialization(report)
            
            report_json = os.path.join(report_dir, f"{self.mode_name}_report_{timestamp}.json")
            with open(report_json, 'w') as f:
                json.dump(clean_report, f, indent=4, default=str)
            
            self.logger.info(f"Performance report saved to {report_json}")
        
        self._log_report_summary(report)
    
    def _prepare_report_for_serialization(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare report for JSON serialization"""
        clean_report = report.copy()
        
        for key in ['equity_curve', 'strategy']:
            if key in clean_report:
                if key == 'equity_curve':
                    clean_report[key] = str(f"{len(clean_report[key])} records")
                elif key == 'strategy':
                    clean_report[key] = clean_report[key].__class__.__name__
        
        return clean_report
    
    def _log_report_summary(self, report: Dict[str, Any]) -> None:
        """Log report summary"""
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
        """Check if the mode should continue running"""
        return self._running and not self.risk_manager.is_risk_breached()
    
    async def _sleep_interval(self, interval: Optional[int] = None) -> None:
        """Wait for the specified time"""
        if interval is None:
            interval = self.config.get("data", "polling_interval", default=60)
        
        if interval > 0:
            await asyncio.sleep(interval)
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown mode specific components"""
        # Common cleanup logic
        self._running = False
        
        # Close execution engine
        if self.execution_engine:
            try:
                await self.execution_engine.close()
            except Exception as e:
                self.logger.error(f"Error closing execution engine: {e}")