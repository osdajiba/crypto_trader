#!/usr/bin/env python3
# src/trading/paper.py

from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime
import time

from src.common.abstract_factory import register_factory_class
from src.common.config_manager import ConfigManager
from src.mode.base import BaseTradingMode, PerformanceCategory
from src.common.helpers import time_func


@register_factory_class('trading_mode_factory', 'paper', 
                       description="Paper trading (uses real market data without real funds)", 
                       features=["real_time_data", "virtual_execution"],
                       category="simulation")
class PaperMode(BaseTradingMode):
    """Paper trading mode implementation"""
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """Initialize paper trading mode"""
        super().__init__(config, params)
        self.data_source_type = "hybrid"
        self.execution_factory = None
        
        # Paper-specific performance metrics
        self.data_fetch_latency = []
        self.signal_generation_latency = []
        self.execution_latency = []
        
        # Performance tracking intervals
        self.perf_log_interval = self.config.get(
            "paper_trading", "perf_log_interval", default=10
        )  # iterations
        self._perf_counter = 0
        
        # Simulate exchange latency (for more realistic paper trading)
        self.simulate_latency = self.config.get(
            "paper_trading", "simulate_latency", default=False
        )
        self.simulated_latency = self.config.get(
            "paper_trading", "simulated_latency", default=0.1
        )  # seconds
    
    @time_func("paper_initialize")
    async def _initialize_mode_specific(self) -> None:
        """Initialize paper trading mode components"""
        self.logger.info("Initializing paper trading mode components")
        
        # Check if data source is available
        if self.data_manager:
            data_check_start = time.time()
            test_symbol = self.config.get("paper_trading", "test_symbol", default="BTC/USDT")
            test_data = await self.data_manager.get_latest_data_for_symbols(test_symbol, "1m")
            data_check_time = time.time() - data_check_start
            
            if test_data is None or test_data.empty:
                self.logger.warning(f"Could not fetch test data for {test_symbol}")
            else:
                self.logger.info(f"Data source check successful (latency: {data_check_time:.4f}s)")
                self.data_fetch_latency.append(data_check_time)
        
        # Log performance tracking settings
        if self.enable_performance_tracking:
            self.logger.info(f"Performance tracking enabled with log interval: {self.perf_log_interval}")
        
        # Log latency simulation settings
        if self.simulate_latency:
            self.logger.info(f"Simulating exchange latency: {self.simulated_latency}s per operation")
    
    @time_func("prepare_paper_trading")
    async def _prepare_run(self, symbols: List[str], timeframe: str) -> None:
        """Paper trading specific preparation"""
        # Initialize performance tracking data
        self.state['performance_data'] = {
            'start_time': datetime.now(),
            'trade_count': 0,
            'cycle_count': 0,
            'avg_cycle_time': 0
        }
        
        # Get data source type
        self.data_source_type = self.config.get("paper_trading", "data_source", default="hybrid")
        
        self.logger.info(f"Paper trading prepared with {self.data_source_type} data source")
    
    @time_func("paper_trading_loop")
    async def _execute_trading_loop(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Execute paper trading loop"""
        try:
            # Trading loop
            while self._should_continue():
                # Start timing this cycle
                cycle_start = time.time()
                
                # Get market data with performance tracking already included
                data_fetch_start = time.time()
                data_map = await self._fetch_market_data(symbols, timeframe)
                data_fetch_time = time.time() - data_fetch_start
                
                # Store in latency tracking
                self.data_fetch_latency.append(data_fetch_time)
                
                # Simulate exchange latency if enabled
                if self.simulate_latency:
                    await asyncio.sleep(self.simulated_latency)
                
                # Process market data with performance tracking already included
                if data_map:
                    process_start = time.time()
                    executed_trades = await self._process_market_data(data_map)
                    process_time = time.time() - process_start
                    
                    # Track signal generation and execution latency
                    if self.enable_performance_tracking:
                        signal_times = self.performance_metrics.get(PerformanceCategory.SIGNAL_GENERATION.value, [])
                        if signal_times:
                            self.signal_generation_latency.append(signal_times[-1])
                            
                        exec_times = self.performance_metrics.get(PerformanceCategory.TRADE_EXECUTION.value, [])
                        if exec_times:
                            self.execution_latency.append(exec_times[-1])
                    
                    # Update performance data
                    if executed_trades:
                        self.state['performance_data']['trade_count'] += len(executed_trades)
                else:
                    self.logger.warning("No market data received, skipping trading cycle")
                
                # Check risk control
                risk_start = time.time()
                if hasattr(self, 'risk_manager') and self.risk_manager and hasattr(self.risk_manager, 'execute_risk_control'):
                    if await self.risk_manager.execute_risk_control():
                        self.logger.critical("Risk control triggered, stopping paper trading")
                        self._running = False
                        break
                
                risk_time = time.time() - risk_start
                if self.enable_performance_tracking:
                    self._record_performance_metric(
                        PerformanceCategory.RISK_MANAGEMENT,
                        risk_time
                    )
                
                # Calculate and record cycle time
                cycle_time = time.time() - cycle_start
                self.state['performance_data']['cycle_count'] += 1
                total_cycles = self.state['performance_data']['cycle_count']
                current_avg = self.state['performance_data']['avg_cycle_time']
                
                # Update rolling average cycle time
                self.state['performance_data']['avg_cycle_time'] = (
                    (current_avg * (total_cycles - 1) + cycle_time) / total_cycles
                    if total_cycles > 1 else cycle_time
                )
                
                # Record total iteration time
                if self.enable_performance_tracking:
                    self._record_performance_metric(
                        PerformanceCategory.TOTAL_ITERATION,
                        cycle_time
                    )
                
                # Log performance metrics periodically
                self._perf_counter += 1
                if self.enable_performance_tracking and self._perf_counter % self.perf_log_interval == 0:
                    self._log_performance_metrics()
                
                # Wait for next interval
                await self._sleep_interval()
            
            # Return basic results
            return {
                "status": "completed" if self._running else "stopped_by_risk_control",
                "symbols": symbols, 
                "timeframe": timeframe,
                "cycles": self.state['performance_data']['cycle_count'],
                "avg_cycle_time": self.state['performance_data']['avg_cycle_time']
            }
            
        except asyncio.CancelledError:
            self.logger.warning("Paper trading cancelled")
            raise
            
        except Exception as e:
            self.logger.error(f"Paper trading error: {e}", exc_info=True)
            raise
    
    def _log_performance_metrics(self) -> None:
        """Log performance metrics for paper trading"""
        if not self.enable_performance_tracking:
            return
            
        # Get cycle metrics
        cycles = self.state['performance_data']['cycle_count']
        avg_cycle = self.state['performance_data']['avg_cycle_time']
        
        # Calculate data fetch metrics
        avg_fetch = sum(self.data_fetch_latency) / len(self.data_fetch_latency) if self.data_fetch_latency else 0
        max_fetch = max(self.data_fetch_latency) if self.data_fetch_latency else 0
        
        # Calculate signal generation metrics
        avg_signal = sum(self.signal_generation_latency) / len(self.signal_generation_latency) if self.signal_generation_latency else 0
        max_signal = max(self.signal_generation_latency) if self.signal_generation_latency else 0
        
        # Calculate execution metrics
        avg_exec = sum(self.execution_latency) / len(self.execution_latency) if self.execution_latency else 0
        max_exec = max(self.execution_latency) if self.execution_latency else 0
        
        # Log the metrics
        self.logger.info("==== Paper Trading Performance ====")
        self.logger.info(f"Cycles: {cycles} | Avg cycle time: {avg_cycle:.4f}s")
        self.logger.info(f"Data fetch: avg={avg_fetch:.4f}s, max={max_fetch:.4f}s")
        self.logger.info(f"Signal generation: avg={avg_signal:.4f}s, max={max_signal:.4f}s")
        self.logger.info(f"Trade execution: avg={avg_exec:.4f}s, max={max_exec:.4f}s")
        
        # Log simulated latency info if enabled
        if self.simulate_latency:
            self.logger.info(f"Simulated exchange latency: {self.simulated_latency}s per operation")
        
        # Log total elapsed time
        elapsed = (datetime.now() - self.state['performance_data']['start_time']).total_seconds() / 60.0
        self.logger.info(f"Total runtime: {elapsed:.2f} minutes")
    
    def _add_mode_specific_metrics(self, report: Dict[str, Any]) -> None:
        """Add paper trading specific metrics to report"""
        # Add paper trading parameters
        report['paper_trading_params'] = {
            'data_source': self.data_source_type,
            'simulate_latency': self.simulate_latency,
            'simulated_latency': self.simulated_latency if self.simulate_latency else 0,
            'run_duration': (datetime.now() - self.state['performance_data']['start_time']).total_seconds() / 60,  # minutes
            'cycles': self.state['performance_data']['cycle_count'],
            'avg_cycle_time': self.state['performance_data']['avg_cycle_time'],
            'trade_count': self.state['performance_data']['trade_count']
        }
        
        # Add strategy info
        if self.strategy:
            report['strategy'] = self.strategy.__class__.__name__
            
        # Add performance metrics if available
        if self.enable_performance_tracking:
            report['performance_metrics'] = {
                'data_fetch': {
                    'avg': sum(self.data_fetch_latency) / len(self.data_fetch_latency) if self.data_fetch_latency else 0,
                    'max': max(self.data_fetch_latency) if self.data_fetch_latency else 0,
                    'min': min(self.data_fetch_latency) if self.data_fetch_latency else 0,
                    'samples': len(self.data_fetch_latency)
                },
                'signal_generation': {
                    'avg': sum(self.signal_generation_latency) / len(self.signal_generation_latency) if self.signal_generation_latency else 0,
                    'max': max(self.signal_generation_latency) if self.signal_generation_latency else 0,
                    'min': min(self.signal_generation_latency) if self.signal_generation_latency else 0,
                    'samples': len(self.signal_generation_latency)
                },
                'trade_execution': {
                    'avg': sum(self.execution_latency) / len(self.execution_latency) if self.execution_latency else 0,
                    'max': max(self.execution_latency) if self.execution_latency else 0,
                    'min': min(self.execution_latency) if self.execution_latency else 0,
                    'samples': len(self.execution_latency)
                }
            }
    
    @time_func("paper_shutdown")
    async def shutdown(self) -> None:
        """Shutdown paper trading mode"""
        self.logger.info("Shutting down paper trading mode")
        
        # Log final performance metrics
        if self.enable_performance_tracking:
            self._log_performance_metrics()
        
        # Call parent shutdown
        await super().shutdown()
            
        self.logger.info("Paper trading mode shutdown complete")