# src\execution\execution_engine.py

import pandas as pd
from typing import Dict, Optional, List, Tuple
from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.execution.order import *
from src.exchange.binance import Binance
from enum import Enum
import ccxt

class Direction(Enum):
    BUY = 'buy'
    SELL = 'sell'

class ExecutionEngine:
    """Engine for executing trading orders with multiple execution modes."""
    
    def __init__(self, config: ConfigManager, mode: str, 
                 historical_data: Optional[Dict[str, pd.DataFrame]] = None,
                 ):
        """
        Initialize the execution engine.

        Args:
            config (ConfigManager): Configuration manager instance.
            mode (str): Trading mode ("live", "backtest", "simple_backtest").
            historical_data (Optional[Dict[str, pd.DataFrame]]): Historical data for backtesting.
        """
        self.config = config
        self.mode = mode.lower()
        self.commission = config.get("default_config", "user_config", "commission", default=0.001)
        self.slippage = config.get("default_config", "user_config", "slippage", default=0.001)
        self.logger = LogManager.get_logger("system.execution")
        self._running = True
        self.historical_data = historical_data or {}
        self.binance = Binance(config) if mode == "live" else None

    async def execute(self, signals: pd.DataFrame, prices: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Optional[Dict[str, pd.DataFrame]]]:
        """
        Execute trading signals and return executed orders and updated historical data (if applicable).

        Args:
            signals (pd.DataFrame): Signals with 'timestamp', 'symbol', 'action', 'quantity', and optional 'price'.
            prices (Optional[Dict[str, float]]): Current prices for each symbol (required for live).

        Returns:
            Tuple[pd.DataFrame, Optional[Dict[str, pd.DataFrame]]]: Executed orders and updated historical data (None for live/simple_backtest).
        """
        if signals.empty:
            self.logger.info("No signals to execute")
            return pd.DataFrame(), None

        self.logger.info("Executing %d signals in %s mode", len(signals), self.mode)
        orders = await self._create_orders(signals)

        if self.mode == "live":
            if not self.binance:
                raise ValueError("Binance interface required for live mode")
            executed_orders = await self._live_execution(orders)
            updated_data = None
        elif self.mode == "backtest":
            if not self.historical_data:
                raise ValueError("Historical data required for backtest mode")
            executed_orders, updated_data = await self._backtest_execution(orders)
        elif self.mode == "simple_backtest":
            if not self.historical_data:
                raise ValueError("Historical data required for simple_backtest mode")
            executed_orders = await self._simple_backtest_execution(orders)
            updated_data = None
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        return executed_orders, updated_data

    async def _create_orders(self, signals: pd.DataFrame) -> List:
        """Create Order objects from signals."""
        orders = []
        for _, signal in signals.iterrows():
            symbol = signal['symbol']
            action = signal['action'].lower()
            timestamp = signal['timestamp']
            quantity = signal.get('quantity', self.config.get("default_config", "user_config", "min_order_size", default=0.001))
            price = signal.get('price', None)
            direction = Direction.BUY if action == 'buy' else Direction.SELL

            try:
                order = MarketOrder(symbol, direction, quantity, timestamp=timestamp)  # Simplified to market orders
                if price:
                    order.price = price  # Optional price for limit-like behavior in backtest
                orders.append(order)
            except ValueError as e:
                self.logger.error("Failed to create order for signal %s: %s", signal.to_dict(), str(e))
                continue

        self.logger.debug("Created %d orders from signals", len(orders))
        return orders

    async def close(self) -> None:
        """Close the execution engine and clean up resources."""
        if not self._running:
            return
        self._running = False
        self.historical_data.clear()
        self.logger.info("Execution engine closed")

    def set_historical_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """Set historical data for backtesting."""
        self.historical_data = data
        self.logger.debug("Historical data set for %d symbols", len(data))
        
    
    async def _live_execution(self, orders: List) -> pd.DataFrame:
        """实盘执行：通过Binance接口下单"""
        executed_orders = []
        for order in orders:
            try:
                # 转换为交易所需要的参数格式
                response = self.binance.create_order(
                    symbol=order.symbol,
                    direction=order.direction,  # 传递Direction枚举
                    order_type='limit' if isinstance(order, LimitOrder) else 'market',
                    quantity=order.quantity,
                    price=order.price
                )
                
                # 记录成交详情
                executed_orders.append({
                    'order_id': response['id'],
                    'symbol': order.symbol,
                    'filled_qty': float(response['filled']),
                    'avg_price': float(response['average']),
                    'status': self._map_exchange_status(response['status'])
                })
                self.logger.info(f"实盘订单 {response['id']} 已提交")
                
            except ccxt.InsufficientFunds as e:
                self.logger.error(f"资金不足: {str(e)}")
            except Exception as e:
                self.logger.error(f"下单失败: {str(e)}")

        return pd.DataFrame(executed_orders)

    async def _backtest_execution(self, orders: List) -> Tuple[pd.DataFrame, Dict]:
        """完整回测：修改历史数据"""
        updated_data = self.historical_data.copy()
        executed = []
        
        for order in orders:
            symbol_data = updated_data[order.symbol]
            execution_bar = symbol_data.loc[order.timestamp:].iloc[0]  # 找到执行时刻的K线
            
            # 计算可成交量（考虑市场深度）
            max_volume = execution_bar['volume'] * 0.1  # 假设最多成交该K线成交量的10%
            filled_qty = min(order.quantity, max_volume)
            
            if filled_qty > 0:
                # 计算执行价格（带滑点）
                if order.direction == Direction.BUY:
                    exec_price = execution_bar['high'] * (1 + self.slippage)
                else:
                    exec_price = execution_bar['low'] * (1 - self.slippage)
                
                # 更新历史成交量
                updated_data[order.symbol].loc[execution_bar.name, 'volume'] -= filled_qty
                
                # 记录成交
                executed.append({
                    'order_id': order.order_id,
                    'filled_qty': filled_qty,
                    'price': exec_price,
                    'commission': exec_price * filled_qty * self.commission
                })

        return pd.DataFrame(executed), updated_data

    async def _simple_backtest_execution(self, orders: List) -> pd.DataFrame:
        """简单回测：仅检测可行性"""
        results = []
        
        for order in orders:
            symbol_data = self.historical_data[order.symbol]
            execution_bar = symbol_data.loc[order.timestamp:].iloc[0]
            
            # 检查价格条件
            if isinstance(order, LimitOrder):
                if order.direction == Direction.BUY and order.price < execution_bar['low']:
                    continue  # 限价未触发
                elif order.direction == Direction.SELL and order.price > execution_bar['high']:
                    continue
            
            # 检查成交量
            fillable_qty = min(order.quantity, execution_bar['volume'])
            
            if fillable_qty > 0:
                results.append({
                    'order_id': order.order_id,
                    'filled': fillable_qty,
                    'price': execution_bar['close']  # 假设以收盘价成交
                })

        return pd.DataFrame(results)

    def _map_exchange_status(self, status: str) -> str:
        """映射交易所状态到本地状态"""
        return {
            'new': 'submitted',
            'filled': 'filled',
            'partially_filled': 'partial',
            'canceled': 'canceled'
        }.get(status, 'unknown')