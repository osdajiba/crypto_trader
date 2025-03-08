# src/backtest/ohlcv_backtest_engine.py

from typing import Dict, List, Any
import pandas as pd
from datetime import datetime

from src.backtest.base_backtest_engine import BacktestEngine
from src.execution.execution_engine import ExecutionEngine


class OHLCVBacktestEngine(BacktestEngine):
    """基于OHLCV数据的回测引擎实现"""
    
    async def _initialize_backtest(self, symbols: List[str], timeframe: str) -> None:
        """
        初始化OHLCV回测
        
        Args:
            symbols: 交易品种列表
            timeframe: 时间周期
        """
        # 保存回测参数
        self.symbols = symbols
        self.timeframe = timeframe
        
        # 获取初始资金等配置参数
        self.initial_capital = self.config.get("backtest", "initial_capital", default=100000)
        self.commission_rate = self.config.get("backtest", "transaction_costs", "commission_rate", default=0.001)
        self.slippage = self.config.get("backtest", "transaction_costs", "slippage", default=0.001)
        
        # 初始化回测状态
        self.state = {
            'timestamp': None,
            'cash': self.initial_capital,
            'positions': {},
            'trades': [],
            'equity_curve': [],
            'market_prices': {},
            'current_equity': self.initial_capital,
            'peak_equity': self.initial_capital,
            'max_drawdown': 0.0
        }
        
        # 初始化执行引擎
        self.execution_engine = ExecutionEngine(
            config=self.config,
            mode="backtest"
        )
        
        # 初始化策略
        await self.strategy.initialize()
        
        self.logger.info(f"初始化OHLCV回测 | 初始资金: {self.initial_capital}")
    
    async def _load_historical_data(self, symbols: List[str], timeframe: str) -> Dict[str, pd.DataFrame]:
        """
        加载OHLCV历史数据
        
        Args:
            symbols: 交易品种列表
            timeframe: 时间周期
            
        Returns:
            Dict[str, pd.DataFrame]: 历史数据
        """
        data_map = {}
        
        # 设置回测时间范围
        start_date = self.config.get("backtest", "start_date")
        end_date = self.config.get("backtest", "end_date")
        
        # 加载每个品种的历史数据
        for symbol in symbols:
            try:
                data = await self.data_manager.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=start_date,
                    end=end_date
                )
                
                if not data.empty:
                    data_map[symbol] = data
                    self.logger.info(f"已加载 {symbol} 历史数据: {len(data)} 条记录")
                else:
                    self.logger.warning(f"{symbol} 无历史数据")
                
            except Exception as e:
                self.logger.error(f"加载 {symbol} 历史数据失败: {e}")
        
        return data_map
    
    def _get_time_points(self, data: Dict[str, pd.DataFrame]) -> List[Any]:
        """
        获取所有时间点
        
        Args:
            data: 历史数据
            
        Returns:
            List[Any]: 排序后的唯一时间点
        """
        all_timestamps = []
        
        for df in data.values():
            if 'datetime' in df.columns:
                all_timestamps.extend(df['datetime'].tolist())
        
        # 返回排序后的唯一时间戳
        return sorted(set(all_timestamps))
    
    def _get_data_at_time_point(self, data: Dict[str, pd.DataFrame], time_point: Any) -> Dict[str, pd.DataFrame]:
        """
        获取指定时间点的数据
        
        Args:
            data: 历史数据
            time_point: 时间点
            
        Returns:
            Dict[str, pd.DataFrame]: 指定时间点的数据
        """
        result = {}
        
        for symbol, df in data.items():
            if 'datetime' in df.columns:
                data_at_time = df[df['datetime'] == time_point]
                if not data_at_time.empty:
                    result[symbol] = data_at_time
        
        return result
    
    async def _process_data_point(self, time_point: Any, data: Dict[str, pd.DataFrame]) -> None:
        """
        处理单个时间点的数据
        
        Args:
            time_point: 时间点
            data: 时间点数据
        """
        # 更新当前时间点
        self.state['timestamp'] = time_point
        
        # 更新市场价格
        self._update_market_prices(data)
        
        # 处理每个交易品种
        for symbol, symbol_data in data.items():
            # 生成信号
            signals = await self.strategy.process_data(symbol_data, symbol)
            
            # 验证信号
            valid_signals = await self.execution_engine.validate_signals(signals)
            
            # 执行有效信号
            if not valid_signals.empty:
                executed_trades = self._execute_trades(valid_signals, data)
                
                # 记录交易
                if executed_trades:
                    self.state['trades'].extend(executed_trades)
        
        # 计算当前权益
        equity = self._calculate_equity()
        
        # 更新权益曲线
        self.state['equity_curve'].append({
            'timestamp': time_point,
            'equity': equity
        })
        
        # 更新最大回撤
        self._update_drawdown(equity)
    
    def _update_market_prices(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        更新市场价格
        
        Args:
            data: 当前时间点数据
        """
        for symbol, df in data.items():
            if not df.empty and 'close' in df.columns:
                self.state['market_prices'][symbol] = df['close'].iloc[0]
    
    def _execute_trades(self, signals: pd.DataFrame, current_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        执行交易信号
        
        Args:
            signals: 交易信号
            current_data: 当前市场数据
            
        Returns:
            List[Dict[str, Any]]: 执行的交易列表
        """
        executed_trades = []
        
        for _, signal in signals.iterrows():
            symbol = signal['symbol']
            action = signal['action'].lower()
            
            # 只处理buy/sell信号
            if action not in ['buy', 'sell']:
                continue
            
            # 获取当前价格
            if symbol not in current_data or current_data[symbol].empty:
                continue
                
            current_price = current_data[symbol]['close'].iloc[0]
            
            # 计算执行价格(考虑滑点)
            execution_price = current_price * (1 + self.slippage) if action == 'buy' else current_price * (1 - self.slippage)
            
            # 计算交易数量
            if 'quantity' in signal:
                quantity = signal['quantity']
            else:
                # 计算仓位大小(基于风险参数)
                risk_per_trade = self.config.get("backtest", "risk_parameters", "risk_per_trade", default=0.01)
                max_position = self.config.get("backtest", "transaction_costs", "max_position", default=0.1)
                quantity = min((self.state['cash'] * risk_per_trade) / execution_price, 
                              self.state['cash'] * max_position / execution_price)
            
            # 验证交易
            if action == 'buy':
                # 检查资金是否足够
                total_cost = quantity * execution_price * (1 + self.commission_rate)
                if total_cost > self.state['cash']:
                    quantity = self.state['cash'] / (execution_price * (1 + self.commission_rate))
                    if quantity <= 0:
                        continue
            elif action == 'sell':
                # 检查是否有足够的持仓
                current_position = self.state['positions'].get(symbol, 0)
                if current_position <= 0:
                    continue
                quantity = min(quantity, current_position)
            
            # 执行交易
            commission_cost = quantity * execution_price * self.commission_rate
            
            if action == 'buy':
                self.state['cash'] -= (quantity * execution_price + commission_cost)
                
                if symbol not in self.state['positions']:
                    self.state['positions'][symbol] = 0
                
                self.state['positions'][symbol] += quantity
                
            else:  # sell
                self.state['cash'] += (quantity * execution_price - commission_cost)
                
                self.state['positions'][symbol] -= quantity
                
                if self.state['positions'][symbol] <= 0:
                    del self.state['positions'][symbol]
            
            # 记录交易
            trade = {
                'timestamp': current_data[symbol]['datetime'].iloc[0],
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': execution_price,
                'commission': commission_cost,
                'slippage': abs(execution_price - current_price),
                'cash_after': self.state['cash']
            }
            
            executed_trades.append(trade)
        
        return executed_trades
    
    def _calculate_equity(self) -> float:
        """
        计算当前权益
        
        Returns:
            float: 当前权益值
        """
        equity = self.state['cash']
        
        # 加上持仓价值
        for symbol, quantity in self.state['positions'].items():
            if symbol in self.state['market_prices']:
                price = self.state['market_prices'][symbol]
                equity += quantity * price
        
        return equity
    
    def _update_drawdown(self, equity: float) -> None:
        """
        更新最大回撤
        
        Args:
            equity: 当前权益
        """
        # 更新峰值
        if equity > self.state['peak_equity']:
            self.state['peak_equity'] = equity
        
        # 计算当前回撤
        if self.state['peak_equity'] > 0:
            drawdown = (self.state['peak_equity'] - equity) / self.state['peak_equity']
            self.state['max_drawdown'] = max(self.state['max_drawdown'], drawdown)
    
    def _generate_backtest_report(self) -> Dict[str, Any]:
        """
        生成回测报告
        
        Returns:
            Dict[str, Any]: 回测报告
        """
        final_equity = self._calculate_equity()
        
        # 计算性能指标
        total_return = final_equity - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100 if self.initial_capital > 0 else 0
        
        # 计算交易统计
        trades = self.state['trades']
        buy_trades = len([t for t in trades if t['action'] == 'buy'])
        sell_trades = len([t for t in trades if t['action'] == 'sell'])
        
        # 计算夏普比率
        equity_curve = pd.DataFrame(self.state['equity_curve'])
        sharpe_ratio = 0
        
        if not equity_curve.empty and 'equity' in equity_curve.columns:
            # 计算每日收益率
            equity_curve['return'] = equity_curve['equity'].pct_change()
            
            # 计算夏普比率(假设无风险收益率为0)
            if not equity_curve['return'].empty and equity_curve['return'].std() > 0:
                sharpe_ratio = (equity_curve['return'].mean() / equity_curve['return'].std()) * (252 ** 0.5)
        
        # 组合报告
        report = {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'max_drawdown_pct': self.state['max_drawdown'] * 100,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(trades),
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'backtest_params': {
                'symbols': self.symbols,
                'timeframe': self.timeframe,
                'start_date': self.config.get("backtest", "start_date", default=""),
                'end_date': self.config.get("backtest", "end_date", default=""),
                'commission_rate': self.commission_rate,
                'slippage': self.slippage
            },
            'strategy': self.strategy.__class__.__name__,
            'open_positions': self.state['positions'],
            'remaining_cash': self.state['cash'],
            'trades': trades
        }
        
        return report