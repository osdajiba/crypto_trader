"""
期权定价核心模块
实现蒙特卡洛期权复制策略定价
"""

import numpy as np
import torch
from scipy import stats
import logging
from typing import Dict, List, Optional, Tuple
import time

class OptionPricer:
    """期权定价器"""
    
    def __init__(self, 
                 S0: float = 100,           # 初始股价
                 K: float = 100,            # 执行价
                 r: float = 0.05,           # 无风险利率
                 sigma: float = 0.2,        # 波动率
                 T: float = 1.0,            # 到期时间(年)
                 n_steps: int = 252,        # 时间步数
                 fixed_cost_ratio: float = 0.0002,  # 固定成本比例
                 slippage: float = 0.0001,  # 滑点成本
                 margin_ratio: float = 0.1, # 保证金比例
                 **kwargs):
        
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.fixed_cost_ratio = fixed_cost_ratio
        self.slippage = slippage
        self.margin_ratio = margin_ratio
        
        # 计算Black-Scholes理论价格
        self.bs_price = self._black_scholes_call()
        
        logging.debug(f"初始化定价器: S0={S0}, K={K}, T={T}, σ={sigma}")
    
    def _black_scholes_call(self) -> float:
        """计算Black-Scholes期权价格"""
        if self.T <= 0:
            return max(self.S0 - self.K, 0)
            
        d1 = (np.log(self.S0/self.K) + (self.r + 0.5*self.sigma**2)*self.T) / (self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma*np.sqrt(self.T)
        
        return self.S0 * stats.norm.cdf(d1) - self.K * np.exp(-self.r*self.T) * stats.norm.cdf(d2)
    
    def _generate_path(self) -> np.ndarray:
        """生成单条资产价格路径"""
        Z = np.random.normal(0, 1, self.n_steps)
        S_path = np.zeros(self.n_steps + 1)
        S_path[0] = self.S0
        
        drift = (self.r - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt)
        
        for i in range(self.n_steps):
            S_path[i+1] = S_path[i] * np.exp(drift + diffusion * Z[i])
            
        return S_path
    
    def _generate_paths_gpu(self, n_paths: int) -> np.ndarray:
        """使用GPU生成多条路径"""
        if not torch.cuda.is_available():
            logging.warning("GPU不可用，使用CPU生成路径")
            return np.array([self._generate_path() for _ in range(n_paths)])
        
        device = torch.device("cuda")
        batch_size = min(10000, n_paths)  # 避免GPU内存溢出
        
        all_paths = []
        for i in range(0, n_paths, batch_size):
            current_batch = min(batch_size, n_paths - i)
            
            # 生成随机数
            Z = torch.randn((current_batch, self.n_steps), device=device)
            
            # 初始化路径
            S_paths = torch.zeros((current_batch, self.n_steps + 1), device=device)
            S_paths[:, 0] = self.S0
            
            # 计算路径
            drift = (self.r - 0.5 * self.sigma**2) * self.dt
            diffusion = self.sigma * np.sqrt(self.dt)
            
            for j in range(self.n_steps):
                S_paths[:, j+1] = S_paths[:, j] * torch.exp(drift + diffusion * Z[:, j])
            
            all_paths.append(S_paths.cpu().numpy())
        
        return np.vstack(all_paths)
    
    def _calculate_transaction_cost(self, price: float, quantity: float) -> float:
        """计算交易成本"""
        fixed_cost = self.fixed_cost_ratio * price * abs(quantity)
        slippage_cost = self.slippage * abs(quantity)
        return fixed_cost + slippage_cost
    
    def _replicate_option(self, S_path: np.ndarray) -> Dict:
        """期权复制策略实现"""
        # 初始化变量
        cash = np.zeros(self.n_steps + 1)
        long_pos = np.zeros(self.n_steps + 1)
        short_pos = np.zeros(self.n_steps + 1)
        trades = []
        
        # 初始设置
        borrowed_amount = self.K * np.exp(-self.r * self.T)
        cash[0] = borrowed_amount
        
        # 买入初始资产
        init_cost = self._calculate_transaction_cost(S_path[0], 1)
        cash[0] -= S_path[0] + init_cost
        long_pos[0] = 1
        
        trades.append({
            'time': 0,
            'action': 'buy',
            'price': S_path[0],
            'quantity': 1,
            'cost': init_cost
        })
        
        # 动态调整
        for i in range(1, self.n_steps + 1):
            long_pos[i] = long_pos[i-1]
            short_pos[i] = short_pos[i-1]
            
            if i < self.n_steps:
                # 更新现金
                margin_cost = (self.margin_ratio * short_pos[i-1] * S_path[i-1] * 
                              self.r * self.dt if short_pos[i-1] > 0 else 0)
                cash[i] = cash[i-1] * np.exp(self.r * self.dt) - margin_cost
                
                # 检查是否需要交易
                prev_price = S_path[i-1]
                curr_price = S_path[i]
                
                # 价格从上穿过K
                if (prev_price > self.K and curr_price <= self.K and 
                    long_pos[i] > 0 and short_pos[i] == 0):
                    
                    cost = self._calculate_transaction_cost(self.K, 1)
                    cash[i] += self.K - cost
                    short_pos[i] = 1
                    
                    trades.append({
                        'time': i * self.dt,
                        'action': 'sell',
                        'price': self.K,
                        'quantity': 1,
                        'cost': cost
                    })
                
                # 价格从下穿过K
                elif (prev_price < self.K and curr_price >= self.K and 
                      long_pos[i] > 0 and short_pos[i] > 0):
                    
                    cost = self._calculate_transaction_cost(self.K, 1)
                    cash[i] -= self.K + cost
                    short_pos[i] = 0
                    
                    trades.append({
                        'time': i * self.dt,
                        'action': 'buy',
                        'price': self.K,
                        'quantity': 1,
                        'cost': cost
                    })
            else:
                cash[i] = cash[i-1]
        
        # 最终清算
        final_cash = cash[-1]
        final_liquidation_cost = 0
        
        if long_pos[-1] > 0:
            cost = self._calculate_transaction_cost(S_path[-1], long_pos[-1])
            final_cash += S_path[-1] * long_pos[-1] - cost
            final_liquidation_cost += cost
        
        if short_pos[-1] > 0:
            cost = self._calculate_transaction_cost(S_path[-1], short_pos[-1])
            final_cash -= S_path[-1] * short_pos[-1] + cost
            final_liquidation_cost += cost
        
        # 计算结果
        final_worth = final_cash - self.K
        option_payoff = max(S_path[-1] - self.K, 0)
        replication_error = final_worth - option_payoff
        total_costs = sum(trade['cost'] for trade in trades)
        
        return {
            'final_price': S_path[-1],
            'option_payoff': option_payoff,
            'final_worth': final_worth,
            'replication_error': replication_error,
            'total_costs': total_costs,
            'trade_count': len(trades),
            'trades': trades
        }
    
    def price_single(self) -> Dict:
        """单次定价"""
        start_time = time.time()
        
        S_path = self._generate_path()
        result = self._replicate_option(S_path)
        
        # 添加基础信息
        result.update({
            'bs_price': self.bs_price,
            'pricing_time': time.time() - start_time,
            'method': 'single_path',
            'parameters': {
                'S0': self.S0, 'K': self.K, 'r': self.r,
                'sigma': self.sigma, 'T': self.T, 'n_steps': self.n_steps
            }
        })
        
        logging.info(f"单次定价完成 - BS价格: {self.bs_price:.4f}, "
                    f"复制价格: {result['final_worth']:.4f}, "
                    f"误差: {result['replication_error']:.4f}")
        
        return result
    
    def price_monte_carlo(self, n_paths: int = 10000, use_gpu: bool = True) -> Dict:
        """蒙特卡洛定价"""
        start_time = time.time()
        
        logging.info(f"开始蒙特卡洛定价，路径数: {n_paths}")
        
        # 生成路径
        if use_gpu and torch.cuda.is_available():
            logging.info("使用GPU生成路径")
            S_paths = self._generate_paths_gpu(n_paths)
        else:
            logging.info("使用CPU生成路径")
            S_paths = np.array([self._generate_path() for _ in range(n_paths)])
        
        # 并行计算复制结果
        results = []
        for i, path in enumerate(S_paths):
            if i % 1000 == 0:
                logging.debug(f"处理路径 {i+1}/{n_paths}")
            results.append(self._replicate_option(path))
        
        # 统计结果
        option_payoffs = [r['option_payoff'] for r in results]
        final_worths = [r['final_worth'] for r in results]
        replication_errors = [r['replication_error'] for r in results]
        total_costs = [r['total_costs'] for r in results]
        trade_counts = [r['trade_count'] for r in results]
        
        mc_result = {
            'bs_price': self.bs_price,
            'avg_option_payoff': np.mean(option_payoffs),
            'avg_final_worth': np.mean(final_worths),
            'avg_replication_error': np.mean(replication_errors),
            'std_replication_error': np.std(replication_errors),
            'avg_total_costs': np.mean(total_costs),
            'avg_trade_count': np.mean(trade_counts),
            'success_rate': np.mean([abs(e) < 0.1 for e in replication_errors]),
            'n_paths': n_paths,
            'pricing_time': time.time() - start_time,
            'method': 'monte_carlo',
            'parameters': {
                'S0': self.S0, 'K': self.K, 'r': self.r,
                'sigma': self.sigma, 'T': self.T, 'n_steps': self.n_steps
            }
        }
        
        logging.info(f"蒙特卡洛定价完成 - BS价格: {self.bs_price:.4f}, "
                    f"平均复制价格: {mc_result['avg_final_worth']:.4f}, "
                    f"平均误差: {mc_result['avg_replication_error']:.4f}")
        
        return mc_result