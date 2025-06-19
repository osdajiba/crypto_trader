"""
期权数据处理器
处理爬取的数据，准备定价参数
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from datetime import datetime
import os

class OptionDataProcessor:
    """期权数据处理器"""
    
    def __init__(self):
        self.data = None
        self.processed_data = None
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        """加载CSV数据"""
        try:
            self.data = pd.read_csv(filename)
            logging.info(f"成功加载数据: {filename}")
            logging.info(f"数据形状: {self.data.shape}")
            return self.data
        except Exception as e:
            logging.error(f"加载CSV失败: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理数据"""
        try:
            # 创建副本
            cleaned_df = df.copy()
            
            # 移除无效数据
            cleaned_df = cleaned_df.dropna(subset=['strike', 'option_type'])
            
            # 确保数值类型正确
            numeric_columns = [
                'strike', 'delta', 'bid_price', 'ask_price', 'mark_price',
                'bid_iv', 'ask_iv', 'mark_iv', 'volume', 'open_interest',
                'underlying_price', 'atm_volatility', 'time_to_expiry'
            ]
            
            for col in numeric_columns:
                if col in cleaned_df.columns:
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            
            # 移除价格为0或负数的期权
            cleaned_df = cleaned_df[cleaned_df['mark_price'] > 0]
            cleaned_df = cleaned_df[cleaned_df['mark_iv'] > 0]
            
            # 移除极端IV值
            cleaned_df = cleaned_df[
                (cleaned_df['mark_iv'] >= 0.05) & 
                (cleaned_df['mark_iv'] <= 5.0)
            ]
            
            logging.info(f"数据清理后形状: {cleaned_df.shape}")
            return cleaned_df
            
        except Exception as e:
            logging.error(f"数据清理失败: {e}")
            raise
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算额外特征"""
        try:
            enhanced_df = df.copy()
            
            # 计算moneyness
            enhanced_df['moneyness'] = enhanced_df['underlying_price'] / enhanced_df['strike']
            
            # 计算内在价值
            enhanced_df['intrinsic_value'] = np.where(
                enhanced_df['option_type'] == 'call',
                np.maximum(enhanced_df['underlying_price'] - enhanced_df['strike'], 0),
                np.maximum(enhanced_df['strike'] - enhanced_df['underlying_price'], 0)
            )
            
            # 计算时间价值
            enhanced_df['time_value'] = enhanced_df['mark_price'] - enhanced_df['intrinsic_value']
            
            # 计算买卖价差
            enhanced_df['bid_ask_spread'] = enhanced_df['ask_price'] - enhanced_df['bid_price']
            enhanced_df['bid_ask_spread_pct'] = enhanced_df['bid_ask_spread'] / enhanced_df['mark_price']
            
            # 分类期权类型（按moneyness）
            def categorize_option(row):
                moneyness = row['moneyness']
                option_type = row['option_type']
                
                if option_type == 'call':
                    if moneyness >= 1.05:
                        return 'ITM'
                    elif moneyness >= 0.95:
                        return 'ATM'
                    else:
                        return 'OTM'
                else:  # put
                    if moneyness <= 0.95:
                        return 'ITM'
                    elif moneyness <= 1.05:
                        return 'ATM'
                    else:
                        return 'OTM'
            
            enhanced_df['money_category'] = enhanced_df.apply(categorize_option, axis=1)
            
            # 计算到期天数
            enhanced_df['days_to_expiry'] = enhanced_df['time_to_expiry'] * 365
            
            logging.info("特征计算完成")
            return enhanced_df
            
        except Exception as e:
            logging.error(f"特征计算失败: {e}")
            raise
    
    def prepare_pricing_params(self, df: pd.DataFrame) -> List[Dict]:
        """准备定价参数"""
        try:
            pricing_params = []
            
            for _, row in df.iterrows():
                # 基础参数
                params = {
                    'S0': float(row['underlying_price']),
                    'K': float(row['strike']),
                    'T': float(row['time_to_expiry']),
                    'sigma': float(row['mark_iv']),
                    'r': 0.05,  # 默认无风险利率，可以从其他源获取
                    
                    # 交易成本参数（根据实际情况调整）
                    'fixed_cost_ratio': 0.0002,  # 2个基点
                    'slippage': 0.0001,          # 1个基点
                    'margin_ratio': 0.5,         # 50%保证金
                    
                    # 模拟参数
                    'n_steps': int(max(252 * row['time_to_expiry'], 50)),  # 至少50步
                    
                    # 元数据
                    'option_type': row['option_type'],
                    'strike_price': float(row['strike']),
                    'market_price': float(row['mark_price']),
                    'market_iv': float(row['mark_iv']),
                    'delta': float(row.get('delta', 0)),
                    'bid_price': float(row.get('bid_price', 0)),
                    'ask_price': float(row.get('ask_price', 0)),
                    'moneyness': float(row.get('moneyness', 1)),
                    'money_category': str(row.get('money_category', 'ATM')),
                    'intrinsic_value': float(row.get('intrinsic_value', 0)),
                    'time_value': float(row.get('time_value', 0)),
                    
                    # 用于标识
                    'option_id': f"{row['option_type'].upper()}_{int(row['strike'])}",
                    'description': f"{row['option_type'].upper()} K={int(row['strike'])} T={row['time_to_expiry']:.3f}年"
                }
                
                pricing_params.append(params)
            
            logging.info(f"准备了 {len(pricing_params)} 个定价参数集")
            return pricing_params
            
        except Exception as e:
            logging.error(f"准备定价参数失败: {e}")
            raise
    
    def filter_options(self, df: pd.DataFrame, filters: Dict = None) -> pd.DataFrame:
        """筛选期权"""
        try:
            filtered_df = df.copy()
            
            if filters is None:
                filters = {}
            
            # 默认筛选条件
            default_filters = {
                'min_volume': 1,           # 最小成交量
                'max_bid_ask_spread': 0.2, # 最大买卖价差比例
                'min_time_to_expiry': 0.001, # 最小到期时间（天）
                'max_time_to_expiry': 1.0,   # 最大到期时间（年）
                'option_types': ['call', 'put'], # 期权类型
                'money_categories': ['ITM', 'ATM', 'OTM'] # moneyness类别
            }
            
            # 合并筛选条件
            filters = {**default_filters, **filters}
            
            # 应用筛选条件
            if 'min_volume' in filters:
                filtered_df = filtered_df[filtered_df['volume'] >= filters['min_volume']]
            
            if 'max_bid_ask_spread' in filters and 'bid_ask_spread_pct' in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df['bid_ask_spread_pct'] <= filters['max_bid_ask_spread']
                ]
            
            if 'min_time_to_expiry' in filters:
                filtered_df = filtered_df[
                    filtered_df['time_to_expiry'] >= filters['min_time_to_expiry']
                ]
            
            if 'max_time_to_expiry' in filters:
                filtered_df = filtered_df[
                    filtered_df['time_to_expiry'] <= filters['max_time_to_expiry']
                ]
            
            if 'option_types' in filters:
                filtered_df = filtered_df[
                    filtered_df['option_type'].isin(filters['option_types'])
                ]
            
            if 'money_categories' in filters and 'money_category' in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df['money_category'].isin(filters['money_categories'])
                ]
            
            # 按成交量和价格筛选最活跃的期权
            if 'top_n_by_volume' in filters:
                n = filters['top_n_by_volume']
                filtered_df = filtered_df.nlargest(n, 'volume')
            
            logging.info(f"筛选后期权数量: {len(filtered_df)}")
            return filtered_df
            
        except Exception as e:
            logging.error(f"期权筛选失败: {e}")
            raise
    
    def process(self, data_source, filters: Dict = None) -> Tuple[pd.DataFrame, List[Dict]]:
        """主处理流程"""
        try:
            # 加载数据
            if isinstance(data_source, str):
                # 从文件加载
                df = self.load_csv(data_source)
            elif isinstance(data_source, pd.DataFrame):
                # 直接使用DataFrame
                df = data_source.copy()
            else:
                # 从字典数据创建DataFrame
                df = pd.DataFrame(data_source.get('options', []))
            
            # 数据处理流程
            logging.info("开始数据处理流程...")
            
            # 1. 清理数据
            df_cleaned = self.clean_data(df)
            
            # 2. 计算特征
            df_enhanced = self.calculate_features(df_cleaned)
            
            # 3. 筛选期权
            df_filtered = self.filter_options(df_enhanced, filters)
            
            # 4. 准备定价参数
            pricing_params = self.prepare_pricing_params(df_filtered)
            
            # 保存处理后的数据
            self.processed_data = df_filtered
            
            logging.info("数据处理完成")
            
            return df_filtered, pricing_params
            
        except Exception as e:
            logging.error(f"数据处理失败: {e}")
            raise
    
    def save_processed_data(self, filename: str):
        """保存处理后的数据"""
        try:
            if self.processed_data is not None:
                self.processed_data.to_csv(filename, index=False, encoding='utf-8-sig')
                logging.info(f"处理后数据已保存到: {filename}")
            else:
                logging.warning("没有处理后的数据可保存")
        except Exception as e:
            logging.error(f"保存处理后数据失败: {e}")
    
    def get_summary_stats(self, df: pd.DataFrame = None) -> Dict:
        """获取数据汇总统计"""
        try:
            if df is None:
                df = self.processed_data
            
            if df is None:
                return {}
            
            stats = {
                'total_options': len(df),
                'calls': len(df[df['option_type'] == 'call']),
                'puts': len(df[df['option_type'] == 'put']),
                'strike_range': {
                    'min': float(df['strike'].min()),
                    'max': float(df['strike'].max()),
                    'count': len(df['strike'].unique())
                },
                'iv_stats': {
                    'mean': float(df['mark_iv'].mean()),
                    'median': float(df['mark_iv'].median()),
                    'std': float(df['mark_iv'].std()),
                    'min': float(df['mark_iv'].min()),
                    'max': float(df['mark_iv'].max())
                },
                'price_stats': {
                    'mean': float(df['mark_price'].mean()),
                    'median': float(df['mark_price'].median()),
                    'std': float(df['mark_price'].std()),
                    'min': float(df['mark_price'].min()),
                    'max': float(df['mark_price'].max())
                },
                'time_to_expiry': {
                    'mean_days': float(df['time_to_expiry'].mean() * 365),
                    'min_days': float(df['time_to_expiry'].min() * 365),
                    'max_days': float(df['time_to_expiry'].max() * 365)
                }
            }
            
            # 按类别统计
            if 'money_category' in df.columns:
                stats['by_category'] = df.groupby(['option_type', 'money_category']).size().to_dict()
            
            return stats
            
        except Exception as e:
            logging.error(f"生成汇总统计失败: {e}")
            return {}

def create_quick_filter_presets():
    """创建快速筛选预设"""
    return {
        'active_options': {
            'min_volume': 10,
            'max_bid_ask_spread': 0.1,
            'top_n_by_volume': 50
        },
        'atm_options': {
            'money_categories': ['ATM'],
            'min_volume': 5
        },
        'all_options': {
            'min_volume': 1,
            'max_bid_ask_spread': 0.5
        },
        'short_term': {
            'max_time_to_expiry': 0.1,  # 约36天
            'min_volume': 5
        },
        'liquid_options': {
            'min_volume': 50,
            'max_bid_ask_spread': 0.05
        }
    }