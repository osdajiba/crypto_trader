"""
期权定价报告生成器
生成markdown格式的定价分析报告
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional
import os
from pathlib import Path

class OptionPricingReportGenerator:
    """期权定价报告生成器"""
    
    def __init__(self):
        self.report_data = {}
        self.template_dir = Path(__file__).parent / 'templates'
    
    def prepare_report_data(self, 
                          market_data: pd.DataFrame,
                          pricing_results: List[Dict],
                          summary_stats: Dict) -> Dict:
        """准备报告数据"""
        try:
            # 合并市场数据和定价结果
            pricing_df = pd.DataFrame(pricing_results)
            
            # 确保有有效的定价结果
            valid_results = [r for r in pricing_results if 'error' not in r]
            
            if not valid_results:
                logging.error("没有有效的定价结果")
                return {}
            
            # 创建分析数据
            analysis_data = self._create_analysis_data(market_data, valid_results, summary_stats)
            
            # 创建比较表格
            comparison_table = self._create_comparison_table(valid_results)
            
            # 创建统计分析
            statistical_analysis = self._create_statistical_analysis(valid_results)
            
            self.report_data = {
                'metadata': {
                    'report_time': datetime.now(),
                    'total_options': len(pricing_results),
                    'successful_pricings': len(valid_results),
                    'failed_pricings': len(pricing_results) - len(valid_results)
                },
                'market_summary': self._create_market_summary(market_data, summary_stats),
                'analysis_data': analysis_data,
                'comparison_table': comparison_table,
                'statistical_analysis': statistical_analysis,
                'pricing_methodology': self._get_pricing_methodology(),
                'risk_analysis': self._create_risk_analysis(valid_results),
                'recommendations': self._create_recommendations(valid_results)
            }
            
            return self.report_data
            
        except Exception as e:
            logging.error(f"准备报告数据失败: {e}")
            return {}
    
    def _create_market_summary(self, market_data: pd.DataFrame, summary_stats: Dict) -> Dict:
        """创建市场概况"""
        try:
            if market_data.empty:
                return {}
            
            # 获取基础市场信息
            underlying_price = market_data['underlying_price'].iloc[0]
            atm_volatility = market_data['atm_volatility'].iloc[0]
            time_to_expiry = market_data['time_to_expiry'].iloc[0]
            
            return {
                'underlying_price': underlying_price,
                'atm_volatility': atm_volatility * 100,  # 转换为百分比
                'time_to_expiry_days': time_to_expiry * 365,
                'total_options_available': len(market_data),
                'strike_range': {
                    'min': float(market_data['strike'].min()),
                    'max': float(market_data['strike'].max())
                },
                'iv_range': {
                    'min': float(market_data['mark_iv'].min() * 100),
                    'max': float(market_data['mark_iv'].max() * 100),
                    'mean': float(market_data['mark_iv'].mean() * 100)
                },
                'volume_info': {
                    'total_volume': float(market_data['volume'].sum()),
                    'avg_volume': float(market_data['volume'].mean())
                }
            }
        except Exception as e:
            logging.error(f"创建市场概况失败: {e}")
            return {}
    
    def _create_analysis_data(self, market_data: pd.DataFrame, pricing_results: List[Dict], summary_stats: Dict) -> Dict:
        """创建分析数据"""
        try:
            analysis = {}
            
            # 按期权类型分组分析
            calls = [r for r in pricing_results if r.get('option_type') == 'call']
            puts = [r for r in pricing_results if r.get('option_type') == 'put']
            
            analysis['by_option_type'] = {
                'calls': {
                    'count': len(calls),
                    'avg_pricing_error': np.mean([abs(r.get('avg_replication_error', 0)) for r in calls]) if calls else 0,
                    'avg_transaction_cost': np.mean([r.get('avg_total_costs', 0) for r in calls]) if calls else 0
                },
                'puts': {
                    'count': len(puts),
                    'avg_pricing_error': np.mean([abs(r.get('avg_replication_error', 0)) for r in puts]) if puts else 0,
                    'avg_transaction_cost': np.mean([r.get('avg_total_costs', 0) for r in puts]) if puts else 0
                }
            }
            
            # 按moneyness分组分析
            itm_options = [r for r in pricing_results if r.get('money_category') == 'ITM']
            atm_options = [r for r in pricing_results if r.get('money_category') == 'ATM']
            otm_options = [r for r in pricing_results if r.get('money_category') == 'OTM']
            
            analysis['by_moneyness'] = {
                'ITM': {
                    'count': len(itm_options),
                    'avg_pricing_accuracy': 1 - np.mean([abs(r.get('avg_replication_error', 0)) / max(r.get('market_price', 0.01), 0.01) for r in itm_options]) if itm_options else 0
                },
                'ATM': {
                    'count': len(atm_options),
                    'avg_pricing_accuracy': 1 - np.mean([abs(r.get('avg_replication_error', 0)) / max(r.get('market_price', 0.01), 0.01) for r in atm_options]) if atm_options else 0
                },
                'OTM': {
                    'count': len(otm_options),
                    'avg_pricing_accuracy': 1 - np.mean([abs(r.get('avg_replication_error', 0)) / max(r.get('market_price', 0.01), 0.01) for r in otm_options]) if otm_options else 0
                }
            }
            
            return analysis
            
        except Exception as e:
            logging.error(f"创建分析数据失败: {e}")
            return {}
    
    def _create_comparison_table(self, pricing_results: List[Dict]) -> List[Dict]:
        """创建比较表格数据"""
        try:
            table_data = []
            
            for result in pricing_results:
                if 'error' in result:
                    continue
                
                # 计算关键指标
                market_price = result.get('market_price', 0)
                model_price = result.get('avg_final_worth', result.get('final_worth', 0))
                bs_price = result.get('bs_price', 0)
                
                price_diff = model_price - market_price
                price_diff_pct = (price_diff / market_price * 100) if market_price > 0 else 0
                
                bs_diff = bs_price - market_price
                bs_diff_pct = (bs_diff / market_price * 100) if market_price > 0 else 0
                
                table_row = {
                    'option_id': result.get('option_id', ''),
                    'option_type': result.get('option_type', '').upper(),
                    'strike': result.get('strike_price', result.get('strike', 0)),
                    'market_price': market_price,
                    'bs_price': bs_price,
                    'model_price': model_price,
                    'market_iv': result.get('market_iv', 0) * 100,
                    'model_vs_market_diff': price_diff,
                    'model_vs_market_pct': price_diff_pct,
                    'bs_vs_market_diff': bs_diff,
                    'bs_vs_market_pct': bs_diff_pct,
                    'delta': result.get('delta', 0),
                    'avg_trade_count': result.get('avg_trade_count', result.get('trade_count', 0)),
                    'avg_costs': result.get('avg_total_costs', result.get('total_costs', 0)),
                    'success_rate': result.get('success_rate', 1.0) * 100,
                    'moneyness': result.get('moneyness', 1.0),
                    'money_category': result.get('money_category', 'ATM')
                }
                
                table_data.append(table_row)
            
            # 按执行价排序
            table_data.sort(key=lambda x: (x['option_type'], x['strike']))
            
            return table_data
            
        except Exception as e:
            logging.error(f"创建比较表格失败: {e}")
            return []
    
    def _create_statistical_analysis(self, pricing_results: List[Dict]) -> Dict:
        """创建统计分析"""
        try:
            if not pricing_results:
                return {}
            
            # 提取关键指标
            replication_errors = []
            pricing_accuracy = []
            transaction_costs = []
            success_rates = []
            
            for result in pricing_results:
                if 'error' in result:
                    continue
                
                error = result.get('avg_replication_error', result.get('replication_error', 0))
                replication_errors.append(abs(error))
                
                market_price = result.get('market_price', 0)
                if market_price > 0:
                    accuracy = 1 - abs(error) / market_price
                    pricing_accuracy.append(max(0, min(1, accuracy)))
                
                cost = result.get('avg_total_costs', result.get('total_costs', 0))
                transaction_costs.append(cost)
                
                success_rate = result.get('success_rate', 1.0)
                success_rates.append(success_rate)
            
            # 计算统计量
            stats = {
                'replication_error': {
                    'mean': np.mean(replication_errors) if replication_errors else 0,
                    'median': np.median(replication_errors) if replication_errors else 0,
                    'std': np.std(replication_errors) if replication_errors else 0,
                    'max': np.max(replication_errors) if replication_errors else 0,
                    'min': np.min(replication_errors) if replication_errors else 0
                },
                'pricing_accuracy': {
                    'mean': np.mean(pricing_accuracy) if pricing_accuracy else 0,
                    'median': np.median(pricing_accuracy) if pricing_accuracy else 0,
                    'std': np.std(pricing_accuracy) if pricing_accuracy else 0
                },
                'transaction_costs': {
                    'mean': np.mean(transaction_costs) if transaction_costs else 0,
                    'median': np.median(transaction_costs) if transaction_costs else 0,
                    'total': np.sum(transaction_costs) if transaction_costs else 0
                },
                'overall_success_rate': np.mean(success_rates) if success_rates else 0,
                'total_samples': len(pricing_results)
            }
            
            return stats
            
        except Exception as e:
            logging.error(f"创建统计分析失败: {e}")
            return {}
    
    def _create_risk_analysis(self, pricing_results: List[Dict]) -> Dict:
        """创建风险分析"""
        try:
            risk_metrics = {
                'hedging_effectiveness': 0,
                'transaction_cost_impact': 0,
                'liquidity_risk': 0,
                'model_risk': 0
            }
            
            if not pricing_results:
                return risk_metrics
            
            # 计算对冲有效性
            successful_hedges = sum(1 for r in pricing_results 
                                  if abs(r.get('avg_replication_error', 1)) < 0.1)
            risk_metrics['hedging_effectiveness'] = successful_hedges / len(pricing_results)
            
            # 交易成本影响
            avg_costs = np.mean([r.get('avg_total_costs', 0) for r in pricing_results])
            avg_option_price = np.mean([r.get('market_price', 0) for r in pricing_results])
            if avg_option_price > 0:
                risk_metrics['transaction_cost_impact'] = avg_costs / avg_option_price
            
            # 流动性风险（基于买卖价差）
            bid_ask_spreads = []
            for r in pricing_results:
                bid = r.get('bid_price', 0)
                ask = r.get('ask_price', 0)
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
                if mid > 0:
                    spread_pct = (ask - bid) / mid
                    bid_ask_spreads.append(spread_pct)
            
            if bid_ask_spreads:
                risk_metrics['liquidity_risk'] = np.mean(bid_ask_spreads)
            
            # 模型风险（定价偏差的标准差）
            pricing_errors = [abs(r.get('avg_replication_error', 0)) for r in pricing_results]
            if pricing_errors:
                risk_metrics['model_risk'] = np.std(pricing_errors)
            
            return risk_metrics
            
        except Exception as e:
            logging.error(f"创建风险分析失败: {e}")
            return {}
    
    def _create_recommendations(self, pricing_results: List[Dict]) -> List[str]:
        """创建投资建议"""
        try:
            recommendations = []
            
            if not pricing_results:
                return ["数据不足，无法提供建议"]
            
            # 分析定价偏差
            undervalued_options = []
            overvalued_options = []
            
            for result in pricing_results:
                market_price = result.get('market_price', 0)
                model_price = result.get('avg_final_worth', result.get('final_worth', 0))
                
                if market_price > 0 and model_price > 0:
                    price_diff_pct = (model_price - market_price) / market_price
                    
                    if price_diff_pct > 0.05:  # 低估超过5%
                        undervalued_options.append(result)
                    elif price_diff_pct < -0.05:  # 高估超过5%
                        overvalued_options.append(result)
            
            # 生成建议
            if undervalued_options:
                top_undervalued = sorted(undervalued_options, 
                                       key=lambda x: (x.get('avg_final_worth', 0) - x.get('market_price', 0)) / x.get('market_price', 1), 
                                       reverse=True)[:3]
                rec_text = "**买入机会**: "
                for opt in top_undervalued:
                    rec_text += f"{opt.get('option_id', '')} (低估{((opt.get('avg_final_worth', 0) - opt.get('market_price', 0)) / opt.get('market_price', 1) * 100):.1f}%), "
                recommendations.append(rec_text.rstrip(', '))
            
            if overvalued_options:
                top_overvalued = sorted(overvalued_options,
                                      key=lambda x: (x.get('market_price', 0) - x.get('avg_final_worth', 0)) / x.get('market_price', 1),
                                      reverse=True)[:3]
                rec_text = "**卖出机会**: "
                for opt in top_overvalued:
                    rec_text += f"{opt.get('option_id', '')} (高估{((opt.get('market_price', 0) - opt.get('avg_final_worth', 0)) / opt.get('market_price', 1) * 100):.1f}%), "
                recommendations.append(rec_text.rstrip(', '))
            
            # 风险提示
            high_cost_options = [r for r in pricing_results 
                               if r.get('avg_total_costs', 0) / max(r.get('market_price', 0.01), 0.01) > 0.1]
            if high_cost_options:
                recommendations.append(f"**风险提示**: {len(high_cost_options)}个期权的交易成本较高，需要谨慎考虑")
            
            # 流动性建议
            low_volume_options = [r for r in pricing_results if r.get('volume', 0) < 10]
            if low_volume_options:
                recommendations.append(f"**流动性提示**: {len(low_volume_options)}个期权流动性较低，建议关注买卖价差")
            
            return recommendations if recommendations else ["市场定价相对合理，暂无明显套利机会"]
            
        except Exception as e:
            logging.error(f"创建投资建议失败: {e}")
            return ["生成建议时出现错误"]
    
    def _get_pricing_methodology(self) -> str:
        """获取定价方法说明"""
        return """
## 定价方法说明

本报告采用**蒙特卡洛期权复制策略定价法**，该方法通过模拟期权的动态对冲过程来确定期权的理论价值。

### 核心原理

1. **初始设置**: 借入执行价现值，买入一单位标的资产
2. **动态调整**: 当资产价格穿越执行价时进行交易
   - 价格从上穿过执行价：卖出资产建立空头
   - 价格从下穿过执行价：买入资产平仓空头  
3. **最终清算**: 到期时清算所有头寸，计算复制误差

### 成本模型

- **固定成本**: 按交易金额的固定比例收取
- **滑点成本**: 按交易数量的固定金额收取
- **保证金成本**: 空头头寸需要支付保证金利息

### 优势特点

- 考虑实际交易成本和市场摩擦
- 基于真实交易策略的定价方法  
- 通过大量蒙特卡洛模拟提高准确性
- 提供详细的风险分析指标
"""
    
    def generate_markdown_report(self, output_file: str = None) -> str:
        """生成markdown格式的报告"""
        try:
            if not self.report_data:
                return "# 错误：没有可用的报告数据"
            
            # 生成报告内容
            report_content = self._build_markdown_content()
            
            # 保存文件
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logging.info(f"报告已保存到: {output_file}")
            
            return report_content
            
        except Exception as e:
            logging.error(f"生成markdown报告失败: {e}")
            return f"# 错误：生成报告失败\n\n{str(e)}"
    
    def _build_markdown_content(self) -> str:
        """构建markdown内容"""
        content = []
        
        # 标题和元数据
        metadata = self.report_data.get('metadata', {})
        content.append(f"# Binance ETH期权定价分析报告")
        content.append(f"\n**报告生成时间**: {metadata.get('report_time', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"**分析期权数量**: {metadata.get('total_options', 0)}")
        content.append(f"**成功定价数量**: {metadata.get('successful_pricings', 0)}")
        content.append(f"**失败定价数量**: {metadata.get('failed_pricings', 0)}")
        
        # 市场概况
        market_summary = self.report_data.get('market_summary', {})
        if market_summary:
            content.append("\n## 市场概况")
            content.append(f"- **ETH现价**: ${market_summary.get('underlying_price', 0):,.2f}")
            content.append(f"- **ATM隐含波动率**: {market_summary.get('atm_volatility', 0):.1f}%")
            content.append(f"- **距离到期**: {market_summary.get('time_to_expiry_days', 0):.1f}天")
            content.append(f"- **执行价范围**: ${market_summary.get('strike_range', {}).get('min', 0):,.0f} - ${market_summary.get('strike_range', {}).get('max', 0):,.0f}")
            content.append(f"- **IV范围**: {market_summary.get('iv_range', {}).get('min', 0):.1f}% - {market_summary.get('iv_range', {}).get('max', 0):.1f}%")
        
        # 定价方法
        content.append(self.report_data.get('pricing_methodology', ''))
        
        # 主要发现
        content.append("\n## 主要发现")
        statistical_analysis = self.report_data.get('statistical_analysis', {})
        if statistical_analysis:
            content.append(f"- **平均复制误差**: {statistical_analysis.get('replication_error', {}).get('mean', 0):.4f}")
            content.append(f"- **定价准确率**: {statistical_analysis.get('pricing_accuracy', {}).get('mean', 0):.2%}")
            content.append(f"- **整体成功率**: {statistical_analysis.get('overall_success_rate', 0):.2%}")
            content.append(f"- **平均交易成本**: {statistical_analysis.get('transaction_costs', {}).get('mean', 0):.4f}")
        
        # 详细定价结果
        content.append("\n## 详细定价结果")
        comparison_table = self.report_data.get('comparison_table', [])
        if comparison_table:
            content.append(self._create_markdown_table(comparison_table))
        
        # 按类型分析
        analysis_data = self.report_data.get('analysis_data', {})
        if analysis_data:
            content.append("\n## 分类分析")
            
            # 按期权类型
            by_type = analysis_data.get('by_option_type', {})
            if by_type:
                content.append("\n### 按期权类型")
                content.append("| 类型 | 数量 | 平均误差 | 平均成本 |")
                content.append("|------|------|----------|----------|")
                for opt_type, stats in by_type.items():
                    content.append(f"| {opt_type.upper()} | {stats.get('count', 0)} | {stats.get('avg_pricing_error', 0):.4f} | {stats.get('avg_transaction_cost', 0):.4f} |")
            
            # 按moneyness
            by_moneyness = analysis_data.get('by_moneyness', {})
            if by_moneyness:
                content.append("\n### 按价值状态")
                content.append("| 状态 | 数量 | 平均准确率 |")
                content.append("|------|------|------------|")
                for money_cat, stats in by_moneyness.items():
                    content.append(f"| {money_cat} | {stats.get('count', 0)} | {stats.get('avg_pricing_accuracy', 0):.2%} |")
        
        # 风险分析
        risk_analysis = self.report_data.get('risk_analysis', {})
        if risk_analysis:
            content.append("\n## 风险分析")
            content.append(f"- **对冲有效性**: {risk_analysis.get('hedging_effectiveness', 0):.2%}")
            content.append(f"- **交易成本影响**: {risk_analysis.get('transaction_cost_impact', 0):.2%}")
            content.append(f"- **流动性风险**: {risk_analysis.get('liquidity_risk', 0):.2%}")
            content.append(f"- **模型风险**: {risk_analysis.get('model_risk', 0):.4f}")
        
        # 投资建议
        recommendations = self.report_data.get('recommendations', [])
        if recommendations:
            content.append("\n## 投资建议")
            for rec in recommendations:
                content.append(f"- {rec}")
        
        # 免责声明
        content.append("\n## 免责声明")
        content.append("本报告仅供参考，不构成投资建议。期权交易存在重大风险，投资者应谨慎决策并承担相应风险。")
        
        return '\n'.join(content)
    
    def _create_markdown_table(self, table_data: List[Dict]) -> str:
        """创建markdown表格"""
        if not table_data:
            return "暂无数据"
        
        # 表头
        headers = [
            "期权ID", "类型", "执行价", "市场价", "BS价格", "模型价格", 
            "市场IV%", "模型差异", "模型差异%", "交易次数", "成本", "成功率%"
        ]
        
        table_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |"
        ]
        
        # 数据行
        for row in table_data:
            row_data = [
                row.get('option_id', ''),
                row.get('option_type', ''),
                f"{row.get('strike', 0):.0f}",
                f"{row.get('market_price', 0):.3f}",
                f"{row.get('bs_price', 0):.3f}",
                f"{row.get('model_price', 0):.3f}",
                f"{row.get('market_iv', 0):.1f}",
                f"{row.get('model_vs_market_diff', 0):.3f}",
                f"{row.get('model_vs_market_pct', 0):.1f}",
                f"{row.get('avg_trade_count', 0):.1f}",
                f"{row.get('avg_costs', 0):.3f}",
                f"{row.get('success_rate', 0):.1f}"
            ]
            table_lines.append("| " + " | ".join(row_data) + " |")
        
        return "\n".join(table_lines)