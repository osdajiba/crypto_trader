#!/usr/bin/env python3
"""
Binance期权定价系统主程序
集成数据爬取、处理、定价和报告生成
"""

import argparse
import logging
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 添加src目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.scraper import BinanceScraper, create_sample_option_data
from src.option_processor import OptionDataProcessor, create_quick_filter_presets
from src.pricer import OptionPricer
from reporter import OptionPricingReportGenerator

class BinanceOptionPricingSystem:
    """Binance期权定价系统"""
    
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        self.setup_logging()
        self.output_dir = Path(self.config.get('output_dir', './output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 组件
        self.scraper = None
        self.processor = OptionDataProcessor()
        self.report_generator = OptionPricingReportGenerator()
        
        logging.info("Binance期权定价系统初始化完成")
    
    def _load_config(self, config_file: str = None) -> dict:
        """加载配置文件"""
        default_config = {
            'scraping': {
                'use_selenium': True,
                'headless': True,
                'use_sample_data': False  # 是否使用示例数据
            },
            'data_processing': {
                'filter_preset': 'active_options',  # 筛选预设
                'custom_filters': {}
            },
            'pricing': {
                'n_paths': 30000,  # 蒙特卡洛路径数
                'use_gpu': True,
                'max_workers': 4,
                'default_risk_free_rate': 0.05
            },
            'output_dir': './output',
            'log_level': 'INFO'
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                    # 深度合并配置
                    default_config.update(file_config)
                    logging.info(f"已加载配置文件: {config_file}")
            except Exception as e:
                logging.warning(f"加载配置文件失败，使用默认配置: {e}")
        
        return default_config
    
    def setup_logging(self):
        """设置日志"""
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        
        # 创建日志目录
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        # 配置日志
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        # 文件处理器
        file_handler = logging.FileHandler(
            log_dir / f'binance_pricing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # 配置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    def scrape_data(self) -> dict:
        """爬取数据"""
        try:
            scraping_config = self.config.get('scraping', {})
            
            if scraping_config.get('use_sample_data', False):
                logging.info("使用示例数据（跳过网页爬取）")
                data = create_sample_option_data()
            else:
                logging.info("开始爬取Binance期权数据...")
                self.scraper = BinanceScraper(
                    use_selenium=scraping_config.get('use_selenium', True),
                    headless=scraping_config.get('headless', True)
                )
                
                # 保存原始数据
                raw_data_file = self.output_dir / 'raw_option_data.csv'
                data = self.scraper.scrape(save_file=str(raw_data_file))
            
            return data
            
        except Exception as e:
            logging.error(f"数据爬取失败: {e}")
            logging.info("尝试使用示例数据...")
            return create_sample_option_data()
    
    def process_data(self, raw_data: dict) -> tuple:
        """处理数据"""
        try:
            logging.info("开始数据处理...")
            
            # 获取筛选配置
            processing_config = self.config.get('data_processing', {})
            filter_preset = processing_config.get('filter_preset', 'active_options')
            custom_filters = processing_config.get('custom_filters', {})
            
            # 获取预设筛选条件
            filter_presets = create_quick_filter_presets()
            filters = filter_presets.get(filter_preset, {})
            filters.update(custom_filters)
            
            logging.info(f"使用筛选预设: {filter_preset}")
            logging.info(f"筛选条件: {filters}")
            
            # 处理数据
            processed_df, pricing_params = self.processor.process(raw_data, filters)
            
            # 保存处理后的数据
            processed_data_file = self.output_dir / 'processed_option_data.csv'
            self.processor.save_processed_data(str(processed_data_file))
            
            # 获取汇总统计
            summary_stats = self.processor.get_summary_stats(processed_df)
            logging.info(f"数据处理完成，准备定价 {len(pricing_params)} 个期权")
            
            return processed_df, pricing_params, summary_stats
            
        except Exception as e:
            logging.error(f"数据处理失败: {e}")
            raise
    
    def price_single_option(self, params: dict) -> dict:
        """定价单个期权"""
        try:
            # 创建定价器
            pricer = OptionPricer(
                S0=params['S0'],
                K=params['K'],
                r=params['r'],
                sigma=params['sigma'],
                T=params['T'],
                n_steps=params['n_steps'],
                fixed_cost_ratio=params['fixed_cost_ratio'],
                slippage=params['slippage'],
                margin_ratio=params['margin_ratio']
            )
            
            # 运行蒙特卡洛定价
            pricing_config = self.config.get('pricing', {})
            result = pricer.price_monte_carlo(
                n_paths=pricing_config.get('n_paths', 30000),
                use_gpu=pricing_config.get('use_gpu', True)
            )
            
            # 添加原始参数信息
            result.update({
                'option_id': params.get('option_id'),
                'option_type': params.get('option_type'),
                'strike_price': params.get('strike_price'),
                'market_price': params.get('market_price'),
                'market_iv': params.get('market_iv'),
                'delta': params.get('delta'),
                'bid_price': params.get('bid_price'),
                'ask_price': params.get('ask_price'),
                'moneyness': params.get('moneyness'),
                'money_category': params.get('money_category'),
                'intrinsic_value': params.get('intrinsic_value'),
                'time_value': params.get('time_value'),
                'description': params.get('description')
            })
            
            return result
            
        except Exception as e:
            logging.error(f"期权定价失败: {params.get('option_id', 'Unknown')}, 错误: {e}")
            return {
                'error': str(e),
                'option_id': params.get('option_id', 'Unknown'),
                'params': params
            }
    
    def price_options_batch(self, pricing_params: list) -> list:
        """批量定价期权"""
        try:
            pricing_config = self.config.get('pricing', {})
            max_workers = min(pricing_config.get('max_workers', 4), mp.cpu_count())
            
            logging.info(f"开始批量定价，共 {len(pricing_params)} 个期权，使用 {max_workers} 个进程")
            
            results = []
            
            if max_workers <= 1:
                # 单进程模式
                for i, params in enumerate(pricing_params):
                    result = self.price_single_option(params)
                    results.append(result)
                    if (i + 1) % 5 == 0:
                        logging.info(f"已完成 {i + 1}/{len(pricing_params)} 个期权定价")
            else:
                # 多进程模式
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # 提交所有任务
                    future_to_params = {
                        executor.submit(self.price_single_option, params): params
                        for params in pricing_params
                    }
                    
                    # 收集结果
                    completed = 0
                    for future in as_completed(future_to_params):
                        result = future.result()
                        results.append(result)
                        completed += 1
                        
                        if completed % 5 == 0 or completed == len(pricing_params):
                            logging.info(f"已完成 {completed}/{len(pricing_params)} 个期权定价")
            
            # 统计结果
            successful = len([r for r in results if 'error' not in r])
            failed = len(results) - successful
            
            logging.info(f"批量定价完成：成功 {successful} 个，失败 {failed} 个")
            
            return results
            
        except Exception as e:
            logging.error(f"批量定价失败: {e}")
            raise
    
    def generate_report(self, market_data, pricing_results, summary_stats) -> str:
        """生成报告"""
        try:
            logging.info("开始生成报告...")
            
            # 准备报告数据
            self.report_generator.prepare_report_data(
                market_data=market_data,
                pricing_results=pricing_results,
                summary_stats=summary_stats
            )
            
            # 生成报告文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f'binance_eth_option_pricing_report_{timestamp}.md'
            
            # 生成markdown报告
            report_content = self.report_generator.generate_markdown_report(str(report_file))
            
            logging.info(f"报告已生成: {report_file}")
            
            return str(report_file)
            
        except Exception as e:
            logging.error(f"报告生成失败: {e}")
            raise
    
    def run_full_analysis(self):
        """运行完整分析流程"""
        try:
            start_time = datetime.now()
            logging.info("="*80)
            logging.info("开始Binance ETH期权定价分析")
            logging.info("="*80)
            
            # 1. 爬取数据
            logging.info("步骤 1/5: 数据爬取")
            raw_data = self.scrape_data()
            
            if not raw_data or not raw_data.get('options'):
                raise Exception("无法获取期权数据")
            
            # 2. 处理数据
            logging.info("步骤 2/5: 数据处理")
            market_data, pricing_params, summary_stats = self.process_data(raw_data)
            
            if not pricing_params:
                raise Exception("没有可用于定价的期权数据")
            
            # 3. 期权定价
            logging.info("步骤 3/5: 期权定价")
            pricing_results = self.price_options_batch(pricing_params)
            
            # 4. 保存结果
            logging.info("步骤 4/5: 保存结果")
            results_file = self.output_dir / f'pricing_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            
            # 转换结果为DataFrame并保存
            import pandas as pd
            results_df = pd.DataFrame(pricing_results)
            results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
            logging.info(f"定价结果已保存: {results_file}")
            
            # 5. 生成报告
            logging.info("步骤 5/5: 生成报告")
            report_file = self.generate_report(market_data, pricing_results, summary_stats)
            
            # 完成
            elapsed_time = datetime.now() - start_time
            logging.info("="*80)
            logging.info(f"分析完成！总耗时: {elapsed_time}")
            logging.info(f"输出目录: {self.output_dir}")
            logging.info(f"报告文件: {report_file}")
            logging.info("="*80)
            
            return {
                'success': True,
                'output_dir': str(self.output_dir),
                'report_file': report_file,
                'results_file': str(results_file),
                'elapsed_time': str(elapsed_time)
            }
            
        except Exception as e:
            logging.error(f"分析失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            # 清理资源
            if self.scraper:
                self.scraper.close()
    
    def cleanup(self):
        """清理资源"""
        if self.scraper:
            self.scraper.close()

def create_default_config():
    """创建默认配置文件"""
    config = {
        'scraping': {
            'use_selenium': True,
            'headless': True,
            'use_sample_data': False
        },
        'data_processing': {
            'filter_preset': 'active_options',
            'custom_filters': {
                'min_volume': 5,
                'max_bid_ask_spread': 0.2
            }
        },
        'pricing': {
            'n_paths': 30000,
            'use_gpu': True,
            'max_workers': 4,
            'default_risk_free_rate': 0.05
        },
        'output_dir': './output',
        'log_level': 'INFO'
    }
    
    config_file = 'binance_option_config.yaml'
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"默认配置文件已创建: {config_file}")
    return config_file

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Binance ETH期权定价系统')
    parser.add_argument('--config', '-c', help='配置文件路径')
    parser.add_argument('--create-config', action='store_true', help='创建默认配置文件')
    parser.add_argument('--sample-data', action='store_true', help='使用示例数据（跳过网页爬取）')
    parser.add_argument('--output', '-o', help='输出目录')
    parser.add_argument('--workers', '-w', type=int, help='并行进程数')
    parser.add_argument('--paths', '-p', type=int, help='蒙特卡洛路径数')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 创建配置文件
    if args.create_config:
        create_default_config()
        return
    
    try:
        # 初始化系统
        system = BinanceOptionPricingSystem(args.config)
        
        # 命令行参数覆盖配置
        if args.sample_data:
            system.config['scraping']['use_sample_data'] = True
        
        if args.output:
            system.config['output_dir'] = args.output
            system.output_dir = Path(args.output)
            system.output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.workers:
            system.config['pricing']['max_workers'] = args.workers
        
        if args.paths:
            system.config['pricing']['n_paths'] = args.paths
        
        if args.verbose:
            system.config['log_level'] = 'DEBUG'
            system.setup_logging()
        
        # 运行分析
        result = system.run_full_analysis()
        
        if result['success']:
            print(f"\n✅ 分析成功完成！")
            print(f"📁 输出目录: {result['output_dir']}")
            print(f"📊 报告文件: {result['report_file']}")
            print(f"⏱️  总耗时: {result['elapsed_time']}")
        else:
            print(f"\n❌ 分析失败: {result['error']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n用户中断执行")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()