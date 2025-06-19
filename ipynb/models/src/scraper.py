"""
Binance期权数据爬虫模块
爬取ETH期权链数据，包含执行价、IV、Delta、买卖价等信息
"""

import requests
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta
import re
from typing import Dict, List, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

class BinanceScraper:
    """Binance期权数据爬虫"""
    
    def __init__(self, use_selenium=True, headless=True):
        self.base_url = "https://www.binance.com/zh-CN/eoptions/ETHUSDT"
        self.session = requests.Session()
        self.use_selenium = use_selenium
        self.headless = headless
        self.driver = None
        
        # 设置请求头
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def _init_selenium(self):
        """初始化Selenium WebDriver"""
        if self.driver is None:
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            try:
                self.driver = webdriver.Chrome(options=chrome_options)
                logging.info("Selenium WebDriver初始化成功")
            except Exception as e:
                logging.error(f"Selenium WebDriver初始化失败: {e}")
                raise
    
    def _parse_expiry_date(self, expiry_str: str) -> datetime:
        """解析到期时间字符串"""
        try:
            # 处理类似 "2025-06-19 04:14:13**(日交割)**" 的格式
            if "**" in expiry_str:
                date_part = expiry_str.split("**")[0].strip()
            else:
                date_part = expiry_str.strip()
            
            # 尝试不同的日期格式
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%m-%d %H:%M:%S",
                "%m-%d"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_part, fmt)
                except ValueError:
                    continue
            
            # 如果都失败，返回默认值（明天）
            return datetime.now() + timedelta(days=1)
            
        except Exception as e:
            logging.warning(f"解析到期时间失败: {expiry_str}, 错误: {e}")
            return datetime.now() + timedelta(days=1)
    
    def _parse_percentage(self, pct_str: str) -> float:
        """解析百分比字符串"""
        try:
            if pct_str and pct_str != '--' and '%' in pct_str:
                return float(pct_str.replace('%', '').strip()) / 100.0
            return 0.0
        except:
            return 0.0
    
    def _parse_float(self, value_str: str) -> float:
        """解析浮点数字符串"""
        try:
            if value_str and value_str != '--' and value_str != '':
                # 移除逗号和其他特殊字符
                clean_str = re.sub(r'[^\d.-]', '', str(value_str))
                return float(clean_str)
            return 0.0
        except:
            return 0.0
    
    def scrape_option_chain_selenium(self) -> Dict:
        """使用Selenium爬取期权链数据"""
        self._init_selenium()
        
        try:
            logging.info("正在访问Binance期权页面...")
            self.driver.get(self.base_url)
            
            # 等待页面加载
            wait = WebDriverWait(self.driver, 20)
            
            # 等待期权数据表格加载
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "option-table")))
            time.sleep(5)  # 额外等待确保数据完全加载
            
            # 获取页面源码
            html = self.driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            
            # 解析基础信息
            basic_info = self._parse_basic_info(soup)
            
            # 解析期权链数据
            option_data = self._parse_option_table(soup)
            
            return {
                'basic_info': basic_info,
                'options': option_data,
                'scrape_time': datetime.now()
            }
            
        except Exception as e:
            logging.error(f"Selenium爬取失败: {e}")
            raise
        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None
    
    def _parse_basic_info(self, soup) -> Dict:
        """解析基础市场信息"""
        basic_info = {
            'underlying_price': 0.0,
            'atm_volatility': 0.0,
            'expiry_date': '',
            'expiry_type': '',
            'time_to_expiry': 0.0
        }
        
        try:
            # 查找标的价格
            price_elements = soup.find_all(text=re.compile(r'ETHUSDT价格'))
            if price_elements:
                for element in price_elements:
                    parent = element.parent
                    if parent:
                        # 查找价格数值
                        siblings = parent.find_next_siblings()
                        for sibling in siblings:
                            price_match = re.search(r'[\d,]+\.?\d*', sibling.get_text())
                            if price_match:
                                basic_info['underlying_price'] = self._parse_float(price_match.group())
                                break
            
            # 查找ATM波动率
            vol_elements = soup.find_all(text=re.compile(r'ATM 波动率'))
            if vol_elements:
                for element in vol_elements:
                    parent = element.parent
                    if parent:
                        vol_text = parent.get_text()
                        vol_match = re.search(r'(\d+\.?\d*)%', vol_text)
                        if vol_match:
                            basic_info['atm_volatility'] = float(vol_match.group(1)) / 100.0
                            break
            
            # 查找到期时间
            expiry_elements = soup.find_all(text=re.compile(r'到期时间'))
            if expiry_elements:
                for element in expiry_elements:
                    parent = element.parent
                    if parent:
                        expiry_text = parent.get_text()
                        basic_info['expiry_date'] = expiry_text
                        
                        # 判断到期类型
                        if '日交割' in expiry_text:
                            basic_info['expiry_type'] = 'daily'
                        elif '周交割' in expiry_text:
                            basic_info['expiry_type'] = 'weekly'
                        
                        # 计算到期时间（天数）
                        if '天' in expiry_text:
                            days_match = re.search(r'(\d+)天', expiry_text)
                            if days_match:
                                basic_info['time_to_expiry'] = float(days_match.group(1)) / 365.0
                        else:
                            # 默认设置为1天
                            basic_info['time_to_expiry'] = 1.0 / 365.0
                        break
            
            logging.info(f"基础信息: {basic_info}")
            
        except Exception as e:
            logging.warning(f"解析基础信息失败: {e}")
        
        return basic_info
    
    def _parse_option_table(self, soup) -> List[Dict]:
        """解析期权表格数据"""
        options = []
        
        try:
            # 查找期权数据行
            # Binance的期权表格通常使用特定的class或结构
            option_rows = soup.find_all('tr') + soup.find_all('div', class_=re.compile(r'option|row'))
            
            for row in option_rows:
                row_text = row.get_text(strip=True)
                
                # 查找包含执行价的行
                strike_match = re.search(r'(\d{4})', row_text)
                if not strike_match:
                    continue
                
                strike = float(strike_match.group(1))
                
                # 解析该行的期权数据
                option_data = self._parse_option_row(row, strike)
                if option_data:
                    options.extend(option_data)
            
            logging.info(f"解析到 {len(options)} 个期权")
            
        except Exception as e:
            logging.warning(f"解析期权表格失败: {e}")
        
        return options
    
    def _parse_option_row(self, row, strike: float) -> List[Dict]:
        """解析单行期权数据"""
        options = []
        row_text = row.get_text()
        
        try:
            # 查找数值
            numbers = re.findall(r'[\d,]+\.?\d*', row_text)
            percentages = re.findall(r'(\d+\.?\d*)%', row_text)
            
            # Call期权数据
            call_option = {
                'strike': strike,
                'option_type': 'call',
                'delta': 0.0,
                'bid_price': 0.0,
                'ask_price': 0.0,
                'mark_price': 0.0,
                'bid_iv': 0.0,
                'ask_iv': 0.0,
                'mark_iv': 0.0,
                'volume': 0.0,
                'open_interest': 0.0
            }
            
            # Put期权数据
            put_option = {
                'strike': strike,
                'option_type': 'put',
                'delta': 0.0,
                'bid_price': 0.0,
                'ask_price': 0.0,
                'mark_price': 0.0,
                'bid_iv': 0.0,
                'ask_iv': 0.0,
                'mark_iv': 0.0,
                'volume': 0.0,
                'open_interest': 0.0
            }
            
            # 尝试解析数值（这里需要根据实际HTML结构调整）
            if len(numbers) >= 6:
                call_option['mark_price'] = self._parse_float(numbers[0])
                call_option['bid_price'] = self._parse_float(numbers[1])
                call_option['ask_price'] = self._parse_float(numbers[2])
                put_option['mark_price'] = self._parse_float(numbers[3])
                put_option['bid_price'] = self._parse_float(numbers[4])
                put_option['ask_price'] = self._parse_float(numbers[5])
            
            if len(percentages) >= 4:
                call_option['mark_iv'] = self._parse_percentage(percentages[0] + '%')
                call_option['bid_iv'] = self._parse_percentage(percentages[1] + '%')
                put_option['mark_iv'] = self._parse_percentage(percentages[2] + '%')
                put_option['ask_iv'] = self._parse_percentage(percentages[3] + '%')
            
            options.extend([call_option, put_option])
            
        except Exception as e:
            logging.debug(f"解析期权行失败: {e}")
        
        return options
    
    def scrape_option_chain_api(self) -> Dict:
        """使用API方式爬取期权链数据（备用方案）"""
        try:
            # Binance API端点（可能需要调整）
            api_url = "https://eapi.binance.com/eapi/v1/ticker"
            
            params = {
                'symbol': 'ETHUSDT'
            }
            
            response = self.session.get(api_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                # 这里需要根据实际API响应格式处理数据
                return self._process_api_data(data)
            else:
                logging.warning(f"API请求失败: {response.status_code}")
                return {}
                
        except Exception as e:
            logging.error(f"API爬取失败: {e}")
            return {}
    
    def _process_api_data(self, data: Dict) -> Dict:
        """处理API返回的数据"""
        # 这里需要根据实际API响应格式实现
        return {
            'basic_info': {},
            'options': [],
            'scrape_time': datetime.now()
        }
    
    def save_to_csv(self, data: Dict, filename: str):
        """将数据保存为CSV格式"""
        try:
            if not data.get('options'):
                logging.warning("没有期权数据可保存")
                return
            
            # 转换为DataFrame
            df = pd.DataFrame(data['options'])
            
            # 添加基础信息列
            basic_info = data.get('basic_info', {})
            df['underlying_price'] = basic_info.get('underlying_price', 0)
            df['atm_volatility'] = basic_info.get('atm_volatility', 0)
            df['time_to_expiry'] = basic_info.get('time_to_expiry', 0)
            df['expiry_type'] = basic_info.get('expiry_type', '')
            df['scrape_time'] = data.get('scrape_time', datetime.now())
            
            # 保存CSV
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            logging.info(f"数据已保存到: {filename}")
            
            # 打印统计信息
            calls = df[df['option_type'] == 'call']
            puts = df[df['option_type'] == 'put']
            
            logging.info(f"总期权数: {len(df)}")
            logging.info(f"看涨期权数: {len(calls)}")
            logging.info(f"看跌期权数: {len(puts)}")
            logging.info(f"执行价范围: {df['strike'].min()} - {df['strike'].max()}")
            
        except Exception as e:
            logging.error(f"保存CSV失败: {e}")
    
    def scrape(self, save_file: Optional[str] = None) -> Dict:
        """主要的爬取方法"""
        logging.info("开始爬取Binance期权数据...")
        
        data = {}
        
        # 优先使用Selenium方式
        if self.use_selenium:
            try:
                data = self.scrape_option_chain_selenium()
            except Exception as e:
                logging.warning(f"Selenium方式失败，尝试API方式: {e}")
                data = self.scrape_option_chain_api()
        else:
            data = self.scrape_option_chain_api()
        
        # 保存数据
        if save_file and data:
            self.save_to_csv(data, save_file)
        
        return data
    
    def close(self):
        """清理资源"""
        if self.driver:
            self.driver.quit()
            self.driver = None

# 辅助函数
def create_sample_option_data():
    """创建示例期权数据（用于测试）"""
    logging.info("创建示例期权数据...")
    
    # 基于提供的Binance数据创建示例
    strikes = [2250, 2300, 2350, 2400, 2425, 2450, 2475, 2500, 2525, 2550, 2575, 2600, 2625, 2650]
    underlying_price = 2522.5
    atm_vol = 0.518
    time_to_expiry = 1.0 / 365  # 1天
    
    options = []
    
    for strike in strikes:
        # 计算moneyness
        moneyness = underlying_price / strike
        
        # 估算IV（基于moneyness调整）
        if moneyness > 1.1:  # Deep ITM
            iv_call = atm_vol * 0.8
            iv_put = atm_vol * 1.3
        elif moneyness > 1.05:  # ITM
            iv_call = atm_vol * 0.9
            iv_put = atm_vol * 1.2
        elif moneyness > 0.95:  # ATM
            iv_call = atm_vol
            iv_put = atm_vol
        elif moneyness > 0.9:  # OTM
            iv_call = atm_vol * 1.1
            iv_put = atm_vol * 0.9
        else:  # Deep OTM
            iv_call = atm_vol * 1.3
            iv_put = atm_vol * 0.8
        
        # 简单的Black-Scholes估价（用于生成合理的价格）
        from scipy.stats import norm
        import math
        
        r = 0.05  # 假设的无风险利率
        
        def bs_call_price(S, K, T, r, sigma):
            if T <= 0:
                return max(S - K, 0)
            d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
            d2 = d1 - sigma*math.sqrt(T)
            return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
        
        def bs_put_price(S, K, T, r, sigma):
            if T <= 0:
                return max(K - S, 0)
            d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
            d2 = d1 - sigma*math.sqrt(T)
            return K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        # Call期权
        call_price = bs_call_price(underlying_price, strike, time_to_expiry, r, iv_call)
        call_delta = norm.cdf((math.log(underlying_price/strike) + (r + 0.5*iv_call**2)*time_to_expiry) / (iv_call*math.sqrt(time_to_expiry)))
        
        # Put期权
        put_price = bs_put_price(underlying_price, strike, time_to_expiry, r, iv_put)
        put_delta = call_delta - 1
        
        # Call期权数据
        options.append({
            'strike': strike,
            'option_type': 'call',
            'delta': call_delta,
            'bid_price': call_price * 0.98,
            'ask_price': call_price * 1.02,
            'mark_price': call_price,
            'bid_iv': iv_call * 0.98,
            'ask_iv': iv_call * 1.02,
            'mark_iv': iv_call,
            'volume': np.random.uniform(10, 1000),
            'open_interest': np.random.uniform(1000, 100000),
            'underlying_price': underlying_price,
            'atm_volatility': atm_vol,
            'time_to_expiry': time_to_expiry,
            'expiry_type': 'daily',
            'scrape_time': datetime.now()
        })
        
        # Put期权数据
        options.append({
            'strike': strike,
            'option_type': 'put',
            'delta': put_delta,
            'bid_price': put_price * 0.98,
            'ask_price': put_price * 1.02,
            'mark_price': put_price,
            'bid_iv': iv_put * 0.98,
            'ask_iv': iv_put * 1.02,
            'mark_iv': iv_put,
            'volume': np.random.uniform(10, 1000),
            'open_interest': np.random.uniform(1000, 100000),
            'underlying_price': underlying_price,
            'atm_volatility': atm_vol,
            'time_to_expiry': time_to_expiry,
            'expiry_type': 'daily',
            'scrape_time': datetime.now()
        })
    
    return {
        'basic_info': {
            'underlying_price': underlying_price,
            'atm_volatility': atm_vol,
            'time_to_expiry': time_to_expiry,
            'expiry_type': 'daily',
            'expiry_date': '2025-06-19 04:14:13**(日交割)**'
        },
        'options': options,
        'scrape_time': datetime.now()
    }