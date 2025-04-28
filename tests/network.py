#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CCXT Binance 连接诊断工具

此工具用于诊断高延迟VPN环境下CCXT-Binance API连接问题。
提供全面的网络连接、代理配置、TLS握手和API请求分析。

用法: python binance_diagnostics.py

可选参数:
  --proxy URL      指定要测试的HTTP代理 (例如: http://127.0.0.1:7890)
  --symbol PAIR    指定要测试的交易对 (默认: BTC/USDT)
  --timeframe TF   指定要测试的时间周期 (默认: 1h)
  --verbose        启用详细日志
"""

import argparse
import asyncio
import json
import random
import socket
import ssl
import statistics
import sys
import time
import traceback
from datetime import datetime, timedelta
from urllib.parse import urlparse

try:
    import requests
    import ccxt
    import ccxt.async_support as ccxt_async
except ImportError:
    print("缺少必要的依赖库。请安装：")
    print("pip install requests ccxt")
    sys.exit(1)


class BinanceDiagnostics:
    """Binance API 连接诊断工具"""

    def __init__(self, proxy=None, symbol="BTC/USDT", timeframe="1h", verbose=False):
        """初始化诊断工具"""
        self.proxy = proxy
        self.symbol = symbol
        self.timeframe = timeframe
        self.verbose = verbose
        self.binance_endpoints = {
            "spot": "https://api.binance.com/api/v3/ping",
            "futures": "https://fapi.binance.com/fapi/v1/ping",
        }
        
    def log(self, message, level="INFO"):
        """打印日志信息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} | {level:<8} | {message}")
    
    def error(self, message):
        """打印错误信息"""
        self.log(message, "ERROR")
    
    def debug(self, message):
        """打印调试信息"""
        if self.verbose:
            self.log(message, "DEBUG")
    
    def print_section(self, title):
        """打印分节标题"""
        print("\n" + "=" * 80)
        print(f" {title}")
        print("=" * 80)
    
    def test_basic_connection(self, url, attempts=5):
        """测试基本连接并测量延迟"""
        self.log(f"测试与 {url} 的基本连接")
        
        results = []
        success = 0
        
        for i in range(attempts):
            try:
                proxies = None
                if self.proxy:
                    proxies = {"http": self.proxy, "https": self.proxy}
                
                start_time = time.time()
                response = requests.get(url, proxies=proxies, timeout=10)
                end_time = time.time()
                
                latency = (end_time - start_time) * 1000  # 毫秒
                status = response.status_code
                results.append((True, status, latency))
                success += 1
                self.log(f"尝试 {i+1}: 成功, 状态码 {status}, 延迟 {latency:.2f}ms")
            except Exception as e:
                self.error(f"尝试 {i+1}: 失败, 错误: {str(e)}")
                results.append((False, None, None))
                
        if success > 0:
            latencies = [r[2] for r in results if r[0]]
            avg_latency = statistics.mean(latencies)
            std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
            
            self.log(f"成功率: {success}/{attempts} ({success/attempts*100:.1f}%)")
            self.log(f"平均延迟: {avg_latency:.2f}ms ± {std_latency:.2f}ms")
            
            if avg_latency > 500:
                self.log(f"警告: 延迟超过 500ms, 可能导致API请求超时", "WARNING")
            
        # DNS解析测试
        try:
            hostname = urlparse(url).netloc
            ip = socket.gethostbyname(hostname)
            self.log(f"DNS解析: {hostname} -> {ip}")
        except Exception as e:
            self.error(f"DNS解析失败: {str(e)}")
            
        return results
    
    def diagnose_proxy_connection(self, test_urls=None):
        """诊断代理连接问题"""
        if not self.proxy:
            self.log("未指定代理，跳过代理诊断")
            return False
            
        if test_urls is None:
            test_urls = list(self.binance_endpoints.values())
            
        self.log(f"正在诊断代理: {self.proxy}")
        
        # 解析代理URL
        parsed = urlparse(self.proxy)
        proxy_host = parsed.hostname
        proxy_port = parsed.port
        
        if not proxy_host or not proxy_port:
            self.error(f"无效的代理URL格式: {self.proxy}")
            return False
        
        # 测试代理TCP连接
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect((proxy_host, proxy_port))
            self.log(f"✓ 代理TCP连接成功 ({proxy_host}:{proxy_port})")
            s.close()
        except Exception as e:
            self.error(f"✗ 代理TCP连接失败: {str(e)}")
            return False
        
        # 测试代理HTTP请求
        success_count = 0
        
        for target in test_urls:
            try:
                proxies = {
                    'http': self.proxy,
                    'https': self.proxy
                }
                
                start = time.time()
                response = requests.get(
                    target, 
                    proxies=proxies,
                    timeout=30
                )
                elapsed = (time.time() - start) * 1000
                
                self.log(f"✓ 通过代理访问 {target} 成功 ({elapsed:.0f}ms)")
                self.log(f"  状态码: {response.status_code}")
                self.log(f"  响应大小: {len(response.content)} bytes")
                
                # 检查TLS握手时间
                if hasattr(response, 'elapsed'):
                    self.log(f"  请求耗时: {response.elapsed.total_seconds()*1000:.0f}ms")
                
                success_count += 1
            except Exception as e:
                self.error(f"✗ 通过代理访问 {target} 失败: {str(e)}")
        
        return success_count > 0

    def analyze_tls_handshake(self, host, port=443, use_proxy=True):
        """分析TLS握手过程"""
        class ProxySocket:
            """通过HTTP代理的套接字实现"""
            def __init__(self, proxy_host, proxy_port):
                self.proxy_host = proxy_host
                self.proxy_port = proxy_port
                self._sock = None
                
            def connect(self, target):
                host, port = target
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.connect((self.proxy_host, self.proxy_port))
                
                # CONNECT请求
                connect_str = f"CONNECT {host}:{port} HTTP/1.1\r\n"
                connect_str += f"Host: {host}:{port}\r\n"
                connect_str += "\r\n"
                self._sock.sendall(connect_str.encode())
                
                # 读取响应
                response = b""
                while b"\r\n\r\n" not in response:
                    chunk = self._sock.recv(4096)
                    if not chunk:
                        break
                    response += chunk
                
                # 检查是否成功
                if not response.startswith(b"HTTP/1.1 200"):
                    raise ConnectionError(f"代理连接失败: {response.decode()}")
            
            def sendall(self, data):
                return self._sock.sendall(data)
                
            def recv(self, bufsize):
                return self._sock.recv(bufsize)
                
            def settimeout(self, timeout):
                return self._sock.settimeout(timeout)
                
            def close(self):
                return self._sock.close()

        self.log(f"分析与 {host}:{port} 的TLS握手")
        
        # 创建适当的socket
        if use_proxy and self.proxy:
            parsed = urlparse(self.proxy)
            proxy_sock = ProxySocket(parsed.hostname, parsed.port)
            proxy_sock.connect((host, port))
            sock = proxy_sock
            self.log(f"使用代理: {self.proxy}")
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            self.log("直接连接 (不使用代理)")
        
        sock.settimeout(30)
        
        try:
            # 记录开始时间
            start = time.time()
            
            # 创建SSL上下文
            context = ssl.create_default_context()
            
            # 开始TLS握手
            secure_sock = context.wrap_socket(sock, server_hostname=host)
            
            # 记录握手完成时间
            handshake_time = (time.time() - start) * 1000
            
            # 获取证书信息
            cert = secure_sock.getpeercert()
            
            self.log(f"✓ TLS握手成功 ({handshake_time:.0f}ms)")
            self.log(f"  协议版本: {secure_sock.version()}")
            self.log(f"  加密套件: {secure_sock.cipher()}")
            
            if cert:
                subject = dict(x[0] for x in cert['subject'])
                issuer = dict(x[0] for x in cert['issuer'])
                
                self.log(f"  证书通用名称: {subject.get('commonName', 'N/A')}")
                self.log(f"  证书颁发者: {issuer.get('commonName', 'N/A')}")
                
                if 'notAfter' in cert:
                    self.log(f"  证书有效期至: {cert['notAfter']}")
            
            # 关闭连接
            secure_sock.close()
            
            return True
            
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            self.error(f"✗ TLS握手失败 ({elapsed:.0f}ms): {str(e)}")
            return False
        
        finally:
            sock.close()

    async def test_ccxt_exchange(self, use_proxy=True, max_retry=3):
        """测试CCXT交易所连接"""
        # 创建交易所实例
        exchange_params = {
            'enableRateLimit': True,
            'timeout': 30000,  # 30秒
            'verbose': self.verbose
        }
        
        # 如果使用代理
        if use_proxy and self.proxy:
            exchange_params['proxies'] = {
                'http': self.proxy,
                'https': self.proxy
            }
        
        exchange = ccxt_async.binance(exchange_params)
        result = False
        
        try:
            # 测试基本市场数据
            self.log("尝试加载市场数据...")
            start = time.time()
            
            # 添加重试机制
            for attempt in range(max_retry):
                try:
                    await exchange.load_markets()
                    elapsed = (time.time() - start) * 1000
                    self.log(f"✓ 市场数据加载成功 ({elapsed:.0f}ms)")
                    result = True
                    break
                except Exception as e:
                    self.error(f"✗ 市场数据加载失败 (尝试 {attempt+1}/{max_retry}): {str(e)}")
                    if attempt < max_retry - 1:
                        backoff = (2 ** attempt) * (0.5 + random.random() * 0.5)
                        self.log(f"等待 {backoff:.2f} 秒后重试...")
                        await asyncio.sleep(backoff)
            
            if result:
                # 打印一些交易所信息
                markets_count = len(exchange.markets)
                symbols = list(exchange.markets.keys())[:5]  # 只取前5个
                
                self.log(f"交易所信息:")
                self.log(f"  API URL: {exchange.urls.get('api', 'N/A')}")
                self.log(f"  市场数量: {markets_count}")
                self.log(f"  示例交易对: {', '.join(symbols)}...")
            
        except Exception as e:
            self.error(f"✗ 交易所连接失败: {str(e)}")
        finally:
            await exchange.close()
        
        return result

    async def diagnose_chunk_errors(self, use_proxy=True):
        """诊断分块请求的具体错误"""
        self.log(f"诊断交易所分块请求错误 ({self.symbol}, {self.timeframe})")
        
        # 创建交易所实例
        exchange_params = {
            'enableRateLimit': True,
            'timeout': 30000,  # 30秒
        }
        
        # 如果使用代理
        if use_proxy and self.proxy:
            exchange_params['proxies'] = {
                'http': self.proxy,
                'https': self.proxy
            }
        
        exchange = ccxt_async.binance(exchange_params)
        
        try:
            # 获取当前时间和一个月前
            end = datetime.now()
            start = end - timedelta(days=30)
            
            # 尝试获取单个小数据块
            try:
                since = int(start.timestamp() * 1000)
                until = int(end.timestamp() * 1000)
                
                self.log(f"尝试获取小数据块 (24小时)...")
                small_end = start + timedelta(days=1)
                small_until = int(small_end.timestamp() * 1000)
                
                self.log(f"请求参数: {self.symbol}, {self.timeframe}, since={since}, limit=100, endTime={small_until}")
                result = await exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    since=since,
                    limit=100,
                    params={"endTime": small_until}
                )
                
                self.log(f"✓ 小数据块请求成功, 获取到 {len(result)} 条记录")
                
                # 尝试获取完整数据段分析错误
                self.log("\n尝试获取完整数据段...")
                try:
                    self.log(f"请求参数: {self.symbol}, {self.timeframe}, since={since}, limit=500, endTime={until}")
                    full_result = await exchange.fetch_ohlcv(
                        symbol=self.symbol,
                        timeframe=self.timeframe,
                        since=since,
                        limit=500,
                        params={"endTime": until}
                    )
                    self.log(f"✓ 完整数据段请求成功, 获取到 {len(full_result)} 条记录")
                    
                    return True
                except Exception as e:
                    self.error(f"✗ 完整数据段请求失败: {str(e)}")
                    if self.verbose:
                        self.debug(f"错误详情: {traceback.format_exc()}")
                    
                    self.log("\n分析错误原因:")
                    
                    # 分析常见错误
                    error_str = str(e).lower()
                    if "timeout" in error_str:
                        self.log(" - 请求超时: 增加timeouts设置，考虑减小请求数据量")
                    elif "rate limit" in error_str:
                        self.log(" - 触发速率限制: 减少并发请求，增加请求间隔")
                    elif "permission denied" in error_str or "unauthorized" in error_str:
                        self.log(" - 授权问题: 检查API密钥权限")
                    elif "connection" in error_str and "reset" in error_str:
                        self.log(" - 连接被重置: 网络不稳定或防火墙干扰")
                    elif "handshake" in error_str:
                        self.log(" - TLS握手失败: 可能是代理SSL拦截或证书问题")
                    else:
                        self.log(" - 其他错误: 检查网络连接和API端点状态")
            
            except Exception as e:
                self.error(f"✗ 小数据块请求也失败: {str(e)}")
                if self.verbose:
                    self.debug(f"错误详情: {traceback.format_exc()}")
                
                return False
                
        except Exception as e:
            self.error(f"✗ 诊断过程失败: {str(e)}")
            return False
        finally:
            await exchange.close()

    async def test_optimized_fetch(self, use_proxy=True):
        """测试优化后的数据获取方法"""
        self.log("测试优化后的数据获取方法")
        
        exchange_params = {
            'enableRateLimit': True,
            'timeout': 60000,  # 60秒 - 增加超时
            'options': {
                'defaultType': 'spot',
                'recvWindow': 60000  # 增加接收窗口
            }
        }
        
        # 如果使用代理
        if use_proxy and self.proxy:
            exchange_params['proxies'] = {
                'http': self.proxy,
                'https': self.proxy
            }
        
        # 添加自定义头部
        exchange_params['headers'] = {
            'Connection': 'keep-alive',
            'Keep-Alive': '60'
        }
        
        exchange = ccxt_async.binance(exchange_params)
        
        try:
            # 手动初始化aiohttp会话
            try:
                import aiohttp
                from aiohttp import ClientTimeout
                
                connector = aiohttp.TCPConnector(
                    keepalive_timeout=60,
                    limit_per_host=10,
                    ttl_dns_cache=300,
                    ssl=False  # 测试禁用SSL以观察性能
                )
                timeout = ClientTimeout(total=60)
                session = aiohttp.ClientSession(connector=connector, timeout=timeout)
                exchange.session = session
                
                self.log("已配置优化的aiohttp会话")
            except Exception as e:
                self.error(f"配置aiohttp会话失败: {str(e)}")
            
            # 实现分页获取策略
            end = datetime.now()
            start = end - timedelta(days=7)  # 只获取7天数据便于测试
            
            since = int(start.timestamp() * 1000)
            
            # 请求限制
            request_semaphore = asyncio.Semaphore(3)
            
            # 定义指数退避函数
            async def backoff_retry(task, max_attempts=3):
                for attempt in range(max_attempts):
                    try:
                        async with request_semaphore:
                            return await task()
                    except Exception as e:
                        if attempt == max_attempts - 1:
                            raise
                        
                        # 计算指数退避
                        delay = (2 ** attempt) * (0.5 + random.random() * 0.5)
                        self.log(f"尝试 {attempt+1} 失败: {str(e)}. 退避 {delay:.2f} 秒...")
                        await asyncio.sleep(delay)
            
            # 分页获取数据
            chunks = []
            current = start
            step = timedelta(days=1)  # 每次获取1天数据
            
            while current < end:
                next_date = min(current + step, end)
                
                since_ms = int(current.timestamp() * 1000)
                until_ms = int(next_date.timestamp() * 1000)
                
                self.log(f"获取 {current} 到 {next_date} 的数据...")
                
                # 定义任务供backoff_retry使用
                async def fetch_task():
                    return await exchange.fetch_ohlcv(
                        symbol=self.symbol,
                        timeframe=self.timeframe,
                        since=since_ms,
                        limit=1000,
                        params={"endTime": until_ms}
                    )
                
                try:
                    chunk = await backoff_retry(fetch_task)
                    if chunk and len(chunk) > 0:
                        chunks.append(chunk)
                        self.log(f"✓ 成功获取 {len(chunk)} 条记录")
                    else:
                        self.log("获取到空数据块")
                except Exception as e:
                    self.error(f"获取数据块失败: {str(e)}")
                
                # 移动到下一个日期
                current = next_date
            
            # 合并数据
            all_data = []
            for chunk in chunks:
                all_data.extend(chunk)
            
            self.log(f"总共获取了 {len(all_data)} 条记录")
            
            if all_data:
                # 转换为DataFrame (仅显示统计信息)
                try:
                    import pandas as pd
                    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # 添加日期时间列
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # 确保按时间排序
                    df = df.sort_values('timestamp')
                    
                    # 显示日期范围
                    start_date = df['datetime'].min()
                    end_date = df['datetime'].max()
                    
                    self.log(f"数据日期范围: {start_date} 到 {end_date}")
                    self.log(f"数据点数量: {len(df)}")
                    
                    # 检查是否有缺失的时间点
                    expected_periods = int((end_date - start_date).total_seconds() / 
                                          (self._timeframe_to_seconds(self.timeframe)))
                    if len(df) < expected_periods * 0.9:  # 允许10%的缺失
                        self.log(f"警告: 可能存在数据缺失. 预期 ~{expected_periods} 条记录，实际 {len(df)} 条", 
                               "WARNING")
                    
                    return True
                except ImportError:
                    self.log("未安装pandas, 跳过数据分析")
                    return len(all_data) > 0
            else:
                self.error("未获取到数据")
                return False
            
        except Exception as e:
            self.error(f"优化获取测试失败: {str(e)}")
            if self.verbose:
                self.debug(traceback.format_exc())
            return False
        finally:
            # 关闭交易所连接
            await exchange.close()
    
    def _timeframe_to_seconds(self, timeframe):
        """将时间周期转换为秒数"""
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 60 * 60
        elif unit == 'd':
            return value * 24 * 60 * 60
        elif unit == 'w':
            return value * 7 * 24 * 60 * 60
        else:
            return value * 60  # 默认为分钟
    
    def suggest_solutions(self, test_results):
        """基于测试结果提供解决方案"""
        self.print_section("诊断结果与解决方案")
        
        # 解析测试结果
        basic_conn = test_results.get('basic_connection', False)
        proxy_test = test_results.get('proxy_test', False)
        tls_test = test_results.get('tls_test', False)
        exchange_test = test_results.get('exchange_test', False)
        chunk_test = test_results.get('chunk_test', False)
        optimized_test = test_results.get('optimized_test', False)
        
        # 分析问题并提供建议
        if not basic_conn:
            self.log("问题: 基本网络连接失败")
            self.log("建议:")
            self.log("1. 检查网络连接是否正常")
            self.log("2. 验证是否可以访问其他网站")
            self.log("3. 检查DNS解析是否正确")
            self.log("4. 考虑更换DNS服务器 (如8.8.8.8或1.1.1.1)")
        
        if not proxy_test and self.proxy:
            self.log("问题: 代理配置测试失败")
            self.log("建议:")
            self.log("1. 验证代理服务器是否正在运行")
            self.log("2. 检查代理URL格式是否正确")
            self.log("3. 尝试更换代理服务器")
            self.log("4. 检查代理服务器是否允许HTTPS流量")
        
        if not tls_test:
            self.log("问题: TLS握手失败")
            self.log("建议:")
            self.log("1. 检查时间是否同步 (NTP)")
            self.log("2. 检查SSL证书是否受信任")
            self.log("3. 尝试更新OpenSSL库")
            self.log("4. 如果使用代理，检查代理是否支持TLS 1.2/1.3")
        
        if not exchange_test:
            self.log("问题: CCXT交易所连接失败")
            self.log("建议:")
            self.log("1. 验证API端点是否正确")
            self.log("2. 检查是否触发了速率限制")
            self.log("3. 确认API密钥和权限")
            self.log("4. 更新CCXT库到最新版本")
        
        if not chunk_test:
            self.log("问题: 分块数据请求失败")
            self.log("建议:")
            self.log("1. 增加请求超时时间 (timeout)")
            self.log("2. 减小数据块大小")
            self.log("3. 实现指数退避重试策略")
            self.log("4. 使用并发限制避免请求风暴")
        
        # 总体建议
        self.print_section("综合优化建议")
        
        if optimized_test:
            self.log("优化后的获取方法测试成功! 建议采用以下配置:")
        else:
            self.log("建议采用以下优化方案:")
            
        self.log("1. 配置优化:")
        self.log("   - 将 connection: close 改为 connection: keep-alive")
        self.log("   - 增加 timeout 到 60000ms (60秒)")
        self.log("   - 设置 recvWindow 到 60000")
        self.log("   - 使用明确的代理配置而非 auto 检测")
        
        self.log("2. HTTP连接优化:")
        self.log("   - 使用 aiohttp TCPConnector 配置持久连接")
        self.log("   - 设置 keepalive_timeout=60, limit_per_host=10")
        self.log("   - 使用 ttl_dns_cache=300 缓存DNS解析")
        
        self.log("3. 重试策略:")
        self.log("   - 实现指数退避策略 (2^attempt)")
        self.log("   - 添加随机抖动防止请求同步")
        self.log("   - 使用 asyncio.Semaphore 限制并发请求")
        
        self.log("4. 数据获取优化:")
        self.log("   - 分块获取并合并数据")
        self.log("   - 减小每个块的大小 (每次请求1天数据)")
        self.log("   - 使用 endTime 参数限制数据范围")
        
        self.log("5. WebSocket替代方案:")
        self.log("   - 对于实时数据，考虑使用CCXT Pro的WebSocket API")
        self.log("   - 使用 watchOHLCV 代替 fetch_ohlcv 获取实时数据")
        
        if not basic_conn or not tls_test:
            self.log("6. 网络级别优化:")
            self.log("   - 考虑在靠近Binance API的服务器上部署应用")
            self.log("   - AWS Tokyo (ap-northeast-1) 是Binance API的主要托管地区")
            self.log("   - 使用高质量的付费代理服务")
            self.log("   - 考虑VPS服务如Vultr、DigitalOcean或Linode")

    async def run_diagnostics(self):
        """运行全面诊断测试"""
        test_results = {}
        
        # 1. 基本连接测试
        self.print_section("1. 基本API连接测试")
        spot_result = self.test_basic_connection(self.binance_endpoints["spot"])
        futures_result = self.test_basic_connection(self.binance_endpoints["futures"])
        
        # 最少一个endpoint连接成功
        test_results['basic_connection'] = any(r[0] for r in spot_result) or any(r[0] for r in futures_result)
        
        # 2. 检查代理连接
        self.print_section("2. 代理连接测试")
        if self.proxy:
            proxy_result = self.diagnose_proxy_connection()
            test_results['proxy_test'] = proxy_result
        else:
            self.log("未指定代理，跳过代理测试")
            test_results['proxy_test'] = None
        
        # 3. TLS握手分析
        self.print_section("3. TLS握手分析")
        
        # 从endpoint URL提取主机名
        spot_host = urlparse(self.binance_endpoints["spot"]).netloc
        futures_host = urlparse(self.binance_endpoints["futures"]).netloc
        
        # 直接连接测试
        self.log("3.1 直接连接TLS握手测试")
        direct_tls_result = self.analyze_tls_handshake(spot_host, 443, use_proxy=False)
        
        # 代理连接测试 (如果指定了代理)
        proxy_tls_result = False
        if self.proxy:
            self.log("3.2 通过代理的TLS握手测试")
            proxy_tls_result = self.analyze_tls_handshake(spot_host, 443, use_proxy=True)
        
        test_results['tls_test'] = direct_tls_result or proxy_tls_result
        
        # 4. CCXT交易所连接测试
        self.print_section("4. CCXT交易所连接测试")
        exchange_result = await self.test_ccxt_exchange(use_proxy=self.proxy is not None)
        test_results['exchange_test'] = exchange_result
        
        # 5. 分块请求诊断
        self.print_section("5. 分块请求诊断")
        chunk_result = await self.diagnose_chunk_errors(use_proxy=self.proxy is not None)
        test_results['chunk_test'] = chunk_result
        
        # 6. 测试优化后的获取方法
        self.print_section("6. 优化方法测试")
        optimized_result = await self.test_optimized_fetch(use_proxy=self.proxy is not None)
        test_results['optimized_test'] = optimized_result
        
        # 提供解决方案建议
        self.suggest_solutions(test_results)
        
        return test_results


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CCXT Binance 连接诊断工具")
    parser.add_argument("--proxy", help="HTTP代理URL (例如: http://127.0.0.1:7890)")
    parser.add_argument("--symbol", default="BTC/USDT", help="要测试的交易对 (默认: BTC/USDT)")
    parser.add_argument("--timeframe", default="1m", help="要测试的时间周期 (默认: 1h)")
    parser.add_argument("--verbose", action="store_true", help="启用详细日志")
    
    args = parser.parse_args()
    
    # 创建诊断工具
    diagnostics = BinanceDiagnostics(
        proxy=args.proxy,
        symbol=args.symbol,
        timeframe=args.timeframe,
        verbose=args.verbose
    )
    
    # 执行诊断
    print(f"开始诊断 - Binance API连接问题排查工具")
    print(f"交易对: {args.symbol}, 时间周期: {args.timeframe}")
    if args.proxy:
        print(f"使用代理: {args.proxy}")
    
    await diagnostics.run_diagnostics()

if __name__ == "__main__":
    asyncio.run(main())