# src/exchange/binance.py

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import asyncio
import time
import os
from typing import Dict, List, Union, Optional, Any, Tuple
from datetime import datetime, timedelta
from functools import wraps

from common.config import ConfigManager
from common.logging import LogManager
from src.common.helpers import TimeUtils


logger = LogManager.get_logger("trading_system")


def retry_exchange_operation(max_attempts=3, delay_base=1):
    """Decorator for retrying exchange operations with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    last_exception = e
                    
                    # Exponential backoff with jitter
                    delay = delay_base * (2 ** attempt)
                    time.sleep(delay)
                except Exception as e:
                    logger.error(f"Unhandled error: {str(e)}")
                    raise
            
            # If all attempts fail
            logger.error(f"All attempts failed. Last error: {last_exception}")
            raise last_exception
        return wrapper
    return decorator


class Binance:
    """Binance exchange interface with optimized data retrieval and caching"""
    
    def __init__(self, config: ConfigManager):
        """Initialize Binance interface with configuration"""
        self.config = config
        
        # Set up cache directory
        self.cache_dir = self._get_cache_dir()
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Extract configuration parameters
        self.params = self._build_params()
        
        # Initialize exchange objects
        self.exchange = None
        self.async_exchange = None
        
        # Initialize exchange
        self._init_exchange()
        
        # Settings
        self.rate_limit = self.config.get('api', 'rate_limits', 'max_calls_per_second', default=10) * 60
        self.current_requests = 0
        self.rate_limit_reset = time.time() + 60
        self.download_chunk_size = 500
        self.max_retry_attempts = 3
        self.retry_delay = 2
        
        logger.info("Binance interface initialized")
    
    def _get_cache_dir(self) -> str:
        """Get cache directory from config"""
        cache_dir = self.config.get('data', 'cache', 'dir', default='./cache')
        return os.path.abspath(os.path.expanduser(cache_dir))
    
    def _init_exchange(self):
        """Initialize exchange connection with better fallback mechanisms"""
        
        # Check whether offline mode is enabled
        offline_mode = self.config.get('system', 'offline_mode', default=False)
        if offline_mode:
            logger.info("Running in offline mode, skipping Binance exchange initialization")
            self.exchange = None
            return
        
        # Track proxy status
        using_proxy = False
        original_params = self.params.copy()
        
        try:
            logger.info("Initializing Binance exchange")
            
            # First try with configured proxy
            if 'proxies' in self.params:
                proxies = self.params.get('proxies', None)
                logger.info(f"Attempting connection with configured proxy: {proxies}")
                
                api_test_url = 'https://api.binance.com/api/v3/ping'
                proxy_works = test_connection(url=api_test_url, proxy=proxies)
                
                if proxy_works:
                    logger.info("Proxy connection test successful")
                    using_proxy = True
                else:
                    logger.warning("Configured proxy failed connection test")
                    # Remove proxy from params
                    params_without_proxy = self.params.copy()
                    if 'proxies' in params_without_proxy:
                        del params_without_proxy['proxies']
                    self.params = params_without_proxy
            
            # If proxy didn't work or no proxy configured, try auto-detection
            if not using_proxy:
                logger.info("Attempting to auto-detect working proxy")
                detected_proxies = detect_proxy('https://api.binance.com/api/v3/ping')
                
                if detected_proxies:
                    logger.info(f"Using auto-detected proxy: {detected_proxies}")
                    self.params['proxies'] = detected_proxies
                    using_proxy = True
                else:
                    logger.info("No working proxy detected, using direct connection")
            
            # Initialize exchange with current params
            try:
                self.exchange = ccxt.binance(self.params)
                
                # Test with load_markets
                try:
                    self.exchange.load_markets()
                    logger.info("Successfully loaded market data")
                    return  # Success, early return
                except Exception as e:
                    logger.warning(f"Failed to load markets: {str(e)}")
                    
                    # If we're using a proxy and it failed, try direct connection
                    if using_proxy:
                        logger.warning("Proxy connection failed during market load, trying direct connection")
                        params_without_proxy = self.params.copy()
                        if 'proxies' in params_without_proxy:
                            del params_without_proxy['proxies']
                        
                        # Try again without proxy
                        self.exchange = ccxt.binance(params_without_proxy)
                        try:
                            self.exchange.load_markets()
                            logger.info("Direct connection successful")
                            return  # Success, early return
                        except Exception as direct_e:
                            logger.warning(f"Direct connection also failed: {str(direct_e)}")
                            # Continue to workaround implementation
                    
                    # Use direct API access as an alternative (workaround)
                    import requests
                    
                    try:
                        # First try direct connection (no proxy)
                        direct_response = requests.get(
                            'https://api.binance.com/api/v3/exchangeInfo',
                            timeout=30
                        )
                        
                        if direct_response.status_code == 200:
                            logger.info("Successfully connected to Binance API via direct request")
                            # Set the minimum market data
                            self.exchange = ccxt.binance(params_without_proxy)
                            self.exchange.markets = {}
                            self.exchange.markets_by_id = {}
                            logger.info("Exchange initialized with minimal market data")
                            return  # Success, early return
                        else:
                            logger.warning(f"Direct API request failed with status {direct_response.status_code}")
                    except Exception as req_error:
                        logger.error(f"Direct API request failed: {str(req_error)}")
                    
                    # Last attempt - try with each detected proxy option
                    if not using_proxy:
                        logger.info("Attempting to find any working proxy...")
                        detector = ProxyDetector(test_url='https://api.binance.com/api/v3/exchangeInfo')
                        common_ports = detector.common_proxy_ports
                        
                        for port in common_ports:
                            try:
                                test_proxy = {'http': f'http://127.0.0.1:{port}', 'https': f'http://127.0.0.1:{port}'}
                                logger.info(f"Testing proxy on port {port}")
                                
                                proxy_response = requests.get(
                                    'https://api.binance.com/api/v3/exchangeInfo',
                                    proxies=test_proxy,
                                    timeout=5
                                )
                                
                                if proxy_response.status_code == 200:
                                    logger.info(f"Found working proxy on port {port}")
                                    self.params['proxies'] = test_proxy
                                    self.exchange = ccxt.binance(self.params)
                                    self.exchange.markets = {}
                                    self.exchange.markets_by_id = {}
                                    return  # Success, early return
                            except Exception:
                                pass  # Try next port
                    
                    # If we got here, all attempts failed
                    raise RuntimeError(f"All connection attempts failed")
                    
            except Exception as e:
                logger.error(f"Error initializing exchange: {str(e)}")
                raise RuntimeError(f"Could not initialize exchange: {str(e)}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Binance exchange: {str(e)}")
            
            # Special handling for configuration with offline support
            if self.config.get('backtest', 'fallback_to_local', default=True):
                logger.warning("Exchange initialization failed but fallback_to_local is enabled. Creating minimal exchange instance.")
                self.exchange = ccxt.binance()
                self.exchange.markets = {}
                self.exchange.markets_by_id = {}
                return
                
            raise RuntimeError(f"Binance initialization failed: {str(e)}")

    def _build_params(self) -> Dict:
        """
        Build CCXT parameters from configuration with enhanced security
        
        Returns:
        Dictionary of CCXT parameters
        """
        params = {
            'apiKey': self.config.get('api', 'binance', 'apiKey', default=''),
            'secret': self.config.get('api', 'binance', 'secret', default=''),
            'timeout': self.config.get('api', 'timeout', default=30000),
            'enableRateLimit': self.config.get('api', 'enableRateLimit', default=True),
            'options': {
                'adjustForTimeDifference': self.config.get(
                    'default_config', 'options', 'adjustForTimeDifference', default=True),
                'recvWindow': self.config.get(
                    'default_config', 'options', 'recvWindow', default=60000),
                'defaultType': self.config.get(
                    'default_config', 'options', 'defaultType', default='spot')
            }
        }
        
        # 检查配置是否明确启用或禁用代理
        use_proxy = self.config.get('api', 'useproxy', default=None)
        
        # 如果没有明确禁用代理，尝试使用代理
        if use_proxy is None or use_proxy:
            # 首先从配置中获取代理
            http_proxy = self.config.get('proxies', 'http', default=None)
            https_proxy = self.config.get('proxies', 'https', default=None)
            
            proxies = {}
            if http_proxy or https_proxy:
                # 使用配置中指定的代理
                proxies = {
                    'http': http_proxy,
                    'https': https_proxy or http_proxy
                }
                logger.info(f"Using proxy from configuration: {proxies}")
            else:
                # 配置中未指定代理，尝试自动检测
                detected_proxies = detect_proxy('https://api.binance.com/api/v3/ping')
                if detected_proxies:
                    proxies = detected_proxies
                    logger.info(f"Using auto-detected proxy: {proxies}")
                else:
                    logger.info("No proxy detected, using direct connection")
                    
            # 如果找到有效代理，添加到参数中
            if proxies:
                params['proxies'] = proxies
        else:
            logger.info("Proxy explicitly disabled in configuration")
        
        return params

    async def _init_async_exchange(self) -> None:
        """Initialize async exchange (lazy initialization)"""
        if self.async_exchange is not None:
            return

        try:
            self.async_exchange = ccxt_async.binance(self.params)
            logger.info("Async Binance exchange initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Async Binance exchange: {str(e)}")
            self.async_exchange = None
    
    def _handle_rate_limit(self) -> None:
        """Manage rate limiting for API requests"""
        current_time = time.time()
        
        # Check if the rate limit window has reset
        if current_time > self.rate_limit_reset:
            self.current_requests = 0
            self.rate_limit_reset = current_time + 60
        
        # Check if we're at the rate limit
        if self.current_requests >= self.rate_limit:
            # Calculate sleep time
            sleep_time = max(0, self.rate_limit_reset - current_time)
            logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            
            # Reset after sleeping
            self.current_requests = 0
            self.rate_limit_reset = time.time() + 60
        
        # Increment the request counter
        self.current_requests += 1

    async def _async_handle_rate_limit(self) -> None:
        """Manage rate limiting for async API requests"""
        current_time = time.time()
        
        # Check if the rate limit window has reset
        if current_time > self.rate_limit_reset:
            self.current_requests = 0
            self.rate_limit_reset = current_time + 60
        
        # Check if we're at the rate limit
        if self.current_requests >= self.rate_limit:
            # Calculate sleep time
            sleep_time = max(0, self.rate_limit_reset - current_time)
            logger.warning(f"Async rate limit reached, sleeping for {sleep_time:.2f} seconds")
            await asyncio.sleep(sleep_time)
            
            # Reset after sleeping
            self.current_requests = 0
            self.rate_limit_reset = time.time() + 60
        
        # Increment the request counter
        self.current_requests += 1
    
    @staticmethod
    def _process_ohlcv_data(ohlcv_data: List) -> pd.DataFrame:
        """Convert OHLCV data to DataFrame"""
        if not ohlcv_data:
            return pd.DataFrame()
        
        try:
            # Create DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing OHLCV data: {str(e)}")
            return pd.DataFrame()
    
    def fetch_latest_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Fetch recent OHLCV data"""
        if not self.exchange:
            self._init_exchange()
            if not self.exchange:
                logger.error("Exchange not initialized, can't fetch data")
                return pd.DataFrame()
        
        for attempt in range(self.max_retry_attempts):
            try:
                # Handle rate limiting
                self._handle_rate_limit()
                
                # Fetch data
                ohlcv = self.exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
                
                # Convert to DataFrame
                df = self._process_ohlcv_data(ohlcv)
                
                if not df.empty:
                    return df
                
                logger.warning(f"Empty response on attempt {attempt+1}")
                time.sleep(self.retry_delay)
                
            except Exception as e:
                logger.error(f"Error fetching OHLCV data (attempt {attempt+1}): {str(e)}")
                time.sleep(self.retry_delay * (attempt + 1))
        
        # All attempts failed
        return pd.DataFrame()
        
    async def fetch_historical_ohlcv(self, 
                                symbol: str, 
                                timeframe: str = '1h',
                                start_date: Optional[Union[str, datetime]] = None,
                                end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """Fetch historical OHLCV data with optimized performance"""
        # Make sure exchange is initialized
        if not self.exchange:
            self._init_exchange()
            if not self.exchange:
                logger.error("Exchange initialization failed")
                return pd.DataFrame()
        
        try:
            # Process dates
            start_dt = TimeUtils.parse_timestamp(start_date, default_days_ago=30)
            end_dt = TimeUtils.parse_timestamp(end_date, default_days_ago=0)
            
            logger.info(f"Fetching historical data for {symbol} from {start_dt} to {end_dt}")
            
            # Initialize async exchange if needed
            if not self.async_exchange:
                await self._init_async_exchange()
            
            # Break the request into chunks to handle large date ranges
            chunks = self._get_date_chunks(start_dt, end_dt, timeframe)
            logger.info(f"Splitting request into {len(chunks)} chunks")
            
            # Track progress
            all_data = []
            chunks_processed = 0
            
            # Process each chunk
            for i, (chunk_start, chunk_end) in enumerate(chunks):
                for retry in range(self.max_retry_attempts):
                    try:
                        # Handle rate limiting
                        await self._async_handle_rate_limit()
                        
                        # Convert to millisecond timestamp
                        chunk_since = int(chunk_start.timestamp() * 1000)
                        chunk_until = int(chunk_end.timestamp() * 1000)
                        
                        # Log progress
                        if i == 0 or i % 5 == 0:
                            logger.info(f"Fetching chunk {i+1}/{len(chunks)}: {chunk_start} to {chunk_end}")
                        
                        # Fetch data for this chunk
                        exchange = self.async_exchange if self.async_exchange else self.exchange
                        
                        ohlcv = await exchange.fetch_ohlcv(
                            symbol=symbol, 
                            timeframe=timeframe, 
                            since=chunk_since,
                            limit=self.download_chunk_size,
                            params={"endTime": chunk_until}
                        )
                        
                        chunks_processed += 1
                        
                        # Check if we got data
                        if not ohlcv or len(ohlcv) == 0:
                            logger.debug(f"No data returned for chunk {i+1}")
                            break
                        
                        # Add data to list
                        all_data.extend(ohlcv)
                        
                        # Short delay
                        await asyncio.sleep(0.5)
                        
                        # Success, break retry loop
                        break
                        
                    except Exception as e:
                        logger.warning(f"Error on chunk {i+1}, attempt {retry+1}: {str(e)}")
                        await asyncio.sleep(self.retry_delay * (retry + 1))
            
            # Process all data
            if not all_data:
                logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = self._process_ohlcv_data(all_data)
            
            # Filter by date range
            if not df.empty:
                df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)]
            
            logger.info(f"Downloaded {len(df)} candles for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _get_date_chunks(self, start_dt: datetime, end_dt: datetime, 
                        timeframe: str, chunk_size_days: Optional[int] = None) -> List[Tuple[datetime, datetime]]:
        """Break date range into manageable chunks"""
        # Determine chunk size based on timeframe
        if chunk_size_days is None:
            if timeframe in ['1m', '5m']:
                chunk_size = timedelta(days=1)
            elif timeframe in ['15m', '30m', '1h']:
                chunk_size = timedelta(days=7)
            else:
                chunk_size = timedelta(days=30)
        else:
            chunk_size = timedelta(days=chunk_size_days)
        
        # Create chunks
        chunks = []
        current_start = start_dt
        
        while current_start < end_dt:
            current_end = min(current_start + chunk_size, end_dt)
            chunks.append((current_start, current_end))
            current_start = current_end
        
        return chunks
        
    async def close(self):
        """Clean up resources"""
        try:
            if self.async_exchange:
                await self.async_exchange.close()
                self.async_exchange = None
                logger.info("Async exchange connection closed")
        except Exception as e:
            logger.error(f"Error closing async exchange: {str(e)}")
            
            
# src/utils/network.py
import os
import socket
import requests
from typing import Dict, Optional, Union
import subprocess
import re
import platform
from urllib.parse import urlparse
from common.logging import LogManager


logger = LogManager.get_logger("data.manager")


class ProxyDetector:
    """
    Utility class for detecting and validating proxy configurations
    to use with external APIs like Binance.
    """
    
    def __init__(self, test_url: str = 'https://api.binance.com/api/v3/ping', timeout: int = 5):
        """
        Initialize the proxy detector
        
        Args:
            test_url: URL to use for testing proxy connections
            timeout: Connection timeout in seconds
        """
        self.test_url = test_url
        self.timeout = timeout
        self.common_proxy_ports = [7890, 8080, 1080, 8118, 3128, 10809, 7070, 9090]
    
    def detect_proxy(self, check_env: bool = True, check_system: bool = True, 
                   check_common_ports: bool = True) -> Optional[Dict[str, str]]:
        """
        Detect available proxy configuration using multiple methods
        
        Args:
            check_env: Whether to check environment variables
            check_system: Whether to check system proxy settings
            check_common_ports: Whether to check common proxy ports
            
        Returns:
            Dict containing http/https proxy URLs or None if no working proxy found
        """
        proxy_sources = []
        
        # 1. Check environment variables
        if check_env:
            env_proxies = self._check_environment_variables()
            if env_proxies:
                proxy_sources.append(("Environment Variables", env_proxies))
        
        # 2. Check system proxy settings
        if check_system:
            system_proxies = self._check_system_proxy()
            if system_proxies:
                proxy_sources.append(("System Settings", system_proxies))
        
        # 3. Check requests library's auto-detection
        requests_proxies = self._check_requests_proxy()
        if requests_proxies:
            proxy_sources.append(("Requests Auto-detection", requests_proxies))
        
        # 4. Check common proxy ports
        if check_common_ports:
            for port in self.common_proxy_ports:
                common_proxy = self._check_common_proxy_port(port)
                if common_proxy:
                    proxy_sources.append((f"Common Port {port}", common_proxy))
                    break
        
        # Test each detected proxy and return the first working one
        for source, proxy in proxy_sources:
            if self.test_proxy(proxy):
                logger.info(f"Found working proxy from {source}: {proxy}")
                return proxy
                
        logger.info("No working proxy found, using direct connection")
        return None
    
    def _check_environment_variables(self) -> Optional[Dict[str, str]]:
        """Check for proxy configuration in environment variables"""
        env_vars = {
            'http': ['HTTP_PROXY', 'http_proxy'],
            'https': ['HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy']
        }
        
        proxies = {}
        
        for protocol, var_names in env_vars.items():
            for var in var_names:
                if var in os.environ and os.environ[var]:
                    proxies[protocol] = os.environ[var]
                    break
        
        if proxies:
            logger.debug(f"Found proxy settings in environment variables: {proxies}")
            return proxies
        
        return None
    
    def _check_system_proxy(self) -> Optional[Dict[str, str]]:
        """Check for system proxy configuration using platform-specific methods"""
        proxies = None
        system = platform.system()
        
        try:
            if system == 'Windows':
                proxies = self._get_windows_proxy()
            elif system == 'Darwin':  # macOS
                proxies = self._get_macos_proxy()
            elif system == 'Linux':
                proxies = self._get_linux_proxy()
            
            if proxies:
                logger.debug(f"Found system proxy settings: {proxies}")
                return proxies
        except Exception as e:
            logger.debug(f"Error checking system proxy: {e}")
        
        return None
    
    def _get_windows_proxy(self) -> Optional[Dict[str, str]]:
        """Get proxy settings from Windows registry"""
        try:
            import winreg
            
            proxy_key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"
            )
            
            # Check if proxy is enabled
            proxy_enable, _ = winreg.QueryValueEx(proxy_key, "ProxyEnable")
            
            if proxy_enable:
                proxy_server, _ = winreg.QueryValueEx(proxy_key, "ProxyServer")
                
                # Handle different proxy formats
                if "=" in proxy_server:  # Protocol-specific format
                    protocols = proxy_server.split(";")
                    proxies = {}
                    
                    for protocol in protocols:
                        if "=" in protocol:
                            proto_name, address = protocol.split("=", 1)
                            if proto_name.lower() in ('http', 'https'):
                                proxies[proto_name.lower()] = address
                    
                    return proxies if proxies else None
                else:  # Same proxy for all protocols
                    return {'http': proxy_server, 'https': proxy_server}
            
        except Exception as e:
            logger.debug(f"Error reading Windows registry: {e}")
        
        return None
    
    def _get_macos_proxy(self) -> Optional[Dict[str, str]]:
        """Get proxy settings from macOS system preferences"""
        try:
            # Check HTTP proxy
            result = subprocess.run(
                ['networksetup', '-getwebproxy', 'Wi-Fi'],
                capture_output=True, text=True
            )
            
            http_enabled = "Enabled: Yes" in result.stdout
            http_proxy = None
            
            if http_enabled:
                server_match = re.search(r'Server: (.*)', result.stdout)
                port_match = re.search(r'Port: (\d+)', result.stdout)
                
                if server_match and port_match:
                    server = server_match.group(1)
                    port = port_match.group(1)
                    http_proxy = f"http://{server}:{port}"
            
            # Check HTTPS proxy
            result = subprocess.run(
                ['networksetup', '-getsecurewebproxy', 'Wi-Fi'],
                capture_output=True, text=True
            )
            
            https_enabled = "Enabled: Yes" in result.stdout
            https_proxy = None
            
            if https_enabled:
                server_match = re.search(r'Server: (.*)', result.stdout)
                port_match = re.search(r'Port: (\d+)', result.stdout)
                
                if server_match and port_match:
                    server = server_match.group(1)
                    port = port_match.group(1)
                    https_proxy = f"http://{server}:{port}"
            
            if http_proxy or https_proxy:
                return {
                    'http': http_proxy,
                    'https': https_proxy or http_proxy
                }
            
        except Exception as e:
            logger.debug(f"Error checking macOS proxy: {e}")
        
        return None
    
    def _get_linux_proxy(self) -> Optional[Dict[str, str]]:
        """Get proxy settings from Linux environments"""
        # Check common environment variables
        proxies = self._check_environment_variables()
        if proxies:
            return proxies
            
        # Check GNOME settings
        try:
            result = subprocess.run(
                ['gsettings', 'get', 'org.gnome.system.proxy', 'mode'],
                capture_output=True, text=True
            )
            
            if 'manual' in result.stdout:
                http_host = subprocess.run(
                    ['gsettings', 'get', 'org.gnome.system.proxy.http', 'host'],
                    capture_output=True, text=True
                ).stdout.strip().strip("'")
                
                http_port = subprocess.run(
                    ['gsettings', 'get', 'org.gnome.system.proxy.http', 'port'],
                    capture_output=True, text=True
                ).stdout.strip()
                
                https_host = subprocess.run(
                    ['gsettings', 'get', 'org.gnome.system.proxy.https', 'host'],
                    capture_output=True, text=True
                ).stdout.strip().strip("'")
                
                https_port = subprocess.run(
                    ['gsettings', 'get', 'org.gnome.system.proxy.https', 'port'],
                    capture_output=True, text=True
                ).stdout.strip()
                
                proxies = {}
                
                if http_host and http_port:
                    proxies['http'] = f"http://{http_host}:{http_port}"
                
                if https_host and https_port:
                    proxies['https'] = f"http://{https_host}:{https_port}"
                elif 'http' in proxies:
                    proxies['https'] = proxies['http']
                
                if proxies:
                    return proxies
        
        except Exception as e:
            logger.debug(f"Error checking Linux proxy settings: {e}")
        
        return None
    
    def _check_requests_proxy(self) -> Optional[Dict[str, str]]:
        """Check for proxy settings using requests library's auto-detection"""
        try:
            session = requests.Session()
            proxies = session.proxies
            
            # Filter out empty proxies
            proxies = {k: v for k, v in proxies.items() if v}
            
            if proxies:
                logger.debug(f"Found proxy settings via requests auto-detection: {proxies}")
                return proxies
        except Exception as e:
            logger.debug(f"Error checking requests proxies: {e}")
        
        return None
    
    def _check_common_proxy_port(self, port: int) -> Optional[Dict[str, str]]:
        """Check if a common proxy port is available on localhost"""
        proxy_url = f"http://127.0.0.1:{port}"
        
        try:
            # Try to connect to the port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            
            if result == 0:  # Port is open
                logger.debug(f"Found open proxy port at: {proxy_url}")
                return {'http': proxy_url, 'https': proxy_url}
        except Exception as e:
            logger.debug(f"Error checking proxy port {port}: {e}")
        
        return None
    
    def test_proxy(self, proxy_settings: Dict[str, str]) -> bool:
        """
        Test if a proxy configuration works with the target API
        
        Args:
            proxy_settings: Dict with 'http' and/or 'https' proxy URLs
            
        Returns:
            bool: True if proxy works, False otherwise
        """
        if not proxy_settings:
            return False
            
        try:
            logger.debug(f"Testing proxy configuration: {proxy_settings}")
            response = requests.get(
                self.test_url,
                proxies=proxy_settings,
                timeout=self.timeout
            )
            
            success = response.status_code == 200
            logger.debug(f"Proxy test {'successful' if success else 'failed'} with status {response.status_code}")
            return success
        except Exception as e:
            logger.debug(f"Proxy test failed with error: {e}")
            return False
    
    def get_proxy_info(self, proxy_url: str) -> Dict[str, Union[str, int]]:
        """
        Parse proxy URL and extract components
        
        Args:
            proxy_url: Proxy URL string
            
        Returns:
            Dict containing protocol, host, port, and auth info
        """
        if not proxy_url:
            return {}
            
        try:
            parsed = urlparse(proxy_url)
            
            # Handle auth info if present
            auth = None
            if parsed.username and parsed.password:
                auth = f"{parsed.username}:{parsed.password}"
            
            return {
                'protocol': parsed.scheme,
                'host': parsed.hostname,
                'port': parsed.port,
                'auth': auth
            }
        except Exception as e:
            logger.error(f"Error parsing proxy URL: {e}")
            return {}

def detect_proxy(test_url: str = 'https://api.binance.com/api/v3/ping') -> Optional[Dict[str, str]]:
    """
    Convenience function to detect available proxy settings for the target URL
    
    Args:
        test_url: URL to test proxy against
        
    Returns:
        Dict with proxy configuration or None if no working proxy found
    """
    detector = ProxyDetector(test_url=test_url)
    return detector.detect_proxy()

def test_connection(url: str = 'https://api.binance.com/api/v3/ping', 
                   proxy: Optional[Dict[str, str]] = None, 
                   timeout: int = 5) -> bool:
    """
    Test connection to a URL with optional proxy
    
    Args:
        url: URL to test
        proxy: Optional proxy configuration
        timeout: Connection timeout in seconds
        
    Returns:
        bool: True if connection succeeded, False otherwise
    """
    try:
        response = requests.get(url, proxies=proxy, timeout=timeout)
        return response.status_code == 200
    except Exception as e:
        logger.debug(f"Connection test failed: {e}")
        return False