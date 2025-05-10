#!/usr/bin/env python3
# src/datasource/engine.py

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union, Any
import gc

from src.common.log_manager import LogManager

class DataEngine:
    """Optimized trading data downloader for reliable data acquisition"""
    
    def __init__(self, 
                 base_url: str = "https://api.binance.com",
                 futures_url: str = "https://fapi.binance.com",
                 api_key: str = None,
                 api_secret: str = None,
                 max_retries: int = 5,
                 timeout: int = 60,
                 rate_limit: int = 2,
                 chunk_size: int = 500,
                 storage_path: str = "data",
                 use_futures: bool = False,
                 log_level: str = "INFO"):
        """Initialize downloader"""
        # Basic setup
        self.base_url = futures_url if use_futures else base_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_futures = use_futures
        self.source_name = "binance"
        
        # Performance settings
        self.max_retries = max_retries
        self.timeout = timeout
        self.rate_limit = rate_limit
        self.request_interval = 1.0 / rate_limit
        self.last_request_time = 0
        self.chunk_size = min(chunk_size, 1000)  # Binance max is 1000
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        self.logger = LogManager.get_logger("datasource.downloader")
        if log_level != "INFO":
            LogManager.set_level(log_level, "datasource.downloader")
        
        # HTTP session settings
        self.session = None
        self.request_semaphore = asyncio.Semaphore(2)
        
        self.logger.info(f"Data downloader initialized: chunk_size={self.chunk_size}, rate_limit={self.rate_limit}/s")
    
    async def initialize(self):
        """Initialize HTTP session"""
        if self.session is not None:
            await self.close()
            
        # Create TCP connection pool
        conn = aiohttp.TCPConnector(
            limit=3,
            limit_per_host=2,
            ttl_dns_cache=300,
            keepalive_timeout=60,
            force_close=False
        )
        
        # Set default timeout
        timeout = aiohttp.ClientTimeout(
            total=self.timeout,
            sock_connect=30,
            sock_read=30
        )
        
        # Create session
        self.session = aiohttp.ClientSession(
            connector=conn,
            timeout=timeout,
            headers={
                'User-Agent': 'OptimizedDataEngine/1.0',
                'Accept': 'application/json',
                'Connection': 'keep-alive'
            }
        )
        
        self.logger.info("HTTP session initialized")
        return self.session
    
    async def close(self):
        """Close session and resources"""
        if self.session:
            await self.session.close()
            self.session = None
            self.logger.info("HTTP session closed")
    
    def _get_endpoint(self, symbol: str, timeframe: str) -> str:
        """Build API endpoint URL"""
        if self.use_futures:
            return f"{self.base_url}/fapi/v1/klines"
        else:
            return f"{self.base_url}/api/v3/klines"
    
    def _timeframe_to_interval(self, timeframe: str) -> str:
        """Convert timeframe to Binance format"""
        if timeframe in ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']:
            return timeframe
            
        # Convert other formats
        mapping = {
            '1min': '1m', '3min': '3m', '5min': '5m', '15min': '15m', '30min': '30m',
            '1hour': '1h', '2hour': '2h', '4hour': '4h', '6hour': '6h', '12hour': '12h',
            '1day': '1d', '3day': '3d', '1week': '1w', '1month': '1M'
        }
        return mapping.get(timeframe, '1h')  # Default to 1h
    
    def _calculate_chunks(self, start_time: int, end_time: int, timeframe: str) -> List[Tuple[int, int]]:
        """Split date range into appropriate chunks"""
        # Calculate milliseconds per timeframe
        tf_milliseconds = {
            '1m': 60000, '3m': 180000, '5m': 300000, '15m': 900000, '30m': 1800000,
            '1h': 3600000, '2h': 7200000, '4h': 14400000, '6h': 21600000, '8h': 28800000, 
            '12h': 43200000, '1d': 86400000, '3d': 259200000, '1w': 604800000, '1M': 2592000000
        }
        
        ms_per_candle = tf_milliseconds.get(self._timeframe_to_interval(timeframe), 3600000)
        
        # Calculate chunk time range (chunk_size candles)
        chunk_ms = ms_per_candle * self.chunk_size
        
        # Create time chunks
        chunks = []
        current_start = start_time
        
        while current_start < end_time:
            current_end = min(current_start + chunk_ms, end_time)
            chunks.append((current_start, current_end))
            current_start = current_end
            
        self.logger.info(f"Split time range into {len(chunks)} chunks")
        return chunks
    
    async def _wait_rate_limit(self):
        """Respect rate limit (token bucket algorithm)"""
        now = time.time()
        elapsed = now - self.last_request_time
        
        if elapsed < self.request_interval:
            delay = self.request_interval - elapsed
            await asyncio.sleep(delay)
            
        self.last_request_time = time.time()
    
    async def _fetch_chunk(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> List[List]:
        """Fetch a single data chunk"""
        endpoint = self._get_endpoint(symbol, interval)
        params = {
            'symbol': symbol.replace('/', ''),
            'interval': interval,
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': self.chunk_size
        }
        
        # Avoid rate limiting
        await self._wait_rate_limit()
        
        # Limit concurrent requests
        async with self.request_semaphore:
            # Retry mechanism
            for attempt in range(1, self.max_retries + 1):
                try:
                    async with self.session.get(endpoint, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            self.logger.debug(f"Successfully fetched data: {symbol} {interval} ({len(data)} records)")
                            return data
                        else:
                            error_text = await response.text()
                            self.logger.warning(f"API request failed (attempt {attempt}/{self.max_retries}): HTTP {response.status}, {error_text}")
                            
                            # Handle specific errors
                            if response.status == 429:  # Rate limit
                                wait_time = int(response.headers.get('Retry-After', 5))
                                self.logger.warning(f"Rate limit hit, waiting {wait_time}s")
                                await asyncio.sleep(wait_time)
                            elif response.status >= 500:  # Server error
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            else:
                                await asyncio.sleep(1)  # Other errors pause 1s
                
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    self.logger.warning(f"Request exception (attempt {attempt}/{self.max_retries}): {e}")
                    # Exponential backoff + random jitter
                    backoff = (2 ** attempt) + (0.1 * np.random.random())
                    await asyncio.sleep(backoff)
            
            # All retries failed
            self.logger.error(f"All retries failed: {symbol} {interval}")
            return []
    
    def _parse_ohlcv(self, data: List[List]) -> pd.DataFrame:
        """Parse OHLCV data to DataFrame"""
        if not data:
            return pd.DataFrame()
            
        # Binance API format:
        # [open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_volume, taker_buy_quote_volume, ignore]
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignored'
        ])
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        keep_cols = ['datetime', 'timestamp', 'open', 'high', 'low', 'close', 'volume']    # Keep essential columns
        return df[keep_cols]
    
    async def download_data(self, 
                           symbol: str, 
                           timeframe: str, 
                           start_date: Union[str, datetime, int] = None,
                           end_date: Union[str, datetime, int] = None) -> pd.DataFrame:
        """Download historical OHLCV data"""
        if self.session is None:
            await self.initialize()
            
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
        if end_date is None:
            end_date = datetime.now()
        start_ms = self._to_timestamp_ms(start_date)
        end_ms = self._to_timestamp_ms(end_date)
        interval = self._timeframe_to_interval(timeframe)
        
        self.logger.info(f"Downloading {symbol} {interval} data: {datetime.fromtimestamp(start_ms/1000)} to {datetime.fromtimestamp(end_ms/1000)}")
        
        # Download in chunks
        chunks = self._calculate_chunks(start_ms, end_ms, interval)
        all_data = []
        
        for i, (chunk_start, chunk_end) in enumerate(chunks):
            self.logger.debug(f"Downloading chunk {i+1}/{len(chunks)}: {datetime.fromtimestamp(chunk_start/1000)} to {datetime.fromtimestamp(chunk_end/1000)}")
            chunk_data = await self._fetch_chunk(symbol, interval, chunk_start, chunk_end)
            all_data.extend(chunk_data)
            
            # Prevent resource leaks
            if i > 0 and i % 10 == 0:
                gc.collect()    # Force GC every 10 chunks
        
        df = self._parse_ohlcv(all_data)
        if df.empty:
            self.logger.warning(f"No data retrieved: {symbol} {interval}")
            return df
            
        df = df.sort_values('datetime').drop_duplicates(subset=['datetime'])    # Sort and deduplicate
        
        self.logger.info(f"Successfully downloaded {symbol} {interval} data: {len(df)} records, {df['datetime'].min()} to {df['datetime'].max()}")
        return df
    
    def _to_timestamp_ms(self, date_input: Union[str, datetime, int]) -> int:
        """Convert various date formats to millisecond timestamp"""
        if isinstance(date_input, int):
            # Ensure milliseconds
            if date_input < 10000000000:  # Second-level timestamp
                return date_input * 1000
            return date_input
            
        elif isinstance(date_input, datetime):
            return int(date_input.timestamp() * 1000)
            
        elif isinstance(date_input, str):
            try:
                dt = pd.to_datetime(date_input)
                return int(dt.timestamp() * 1000)
            except:
                raise ValueError(f"Cannot parse date: {date_input}")
                
        else:
            raise TypeError(f"Unsupported date type: {type(date_input)}")
    
    def _generate_storage_path(self, symbol: str, timeframe: str, 
                               start_date: datetime, end_date: datetime) -> str:
        """
        Generate standardized storage path
        Format: data_source/symbol/timeframe/year/month/start_to_end.parquet
        """
        symbol_safe = symbol.replace('/', '_')
        year = start_date.year
        month = f"{start_date.month:02d}"
        start_iso = start_date.astimezone(timezone.utc).isoformat()
        end_iso = end_date.astimezone(timezone.utc).isoformat()        
        filename = f"{start_iso}to{end_iso}.parquet"
        
        path = os.path.join(
            self.storage_path,
            self.source_name,
            symbol_safe,
            timeframe,
            str(year),
            month
        )
        
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, filename)
        
    async def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str, 
                      start_date: datetime, end_date: datetime) -> str:
        """Save data to standardized path format"""
        if df.empty:
            self.logger.warning(f"Attempted to save empty data: {symbol} {timeframe}")
            return None
            
        file_path = self._generate_storage_path(symbol, timeframe, start_date, end_date)
        
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_parquet(file_path, index=False)
            self.logger.info(f"Saved data to: {file_path} ({len(df)} records)")
            
            return file_path
        except Exception as e:
            self.logger.error(f"Failed to save data to {file_path}: {e}")
            return None
    
    async def download_and_save(self, symbol: str, timeframe: str, 
                             start_date: Union[str, datetime, int] = None,
                             end_date: Union[str, datetime, int] = None) -> bool:
        """Download and save data in one operation"""
        try:
            # Ensure datetime objects for start/end, default start is: start '7d' / end 'now' 
            if start_date is not None and not isinstance(start_date, datetime):
                start_date = pd.to_datetime(start_date)
            if end_date is not None and not isinstance(end_date, datetime):
                end_date = pd.to_datetime(end_date)
            if start_date is None:
                start_date = datetime.now() - timedelta(days=7)  
            if end_date is None:
                end_date = datetime.now()
            
            # Download data
            df = await self.download_data(symbol, timeframe, start_date, end_date)
            
            if df.empty:
                self.logger.warning(f"No data to save: {symbol} {timeframe}")
                return False
                
            # Save data with standardized path
            file_path = await self.save_data(df, symbol, timeframe, start_date, end_date)
            return file_path is not None
            
        except Exception as e:
            self.logger.error(f"Error during download and save: {e}")
            return False
    
    async def download_many(self, symbols: List[str], timeframes: List[str],
                          start_date: Union[str, datetime, int] = None,
                          end_date: Union[str, datetime, int] = None,
                          max_concurrency: int = 2) -> Dict[str, bool]:
        """Download multiple symbols and timeframes"""
        tasks = []
        for symbol in symbols:
            for timeframe in timeframes:
                tasks.append((symbol, timeframe))
                
        results = {}
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def download_with_semaphore(symbol, timeframe):
            async with semaphore:
                key = f"{symbol}_{timeframe}"
                try:
                    success = await self.download_and_save(
                        symbol, timeframe, start_date, end_date
                    )
                    results[key] = success
                except Exception as e:
                    self.logger.error(f"Download task failed {key}: {e}")
                    results[key] = False
        
        # Create download tasks
        download_tasks = [download_with_semaphore(s, tf) for s, tf in tasks]
        
        # Execute all tasks
        await asyncio.gather(*download_tasks)
        
        # Count successes and failures
        success_count = sum(1 for success in results.values() if success)
        fail_count = len(results) - success_count
        
        self.logger.info(f"Batch download complete: {success_count} successful, {fail_count} failed")
        return results
    
    def check_local_file(self, symbol: str, timeframe: str, 
                        start_date: datetime, end_date: datetime) -> Optional[str]:
        """Check if local file exists for date range"""
        file_path = self._generate_storage_path(symbol, timeframe, start_date, end_date)
        if os.path.exists(file_path):
            return file_path
        return None
        
    async def ensure_data_available(self, symbol: str, timeframe: str,
                                  start_date: Union[str, datetime, int] = None,
                                  end_date: Union[str, datetime, int] = None) -> str:
        """Ensure data is available locally, downloading if needed"""
        # Convert dates to datetime
        if not isinstance(start_date, datetime) and start_date is not None:
            start_date = pd.to_datetime(start_date)
        
        if not isinstance(end_date, datetime) and end_date is not None:
            end_date = pd.to_datetime(end_date)
            
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
            
        if end_date is None:
            end_date = datetime.now()
        
        # Check for local file
        file_path = self.check_local_file(symbol, timeframe, start_date, end_date)
        
        if file_path:
            self.logger.info(f"Data for {symbol} {timeframe} already exists locally")
            return file_path
        
        # Download if not available
        self.logger.info(f"Data for {symbol} {timeframe} not found locally, downloading...")
        success = await self.download_and_save(symbol, timeframe, start_date, end_date)
        
        if success:
            return self._generate_storage_path(symbol, timeframe, start_date, end_date)
        
        return None