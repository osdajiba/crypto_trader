# src/exchange/binance.py

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import asyncio
from typing import Dict, List, Union, Optional, Any, Callable
from datetime import datetime, timedelta
import time

from src.common.log_manager import LogManager

logger = LogManager.getLogger("trading_system")

class Binance:
    """
    Optimized Binance interface for data retrieval with unified logic
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize Binance interface with configuration
        
        Args:
            config: Configuration dictionary with API credentials and settings
        """
        self.config = config or {}
        
        # Extract configuration parameters with safe defaults
        self.params = self._build_params()
        
        # Initialize exchange objects
        self.exchange = None
        self.async_exchange = None
        self._init_exchange()
        
        # Rate limiting configuration
        self.rate_limit = 1200  # Default requests per minute
        self.current_requests = 0
        self.rate_limit_reset = time.time() + 60
        
        # Download settings
        self.download_chunk_size = 1000  # Number of candles per API call
        
        logger.info("Binance interface initialized")
    
    

    def _build_params(self) -> Dict:
        """
        Build CCXT parameters from configuration

        Returns:
        Dictionary of CCXT parameters
        """
        # Extract API credentials safely
        api_key = self.config.get('api', 'binance', 'apiKey')
        api_secret = self.config.get('api', 'binance', 'secret')

        # Basic parameters
        params = {
            'timeout': self.config.get('api', 'timeout', default=30000),
            'enableRateLimit': self.config.get('api', 'enableRateLimit', default=True),
            'options': {
            'adjustForTimeDifference': self.config.get(
            'default_config', 'options', 'adjustForTimeDifference', default=True),
            'recvWindow': self.config.get(
            'default_config', 'options', 'recvWindow', default=60000),
            'defaultType': self.config.get(
            'default_config', 'options', 'defaultType', default='spot')
            },
            'http': self.config.get('proxies', 'http'),
            'https': self.config.get('proxies', 'https')
        }

        # Add API credentials if provided
        if api_key and api_secret:
            params['apiKey'] = api_key
            params['secret'] = api_secret

        return params
    
    def _nested_get(self, dictionary: Dict, keys: List, default=None) -> Any:
        """
        Safely get a nested value from a dictionary
        
        Args:
            dictionary: Dictionary to search in
            keys: List of keys forming the path
            default: Default value if path doesn't exist
            
        Returns:
            Value at path or default
        """
        current = dictionary
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current
    
    def _init_exchange(self) -> None:
        """Initialize the synchronous exchange connection"""
        try:
            self.exchange = ccxt.binance(self.params)
            logger.info("CCXT Binance exchange initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Binance exchange: {str(e)}")
            self.exchange = None
    
    async def _init_async_exchange(self) -> None:
        """Initialize the asynchronous exchange connection (lazy initialization)"""
        if self.async_exchange is not None:
            return

        try:
            self.async_exchange = ccxt_async.binance(self.params)
            logger.info("CCXT Async Binance exchange initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Async Binance exchange: {str(e)}")
            self.async_exchange = None
    
    def _handle_rate_limit(self, is_async: bool = False) -> Callable:
        """
        Handle rate limiting for API requests
        
        Args:
            is_async: Whether this is for an async context
            
        Returns:
            Coroutine for async handling or None for sync handling
        """
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
            
            # Handle sleeping based on context
            if is_async:
                return asyncio.sleep(sleep_time)
            else:
                time.sleep(sleep_time)
                # Reset after sleeping
                self.current_requests = 0
                self.rate_limit_reset = time.time() + 60
        
        # Increment the request counter
        self.current_requests += 1
        
        # For async context, return completed future
        if is_async:
            return asyncio.sleep(0)
    
    def _process_ohlcv_data(self, ohlcv_data: List) -> pd.DataFrame:
        """
        Process OHLCV data into a DataFrame
        
        Args:
            ohlcv_data: Raw OHLCV data from CCXT
        
        Returns:
            Processed DataFrame
        """
        if not ohlcv_data:
            return pd.DataFrame()
        
        try:
            # Create DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Set datetime as index
            df.set_index('datetime', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error processing OHLCV data: {str(e)}")
            return pd.DataFrame()
    
    def fetch_latest_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """
        Fetch OHLCV data synchronously
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1m', '1h', '1d')
            limit: Number of candles to fetch
        
        Returns:
            DataFrame with OHLCV data
        """
        if not self.exchange:
            logger.error("Exchange not initialized")
            return pd.DataFrame()
        
        try:
            # Handle rate limiting
            self._handle_rate_limit()
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
            
            # Convert to DataFrame
            return self._process_ohlcv_data(ohlcv)
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {str(e)}")
            return pd.DataFrame()
    
    async def fetch_ohlcv_async(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """
        Fetch OHLCV data asynchronously
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            limit: Number of candles to fetch
        
        Returns:
            DataFrame with OHLCV data
        """
        # Initialize async exchange if needed
        await self._init_async_exchange()
        
        if not self.async_exchange:
            logger.error("Async exchange not initialized")
            return pd.DataFrame()
        
        try:
            # Handle rate limiting
            await self._handle_rate_limit(is_async=True)
            
            # Fetch OHLCV data
            ohlcv = await self.async_exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
            
            # Convert to DataFrame
            return self._process_ohlcv_data(ohlcv)
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data async: {str(e)}")
            return pd.DataFrame()
    
    async def fetch_historical_ohlcv(self, 
                                   symbol: str, 
                                   timeframe: str = '1h',
                                   start_date: Optional[Union[str, datetime]] = None,
                                   end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Fetch historical OHLCV data with smart chunking
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_date: Start date/time
            end_date: End date/time
        
        Returns:
            DataFrame with historical OHLCV data
        """
        # Make sure exchange is initialized
        if not self.exchange:
            self._init_exchange()
            if not self.exchange:
                logger.error("Exchange initialization failed")
                return pd.DataFrame()
        
        try:
            # Process dates
            start_dt = self._parse_date(start_date, default_days_ago=30)
            end_dt = self._parse_date(end_date, default_days_ago=0)
            
            logger.info(f"Fetching historical data for {symbol} from {start_dt} to {end_dt}")
            
            # Convert dates to timestamps
            since = int(start_dt.timestamp() * 1000)
            until = int(end_dt.timestamp() * 1000)
            
            # Get all data chunks
            chunks = []
            current_since = since
            
            while current_since < until:
                try:
                    # Handle rate limiting
                    self._handle_rate_limit()
                    
                    # Fetch a chunk of data using synchronous exchange
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol=symbol, 
                        timeframe=timeframe, 
                        since=current_since,
                        limit=self.download_chunk_size
                    )
                    
                    if not ohlcv:
                        logger.warning(f"No data returned for {symbol} at {current_since}")
                        break
                    
                    # Add to chunks
                    chunks.extend(ohlcv)
                    
                    # Get the last timestamp
                    last_timestamp = ohlcv[-1][0]
                    
                    # If we've reached the end date, stop
                    if last_timestamp >= until:
                        break
                    
                    # Set the next since to the last timestamp + 1ms
                    current_since = last_timestamp + 1
                    
                    # Log progress
                    current_date = datetime.fromtimestamp(current_since / 1000)
                    logger.info(f"Downloaded data up to {current_date}")
                    
                    # Small delay to avoid overwhelming the API
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error fetching historical chunk: {str(e)}")
                    # Wait a bit longer on error
                    await asyncio.sleep(2)
                    # Try to continue from the next timestamp
                    if chunks:
                        current_since = chunks[-1][0] + 1
                    else:
                        # If no data fetched yet, move forward in time
                        current_since += 86400000  # Move forward by 1 day
            
            # Process all chunks
            if not chunks:
                logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
            
            # Process the data
            df = self._process_ohlcv_data(chunks)
            
            # Filter by date range
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            
            logger.info(f"Downloaded {len(df)} candles for {symbol}")
            return df
        
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _parse_date(self, date_input: Optional[Union[str, datetime]], default_days_ago: int = 0) -> datetime:
        """
        Parse date input into datetime object
        
        Args:
            date_input: Date string or datetime object
            default_days_ago: Days ago to use if input is None
            
        Returns:
            Datetime object
        """
        if date_input is None:
            return datetime.now() - timedelta(days=default_days_ago)
            
        if isinstance(date_input, datetime):
            return date_input
            
        # Try to parse string
        try:
            return pd.to_datetime(date_input)
        except Exception as e:
            logger.warning(f"Failed to parse date '{date_input}': {str(e)}. Using default.")
            return datetime.now() - timedelta(days=default_days_ago)
    
    # Compatibility methods for existing code
    
    async def fetch_historical(self, symbol: str, timeframe: str, start=None, end=None) -> pd.DataFrame:
        """Compatibility method for DataSource interface"""
        return await self.fetch_historical_ohlcv(symbol, timeframe, start, end)
    
    async def fetch_realtime(self, symbol: str, timeframe: str, limit=100) -> pd.DataFrame:
        """Compatibility method for DataSource interface"""
        # We're using the synchronous version for better reliability
        return self.fetch_latest_ohlcv(symbol, timeframe, limit)
    
    async def smart_fetch_ohlcv(self, symbol, timeframe, start, end):
        """Compatibility method for existing code"""
        return await self.fetch_historical_ohlcv(symbol, timeframe, start, end)
    
    async def close(self):
        """Close all connections"""
        try:
            if self.async_exchange:
                await self.async_exchange.close()
                self.async_exchange = None
                logger.info("Async exchange connection closed")
        except Exception as e:
            logger.error(f"Error closing async exchange: {str(e)}")