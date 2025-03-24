# src/data/data_manager.py

import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from .sources import DataSourceFactory, LocalSource, ExchangeSource, DatabaseSource
from .integrity_checker import DataIntegrityChecker


class DataManager:
    """
    Data Manager responsible for fetching, caching, and managing market data
    """
    
    def __init__(self, source_type: str = "local", config: Optional[ConfigManager] = None):
        """
        Initialize the Data Manager
        
        Args:
            source_type: Type of data source ("local", "exchange", "hybrid")
            config: Configuration manager instance
        """
        self.config = config
        self.source_type = source_type.lower()
        self.logger = LogManager.get_logger("data.manager")
        
        # Data sources
        self.primary_source = None
        self.backup_source = None
        
        # Data cache
        self.data_cache = {}
        self.use_cache = config.get("data", "cache", "enabled", default=True) if config else True
        
        # Data integrity checker
        self.integrity_checker = DataIntegrityChecker(
            timeframe=config.get("data", "default_timeframe", default="1h") if config else "1h"
        )
        self.validate_data = config.get("data", "validation", "enabled", default=True) if config else True
        
        # Pagination settings
        self.page_size = config.get("data", "pagination", "page_size", default=1000) if config else 1000
        self.max_concurrent_pages = config.get("data", "pagination", "max_concurrent", default=5) if config else 5
        
        # Create data sources based on source_type
        self._initialize_data_sources()
        
        self.logger.info(f"DataManager initialized: source={self.source_type}, cache={self.use_cache}, mode={self._get_trading_mode()}")
    
    def _initialize_data_sources(self):
        """Initialize appropriate data sources based on configuration"""
        try:
            factory = DataSourceFactory()
            primary_type = self.config.get("data", "primary_source", default="local")
            backup_type = self.config.get("data", "backup_source", default="exchange")
            
            # 初始化primary_source基于source_type
            if self.source_type == "local":
                self.primary_source = factory.create_source("local", self.config)
                    
            elif self.source_type == "exchange":
                self.primary_source = factory.create_source("exchange", self.config)
                
            elif self.source_type == "database":
                self.primary_source = factory.create_source("database", self.config)
                    
            elif self.source_type == "hybrid":
                self.primary_source = factory.create_source(primary_type, self.config)
                self.backup_source = factory.create_source(backup_type, self.config)
                    
            else:
                raise ValueError(f"Unsupported data source type: {self.source_type}")
                    
            if self._get_trading_mode() == "BACKTEST" and not self.backup_source:
                if self.source_type == "local":
                    self.backup_source = factory.create_source(backup_type, self.config)
                    
        except Exception as e:
            self.logger.error(f"Error initializing data sources: {e}", exc_info=True)
            raise
    
    def _get_trading_mode(self) -> str:
        """Get current trading mode from config"""
        if not self.config:
            return "backtest"  # Default to backtest
            
        mode = self.config.get("system", "operational_mode", default="backtest")
        return mode.upper()
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start: Optional[Union[str, datetime]] = None, 
        end: Optional[Union[str, datetime]] = None,
        use_pagination: bool = True
    ) -> pd.DataFrame:
        """
        Get historical market data with improved pagination and error handling
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Data timeframe/interval (e.g., "1h", "1d")
            start: Start datetime or date string
            end: End datetime or date string
            use_pagination: Whether to use pagination for large data requests
            
        Returns:
            pd.DataFrame: Historical market data
        """
        # Check cache first
        cache_key = f"{symbol}|{timeframe}|{start}|{end}"
        if self.use_cache and cache_key in self.data_cache:
            self.logger.info(f"Historical data loaded from cache: {cache_key}")
            return self.data_cache[cache_key]
        
        try:
            # Try primary source first
            try:
                data = await self.primary_source.fetch_historical(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=start,
                    end=end
                )
                
                if not data.empty:
                    # Validate data if enabled
                    if self.validate_data:
                        valid, results = await self.integrity_checker.check(data)
                        if not valid:
                            self.logger.warning(f"Data integrity issues found for {symbol} {timeframe}")
                            data = await self._fix_data_issues(data, results)
                    
                    # Cache the result
                    if self.use_cache:
                        self.data_cache[cache_key] = data
                        self.logger.info(f"Historical data cached: {cache_key}")
                    
                    return data
                else:
                    self.logger.warning(f"No historical data found in primary source for {symbol} {timeframe}")
                
            except Exception as e:
                self.logger.warning(f"Primary source failed for {symbol} {timeframe}: {e}")
                
            # If primary source fails or returns empty and we have a backup, try it
            if self.backup_source:
                self.logger.info(f"Trying backup source for {symbol} {timeframe}")
                try:
                    data = await self.backup_source.fetch_historical(
                        symbol=symbol,
                        timeframe=timeframe,
                        start=start,
                        end=end
                    )
                    
                    if not data.empty:
                        # Validate data if enabled
                        if self.validate_data:
                            valid, results = await self.integrity_checker.check(data)
                            if not valid:
                                self.logger.warning(f"Data integrity issues found in backup data for {symbol} {timeframe}")
                                data = await self._fix_data_issues(data, results)
                        
                        # Cache the result
                        if self.use_cache:
                            self.data_cache[cache_key] = data
                            self.logger.info(f"Backup historical data cached: {cache_key}")
                        
                        # If local is backup and exchange is primary, store the downloaded data
                        if isinstance(self.backup_source, LocalSource) and hasattr(self.backup_source, 'update_data'):
                            await self.backup_source.update_data(symbol, timeframe, data)
                            self.logger.info(f"Data saved to local storage: {symbol} {timeframe}")
                        
                        return data
                
                except Exception as e:
                    self.logger.error(f"Backup source also failed for {symbol} {timeframe}: {e}")
            
            # If we got here, both sources failed or returned empty
            self.logger.error(f"Failed to get historical data for {symbol} {timeframe}")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}", exc_info=True)
            return pd.DataFrame()

    async def get_latest_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Get the most recent market data
        
        Args:
            symbol: Trading pair symbol
            timeframe: Data timeframe
            limit: Maximum number of records to fetch
            
        Returns:
            pd.DataFrame: Latest market data
        """
        # Only relevant for exchange or hybrid sources
        if self.source_type in ["exchange", "hybrid"]:
            try:
                data = await self.primary_source.fetch_realtime(symbol, timeframe)
                
                if self.validate_data and not data.empty:
                    valid, results = await self.integrity_checker.check(data)
                    if not valid:
                        self.logger.warning(f"Data integrity issues found in latest data for {symbol} {timeframe}")
                        data = await self._fix_data_issues(data, results)
                        
                return data
                
            except Exception as e:
                self.logger.error(f"Error fetching latest data: {e}")
                return pd.DataFrame()
        else:
            # For local source, get most recent data from historical
            end = datetime.now()
            start = end - timedelta(days=7)  # Fetch last 7 days
            return await self.get_historical_data(symbol, timeframe, start, end)
    
    async def download_and_store_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start: Optional[Union[str, datetime]] = None, 
        end: Optional[Union[str, datetime]] = None
    ) -> bool:
        """
        Download data from exchange and store it
        
        Args:
            symbol: Trading pair symbol
            timeframe: Data timeframe
            start: Start date
            end: End date
            
        Returns:
            bool: Success status
        """
        # Need an exchange source
        exchange_source = None
        if self.source_type == "exchange":
            exchange_source = self.primary_source
        elif self.backup_source and hasattr(self.backup_source, 'fetch_historical'):
            exchange_source = self.backup_source
            
        if not exchange_source:
            self.logger.warning(f"No exchange source available for downloading {symbol} {timeframe}")
            return False
            
        try:
            # Parse dates
            from src.utils.time_utils import TimeUtils
            start_dt = TimeUtils.parse_timestamp(start) if start else None
            end_dt = TimeUtils.parse_timestamp(end) if end else datetime.now()
            
            # Default start date is 1 year ago if not specified
            if not start_dt:
                start_dt = end_dt - timedelta(days=365)
                
            self.logger.info(f"Downloading data for {symbol} {timeframe} from {start_dt} to {end_dt}")
            
            # Fetch data from exchange
            data = await exchange_source.fetch_historical(
                symbol=symbol,
                timeframe=timeframe,
                start=start_dt,
                end=end_dt
            )
            
            if data.empty:
                self.logger.error(f"Failed to download data for {symbol} {timeframe}")
                return False
                
            # Get a local source to store the data
            local_source = None
            if self.source_type == "local":
                local_source = self.primary_source
            elif hasattr(self.backup_source, 'update_data'):
                local_source = self.backup_source
                
            if not local_source:
                self.logger.error(f"No local source available for storing data")
                return False
                
            # Store the data
            success = await local_source.update_data(symbol, timeframe, data)
            
            if success:
                self.logger.info(f"Successfully downloaded and stored {len(data)} records for {symbol} {timeframe}")
            else:
                self.logger.error(f"Failed to store downloaded data for {symbol} {timeframe}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error downloading and storing data: {e}")
            return False

    async def _fix_data_issues(self, df: pd.DataFrame, check_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply automatic fixes to common data issues
        
        Args:
            df: Original DataFrame with issues
            check_results: Results from integrity checker
            
        Returns:
            pd.DataFrame: Fixed DataFrame
        """
        fixed_df = df.copy()
        
        try:
            # Get timestamp column
            timestamp_col = None
            for col in ['datetime', 'timestamp', 'date', 'time']:
                if col in fixed_df.columns:
                    timestamp_col = col
                    break
            
            # Fix: Sort by timestamp
            if timestamp_col and check_results.get("timestamp_sequence", {}).get("passed") is False:
                self.logger.info("Fixing timestamp sequence by sorting")
                fixed_df = fixed_df.sort_values(by=timestamp_col)
            
            # Fix: Remove duplicates
            if timestamp_col and check_results.get("duplicate_timestamps", {}).get("passed") is False:
                self.logger.info("Fixing duplicate timestamps by keeping last occurrence")
                fixed_df = fixed_df.drop_duplicates(subset=[timestamp_col], keep='last')
            
            # Fix: Handle missing values
            if check_results.get("missing_values", {}).get("passed") is False:
                self.logger.info("Fixing missing values using forward fill")
                fixed_df = fixed_df.ffill()
            
            # Fix: OHLC consistency
            if check_results.get("ohlc_consistency", {}).get("passed") is False:
                self.logger.info("Fixing OHLC consistency issues")
                
                # Ensure high is highest
                fixed_df['high'] = fixed_df[['high', 'open', 'close']].max(axis=1)
                
                # Ensure low is lowest
                fixed_df['low'] = fixed_df[['low', 'open', 'close']].min(axis=1)
            
            # Fix: Zero prices
            if check_results.get("zero_values", {}).get("passed") is False:
                price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in fixed_df.columns]
                for col in price_cols:
                    zero_mask = fixed_df[col] == 0
                    if zero_mask.any():
                        self.logger.info(f"Replacing zero values in {col}")
                        fixed_df.loc[zero_mask, col] = None  # Convert zeros to NaN
                        fixed_df[col] = fixed_df[col].ffill()  # Forward fill NaNs
            
            return fixed_df
            
        except Exception as e:
            self.logger.error(f"Error fixing data issues: {e}")
            return df  # Return original if fixes fail
    
    async def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> None:
        """
        Clear the data cache for specific symbol/timeframe or all
        
        Args:
            symbol: Symbol to clear (None for all)
            timeframe: Timeframe to clear (None for all)
        """
        if not self.use_cache:
            return
            
        if symbol and timeframe:
            # Clear specific symbol and timeframe
            prefix = f"{symbol}|{timeframe}"
            keys_to_remove = [k for k in self.data_cache.keys() if k.startswith(prefix)]
            for key in keys_to_remove:
                del self.data_cache[key]
            self.logger.info(f"Cleared cache for {symbol} {timeframe} ({len(keys_to_remove)} entries)")
            
        elif symbol:
            # Clear all timeframes for a symbol
            keys_to_remove = [k for k in self.data_cache.keys() if k.startswith(f"{symbol}|")]
            for key in keys_to_remove:
                del self.data_cache[key]
            self.logger.info(f"Cleared cache for {symbol} ({len(keys_to_remove)} entries)")
            
        elif timeframe:
            # Clear all symbols for a timeframe
            keys_to_remove = [k for k in self.data_cache.keys() if f"|{timeframe}|" in k]
            for key in keys_to_remove:
                del self.data_cache[key]
            self.logger.info(f"Cleared cache for timeframe {timeframe} ({len(keys_to_remove)} entries)")
            
        else:
            # Clear all cache
            count = len(self.data_cache)
            self.data_cache.clear()
            self.logger.info(f"Cleared entire data cache ({count} entries)")
    
    async def preload_data(self, symbols: List[str], timeframe: str, days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Preload data for multiple symbols to improve performance
        
        Args:
            symbols: List of symbols to preload
            timeframe: Timeframe to preload
            days: Number of days to preload
            
        Returns:
            Dict[str, pd.DataFrame]: Preloaded data keyed by symbol
        """
        if not symbols:
            return {}
            
        # Calculate date range
        end = datetime.now()
        start = end - timedelta(days=days)
        
        # Create tasks to load data for each symbol
        results = {}
        for symbol in symbols:
            try:
                data = await self.get_historical_data(symbol, timeframe, start, end)
                if not data.empty:
                    results[symbol] = data
                    self.logger.debug(f"Preloaded {len(data)} records for {symbol} {timeframe}")
                else:
                    self.logger.warning(f"No data available when preloading {symbol} {timeframe}")
            except Exception as e:
                self.logger.error(f"Error preloading data for {symbol}: {e}")
                
        self.logger.info(f"Preloaded data for {len(results)}/{len(symbols)} symbols")
        return results
    
    async def close(self) -> None:
        """Clean up resources"""
        # Close data sources
        if self.primary_source and hasattr(self.primary_source, 'close'):
            try:
                await self.primary_source.close()
                self.logger.debug("Closed primary data source")
            except Exception as e:
                self.logger.error(f"Error closing primary data source: {e}")
                
        if self.backup_source and hasattr(self.backup_source, 'close'):
            try:
                await self.backup_source.close()
                self.logger.debug("Closed backup data source")
            except Exception as e:
                self.logger.error(f"Error closing backup data source: {e}")
                
        # Clear cache
        self.data_cache.clear()