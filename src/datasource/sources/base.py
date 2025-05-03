#!/usr/bin/env python3
# src/datasource/sources/base.py

"""
Base data source implementation.
Provides abstract base class for all data sources.
"""

import asyncio
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from src.common.helpers import TimeUtils
from src.common.log_manager import LogManager
from src.common.config_manager import ConfigManager
from src.common.async_executor import AsyncExecutor


class DataSourceError(Exception):
    """Base exception for data source errors"""
    pass


class DataSourceConnectionError(DataSourceError):
    """Connection error with data source"""
    pass


class DataSourceFetchError(DataSourceError):
    """Error fetching data from source"""
    pass


class DataValidationError(DataSourceError):
    """Error validating data format or integrity"""
    pass


class BaseDataSource(ABC):
    """
    Abstract base class defining interfaces for all data sources
    
    This class provides the foundation for all data source implementations with
    standardized data fetching, validation, and resource management.
    """
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        Initialize data source
        
        Args:
            config: Configuration manager
            params: Additional parameters for data source configuration
        """
        self.config = config
        self.params = params or {}
        self.executor = AsyncExecutor()
        self.logger = LogManager.get_logger(f"datasource.{self.__class__.__name__.lower()}")
        
        # Set default parameters
        self._is_initialized = False
        self._is_running = False
    
    async def initialize(self) -> None:
        """
        Initialize data source connections and resources
        
        This method should be called before using the data source.
        """
        if self._is_initialized:
            return
            
        try:
            self.logger.info(f"Initializing {self.__class__.__name__}")
            await self._initialize_source_specific()
            self._is_initialized = True
            self._is_running = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.__class__.__name__}: {str(e)}")
            raise DataSourceConnectionError(f"Initialization failed: {str(e)}")
    
    async def _initialize_source_specific(self) -> None:
        """
        Initialize data source specific resources and connections
        
        This method should be implemented by subclasses to perform
        source-specific initialization.
        """
        pass
    
    @abstractmethod
    async def fetch_historical(self, symbol: str, timeframe: str, 
                             start: Optional[Union[str, datetime]] = None, 
                             end: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Get historical OHLCV data
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Timeframe for candles (e.g., "1h", "1d")
            start: Start datetime or string
            end: End datetime or string
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
            
        Raises:
            DataSourceFetchError: If data fetching fails
        """
        pass

    @abstractmethod
    async def fetch_realtime(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Get real-time OHLCV data
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Timeframe for candles (e.g., "1h", "1d")
            
        Returns:
            pd.DataFrame: DataFrame with latest OHLCV data
            
        Raises:
            DataSourceFetchError: If data fetching fails
        """
        pass
    
    async def fetch_all_data_for_symbols(self, symbols: List[str], timeframe: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols with concurrent processing
        
        Args:
            symbols: List of trading pair symbols
            timeframe: Timeframe for candles
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their data
        """
        results = {}
        
        # Define fetch task for a single symbol
        async def fetch_symbol_data(symbol):
            try:
                data = await self.fetch_realtime(symbol, timeframe)
                if not data.empty:
                    return symbol, data
                else:
                    self.logger.warning(f"No data fetched for {symbol}")
                    return symbol, None
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
                return symbol, None
        
        # Create tasks for all symbols
        tasks = [fetch_symbol_data(symbol) for symbol in symbols]
        
        # Run concurrently and process results
        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in fetch_results:
            if isinstance(result, Exception):
                self.logger.error(f"Fetch task error: {str(result)}")
                continue
                
            symbol, data = result
            if data is not None and not data.empty:
                results[symbol] = data
        
        return results

    async def shutdown(self) -> None:
        """
        Close data source connections and clean up resources
        
        This method performs standard cleanup and then calls
        shutdown_source_specific for subclass-specific cleanup.
        """
        if not self._is_running:
            return
            
        try:
            # Perform source-specific cleanup
            await self._shutdown_source_specific()
            
            # Close executor if available
            if hasattr(self.executor, 'close'):
                await self.executor.close()
                
            self._is_running = False
            self._is_initialized = False
            
            self.logger.info(f"{self.__class__.__name__} closed")
            
        except Exception as e:
            self.logger.error(f"Error during {self.__class__.__name__} shutdown: {str(e)}")
    
    async def _shutdown_source_specific(self) -> None:
        """
        Perform source-specific cleanup
        
        This method should be implemented by subclasses to clean up
        source-specific resources.
        """
        pass
    
    def validate_ohlcv_data(self, df: pd.DataFrame) -> bool:
        """
        Validate OHLCV data format and integrity
        
        Args:
            df: DataFrame to validate
            
        Returns:
            bool: True if data is valid
            
        Raises:
            DataValidationError: If validation fails
        """
        if df.empty:
            return True
            
        # Check for required columns
        required_columns = ['datetime', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")
            
        # Check for numeric values in price columns
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise DataValidationError(f"Column {col} must contain numeric values")
                
        # Check for datetime column
        if not pd.api.types.is_datetime64_dtype(df['datetime']):
            try:
                df['datetime'] = pd.to_datetime(df['datetime'])
            except:
                raise DataValidationError("Cannot convert 'datetime' column to datetime type")
                
        # Check for logical price relationships
        invalid_rows = (
            (df['high'] < df['low']) | 
            (df['high'] < df['open']) | 
            (df['high'] < df['close']) | 
            (df['low'] > df['open']) | 
            (df['low'] > df['close'])
        )
        
        if invalid_rows.any():
            self.logger.warning(f"Found {invalid_rows.sum()} rows with invalid price relationships")
            
        return True
    
    @staticmethod
    def get_optimal_data_ranges(start_dt: datetime, end_dt: datetime, 
                              timeframe: str, max_points: int = 1000) -> List[Tuple[datetime, datetime]]:
        """
        Break large date ranges into smaller chunks for optimized fetching
        
        Args:
            start_dt: Start datetime
            end_dt: End datetime
            timeframe: Timeframe for candles
            max_points: Maximum number of data points per request
            
        Returns:
            List[Tuple[datetime, datetime]]: List of date ranges
        """
        seconds_per_candle = TimeUtils.timeframe_to_seconds(timeframe)
        total_seconds = (end_dt - start_dt).total_seconds()
        estimated_points = total_seconds / seconds_per_candle
        
        if estimated_points <= max_points:
            return [(start_dt, end_dt)]
            
        num_chunks = int(estimated_points / max_points) + 1
        chunk_seconds = total_seconds / num_chunks
        
        ranges = []
        for i in range(num_chunks):
            chunk_start = start_dt + timedelta(seconds=i * chunk_seconds)
            chunk_end = start_dt + timedelta(seconds=(i+1) * chunk_seconds)
            
            if i == num_chunks - 1:
                chunk_end = end_dt
                
            ranges.append((chunk_start, chunk_end))
            
        return ranges