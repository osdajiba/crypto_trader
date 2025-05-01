#!/usr/bin/env python3
# src/datasource/sources/hybrid.py

"""
Hybrid data source implementation.
Combines local and exchange data sources with fallback support.
"""

from typing import Dict, Optional, Union, Any, List
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import time

from src.common.abstract_factory import register_factory_class
from src.common.helpers import TimeUtils
from src.common.config import ConfigManager
from src.datasource.sources.base import BaseDataSource, DataSourceFetchError


@register_factory_class('datasource_factory', 'hybrid', 
    description="Hybrid data source combining local and exchange data",
    features=["historical", "realtime", "online_access", "offline_access"],
    category="datasource")
class HybridDataSource(BaseDataSource):
    """
    Hybrid data source combining local and exchange data sources
    
    This data source provides a unified interface to both local and exchange
    data, with configurable preferences and automatic fallback. It optimizes
    for data availability and reduces unnecessary exchange API calls.
    """
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        Initialize hybrid data source
        
        Args:
            config: Configuration manager
            params: Additional parameters
        """
        super().__init__(config, params)
        
        self.local_source = None
        self.exchange_source = None
        self.fallback_enabled = True
        self.prefer_local = True
        self.cache_results = True
    
    async def _initialize_source_specific(self) -> None:
        """Initialize both local and exchange data sources"""
        from src.datasource.sources.factory import get_datasource_factory
        
        factory = get_datasource_factory(self.config)
        
        # Create local source
        self.local_source = await factory.create('local')
        self.logger.info("Local source initialized successfully")
        
        # Create exchange source if online
        try:
            self.exchange_source = await factory.create('exchange')
            self.logger.info("Exchange source initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize exchange source: {e}")
            self.exchange_source = None
        
        # Configure behavior
        self.fallback_enabled = self.config.get('data', 'sources', 'hybrid', 'fallback_enabled', default=True)
        self.prefer_local = self.config.get('data', 'sources', 'hybrid', 'prefer_local', default=True)
        self.cache_results = self.config.get('data', 'sources', 'hybrid', 'cache_results', default=True)
        
        self.logger.info(f"HybridDataSource initialized - prefer_local: {self.prefer_local}, fallback_enabled: {self.fallback_enabled}")
    
    async def fetch_historical(self, symbol: str, timeframe: str, 
                             start: Optional[Union[str, datetime]] = None, 
                             end: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Fetch historical data with intelligent source selection
        
        This method intelligently selects between local and exchange data
        sources based on configuration, recency, and data availability.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for candles
            start: Start datetime or string
            end: End datetime or string
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
            
        Raises:
            DataSourceFetchError: If data fetching fails from all sources
        """
        if not self._is_initialized:
            await self.initialize()
            
        self.logger.info(f"Fetching historical data for: {symbol} {timeframe}")

        # Calculate date range
        end_dt = TimeUtils.parse_timestamp(end, default_days_ago=0) if end else datetime.now()
        start_dt = TimeUtils.parse_timestamp(start, default_days_ago=30) if start else end_dt - timedelta(days=30)
        
        # Determine if data is recent (within last day)
        now = datetime.now()
        is_recent = end_dt > (now - timedelta(days=1))
        
        # Choose source based on recency and preferences
        try:
            # Strategy 1: Use exchange for recent data if available and preferred
            if is_recent and self.exchange_source and not self.prefer_local:
                self.logger.debug(f"Using exchange source for recent data: {symbol} {timeframe}")
                try:
                    df = await self.exchange_source.fetch_historical(symbol, timeframe, start_dt, end_dt)
                    
                    # Save to local if cache enabled and data retrieved
                    if not df.empty and self.cache_results and self.local_source:
                        try:
                            await self._cache_data(df, symbol, timeframe)
                        except Exception as e:
                            self.logger.warning(f"Failed to cache data: {e}")
                            
                    if not df.empty:
                        return df
                except Exception as e:
                    self.logger.warning(f"Exchange fetch failed, trying local: {e}")
                    # Continue to local source fallback
            
            # Strategy 2: Try local source
            if self.local_source:
                self.logger.debug(f"Using local source for historical data: {symbol} {timeframe}")
                try:
                    df = await self.local_source.fetch_historical(symbol, timeframe, start_dt, end_dt)
                    
                    # If local data is sufficient, return it
                    if not df.empty:
                        return df
                except Exception as e:
                    self.logger.warning(f"Local fetch failed: {e}")
                    # Continue to exchange fallback if enabled
            
            # Strategy 3: Fallback to exchange if local failed or was empty
            if df.empty and self.fallback_enabled and self.exchange_source:
                self.logger.debug(f"Falling back to exchange source for {symbol} {timeframe}")
                try:
                    df = await self.exchange_source.fetch_historical(symbol, timeframe, start_dt, end_dt)
                    
                    # Save to local if cache enabled and data retrieved
                    if not df.empty and self.cache_results and self.local_source:
                        try:
                            await self._cache_data(df, symbol, timeframe)
                        except Exception as e:
                            self.logger.warning(f"Failed to cache data: {e}")
                            
                    return df
                except Exception as e:
                    self.logger.error(f"Exchange fallback failed: {e}")
                    # Last attempt failed, return empty DataFrame
            
            # If we reach here without returning, return empty DataFrame
            return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            
            # Last resort fallbacks in case of unexpected errors
            if self.fallback_enabled:
                if self.exchange_source and not self.prefer_local:
                    self.logger.info(f"Error fallback to local source for {symbol} {timeframe}")
                    try:
                        return await self.local_source.fetch_historical(symbol, timeframe, start_dt, end_dt)
                    except Exception:
                        pass
                elif self.exchange_source:
                    self.logger.info(f"Error fallback to exchange source for {symbol} {timeframe}")
                    try:
                        return await self.exchange_source.fetch_historical(symbol, timeframe, start_dt, end_dt)
                    except Exception:
                        pass
            
            # All attempts failed
            raise DataSourceFetchError(f"Failed to fetch historical data for {symbol} {timeframe} from all sources")
    
    async def fetch_realtime(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Get real-time data with automatic fallback
        
        This method prioritizes exchange data for real-time updates but
        falls back to local data if exchange is unavailable.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for candles
            
        Returns:
            pd.DataFrame: DataFrame with latest OHLCV data
            
        Raises:
            DataSourceFetchError: If data fetching fails from all sources
        """
        if not self._is_initialized:
            await self.initialize()
            
        self.logger.info(f"Fetching real-time data for: {symbol} {timeframe}")
        
        # Always try exchange first for real-time data (if available)
        if self.exchange_source:
            try:
                df = await self.exchange_source.fetch_realtime(symbol, timeframe)
                if not df.empty:
                    # Cache if enabled
                    if self.cache_results and self.local_source:
                        try:
                            await self._cache_data(df, symbol, timeframe)
                        except Exception as e:
                            self.logger.warning(f"Failed to cache real-time data: {e}")
                    
                    return df
            except Exception as e:
                self.logger.warning(f"Exchange real-time fetch failed: {e}")
                # Continue to fallback
        
        # Fallback to local if enabled and exchange failed or is unavailable
        if self.fallback_enabled and self.local_source:
            self.logger.debug(f"Falling back to local source for real-time data: {symbol} {timeframe}")
            try:
                return await self.local_source.fetch_realtime(symbol, timeframe)
            except Exception as e:
                self.logger.error(f"Local real-time fetch failed: {e}")
                # Both sources failed
        
        # If all fails, return empty DataFrame
        self.logger.warning(f"No real-time data available for {symbol} {timeframe}")
        return pd.DataFrame()
    
    async def _cache_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """
        Cache data to local storage using local data source
        
        Args:
            df: DataFrame to cache
            symbol: Trading pair symbol
            timeframe: Timeframe for candles
        """
        if not self.local_source or df.empty:
            return
        
        # Use local source's save_data method if available
        if hasattr(self.local_source, 'save_data'):
            try:
                await self.local_source.save_data(df, symbol, timeframe)
                self.logger.debug(f"Cached {len(df)} rows for {symbol} {timeframe}")
            except Exception as e:
                self.logger.warning(f"Failed to cache data via local source: {e}")
        else:
            # Fallback - log that we would cache but method not available
            self.logger.debug(f"Would cache {len(df)} rows for {symbol} {timeframe} but save_data not available")
    
    async def fetch_all_data_for_symbols(self, symbols: List[str], timeframe: str) -> Dict[str, pd.DataFrame]:
        """
        Optimized multi-symbol data fetch with concurrency
        
        This method efficiently fetches data for multiple symbols using
        asynchronous processing for improved performance.
        
        Args:
            symbols: List of trading pair symbols
            timeframe: Timeframe for candles
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their data
        """
        if not self._is_initialized:
            await self.initialize()
            
        # Define fetch task for a single symbol
        async def fetch_symbol_data(symbol):
            try:
                return symbol, await self.fetch_realtime(symbol, timeframe)
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {symbol}: {e}")
                return symbol, pd.DataFrame()
        
        # Create tasks for each symbol
        tasks = [fetch_symbol_data(symbol) for symbol in symbols]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        data_map = {}
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Task error during multi-symbol fetch: {result}")
                continue
                
            symbol, df = result
            if not df.empty:
                data_map[symbol] = df
            else:
                self.logger.warning(f"No data for {symbol}")
        
        return data_map
    
    async def _shutdown_source_specific(self) -> None:
        """Close all data sources and clean up"""
        tasks = []
        
        if self.local_source:
            tasks.append(self.local_source.shutdown())
            
        if self.exchange_source:
            tasks.append(self.exchange_source.shutdown())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
        self.logger.info("HybridDataSource closed")