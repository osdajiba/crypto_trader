# src/strategy/nn.py
import pandas as pd
from typing import Dict, Optional, Any
from src.strategy.base_strategy import BaseStrategy
from src.common.config_manager import ConfigManager
from src.common.data_processor import DataProcessor

class NeuralNetStrategy(BaseStrategy):
    """Multi-factor ranking strategy with historical preloading and real-time data splicing."""

    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the NeuralNetStrategy.

        Args:
            config (ConfigManager): Configuration manager instance.
            params (Optional[Dict[str, Any]]): Strategy-specific parameters.
                Expected keys: select_coin_num, leverage, period, c_rate, factor_class_dict, 
                              multi_symbol_data, lookback_period.
        """
        super().__init__(config, params)
        self.select_coin_num = self.params.get("select_coin_num", 1)
        self.leverage = self.params.get("leverage", 1)
        self.c_rate = self.params.get("c_rate", 2.5 / 10000)
        self.factor_class_dict = self.params.get("factor_class_dict", {'Bias_6': False, 'Psy_60': True})

    async def initialize(self) -> None:
        """Initialize resources and validate parameters."""
        await super().initialize()  # Calls BaseStrategy.initialize, which includes preloading
        if not self.factor_class_dict:
            self.logger.error("No factors defined in factor_class_dict")
            raise ValueError("factor_class_dict must contain at least one factor")
        self.logger.info("Initialized NeuralNetStrategy with factors: %s", list(self.factor_class_dict.keys()))

    async def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using spliced historical and real-time data.

        Args:
            data (pd.DataFrame): Real-time OHLCV data for a single symbol.

        Returns:
            pd.DataFrame: Signals with 'timestamp', 'symbol', 'action' (buy/sell).
        """
        self.logger.info("Generating signals for NeuralNetStrategy")
        await self._execute_hook("pre_signal_generation", data=data)

        real_time_data = DataProcessor.clean_ohlcv(data)
        real_time_data = DataProcessor.resample(real_time_data, self.period)
        if not await self.validate_data(real_time_data):  # Uses BaseStrategy.validate_data
            return pd.DataFrame()

        primary_symbol = self.params.get("symbol", "unknown")
        all_data = DataProcessor.splice_data(
            self.historical_data, real_time_data, primary_symbol, 
            self.lookback_period, self.multi_symbol_data
        )
        if all_data.empty:
            self.logger.warning("No valid data after splicing")
            return pd.DataFrame()

        all_data = all_data[['time', '_symbol', 'close'] + list(self.factor_class_dict.keys())]
        signals = await self._calculate_signals(all_data)
        signals = self.filter_signals(signals)

        DataProcessor.update_historical_data(self.historical_data, real_time_data, primary_symbol, self.lookback_period)
        self.logger.debug("Updated historical data for %s with %d new rows", primary_symbol, len(real_time_data))

        await self._execute_hook("post_signal_generation", signals=signals)
        return signals

    async def _calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate signals based on factor rankings."""
        if not self.factor_class_dict:
            self.logger.warning("No factors provided, returning empty signals")
            return pd.DataFrame()

        df = df.copy()
        factor_rank_cols = []

        for factor, ascending in self.factor_class_dict.items():
            if factor not in df.columns:
                self.logger.debug("Calculating factor %s as it's missing from data", factor)
                df[factor] = await self._calculate_factor(df, factor)
            
            rank_col = f'{factor}_rank'
            df[rank_col] = df.groupby('time')[factor].rank(method='first', ascending=ascending)
            factor_rank_cols.append(rank_col)

        if not factor_rank_cols:
            self.logger.warning("No factor rankings calculated, returning empty signals")
            return pd.DataFrame()
        
        df['combined_score'] = df[factor_rank_cols].mean(axis=1, skipna=False)
        self.logger.debug("Calculated combined score using %d factors", len(factor_rank_cols))

        df['long_rank'] = df.groupby('time')['combined_score'].rank(method='first')
        df_long = df[df['long_rank'] <= self.select_coin_num].copy()
        df_long['direction'] = 1

        df['short_rank'] = df.groupby('time')['combined_score'].rank(method='first', ascending=False)
        df_short = df[df['short_rank'] <= self.select_coin_num].copy()
        df_short['direction'] = -1

        signals = pd.concat([df_long, df_short], ignore_index=True)
        signals['action'] = signals['direction'].map({1: 'buy', -1: 'sell'})
        signals = signals[['time', '_symbol', 'action']].rename(columns={'time': 'timestamp', '_symbol': 'symbol'})
        
        self.logger.debug("Generated %d signals (%d buy, %d sell)", 
                         len(signals), len(df_long), len(df_short))
        return signals

    async def _calculate_factor(self, df: pd.DataFrame, factor: str) -> pd.Series:
        """Calculate a factor if not provided in data."""
        close = df['close']
        if factor.startswith('Bias_'):
            n = int(factor.split('_')[1])
            return (close - close.rolling(n).mean()) / close.rolling(n).mean()
        elif factor.startswith('Psy_'):
            n = int(factor.split('_')[1])
            return (close > close.shift(1)).rolling(n).mean()
        self.logger.warning("Factor %s not implemented, returning zero", factor)
        return pd.Series(0, index=df.index)
