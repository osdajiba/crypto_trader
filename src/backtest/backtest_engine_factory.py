# src/backtest/backtest_engine_factory.py

from typing import Type, Dict, Any

from src.backtest.base_backtest_engine import BacktestEngine
from src.backtest.ohlcv_backtest_engine import OHLCVBacktestEngine
from src.backtest.market_replay_backtest_engine import MarketReplayBacktestEngine


class BacktestEngineFactory:
    """
    Factory for creating different types of backtest engines
    """
    
    ENGINE_TYPES = {
        # Basic type names
        'ohlcv': OHLCVBacktestEngine,
        'market_replay': MarketReplayBacktestEngine,
        'replay': MarketReplayBacktestEngine,
        
        # Aliases for better discoverability
        'basic': OHLCVBacktestEngine,
        'simple': OHLCVBacktestEngine,
        'standard': OHLCVBacktestEngine,
        'advanced': MarketReplayBacktestEngine,
        'realistic': MarketReplayBacktestEngine,
        'high_fidelity': MarketReplayBacktestEngine,
    }
    
    @classmethod
    def get_available_engines(cls) -> Dict[str, str]:
        """
        Get a mapping of available engine types with descriptions
        
        Returns:
            Dict[str, str]: Dictionary of engine types and descriptions
        """
        return {
            'ohlcv': 'Standard OHLCV engine for basic backtesting',
            'market_replay': 'High fidelity market replay engine with realistic order execution',
            'basic': 'Alias for OHLCV engine',
            'advanced': 'Alias for market replay engine'
        }
        
    @staticmethod
    def create_engine(engine_type: str, config, strategy, data_manager) -> BacktestEngine:
        """
        Create a backtest engine instance based on the specified type
        
        Args:
            engine_type: Type of backtest engine ('ohlcv', 'market_replay')
            config: Configuration manager
            strategy: Trading strategy
            data_manager: Data manager
            
        Returns:
            BacktestEngine: Configured backtest engine instance
            
        Raises:
            ValueError: If an unsupported engine type is requested
        """
        engine_type = engine_type.lower()
        
        # Get engine class based on type
        engine_class = BacktestEngineFactory.get_engine_class(engine_type)
        
        # Create and return the engine instance
        return engine_class(config, strategy, data_manager)
    
    @classmethod
    def get_engine_class(cls, engine_type: str) -> Type[BacktestEngine]:
        """
        Get the class for a backtest engine type
        
        Args:
            engine_type: Type of backtest engine ('ohlcv', 'market_replay', etc.)
            
        Returns:
            Type[BacktestEngine]: Backtest engine class
            
        Raises:
            ValueError: If an unsupported engine type is requested
        """
        engine_type = engine_type.lower()
        
        # Check if the requested engine type is supported
        if engine_type in cls.ENGINE_TYPES:
            return cls.ENGINE_TYPES[engine_type]
        
        # If not found, raise an error with available options
        available_types = ", ".join(cls.ENGINE_TYPES.keys())
        raise ValueError(f"Unsupported backtest engine type: {engine_type}. Available types: {available_types}")
    
    @classmethod
    def get_engine_type_description(cls, engine_type: str) -> str:
        """
        Get a description of the engine type
        
        Args:
            engine_type: Type of backtest engine
            
        Returns:
            str: Description of the engine
        """
        engine_type = engine_type.lower()
        
        descriptions = {
            'ohlcv': 'Standard OHLCV price data engine with basic execution model',
            'market_replay': 'High fidelity market replay engine with realistic order execution',
            'basic': 'Standard OHLCV price data engine (alias)',
            'advanced': 'High fidelity market replay engine (alias)',
        }
        
        return descriptions.get(engine_type, "Unknown engine type")