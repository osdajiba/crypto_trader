#!/usr/bin/env python3
# src/strategy/implementations/neural_network.py

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List, Tuple
import asyncio
import time
import os
from pathlib import Path

from src.common.abstract_factory import register_factory_class
from src.common.config import ConfigManager
from src.strategy.base import BaseStrategy
from src.strategy.factors import get_factor_registry
from src.strategy.factors.trend import SMA, EMA
from src.strategy.factors.momentum import RSI, MACD
from src.strategy.factors.volatility import BollingerBands, ATR
from src.strategy.factors.custom import PriceChangeRate, VolatilityIndex


@register_factory_class('strategy_factory', 'neural_network', 
                      description="Neural Network based prediction strategy",
                      category="ml",
                      features=["machine_learning", "prediction", "advanced"],
                      parameters=[
                          {"name": "model_path", "type": "str", "default": "", "description": "Path to saved model file"},
                          {"name": "window_size", "type": "int", "default": 30, "description": "Input window size for prediction"},
                          {"name": "threshold", "type": "float", "default": 0.6, "description": "Prediction confidence threshold"},
                          {"name": "risk_factor", "type": "float", "default": 0.02, "description": "Risk factor for position sizing"}
                      ])
class NeuralNetworkStrategy(BaseStrategy):
    """
    Neural Network based prediction strategy.
    
    Uses a pre-trained neural network model to predict future price movements
    and generate trading signals based on those predictions.
    """

    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Neural Network strategy
        
        Args:
            config: Configuration manager instance
            params: Strategy-specific parameters
        """
        # Set default parameters
        params = params or {}
        
        # Get default values from config
        nn_config = config.get("strategy", "neural_network", default={})
        
        # Set parameters with defaults
        params.setdefault("model_path", nn_config.get("model_path", "models/price_predictor.pkl"))
        params.setdefault("window_size", nn_config.get("window_size", 30))
        params.setdefault("threshold", nn_config.get("threshold", 0.6))
        params.setdefault("risk_factor", nn_config.get("risk_factor", 0.02))
        params.setdefault("lookback_period", max(100, 2 * params.get("window_size", 30)))
        
        # Initialize parent class
        super().__init__(config, params)
        
        # Extract parameters
        self.model_path = self.params["model_path"]
        self.window_size = self.params["window_size"]
        self.threshold = self.params["threshold"]
        self.risk_factor = self.params["risk_factor"]
        
        # Initialize model
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.prediction_horizon = 5  # Predict 5 time periods ahead by default
        
        # Initialize factors
        self.factor_registry = get_factor_registry()
        self.factors = {}
        
        # Tracking for performance
        self.prediction_accuracy = {
            'total': 0,
            'correct': 0
        }

    def _init_factors(self) -> None:
        """Initialize technical indicators used as model features"""
        # Create price-based factors
        self.factors['sma_5'] = self.factor_registry.create_factor('sma', period=5, price_col='close')
        self.factors['sma_10'] = self.factor_registry.create_factor('sma', period=10, price_col='close')
        self.factors['sma_20'] = self.factor_registry.create_factor('sma', period=20, price_col='close')
        self.factors['ema_5'] = self.factor_registry.create_factor('ema', period=5, price_col='close')
        self.factors['ema_10'] = self.factor_registry.create_factor('ema', period=10, price_col='close')
        
        # Create volume-based factors
        self.factors['volume_sma'] = self.factor_registry.create_factor('sma', period=5, price_col='volume')
        
        # Create advanced factors
        self.factors['rsi'] = self.factor_registry.create_factor('rsi', period=14, price_col='close')
        self.factors['macd'] = self.factor_registry.create_factor('macd', fast_period=12, slow_period=26, signal_period=9, price_col='close')
        self.factors['bbands'] = self.factor_registry.create_factor('bollinger', period=20, std_dev=2.0, price_col='close')
        self.factors['atr'] = self.factor_registry.create_factor('atr', period=14)
        
        # Create custom factors
        self.factors['volatility'] = VolatilityIndex(period=20, price_col='close')
        self.factors['price_change_1'] = PriceChangeRate(period=1, price_col='close')
        self.factors['price_change_5'] = PriceChangeRate(period=5, price_col='close')
        
        # Define feature columns for prediction
        self.feature_columns = [
            'close', 'open', 'high', 'low', 'volume',
            'sma_5', 'sma_10', 'sma_20', 
            'ema_5', 'ema_10',
            'volume_sma',
            'rsi',
            'volatility',
            'price_change_1', 'price_change_5',
            # Add derived features from dataframes
            'macd_value', 'macd_signal', 'macd_hist',
            'bbands_width', 'bbands_percent_b',
            'atr_value'
        ]

    def _calculate_volatility(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate price volatility as rolling standard deviation / price"""
        if 'close' not in data.columns:
            return pd.Series(index=data.index)
        
        return data['close'].rolling(window=20).std() / data['close']
    
    def _calculate_pct_change_1(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate 1-period percentage change"""
        if 'close' not in data.columns:
            return pd.Series(index=data.index)
        
        return data['close'].pct_change(1)
    
    def _calculate_pct_change_5(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate 5-period percentage change"""
        if 'close' not in data.columns:
            return pd.Series(index=data.index)
        
        return data['close'].pct_change(5)

    async def initialize(self) -> None:
        """Initialize strategy and load model"""
        # Initialize parent class
        await super().initialize()
        
        # Load the model
        try:
            await self._load_model()
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            # Continue without model - will use fallback logic
            self.model = None
        
        self.logger.info(f"Initialized NeuralNetworkStrategy with window_size={self.window_size}")

    async def _load_model(self) -> None:
        """Load the pre-trained model from file"""
        try:
            # Use a placeholder for model loading since we don't have a specific ML framework
            # In a real implementation, this would use the appropriate framework 
            # (e.g., TensorFlow, PyTorch, scikit-learn)
            self.logger.info(f"Loading model from {self.model_path}")
            
            if os.path.exists(self.model_path):
                # Placeholder for model loading
                # self.model = joblib.load(self.model_path)
                # Mocking model loading for example purpose
                self.model = "model_loaded"
                self.logger.info(f"Model loaded successfully")
            else:
                self.logger.warning(f"Model file not found: {self.model_path}")
                self.model = None
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def _prepare_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Prepare feature data for prediction
        
        Args:
            data: Raw market data
            symbol: Trading symbol
            
        Returns:
            Feature DataFrame ready for model input
        """
        # Start with a copy of the data
        features = data.copy()
        
        # Calculate all factors and add to features
        for name, factor in self.factors.items():
            try:
                result = factor.calculate(data)
                
                # Handle different return types
                if isinstance(result, pd.DataFrame):
                    # For MACD
                    if name == 'macd':
                        if 'macd' in result.columns:
                            features['macd_value'] = result['macd']
                        if 'signal' in result.columns:
                            features['macd_signal'] = result['signal']
                        if 'histogram' in result.columns:
                            features['macd_hist'] = result['histogram']
                    # For Bollinger Bands
                    elif name == 'bbands':
                        if 'width' in result.columns:
                            features['bbands_width'] = result['width']
                        if 'percent_b' in result.columns:
                            features['bbands_percent_b'] = result['percent_b']
                else:
                    # For simple series results
                    if name == 'atr':
                        features['atr_value'] = result
                    else:
                        features[name] = result
            except Exception as e:
                self.logger.warning(f"Error calculating factor {name}: {e}")
        
        # Select only feature columns that exist and drop NaN values
        available_columns = [col for col in self.feature_columns if col in features.columns]
        if available_columns:
            features = features[available_columns].tail(self.window_size)
            return features.dropna()
        else:
            self.logger.warning(f"No valid feature columns found. Available: {list(features.columns)}")
            return pd.DataFrame()

    def _standardize_features(self, features: pd.DataFrame) -> np.ndarray:
        """
        Standardize features for model input
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Normalized numpy array
        """
        # Simple standardization (mean=0, std=1)
        # In a real implementation, this would use the same scaler as during training
        if features.empty:
            return np.array([])
            
        # Basic standardization logic
        normalized = features.copy()
        for col in normalized.columns:
            if normalized[col].std() > 0:
                normalized[col] = (normalized[col] - normalized[col].mean()) / normalized[col].std()
            else:
                normalized[col] = 0
                
        return normalized.to_numpy()

    def _predict_next_movement(self, features: pd.DataFrame) -> Tuple[float, float]:
        """
        Make prediction using the loaded model
        
        Args:
            features: Prepared and normalized features
            
        Returns:
            Tuple[float, float]: (direction, confidence)
            Direction: 1.0 for up, -1.0 for down, 0.0 for no change
            Confidence: 0.0-1.0 prediction confidence
        """
        if self.model is None or features.empty:
            return 0.0, 0.0
        
        try:
            # Prepare features for model
            normalized_features = self._standardize_features(features)
            if len(normalized_features) == 0:
                return 0.0, 0.0
            
            # In a real implementation, this would use the actual model to make predictions
            # Mocking prediction for example purposes
            # For a real implementation:
            # prediction = self.model.predict(normalized_features.reshape(1, self.window_size, -1))
            
            # Simple mock prediction based on recent price movement
            if 'close' in features.columns and len(features) >= 2:
                recent_change = (features['close'].iloc[-1] / features['close'].iloc[0]) - 1
                
                # Simulate directional prediction with random noise
                rand_component = np.random.random() * 0.4 - 0.2  # -0.2 to 0.2
                signal = np.sign(recent_change + rand_component)
                confidence = min(0.5 + abs(recent_change) * 5, 0.95)  # Scale confidence with magnitude
                
                return float(signal), float(confidence)
            else:
                return 0.0, 0.0
                
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return 0.0, 0.0

    async def _generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on neural network predictions
        
        Args:
            data: Market data for signal generation
            
        Returns:
            pd.DataFrame: Trading signals
        """
        if data.empty:
            return pd.DataFrame()
        
        # Get symbol
        symbol = "unknown"
        if 'symbol' in data.columns and not data.empty:
            symbol = data['symbol'].iloc[0]
        
        try:
            # Prepare features for the model
            features = self._prepare_features(data, symbol)
            if features.empty:
                self.logger.warning(f"Unable to prepare features for {symbol}")
                return pd.DataFrame()
            
            # Make prediction
            direction, confidence = self._predict_next_movement(features)
            
            # Generate signal if confidence exceeds threshold
            signals = pd.DataFrame()
            current_price = data['close'].iloc[-1] if 'close' in data.columns and not data.empty else 0
            
            if confidence >= self.threshold:
                # Determine action from prediction direction
                action = 'buy' if direction > 0 else 'sell' if direction < 0 else None
                
                if action and current_price > 0:
                    # Calculate position size based on confidence and risk
                    capital = self.config.get("trading", "capital", "initial", default=100000)
                    position_size = self.risk_factor * confidence
                    quantity = (capital * position_size) / current_price
                    
                    # Get timestamp
                    timestamp = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else (
                        data['datetime'].iloc[-1] if 'datetime' in data.columns else pd.Timestamp.now()
                    )
                    
                    signal_data = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'action': action,
                        'price': float(current_price),
                        'quantity': float(quantity),
                        'confidence': float(confidence),
                        'reason': f"Neural network prediction: {direction:.2f} with {confidence:.2f} confidence"
                    }
                    
                    signals = pd.DataFrame([signal_data])
                    
                    self.logger.info(
                        f"Generated {action} signal for {symbol} @ ${float(current_price):.2f}, "
                        f"quantity: {quantity:.6f}, confidence: {confidence:.2f}"
                    )
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return pd.DataFrame()
            
    def _verify_prediction_accuracy(self, prediction: float, actual_change: float) -> None:
        """
        Verify prediction accuracy for model performance tracking
        
        Args:
            prediction: Predicted direction (1.0 up, -1.0 down)
            actual_change: Actual price change percentage
        """
        self.prediction_accuracy['total'] += 1
        
        # Prediction is correct if signs match (both positive or both negative)
        if (prediction > 0 and actual_change > 0) or (prediction < 0 and actual_change < 0):
            self.prediction_accuracy['correct'] += 1
            
        # Log accuracy periodically
        if self.prediction_accuracy['total'] % 10 == 0:
            accuracy = self.prediction_accuracy['correct'] / self.prediction_accuracy['total'] if self.prediction_accuracy['total'] > 0 else 0
            self.logger.info(f"Prediction accuracy: {accuracy:.2f} ({self.prediction_accuracy['correct']}/{self.prediction_accuracy['total']})")
            
    async def shutdown(self) -> None:
        """Clean up resources and log performance stats"""
        # Log final prediction accuracy
        if self.prediction_accuracy['total'] > 0:
            accuracy = self.prediction_accuracy['correct'] / self.prediction_accuracy['total']
            self.logger.info(f"Final prediction accuracy: {accuracy:.2f} ({self.prediction_accuracy['correct']}/{self.prediction_accuracy['total']})")
        
        # Call parent shutdown
        await super().shutdown()