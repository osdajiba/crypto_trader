#!/usr/bin/env python3
# src/strategy/implementations/neural_network.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import os
import asyncio
from pathlib import Path

from src.common.abstract_factory import register_factory_class
from src.common.config_manager import ConfigManager
from src.strategy.base import BaseStrategy
from src.strategy.factors.factory import get_factor_factory
from src.strategy.factors.base import BaseFactor


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
    Neural Network Based Prediction Strategy
    
    Uses a pre-trained neural network model to predict future price movements
    and generate trading signals based on those predictions.
    """
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Neural Network Strategy
        
        Args:
            config: System configuration manager
            params: Strategy-specific parameters (optional)
        """
        # Set default parameters if not provided
        params = params or {}
        
        # Get default values from config
        nn_config = config.get("strategy", "neural_network", default={})
        
        # Set parameters with defaults
        params.setdefault("model_path", nn_config.get("model_path", "models/price_predictor.pkl"))
        params.setdefault("window_size", nn_config.get("window_size", 30))
        params.setdefault("threshold", nn_config.get("threshold", 0.6))
        params.setdefault("risk_factor", nn_config.get("risk_factor", 0.02))
        
        # Initialize base class
        super().__init__(config, params)
        
        # Store parameters as instance variables
        self.model_path = self.params["model_path"]
        self.window_size = self.params["window_size"]
        self.threshold = self.params["threshold"]
        self.risk_factor = self.params["risk_factor"]
        
        # Model components
        self.model = None
        self.scaler = None
        self.feature_columns = []
        
        # Factor instances and cache
        self.factor_factory = None
        self.factors = {}
        
        # Prediction tracking
        self.prediction_stats = {
            'total': 0,
            'correct': 0,
            'last_prediction': None,
            'last_price': None
        }
    
    def _determine_required_history(self) -> int:
        """
        Determine required historical data points for strategy
        
        Returns:
            int: Number of required data points
        """
        # Need enough data for feature calculation plus window size
        # Technical indicators like MACD need more history
        return max(100, self.window_size * 2)
    
    async def _initialize_strategy(self) -> None:
        """Strategy-specific initialization"""
        # Initialize factor factory
        self.factor_factory = get_factor_factory(self.config)
        
        # Initialize feature definitions
        self._define_features()
        
        # Load the model
        try:
            await self._load_model()
            if self.model:
                self.logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                self.logger.warning(f"No model loaded, will use fallback prediction logic")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.model = None
        
        self.logger.info(
            f"Initialized NeuralNetworkStrategy with window_size={self.window_size}, "
            f"threshold={self.threshold}, risk_factor={self.risk_factor}"
        )
    
    def _define_features(self) -> None:
        """Define features used by the model"""
        # Define core feature columns
        self.feature_columns = [
            # Price and volume
            'close', 'open', 'high', 'low', 'volume',
            
            # Moving averages
            'sma_5', 'sma_10', 'sma_20', 
            'ema_5', 'ema_10',
            
            # Volume indicators
            'volume_sma',
            
            # Momentum indicators
            'rsi', 'macd_hist', 'macd_value', 'macd_signal',
            
            # Volatility indicators
            'bbands_width', 'bbands_percent_b', 'atr_value',
            
            # Custom indicators
            'volatility', 'price_change_1', 'price_change_5'
        ]
    
    async def _load_model(self) -> None:
        """
        Load the neural network model from file
        
        In a real implementation, this would use an actual ML framework
        like TensorFlow, PyTorch, scikit-learn, etc.
        """
        if not self.model_path or not os.path.exists(self.model_path):
            self.logger.warning(f"Model file not found: {self.model_path}")
            self.model = None
            return
            
        try:
            # Placeholder for model loading
            # In a real implementation:
            # import joblib
            # self.model = joblib.load(self.model_path)
            # or
            # import tensorflow as tf
            # self.model = tf.keras.models.load_model(self.model_path)
            
            # Mock model for example purpose
            self.model = "mock_model"
            self.logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.model = None
    
    async def _generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on neural network predictions
        
        Args:
            data: Market data with sufficient history
            
        Returns:
            pd.DataFrame: Trading signals
        """
        if data.empty:
            return pd.DataFrame()
        
        # Get symbol from data
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else self.params.get("symbol", "BTC/USDT")
        
        try:
            # Calculate features
            features = self._prepare_features(data)
            
            if features.empty:
                self.logger.warning(f"Unable to prepare features for {symbol}")
                return pd.DataFrame()
            
            # Make prediction
            direction, confidence = self._predict_price_movement(features)
            
            # Check if confidence exceeds threshold
            if confidence < self.threshold:
                return pd.DataFrame()  # No signal if confidence is too low
            
            # Determine action based on predicted direction
            action = 'buy' if direction > 0 else 'sell' if direction < 0 else None
            
            if not action:
                return pd.DataFrame()  # No action if direction is neutral
            
            # Generate signal
            current_price = data['close'].iloc[-1]
            quantity = self._calculate_position_size(float(current_price), symbol, confidence)
            
            # Get timestamp
            timestamp = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else (
                data['datetime'].iloc[-1] if 'datetime' in data.columns else 
                data['timestamp'].iloc[-1] if 'timestamp' in data.columns else 
                pd.Timestamp.now()
            )
            
            signal_data = {
                'timestamp': timestamp,
                'symbol': symbol,
                'action': action,
                'price': float(current_price),
                'quantity': quantity,
                'confidence': float(confidence),
                'reason': f"Neural network prediction: {direction:.2f} with {confidence:.2f} confidence"
            }
            
            self.logger.info(
                f"Generated {action} signal for {symbol} @ ${float(current_price):.2f}, "
                f"quantity: {quantity:.6f}, confidence: {confidence:.2f}"
            )
            
            # Store prediction for later validation
            self.prediction_stats['last_prediction'] = direction
            self.prediction_stats['last_price'] = float(current_price)
            
            return pd.DataFrame([signal_data])
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return pd.DataFrame()
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for the neural network model
        
        Args:
            data: Market data
            
        Returns:
            pd.DataFrame: Prepared features
        """
        if data.empty:
            return pd.DataFrame()
            
        # Create features DataFrame
        features = data.copy()
        
        try:
            # Calculate technical indicators
            if not self.factors:
                self._initialize_factors(data)
            
            # Calculate each factor
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
                        elif name == 'bollinger':
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
            
            # Calculate additional features
            features['volatility'] = self._calculate_volatility(data)
            features['price_change_1'] = data['close'].pct_change(1)
            features['price_change_5'] = data['close'].pct_change(5)
            
            # Select only the feature columns we need
            available_columns = [col for col in self.feature_columns if col in features.columns]
            if available_columns:
                features = features[available_columns]
                
                # Drop rows with NaN values
                features = features.dropna()
                
                # Use only the last window_size rows
                if len(features) > self.window_size:
                    features = features.tail(self.window_size)
                
                return features
            else:
                self.logger.warning(f"No valid features available. Columns found: {list(features.columns)}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def _initialize_factors(self, data: pd.DataFrame) -> None:
        """
        Initialize technical factors
        
        Args:
            data: Sample market data for initialization
        """
        try:
            # Create factor instances
            self.factors['sma_5'] = self._create_factor('sma', {'period': 5, 'price_col': 'close'})
            self.factors['sma_10'] = self._create_factor('sma', {'period': 10, 'price_col': 'close'})
            self.factors['sma_20'] = self._create_factor('sma', {'period': 20, 'price_col': 'close'})
            self.factors['ema_5'] = self._create_factor('ema', {'period': 5, 'price_col': 'close'})
            self.factors['ema_10'] = self._create_factor('ema', {'period': 10, 'price_col': 'close'})
            self.factors['volume_sma'] = self._create_factor('sma', {'period': 5, 'price_col': 'volume'})
            self.factors['rsi'] = self._create_factor('rsi', {'period': 14, 'price_col': 'close'})
            self.factors['macd'] = self._create_factor('macd', {
                'fast_period': 12, 
                'slow_period': 26, 
                'signal_period': 9, 
                'price_col': 'close'
            })
            self.factors['bollinger'] = self._create_factor('bollinger', {
                'period': 20, 
                'std_dev': 2.0, 
                'price_col': 'close'
            })
            self.factors['atr'] = self._create_factor('atr', {'period': 14})
            
            self.logger.debug(f"Initialized {len(self.factors)} factors")
            
        except Exception as e:
            self.logger.error(f"Error initializing factors: {e}")
    
    def _create_factor(self, name: str, params: Dict[str, Any]) -> Optional[BaseFactor]:
        """
        Create a factor instance
        
        Args:
            name: Factor name
            params: Factor parameters
            
        Returns:
            Optional[BaseFactor]: Factor instance or None if creation fails
        """
        try:
            # In a real implementation, this would be async:
            # factor = await self.factor_factory.create(name, params)
            factor = self.factor_factory.get_factor_info(name)
            if not factor:
                self.logger.warning(f"Factor not available: {name}")
                return None
                
            # Mock factor creation for example purposes
            class MockFactor:
                def calculate(self, data):
                    if name == 'sma' or name == 'ema':
                        col = params.get('price_col', 'close')
                        period = params.get('period', 20)
                        if col in data.columns:
                            return data[col].rolling(window=period).mean()
                    elif name == 'rsi':
                        return pd.Series(np.random.rand(len(data)) * 100, index=data.index)
                    elif name == 'macd':
                        result = pd.DataFrame(index=data.index)
                        result['macd'] = np.random.randn(len(data)) * 0.5
                        result['signal'] = np.random.randn(len(data)) * 0.5
                        result['histogram'] = result['macd'] - result['signal']
                        return result
                    elif name == 'bollinger':
                        result = pd.DataFrame(index=data.index)
                        result['width'] = np.random.rand(len(data)) * 0.1
                        result['percent_b'] = np.random.rand(len(data))
                        return result
                    elif name == 'atr':
                        return pd.Series(np.random.rand(len(data)) * 5, index=data.index)
                    return pd.Series(index=data.index)
            
            return MockFactor()
            
        except Exception as e:
            self.logger.error(f"Error creating factor {name}: {e}")
            return None
    
    def _calculate_volatility(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate price volatility
        
        Args:
            data: Market data
            
        Returns:
            pd.Series: Volatility values
        """
        if 'close' not in data.columns:
            return pd.Series(index=data.index)
            
        # Calculate rolling standard deviation normalized by price
        return data['close'].rolling(window=20).std() / data['close']
    
    def _predict_price_movement(self, features: pd.DataFrame) -> Tuple[float, float]:
        """
        Predict future price movement using the neural network model
        
        Args:
            features: Prepared features
            
        Returns:
            Tuple[float, float]: (direction, confidence)
            direction: positive for up, negative for down, zero for neutral
            confidence: prediction confidence (0.0 to 1.0)
        """
        if features.empty or not self.model:
            return 0.0, 0.0
            
        try:
            # Normalize features
            normalized_features = self._normalize_features(features)
            
            # In a real implementation, we would use the model:
            # prediction = self.model.predict(normalized_features)
            
            # Mock prediction for example purposes
            # Base prediction on recent price trends and add noise
            if 'close' in features.columns and len(features) >= 2:
                recent_change = (features['close'].iloc[-1] / features['close'].iloc[0]) - 1
                
                # Add noise to simulate neural network behavior
                noise = np.random.normal(0, 0.3)
                direction = np.sign(recent_change + noise)
                
                # Calculate confidence based on trend strength and consistency
                rsi_trend = 0
                if 'rsi' in features.columns:
                    rsi = features['rsi'].iloc[-1]
                    rsi_trend = 1 if rsi > 70 else -1 if rsi < 30 else 0
                
                macd_trend = 0
                if 'macd_hist' in features.columns:
                    macd_hist = features['macd_hist'].iloc[-1]
                    macd_trend = 1 if macd_hist > 0 else -1 if macd_hist < 0 else 0
                
                # Higher confidence when indicators agree
                base_confidence = 0.5
                if (rsi_trend * direction > 0) and (macd_trend * direction > 0):
                    confidence_boost = 0.3
                elif (rsi_trend * direction > 0) or (macd_trend * direction > 0):
                    confidence_boost = 0.15
                else:
                    confidence_boost = 0
                
                # Add magnitude component
                magnitude_component = min(abs(recent_change) * 5, 0.2)
                
                confidence = min(base_confidence + confidence_boost + magnitude_component, 0.95)
                
                # Update stats
                self.prediction_stats['total'] += 1
                
                return float(direction), float(confidence)
            
            return 0.0, 0.5
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return 0.0, 0.0
    
    def _normalize_features(self, features: pd.DataFrame) -> np.ndarray:
        """
        Normalize features for model input
        
        Args:
            features: Feature DataFrame
            
        Returns:
            np.ndarray: Normalized features
        """
        if features.empty:
            return np.array([])
            
        # Simple Z-score normalization (mean=0, std=1)
        # In a real implementation, we would use the scaler from training
        normalized = pd.DataFrame(index=features.index)
        
        for col in features.columns:
            if features[col].std() > 0:
                normalized[col] = (features[col] - features[col].mean()) / features[col].std()
            else:
                normalized[col] = 0
        
        return normalized.values
    
    def _calculate_position_size(self, price: float, symbol: str, confidence: float) -> float:
        """
        Calculate position size based on risk management rules and prediction confidence
        
        Args:
            price: Current asset price
            symbol: Trading symbol
            confidence: Prediction confidence
            
        Returns:
            float: Position size (quantity)
        """
        # Get available capital
        capital = self.config.get("trading", "capital", "initial", default=100000)
        
        # Scale position size by confidence and risk factor
        position_value = capital * self.risk_factor * confidence
        
        # Convert to quantity
        quantity = position_value / price if price > 0 else 0
        
        # Apply minimum order size
        min_order = self.config.get("trading", "capital", "min_order", default=0.001)
        if quantity < min_order:
            quantity = min_order
        
        return quantity
    
    def validate_prediction(self, current_price: float) -> None:
        """
        Validate previous prediction against actual price movement
        
        Args:
            current_price: Current price for validation
        """
        if (self.prediction_stats['last_prediction'] is not None and 
            self.prediction_stats['last_price'] is not None):
            
            # Calculate actual price change
            price_change = current_price - self.prediction_stats['last_price']
            direction = np.sign(price_change)
            
            # Check if prediction was correct
            predicted_direction = self.prediction_stats['last_prediction']
            if (predicted_direction > 0 and direction > 0) or (predicted_direction < 0 and direction < 0):
                self.prediction_stats['correct'] += 1
                
            # Log accuracy periodically
            if self.prediction_stats['total'] % 10 == 0:
                accuracy = (self.prediction_stats['correct'] / self.prediction_stats['total'] 
                          if self.prediction_stats['total'] > 0 else 0)
                self.logger.info(
                    f"Prediction accuracy: {accuracy:.2f} "
                    f"({self.prediction_stats['correct']}/{self.prediction_stats['total']})"
                )
            
            # Reset last prediction
            self.prediction_stats['last_prediction'] = None
            self.prediction_stats['last_price'] = None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get strategy status and configuration
        
        Returns:
            Dict with strategy status
        """
        accuracy = (self.prediction_stats['correct'] / self.prediction_stats['total'] 
                  if self.prediction_stats['total'] > 0 else 0)
                  
        return {
            'name': 'neural_network',
            'description': 'Neural Network based prediction strategy',
            'parameters': {
                'window_size': self.window_size,
                'threshold': self.threshold,
                'risk_factor': self.risk_factor,
                'model_path': self.model_path
            },
            'model_loaded': self.model is not None,
            'performance': self.get_performance_stats(),
            'prediction_stats': {
                'total_predictions': self.prediction_stats['total'],
                'correct_predictions': self.prediction_stats['correct'],
                'accuracy': accuracy
            },
            'features': self.feature_columns,
            'data_buffers': {
                symbol: len(df) for symbol, df in self._data_buffer.items()
            }
        }
    
    async def shutdown(self) -> None:
        """Clean up resources"""
        # Log prediction stats
        if self.prediction_stats['total'] > 0:
            accuracy = self.prediction_stats['correct'] / self.prediction_stats['total']
            self.logger.info(
                f"Final prediction accuracy: {accuracy:.2f} "
                f"({self.prediction_stats['correct']}/{self.prediction_stats['total']})"
            )
        
        # Clean up factors
        self.factors.clear()
        
        # Call base class shutdown
        await super().shutdown()