#!/usr/bin/env python3
# src/strategy/implementations/__init__.py

from src.strategy.implementations.dual_ma import DualMAStrategy
from src.strategy.implementations.multi_factors import MultiFactorsStrategy
from src.strategy.implementations.neural_network import NeuralNetworkStrategy

__all__ = [
    'DualMAStrategy',
    'MultiFactorsStrategy',
    'NeuralNetworkStrategy'
]