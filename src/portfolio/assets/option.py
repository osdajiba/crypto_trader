#!/usr/bin/env python3
# src/portfolio/assets/option.py

from datetime import datetime
from math import exp, log, sqrt
from scipy.stats import norm
import numpy as np

from src.portfolio.assets.base import Asset
from src.common.abstract_factory import register_factory_class


@register_factory_class('asset_factory', 'option')
class Option(Asset):
    """Option contract asset"""
    
    PRICING_METHODS = ['Black-Scholes', 'Monte-Carlo', 'Binomial-Tree', 'BAW', 'Heston']
    
    def __init__(self, config, params):
        name = params.get('contract_code', '')
        super().__init__(name)
        
        # Required parameters
        self.option_type = params.get('option_type', 'call')  # call or put
        self.option_price = params.get('option_price', 0.0)
        self.underlying_price = params.get('underlying_price', 0.0)
        self.strike_price = params.get('strike_price', 0.0)
        self.risk_free_rate = params.get('risk_free_rate', 0.0)
        
        # Optional parameters
        self.operation = params.get('operation', 'buy')  # buy or sell
        self.volatility = params.get('volatility', 0.0)
        self.implied_volatility = params.get('implied_volatility', None)
        self.end_date = params.get('end_date', None)
        self.tau = params.get('tau', None)  # Time to expiration in years
        self.pricing_method = params.get('pricing_method', 'Black-Scholes')
        
        # Calculate time to maturity if not provided
        if self.tau is None:
            time_to_maturity = params.get('time_to_maturity', None)
            if time_to_maturity:
                self.tau = time_to_maturity / 365.0
            elif self.end_date:
                self.tau = (self.end_date - datetime.now()).days / 365.0
            else:
                self.tau = 0.0

    def get_value(self) -> float:
        """Calculate the current value of the option"""
        return self.calculate_price()

    def buy(self, amount: float):
        """Buy options (implementation depends on business logic)"""
        # This would be implemented with actual trading logic
        pass

    def sell(self, amount: float):
        """Sell options (implementation depends on business logic)"""
        # This would be implemented with actual trading logic
        pass
        
    def calculate_price(self) -> float:
        """Calculate option price using the specified pricing method"""
        if self.pricing_method == "Black-Scholes":
            return self._calculate_black_scholes_price()
        elif self.pricing_method == "Monte-Carlo":
            return self._calculate_monte_carlo_price()
        else:
            # Fallback to Black-Scholes for unsupported methods
            return self._calculate_black_scholes_price()
            
    def _calculate_black_scholes_price(self) -> float:
        """Calculate option price using Black-Scholes model"""
        S = self.underlying_price
        K = self.strike_price
        T = self.tau
        r = self.risk_free_rate
        sigma = self.implied_volatility if self.implied_volatility else self.volatility

        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        if self.option_type == "call":
            if self.operation == "buy":
                return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
            else:  # sell
                return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:  # put
            if self.operation == "buy":
                return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            else:  # sell
                return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
                
    def _calculate_monte_carlo_price(self) -> float:
        """Calculate option price using Monte Carlo simulation"""
        np.random.seed(42)  # For reproducibility
        S = self.underlying_price
        K = self.strike_price
        T = self.tau
        r = self.risk_free_rate
        sigma = self.volatility
        n = 10000  # Number of simulations

        z = np.random.normal(0, 1, n)
        ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * sqrt(T) * z)

        if self.option_type == "call":
            payoff = np.maximum(ST - K, 0)
        else:  # put
            payoff = np.maximum(K - ST, 0)

        return np.exp(-r * T) * np.mean(payoff)
        
    def calculate_greeks(self):
        """Calculate option greeks (delta, gamma, vega, theta, rho)"""
        S = self.underlying_price
        K = self.strike_price
        T = self.tau
        r = self.risk_free_rate
        sigma = self.volatility

        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))

        delta = norm.cdf(d1) if self.option_type == "call" else norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
        vega = S * norm.pdf(d1) * sqrt(T)
        theta = -S * norm.pdf(d1) * sigma / (2 * sqrt(T)) - r * K * exp(-r * T) * norm.cdf(
            d1 if self.option_type == "call" else -d1)
        rho = K * T * exp(-r * T) * norm.cdf(d1 if self.option_type == "call" else -d1)

        return delta, gamma, vega, theta, rho