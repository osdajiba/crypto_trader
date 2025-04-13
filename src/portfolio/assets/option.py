# src\portfolio\option.py

from scipy.optimize import newton
from datetime import datetime
from math import exp, log, sqrt
from scipy.stats import norm
import numpy as np
from .base import Asset

class Option:
    def __init__(self, contract_code, option_type, option_price, underlying_price, strike_price, risk_free_rate,
                 operation="buy", volatility=None, implied_volatility=None, end_date=None, tau=None,
                 Time_to_maturity=None, pricing_method="Black-Scholes"):
        """A class representing an option contract.

        Attributes:
            option_price (float): The price of the option.
            option_type (str): The type of option, either 'call' or 'put'.
            underlying_price (float): The price of the underlying asset.
            end_date (datetime): The expiration date of the option.
            strike_price (float): The strike price of the option.
            risk_free_rate (float): The risk-free interest rate.
            operation (str): The operation type, either 'buy' or 'sell'.
            volatility (float): The volatility of the underlying asset.
            implied_volatility (float): The implied volatility of the option.
            tau (float): Time to expiration in years.
            pricing_method (str): The pricing method for the option.

        """
        self.contract_code = contract_code
        self.type = option_type
        self.option_price = option_price
        self.underlying_price = underlying_price
        self.end_date = end_date
        self.strike_price = strike_price
        self.risk_free_rate = risk_free_rate

        self.operation = operation
        if volatility is None:
            self.volatility = np.var(option_price)
        else:
            self.volatility = volatility
        self.implied_volatility = implied_volatility
        self.tau = tau
        self.Time_to_maturity = Time_to_maturity
        if self.tau is None:
            if Time_to_maturity:
                self.tau = self.Time_to_maturity / 365.0
            else:
                self.tau = (self.end_date - datetime.now()).days / 365.0
        self.pricing_method = pricing_method

    @property
    def _args(self):
        """
        Return the order parameters as a dictionary.

        Returns:
            dict: A dictionary containing the order parameters.
        """
        args = {
            'contract_code': self.contract_code,
            'option_type': self.type,
            'option_price': self.option_price,
            'underlying_price': self.underlying_price,
            'strike_price': self.strike_price,
            'risk_free_rate': self.risk_free_rate,
            'operation': self.operation,
            'volatility': self.volatility,
            'implied_volatility': self.implied_volatility,
            'end_date': self.end_date,
            'tau': self.tau,
            'Time_to_maturity': self.Time_to_maturity,
        }
        return args

    def calculate_price(self):
        """Calculate the price of the option using the specified pricing method.

        Returns:
            float: The price of the option.

        Raises:
            ValueError: If the pricing method is not supported.

        """
        if self.pricing_method == "Black-Scholes":
            return self._calculate_black_scholes_price()
        elif self.pricing_method == "BAW":
            return self._calculate_baw_price()
        elif self.pricing_method == "Binomial-Tree":
            return self._calculate_binomial_tree_price()
        elif self.pricing_method == "Monte-Carlo":
            return self._calculate_monte_carlo_price()
        elif self.pricing_method == "Heston":
            return self._calculate_heston_price()
        else:
            raise ValueError(
                "Invalid pricing method. Supported methods are 'Black-Scholes', 'BAW', 'Binomial-Tree',\
                 'Monte-Carlo', and 'Heston'.")

    def _calculate_baw_price(self):
        """Calculate the price of the option using the Barone-Adesi and Whaley approximation method.

        Returns:
            float: The price of the option.

        """

        def calculate_bs_iv():
            bs_price = self._calculate_black_scholes_price()
            return bs_price - self._calculate_black_scholes_price()

        self.implied_volatility = newton(calculate_bs_iv, self.volatility)
        baw_price = self._calculate_black_scholes_price()
        return baw_price

    def _calculate_binomial_tree_price(self):
        """Calculate the price of the option using the binomial tree model.

        Returns:
            float: The price of the option.

        """
        raise NotImplementedError("Binomial-Tree model not implemented yet.")

    def _calculate_heston_price(self, kappa=1.5, theta=0.02, sigma=0.3, rho=-0.5, v0=0.02):
        """Calculate the price of the option using the Heston model.

        Args:
            kappa (float, optional): The mean reversion speed parameter. Defaults to 1.5.
            theta (float, optional): The long-term mean of the variance. Defaults to 0.02.
            sigma (float, optional): The volatility of volatility parameter. Defaults to 0.3.
            rho (float, optional): The correlation between the asset price and its variance. Defaults to -0.5.
            v0 (float, optional): The initial variance. Defaults to 0.02.

        Returns:
            float: The price of the option.

        """
        S0 = self.underlying_price

        K = self.strike_price
        T = self.tau
        r = self.risk_free_rate

        n_simulations = 10000
        n_steps = 100
        dt = T / n_steps

        z1 = np.random.normal(size=(n_simulations, n_steps))
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=(n_simulations, n_steps))
        vt = np.zeros((n_simulations, n_steps + 1))
        vt[:, 0] = v0

        for i in range(1, n_steps + 1):
            vt[:, i] = np.maximum(
                vt[:, i - 1] + kappa * (theta - vt[:, i - 1]) * dt + sigma * np.sqrt(vt[:, i - 1] * dt) * z2[:, i - 1],
                0)

        st = np.zeros((n_simulations, n_steps + 1))
        st[:, 0] = S0

        for i in range(1, n_steps + 1):
            st[:, i] = st[:, i - 1] * np.exp((r - 0.5 * vt[:, i]) * dt + np.sqrt(vt[:, i] * dt) * z1[:, i - 1])

        payoff = np.maximum(st[:, -1] - K, 0)
        price = np.mean(payoff) * np.exp(-r * T)

        return price

    def _calculate_monte_carlo_price(self):
        """Calculate the price of the option using Monte Carlo simulation.

        Returns:
            float: The price of the option.

        """
        np.random.seed(42)  # Set random seed for reproducibility
        S = self.underlying_price
        K = self.strike_price
        T = self.tau
        r = self.risk_free_rate
        sigma = self.volatility
        n = 10000  # Number of Monte Carlo simulations

        z = np.random.normal(0, 1, n)
        ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * sqrt(T) * z)

        if self.type == "call":
            payoff = np.maximum(ST - K, 0)
        elif self.type == "put":
            payoff = np.maximum(K - ST, 0)
        else:
            raise ValueError("Invalid option Option_type. Must be 'call' or 'put'.")

        price = np.exp(-r * T) * np.mean(payoff)

        return price

    def _calculate_black_scholes_price(self):
        """Calculate the price of the option using the Black-Scholes model.

        Returns:
            float: The price of the option.

        """
        S = self.underlying_price
        K = self.strike_price
        T = self.tau
        r = self.risk_free_rate
        sigma = self.volatility
        if self.implied_volatility:
            sigma = self.implied_volatility

        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        if self.type == "call":
            if self.operation == "buy":
                return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
            elif self.operation == "sell":
                return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            else:
                raise ValueError("Invalid operation. Must be 'buy' or 'sell' for call option.")
        elif self.type == "put":
            if self.operation == "buy":
                return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            elif self.operation == "sell":
                return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
            else:
                raise ValueError("Invalid operation. Must be 'buy' or 'sell' for put option.")
        else:
            raise ValueError("Invalid option Option_type. Must be 'call' or 'put'.")

    def _calculate_bs_greeks(self):
        """Calculate the option greeks using the Black-Scholes model.

        Returns:
            tuple: A tuple containing the option delta, gamma, vega, theta, and rho.

        """
        S = self.underlying_price
        K = self.strike_price
        T = self.tau
        r = self.risk_free_rate
        sigma = self.volatility

        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

        delta = norm.cdf(d1) if self.type == "call" else norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
        vega = S * norm.pdf(d1) * sqrt(T)
        theta = -S * norm.pdf(d1) * sigma / (2 * sqrt(T)) - r * K * exp(-r * T) * norm.cdf(
            d1 if self.type == "call" else -d1)
        rho = K * T * exp(-r * T) * norm.cdf(d1 if self.type == "call" else -d1)

        return delta, gamma, vega, theta, rho

    def calculate_greeks(self):
        """Calculate the option greeks using the specified pricing method.

        Returns:
            tuple: A tuple containing the option delta, gamma, vega, theta, and rho.

        Raises:
            ValueError: If the pricing method is not supported.

        """
        if self.pricing_method == "Black-Scholes":
            return self._calculate_bs_greeks()
        else:
            raise ValueError(f"Invalid pricing method. {self.pricing_method}  is NOT supported!")
