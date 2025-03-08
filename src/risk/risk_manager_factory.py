# src/risk/risk_manager_factory.py
from src.risk.risk_manager import BaseRiskManager, BacktestRiskManager, LiveRiskManager

class RiskManagerFactory:
    @staticmethod
    def create_risk_manager(mode: str, config: dict) -> BaseRiskManager:
        """
        Creates appropriate risk manager based on mode
        
        Args:
            mode: Trading mode ('backtest', 'paper', 'live')
            config: Configuration dictionary
            
        Returns:
            Appropriate risk manager instance
        """
        if mode == "backtest":
            return BacktestRiskManager(config)
        elif mode == "paper":
            # Paper trading can use a modified live risk manager
            return LiveRiskManager(config)
        elif mode == "live":
            return LiveRiskManager(config)
        else:
            # Default to backtest risk manager
            return BacktestRiskManager(config)