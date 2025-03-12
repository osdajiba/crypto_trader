# src/common/config_parser.py

import yaml
import json
from pathlib import Path
from typing import List, Dict, Any, Union

class ConfigParser:
    """
    Helper utility to parse configuration files and extract available options
    for command-line interfaces and scripts.
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the config parser with a config file path.
        
        Args:
            config_path: Path to the YAML or JSON configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file
        
        Returns:
            Dict containing configuration values
        """
        if not self.config_path.exists():
            return {}
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                elif self.config_path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    raise ValueError(f"Unsupported config file extension: {self.config_path.suffix}")
        except Exception as e:
            print(f"Error loading config file: {e}")
            return {}
    
    def get_available_strategies(self) -> List[str]:
        """
        Get a list of available trading strategies from the config
        
        Returns:
            List of strategy names
        """
        strategies = ["dual_ma", "neural_network"]  # Default strategies
        
        # Try to find additional strategies from config
        if 'strategy' in self.config:
            strat_section = self.config['strategy']
            if 'active' in strat_section and strat_section['active'] not in strategies:
                strategies.append(strat_section['active'])
            if 'fallback_strategy' in strat_section and strat_section['fallback_strategy'] not in strategies:
                strategies.append(strat_section['fallback_strategy'])
        
        return strategies
    
    def get_available_timeframes(self) -> List[str]:
        """
        Get a list of available timeframes
        
        Returns:
            List of timeframe strings
        """
        # Standard timeframes
        return ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
    
    def get_available_symbols(self) -> List[str]:
        """
        Get a list of available trading symbols from the config
        
        Returns:
            List of symbol strings
        """
        default_symbols = ["BTC/USDT", "ETH/USDT"]
        
        if 'trading' in self.config and 'instruments' in self.config['trading']:
            return self.config['trading']['instruments']
        
        return default_symbols
    
    def get_default_dates(self) -> Dict[str, str]:
        """
        Get default date range for backtesting
        
        Returns:
            Dict with start_date and end_date
        """
        dates = {
            'start_date': '2023-01-01',
            'end_date': '2023-12-31'
        }
        
        if 'backtest' in self.config and 'period' in self.config['backtest']:
            period = self.config['backtest']['period']
            if 'start' in period:
                dates['start_date'] = period['start']
            if 'end' in period:
                dates['end_date'] = period['end']
        
        return dates
    
    def get_config_value(self, *keys, default=None) -> Any:
        """
        Get a configuration value using a sequence of keys
        
        Args:
            *keys: Sequence of keys to navigate the config
            default: Default value if the key doesn't exist
            
        Returns:
            The configuration value or default
        """
        result = self.config
        for key in keys:
            if key in result:
                result = result[key]
            else:
                return default
        return result