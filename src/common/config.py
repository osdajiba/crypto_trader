#!/usr/bin/env python3

# src/common/config.py

from datetime import datetime, timedelta
import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import copy
from abc import ABC, abstractmethod


class ConfigError(Exception):
    """Base class for configuration errors"""
    pass


class ConfigValidationError(ConfigError):
    """Configuration validation error exception"""
    pass


class ConfigLoadError(ConfigError):
    """Configuration loading error exception"""
    pass


class ConfigBackend(ABC):
    """Abstract base class for configuration backends"""
    
    @abstractmethod
    def load(self, path: Path) -> Dict[str, Any]:
        """Load configuration from specified path"""
        pass
    
    @abstractmethod
    def save(self, config: Dict[str, Any], path: Path) -> None:
        """Save configuration to specified path"""
        pass


class JsonConfigBackend(ConfigBackend):
    """JSON configuration backend"""
    
    def load(self, path: Path) -> Dict[str, Any]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigLoadError(f"Invalid JSON in {path}: {str(e)}")
        except Exception as e:
            raise ConfigLoadError(f"Error loading config from {path}: {str(e)}")
    
    def save(self, config: Dict[str, Any], path: Path) -> None:
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            raise ConfigLoadError(f"Error saving config to {path}: {str(e)}")


class YamlConfigBackend(ConfigBackend):
    """YAML configuration backend"""
    
    def load(self, path: Path) -> Dict[str, Any]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"Invalid YAML in {path}: {str(e)}")
        except Exception as e:
            raise ConfigLoadError(f"Error loading config from {path}: {str(e)}")
    
    def save(self, config: Dict[str, Any], path: Path) -> None:
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            raise ConfigLoadError(f"Error saving config to {path}: {str(e)}")


class ConfigManager:
    """
    Enhanced configuration manager with strict validation and consistent default handling.
    Provides a unified interface for loading, validating, and accessing configuration values.
    """
    
    # Default configuration structure
    _DEFAULT_CONFIG = {
        'api': {
            'base_url': 'https://api.example.com',
            'timeout': 60000,
            'retry_attempts': 3,
            'retry_delay': 1.0,
            'enable_rate_limit': True,
            'binance': {
                'api_key': '',
                'secret': 'YOUR_SECRET_KEY',
                'options': {
                    'adjust_time_diff': True,
                    'recv_window': 60000,
                    'default_type': 'spot'
                }
            },
            'headers': {
                'connection': 'close'
            }
        },
        'data': {
            'source': {
                'live': False,
                'type': 'local'
            },
            'storage': {
                'historical': 'data/historical/',
                'orders': 'data/order_history.csv',
                'trades': 'data/trade_records.json'
            },
            'cache': {
                'enabled': True,
                'dir': './cache',
                'max_size': 104857600,
                'retention_days': 30
            }
        },
        'logging': {
            'base_path': './logs',
            'level': 'INFO',
            'async_write': True,
            'combined_log': True,
            'format': "%(asctime)s | %(levelname)-8s | %(module)-18s | [%(filename)s:%(lineno)d] | %(message)s",
            'date_format': '%Y-%m-%d %H:%M:%S%z',
            'detailed_format': True,
            'handlers': {
                'console': {
                    'enabled': True,
                    'level': 'WARNING'
                },
                'file': {
                    'enabled': True,
                    'compression': 'gz',
                    'encoding': 'utf-8',
                    'buffer': {
                        'enabled': True,
                        'capacity': 1000,
                        'flush_interval': 5
                    }
                }
            },
            'categories': {
                'core': {
                    'level': 'INFO',
                    'modules': ['core', 'trading_system', 'mode', 'performance']
                },
                'data': {
                    'level': 'INFO',
                    'modules': ['datasource', 'dataframe', 'database', 'download_data']
                },
                'strategy': {
                    'level': 'INFO',
                    'modules': ['strategy', 'indicators', 'signals', 'backtest']
                },
                'exchange': {
                    'level': 'WARNING',
                    'modules': ['exchange', 'api', 'binance', 'websocket']
                },
                'system': {
                    'level': 'INFO',
                    'modules': ['system', 'cli', 'launcher', 'gui', 'async_executor', 'risk']
                },
                'utils': {
                    'level': 'INFO',
                    'modules': ['utils', 'network', 'config_manager']
                }
            },
            'rotation': {
                'strategy': 'size',
                'max_file_size': 10485760,
                'backup_count': 5,
                'time_interval': 'D'
            }
        },
        'system': {
            'operational_mode': 'backtest',
            'timezone': 'UTC',
            'performance': {
                'max_threads': 4,
                'max_memory': 1073741824,  # 1GB
                'task_timeout': 300
            }
        },
        'trading': {
            'capital': {
                'initial': 100000,
                'min_order': 0.001
            },
            'instruments': ["BTC/USDT", "ETH/USDT"]
        },
        'backtest': {
            'period': {
                'start': '2023-01-01',
                'end': '2023-12-31'
            }
        }
    }
    
    # Mapping of file extensions to backends
    _BACKENDS = {
        '.json': JsonConfigBackend(),
        '.yaml': YamlConfigBackend(),
        '.yml': YamlConfigBackend()
    }
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._config = copy.deepcopy(self._DEFAULT_CONFIG)
        self._config_path = Path(config_path) if config_path else None
        
        # Initialize runtime date context for log rotation
        self._runtime_date = datetime.now().date()
        self._next_date_check = datetime.now() + timedelta(days=1)
        
        # Load the configuration if a path was provided
        if config_path:
            self.load()
    
    def _check_date_rollover(self) -> None:
        """Check if date directory needs rotation for logging"""
        now = datetime.now()
        if now > self._next_date_check:
            current_date = now.date()
            if current_date != self._runtime_date:
                self._logger.info("Date change detected, reinitializing paths")
                self._runtime_date = current_date
            # Schedule next check for next day 00:10
            self._next_date_check = now.replace(
                hour=0, minute=10, second=0
            ) + timedelta(days=1)
    
    def get_log_path(self) -> Path:
        """
        Get current log directory (with date check)
        
        Returns:
            Path to log directory
        """
        self._check_date_rollover()
        return Path(self.get("logging", "base_path", default="./logs"))

    def validate(self) -> None:
        """
        Basic validation of required configuration values
        
        Raises:
            ValueError: If required configuration is missing
        """
        required_keys = [
            ("system", "operational_mode"),
            ("logging", "base_path"),
            ("trading", "capital", "initial")
        ]
        for keys in required_keys:
            if not self.get(*keys):
                raise ValueError(f"Missing configuration: {'.'.join(keys)}")
    
    def load(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Load configuration from file
        
        Args:
            path: Configuration file path (optional, uses initialized path if not provided)
            
        Raises:
            ConfigLoadError: If configuration cannot be loaded
            ConfigValidationError: If configuration is invalid
        """
        if path:
            self._config_path = Path(path)
        
        if not self._config_path:
            return  # No path specified, use default config
        
        if not self._config_path.exists():
            self._logger.warning(f"Configuration file not found: {self._config_path}, using defaults")
            return
        
        # Determine backend based on file extension
        extension = self._config_path.suffix.lower()
        backend = self._BACKENDS.get(extension)
        
        if not backend:
            supported = ', '.join(self._BACKENDS.keys())
            raise ConfigLoadError(f"Unsupported configuration format: {extension}. Supported formats: {supported}")
        
        try:
            loaded_config = backend.load(self._config_path)
            self._merge_config(loaded_config)
            # Validate configuration structure
            try:
                self._validate_config_structure()
            except Exception as e:
                self._logger.warning(f"Config validation warning: {e}")
                
            self._logger.info(f"Configuration loaded from {self._config_path}")
        except ConfigLoadError:
            raise
        except Exception as e:
            raise ConfigLoadError(f"Error processing configuration from {self._config_path}: {str(e)}")
    
    def _merge_config(self, loaded_config: Dict[str, Any]) -> None:
        """
        Merge loaded configuration with defaults while preserving structure
        
        Args:
            loaded_config: Loaded configuration to merge
        """
        def deep_merge(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    deep_merge(target[key], value)
                else:
                    target[key] = value
        
        # Start with fresh default copy
        merged_config = copy.deepcopy(self._DEFAULT_CONFIG)
        
        # Deep merge configurations
        deep_merge(merged_config, loaded_config)
        
        # Update current config
        self._config = merged_config
    
    def _validate_config_structure(self) -> None:
        """
        Enhanced configuration validation with strict schema checking
        
        Raises:
            ConfigValidationError: If configuration doesn't match expected schema
        """
        schema = {
            'api': {
                'required': True,
                'type': dict,
                'children': {
                    'base_url': {'type': str, 'required': True},
                    'timeout': {'type': (int, float), 'required': True, 'min': 1},
                    'retry_attempts': {'type': int, 'required': True, 'min': 0},
                    'retry_delay': {'type': (int, float), 'required': True, 'min': 0}
                }
            },
            'data': {
                'required': True,
                'type': dict,
                'children': {
                    'cache': {'type': dict, 'required': True},
                    'storage': {'type': dict, 'required': True},
                    'source': {'type': dict, 'required': True}
                }
            },
            'logging': {
                'required': True,
                'type': dict,
                'children': {
                    'base_path': {'type': str, 'required': True},
                    'level': {'type': str, 'required': True},
                    'handlers': {'type': dict, 'required': True},
                    'categories': {'type': dict, 'required': True},
                    'rotation': {'type': dict, 'required': True}
                }
            },
            'system': {
                'required': True,
                'type': dict,
                'children': {
                    'operational_mode': {'type': str, 'required': True},
                    'timezone': {'type': str, 'required': True},
                    'performance': {'type': dict, 'required': True}
                }
            }
        }
        
        self._validate_against_schema(self._config, schema)
    
    def _validate_against_schema(self, config: Dict[str, Any], schema: Dict[str, Any], path: str = "") -> None:
        """
        Validate configuration against schema
        
        Args:
            config: Configuration to validate
            schema: Schema to validate against
            path: Current path in configuration structure for error reporting
            
        Raises:
            ConfigValidationError: If configuration doesn't match schema
        """
        for key, spec in schema.items():
            current_path = f"{path}.{key}" if path else key
            
            if spec.get('required', False) and key not in config:
                raise ConfigValidationError(f"Missing required configuration key: {current_path}")
            
            if key not in config:
                continue
            
            value = config[key]
            
            expected_type = spec.get('type')
            if expected_type and not isinstance(value, expected_type):
                type_names = [t.__name__ for t in (expected_type if isinstance(expected_type, tuple) else (expected_type,))]
                raise ConfigValidationError(
                    f"Invalid type for {current_path}: expected {' or '.join(type_names)}, got {type(value).__name__}"
                )
            
            if 'min' in spec and value < spec['min']:
                raise ConfigValidationError(
                    f"Value for {current_path} is too small: minimum is {spec['min']}, got {value}"
                )
            
            if 'max' in spec and value > spec['max']:
                raise ConfigValidationError(
                    f"Value for {current_path} is too large: maximum is {spec['max']}, got {value}"
                )
            
            if 'values' in spec and value not in spec['values']:
                raise ConfigValidationError(
                    f"Invalid value for {current_path}: must be one of {spec['values']}, got {value}"
                )
            
            if isinstance(value, dict) and 'children' in spec:
                self._validate_against_schema(value, spec['children'], current_path)
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Safely retrieve nested configuration values
        
        Args:
            *keys: Configuration key path
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        current_level = self._config
        
        for key in keys:
            if not isinstance(current_level, dict):
                return default
                
            if key not in current_level:
                return default
                
            current_level = current_level[key]
        
        return current_level
    
    def set(self, *keys: str, value: Any = None) -> None:
        """
        Set configuration value
        
        Args:
            *keys: Configuration key path
            value: Value to set
            
        Raises:
            ValueError: If key path is invalid
        """
        if not keys:
            raise ValueError("No keys provided")
        
        if value is None and len(keys) > 1:
            keys, value = keys[:-1], keys[-1]
        
        config = self._config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            elif not isinstance(config[key], dict):
                config[key] = {}
            
            config = config[key]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save configuration to file
        
        Args:
            path: Path to save configuration (optional, uses current path if not provided)
            
        Raises:
            ConfigLoadError: If configuration cannot be saved
        """
        if path:
            save_path = Path(path)
        elif self._config_path:
            save_path = self._config_path
        else:
            raise ConfigLoadError("No configuration path specified for saving")
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        extension = save_path.suffix.lower()
        backend = self._BACKENDS.get(extension)
        
        if not backend:
            supported = ', '.join(self._BACKENDS.keys())
            raise ConfigLoadError(f"Unsupported configuration format: {extension}. Supported formats: {supported}")
        
        try:
            backend.save(self._config, save_path)
            self._logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            raise ConfigLoadError(f"Error saving configuration to {save_path}: {str(e)}")
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get deep copy of entire configuration
        
        Returns:
            Deep copy of configuration
        """
        return copy.deepcopy(self._config)
    
    def update(self, new_config: Dict[str, Any]) -> None:
        """
        Update configuration with new values
        
        Args:
            new_config: New configuration values
            
        Raises:
            ConfigValidationError: If updated configuration is invalid
        """
        self._merge_config(new_config)
        self._validate_config_structure()
    
    def reset(self) -> None:
        """Reset to default configuration"""
        self._config = copy.deepcopy(self._DEFAULT_CONFIG)
        
    def get_log_level(self, level_str: str) -> int:
        """
        Convert string log level to numeric value
        
        Args:
            level_str: String representation of log level
            
        Returns:
            int: Numeric log level
        """
        log_levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        level_upper = level_str.upper()
        return log_levels.get(level_upper, logging.INFO)  # Default to INFO level


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