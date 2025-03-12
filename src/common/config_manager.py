# src/common/config_manager.py

from datetime import datetime, timedelta
import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import copy
from abc import ABC, abstractmethod

class ConfigValidationError(Exception):
    """Configuration validation error exception"""
    pass

class ConfigLoadError(Exception):
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
    Enhanced configuration manager with strict validation and consistent default handling
    Decoupled loading logic with retry and validation mechanisms
    """
    
    _DEFAULT_CONFIG = {
        'api': {
            'base_url': 'https://api.example.com',
            'timeout': 60000,                    # Updated to match YAML (in milliseconds)
            'retry_attempts': 3,
            'retry_delay': 1.0,                  # In seconds, matches YAML
            'enable_rate_limit': True,           # Added from YAML
            'binance': {                         # Added exchange-specific settings
                'api_key': '',
                'secret': 'YOUR_SECRET_KEY',
                'options': {
                    'adjust_time_diff': True,
                    'recv_window': 60000,        # In milliseconds
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
                'dir': './cache',                # Matches original key 'cache_dir'
                'max_size': 104857600,           # Updated to 100 MB (100 * 1024 * 1024 bytes)
                'retention_days': 30
            }
        },
        'logging': {
            'base_path': './logs',               # Updated from 'file_path' to match YAML
            'level': 'INFO',
            'async_write': True,                 # Added from YAML
            'format': '%(asctime)s | %(levelname)-8s | %(module)-15s | %(message)s',
            'date_format': '%Y-%m-%d %H:%M:%S%z',  # ISO-8601 format from YAML
            'detailed_format': True,             # Added from YAML
            'rotation': {
                'strategy': 'size',
                'max_file_size': 10485760,       # Updated to 10 MB (10 * 1024 * 1024 bytes)
                'backup_count': 5,
                'time_interval': 'D'             # Added for completeness, though strategy is size-based
            },
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
                        'flush_interval': 5      # In seconds
                    }
                }
            }
        },
        'system': {
            'operational_mode': 'backtest',      # Added from YAML
            'timezone': 'UTC',                   # Added from YAML
            'performance': {                     # Nested under 'performance' in YAML
                'max_threads': 4,
                'max_memory': 1073741824,        # Updated to 1 GB (1 * 1024 * 1024 * 1024 bytes)
                'task_timeout': 300              # In seconds
            }
        }
    }
    
    _BACKENDS = {
        '.json': JsonConfigBackend(),
        '.yaml': YamlConfigBackend(),
        '.yml': YamlConfigBackend()
    }
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._config = copy.deepcopy(self._DEFAULT_CONFIG)
        self._config_path = Path(config_path) if config_path else None
        
        # Initialize runtime date context
        self._runtime_date = datetime.now().date()
        self.load()
    
    def _check_date_rollover(self):
        """Check if date directory needs rotation"""
        if datetime.now() > self._next_date_check:
            current_date = datetime.now().date()
            if current_date != self._runtime_date:
                self._logger.info("Date change detected, reinitializing paths")
                self._runtime_date = current_date
            # Schedule next check for next day 00:10
            self._next_date_check = datetime.now().replace(
                hour=0, minute=10, second=0
            ) + timedelta(days=1)
    
    def get_log_path(self) -> Path:
        """Get current log directory (with date check)"""
        self._check_date_rollover()
        return Path(self._config['logging']['file_path'])

    def validate(self):
        required_keys = [
            ("mode", "type"),
            ("strategy", "type"),
            ("symbols", "list")
        ]
        for section, key in required_keys:
            if not self.get(section, key):
                raise ValueError(f"Missing configuration: {section}.{key}")
    
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
            raise ConfigLoadError("No configuration path specified")
        
        if not self._config_path.exists():
            raise ConfigLoadError(f"Configuration file not found: {self._config_path}")
        
        # Determine backend based on file extension
        extension = self._config_path.suffix.lower()
        backend = self._BACKENDS.get(extension)
        
        if not backend:
            supported = ', '.join(self._BACKENDS.keys())
            raise ConfigLoadError(f"Unsupported configuration format: {extension}. Supported formats: {supported}")
        
        try:
            loaded_config = backend.load(self._config_path)
            self._merge_config(loaded_config)
            self._validate_config_structure()
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
                    'cache_dir': {'type': str, 'required': True},
                    'max_cache_size': {'type': int, 'required': True, 'min': 0},
                    'retention_days': {'type': int, 'required': True, 'min': 1}
                }
            },
            'logging': {
                'required': True,
                'type': dict,
                'children': {
                    'level': {'type': str, 'required': True, 'values': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']},
                    'file_path': {'type': str, 'required': True},
                    'max_file_size': {'type': int, 'required': True, 'min': 1024},
                    'backup_count': {'type': int, 'required': True, 'min': 0}
                }
            },
            'system': {
                'required': True,
                'type': dict,
                'children': {
                    'max_threads': {'type': int, 'required': True, 'min': 1},
                    'max_memory': {'type': int, 'required': True, 'min': 1024 * 1024}
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
