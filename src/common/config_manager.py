# # src/common/config_manager.py

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import copy
from abc import ABC, abstractmethod

class ConfigValidationError(Exception):
    """配置验证错误异常"""
    pass

class ConfigLoadError(Exception):
    """配置加载错误异常"""
    pass

class ConfigBackend(ABC):
    """配置后端抽象基类"""
    
    @abstractmethod
    def load(self, path: Path) -> Dict[str, Any]:
        """从指定路径加载配置"""
        pass
    
    @abstractmethod
    def save(self, config: Dict[str, Any], path: Path) -> None:
        """将配置保存到指定路径"""
        pass

class JsonConfigBackend(ConfigBackend):
    """JSON配置后端"""
    
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
    """YAML配置后端"""
    
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
    增强的配置管理器，具有严格验证和一致的默认处理
    将加载逻辑与重试和验证逻辑分离
    """
    
    _DEFAULT_CONFIG = {
        'api': {
            'base_url': 'https://api.example.com',
            'timeout': 30,
            'retry_attempts': 3,
            'retry_delay': 1.0
        },
        'data': {
            'cache_dir': './cache',
            'max_cache_size': 1024 * 1024 * 100,  # 100 MB
            'retention_days': 30
        },
        'logging': {
            'level': 'INFO',
            'file_path': './logs',
            'max_file_size': 10 * 1024 * 1024,  # 10 MB
            'backup_count': 5
        },
        'system': {
            'max_threads': 4,
            'max_memory': 1024 * 1024 * 1024  # 1 GB
        }
    }
    
    _BACKENDS = {
        '.json': JsonConfigBackend(),
        '.yaml': YamlConfigBackend(),
        '.yml': YamlConfigBackend()
    }
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径（可选）
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._config = copy.deepcopy(self._DEFAULT_CONFIG)
        self._config_path = Path(config_path) if config_path else None
        
        if self._config_path and self._config_path.exists():
            self.load()
    
    def load(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        从文件加载配置
        
        Args:
            path: 配置文件路径（可选，如果未提供则使用初始化时的路径）
            
        Raises:
            ConfigLoadError: 如果配置无法加载
            ConfigValidationError: 如果配置无效
        """
        if path:
            self._config_path = Path(path)
        
        if not self._config_path:
            raise ConfigLoadError("No configuration path specified")
        
        if not self._config_path.exists():
            raise ConfigLoadError(f"Configuration file not found: {self._config_path}")
        
        # 根据文件扩展名确定适当的后端
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
            # 重新抛出加载错误
            raise
        except Exception as e:
            raise ConfigLoadError(f"Error processing configuration from {self._config_path}: {str(e)}")
    
    def _merge_config(self, loaded_config: Dict[str, Any]) -> None:
        """
        将加载的配置与默认值合并，保留结构
        
        Args:
            loaded_config: 要合并的加载配置
        """
        def deep_merge(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    deep_merge(target[key], value)
                else:
                    target[key] = value
        
        # 从默认值的新副本开始
        merged_config = copy.deepcopy(self._DEFAULT_CONFIG)
        
        # 深度合并加载的配置
        deep_merge(merged_config, loaded_config)
        
        # 用合并结果替换当前配置
        self._config = merged_config
    
    def _validate_config_structure(self) -> None:
        """
        改进的配置验证，具有严格的架构检查
        
        Raises:
            ConfigValidationError: 如果配置不符合预期架构
        """
        # 定义完整的配置架构，包含类型和要求信息
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
        根据提供的架构验证配置
        
        Args:
            config: 要验证的配置
            schema: 要验证的架构
            path: 配置结构中的当前路径，用于错误报告
            
        Raises:
            ConfigValidationError: 如果配置不符合架构
        """
        for key, spec in schema.items():
            current_path = f"{path}.{key}" if path else key
            
            # 检查是否存在必需的键
            if spec.get('required', False) and key not in config:
                raise ConfigValidationError(f"Missing required configuration key: {current_path}")
            
            # 如果键不存在且不是必需的，则跳过验证
            if key not in config:
                continue
            
            value = config[key]
            
            # 验证类型
            expected_type = spec.get('type')
            if expected_type and not isinstance(value, expected_type):
                type_names = [t.__name__ for t in (expected_type if isinstance(expected_type, tuple) else (expected_type,))]
                raise ConfigValidationError(
                    f"Invalid type for {current_path}: expected {' or '.join(type_names)}, got {type(value).__name__}"
                )
            
            # 验证最小值
            if 'min' in spec and value < spec['min']:
                raise ConfigValidationError(
                    f"Value for {current_path} is too small: minimum is {spec['min']}, got {value}"
                )
            
            # 验证最大值
            if 'max' in spec and value > spec['max']:
                raise ConfigValidationError(
                    f"Value for {current_path} is too large: maximum is {spec['max']}, got {value}"
                )
            
            # 验证值是否在允许的集合中
            if 'values' in spec and value not in spec['values']:
                raise ConfigValidationError(
                    f"Invalid value for {current_path}: must be one of {spec['values']}, got {value}"
                )
            
            # 递归验证嵌套字典
            if isinstance(value, dict) and 'children' in spec:
                self._validate_against_schema(value, spec['children'], current_path)
    
    def get(self, *keys: str, default: Any = None) -> Any:
            """
            通过键路径获取配置值
            
            Args:
                *keys: 配置值的键路径
                default: 如果找不到键路径，则返回的默认值
                
            Returns:
                配置值，如果找不到则返回默认值
            """
            config = self._config
            
            for key in keys:
                if not isinstance(config, dict) or key not in config:
                    return default
                config = config[key]
            
            return config
    
    def set(self, value: Any, *keys: str) -> None:
        """
        设置配置值
        
        Args:
            value: 要设置的值
            *keys: 配置值的键路径
            
        Raises:
            ValueError: 如果键路径无效
        """
        if not keys:
            raise ValueError("No keys provided")
        
        config = self._config
        
        # 遍历到最后一个键前的所有键
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            elif not isinstance(config[key], dict):
                config[key] = {}
            
            config = config[key]
        
        # 设置最后一个键的值
        config[keys[-1]] = value
    
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        将配置保存到文件
        
        Args:
            path: 保存配置的路径（可选，如果未提供则使用当前路径）
            
        Raises:
            ConfigLoadError: 如果配置无法保存
        """
        if path:
            save_path = Path(path)
        elif self._config_path:
            save_path = self._config_path
        else:
            raise ConfigLoadError("No configuration path specified for saving")
        
        # 创建父目录（如果不存在）
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 根据文件扩展名确定适当的后端
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
        获取整个配置的深拷贝
        
        Returns:
            配置的深拷贝
        """
        return copy.deepcopy(self._config)
    
    def update(self, new_config: Dict[str, Any]) -> None:
        """
        使用新配置更新当前配置
        
        Args:
            new_config: 更新配置的新值
            
        Raises:
            ConfigValidationError: 如果更新后的配置无效
        """
        self._merge_config(new_config)
        self._validate_config_structure()
    
    def reset(self) -> None:
        """重置为默认配置"""
        self._config = copy.deepcopy(self._DEFAULT_CONFIG)