# src/common/log_manager.py

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
from typing import Dict, Any, Optional, Union
from datetime import datetime

from src.common.config_manager import ConfigManager 


class LogManager:
    """
    增强的日志管理器
    支持灵活的日志轮替策略和一致的日志级别处理
    实现为单例模式
    """
    
    _instance = None
    
    # 定义标准日志级别
    LOG_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    # 默认格式化器
    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DETAILED_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    
    def __new__(cls, config: ConfigManager):
        """
        创建LogManager单例
        
        Args:
            config: log settings
        """
        if cls._instance is None:
            if not config:
                raise ValueError("Config must be provided for initial initialization")
            cls._instance = super(LogManager, cls).__new__(cls)
            cls._instance._init_logger(config)
        return cls._instance
    
    def _init_logger(self, config):
        """
        初始化日志管理器（只在第一次创建实例时调用）
        """
        # 从配置获取参数
        self.base_dir = Path(config.get("logging", "base_path", default="./logs"))
        self.default_level = self._get_log_level(config.get("logging", "level", default="INFO"))
        self.log_to_console = config.get("logging", "handlers", "console", "enabled", default=True)
        self.log_to_file = config.get("logging", "handlers", "file", "enabled", default=True)
        
        # 日志轮转配置
        self.rotation_strategy = config.get("logging", "rotation", "strategy", default="size")
        self.max_file_size = int(config.get("logging", "rotation", "max_file_size", default=10 * 1024 * 1024))  # 默认10MB
        self.backup_count = int(config.get("logging", "rotation", "backup_count", default=5))
        
        # 格式配置
        self.format_str = self.DETAILED_FORMAT if config.get("logging", "format", default=False) else self.DEFAULT_FORMAT
        
        # 初始化根日志记录器
        self.configured_loggers = set()
        self.setup_root_logger()
        
        # 自动创建日志目录
        self.base_dir.mkdir(parents=True, exist_ok=True)
            
    def setup_logger(self, name: str, base_dir: Optional[Union[str, Path]] = None, 
                level: Optional[Union[str, int]] = None) -> logging.Logger:
        # 如果提供了新的基本目录，则更新实例
        if base_dir:
            self.base_dir = Path(base_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用新的 API 获取日志记录器
        return self._get_logger_instance(name, level)
    
    def _get_log_level(self, level: Union[str, int]) -> int:
        if isinstance(level, int):
            return level
        
        level_upper = level.upper()
        if level_upper in self.LOG_LEVELS:
            return self.LOG_LEVELS[level_upper]
        
        raise ValueError(f"Invalid log level: {level}. Valid levels are: {', '.join(self.LOG_LEVELS.keys())}")
    
    def setup_root_logger(self):
        """配置根日志记录器"""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.default_level)
        
        # 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # 控制台处理器
        if self.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(self.format_str))
            console_handler.setLevel(self.default_level)
            root_logger.addHandler(console_handler)

        # 文件处理器
        if self.log_to_file:
            file_handler = self._create_file_handler("system")
            file_handler.setLevel(self.default_level)
            root_logger.addHandler(file_handler)
    
    def _create_file_handler(self, name: str) -> logging.Handler:
        """创建文件处理器"""
        log_file = self.base_dir / f"{name}.log"
        
        if self.rotation_strategy == "size":
            handler = RotatingFileHandler(
                log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding="utf-8"
            )
        else:
            handler = TimedRotatingFileHandler(
                log_file,
                when="midnight",
                interval=1,
                backupCount=self.backup_count,
                encoding="utf-8"
            )
            
        handler.setFormatter(logging.Formatter(self.format_str))
        return handler
        
    @classmethod
    def level_from_string(cls, level_str: str) -> int:
        """
        将日志级别字符串转换为日志级别整数（兼容旧版API）
        
        Args:
            level_str: 日志级别字符串
            
        Returns:
            日志级别整数
        """
        instance = cls()
        return instance._get_log_level(level_str)
    
    @classmethod
    def get_logger(cls, name: str, level: Optional[Union[str, int]] = None) -> logging.Logger:
        """获取配置好的日志记录器"""
        instance = cls._instance
        if not instance:
            raise RuntimeError("LogManager not initialized. Call LogManager(config) first.")
            
        logger = logging.getLogger(name)
        if name not in instance.configured_loggers:
            logger.setLevel(instance._get_log_level(level) if level else instance.default_level)
            if instance.log_to_file:
                handler = instance._create_file_handler(name)
                logger.addHandler(handler)
            instance.configured_loggers.add(name)
        return logger
    
    def _get_logger_instance(self, name: str, level: Optional[Union[str, int]] = None) -> logging.Logger:
        """
        获取配置的日志记录器（实例方法）
        
        Args:
            name: 日志记录器名称
            level: 可选的日志级别（默认使用默认级别）
            
        Returns:
            配置的日志记录器
        """
        logger = logging.getLogger(name)
        
        # 如果已经配置过，只更新级别
        if name in self.configured_loggers:
            if level is not None:
                logger.setLevel(self._get_log_level(level))
            return logger
        
        # 设置日志级别
        logger_level = self._get_log_level(level) if level is not None else self.default_level
        logger.setLevel(logger_level)
        
        # 添加文件处理器（如果启用）
        if self.log_to_file:
            module_name = name.split('.')[-1]
            file_handler = self.create_file_handler(module_name)
            logger.addHandler(file_handler)
        
        # 标记为已配置
        self.configured_loggers.add(name)
        
        return logger
    
    @classmethod
    def set_level(cls, level: Union[str, int], logger_name: Optional[str] = None) -> None:
        """
        设置一个或所有日志记录器的日志级别（类方法）
        
        Args:
            level: 要设置的日志级别
            logger_name: 要更新的特定日志记录器名称，如果为None则更新所有日志记录器
        """
        # 确保单例已创建
        instance = cls()
        instance._set_level_instance(level, logger_name)
    
    def _set_level_instance(self, level: Union[str, int], logger_name: Optional[str] = None) -> None:
        """
        设置一个或所有日志记录器的日志级别（实例方法）
        
        Args:
            level: 要设置的日志级别
            logger_name: 要更新的特定日志记录器名称，如果为None则更新所有日志记录器
        """
        numeric_level = self._get_log_level(level)
        
        if logger_name is not None:
            # 更新单个日志记录器
            logger = logging.getLogger(logger_name)
            logger.setLevel(numeric_level)
            
            # 更新此日志记录器的所有处理器
            for handler in logger.handlers:
                handler.setLevel(numeric_level)
        else:
            # 更新所有已配置的日志记录器
            for name in self.configured_loggers:
                logger = logging.getLogger(name if name != 'root' else '')
                logger.setLevel(numeric_level)
                
                # 更新所有处理器
                for handler in logger.handlers:
                    handler.setLevel(numeric_level)
            
            # 更新默认级别
            self.default_level = numeric_level
    
    def add_json_handler(self, logger_name: str):
        """添加JSON格式日志处理器"""
        logger = logging.getLogger(logger_name)
        json_file = self.base_dir / f"{logger_name}.json.log"
        
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "line": record.lineno
                }
                if record.exc_info:
                    log_entry["exception"] = self.formatException(record.exc_info)
                return json.dumps(log_entry)
        
        handler = RotatingFileHandler(
            json_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding="utf-8"
        )
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    
    def configure_from_dict(self, config: Dict[str, Any]) -> None:
        """
        从配置字典配置日志管理器
        
        Args:
            config: 包含日志配置的字典
        """
        # 更新基本配置
        if 'base_dir' in config:
            self.base_dir = Path(config['base_dir'])
            self.base_dir.mkdir(parents=True, exist_ok=True)
        
        if 'default_level' in config:
            self.default_level = self._get_log_level(config['default_level'])
        
        if 'log_to_console' in config:
            self.log_to_console = bool(config['log_to_console'])
        
        if 'log_to_file' in config:
            self.log_to_file = bool(config['log_to_file'])
        
        if 'rotation_strategy' in config and config['rotation_strategy'] in ('size', 'time'):
            self.rotation_strategy = config['rotation_strategy']
        
        if 'max_file_size' in config:
            self.max_file_size = int(config['max_file_size'])
        
        if 'backup_count' in config:
            self.backup_count = int(config['backup_count'])
        
        if 'detailed_format' in config:
            self.format_str = self.DETAILED_FORMAT if config['detailed_format'] else self.DEFAULT_FORMAT
        
        # 重新配置根日志记录器
        self.setup_root_logger()
        
        # 配置特定的日志记录器
        if 'loggers' in config and isinstance(config['loggers'], dict):
            for logger_name, logger_config in config['loggers'].items():
                level = logger_config.get('level', self.default_level)
                logger = self.get_logger(logger_name, level)
                
                # 添加JSON处理器（如果配置了）
                if logger_config.get('json_output'):
                    json_file = logger_config.get('json_file')
                    self.add_json_handler(logger_name, json_file)