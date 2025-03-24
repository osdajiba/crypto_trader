# src/common/log_manager.py

import logging
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
from typing import Dict, Any, Optional, Union, List
from datetime import datetime

from src.common.config_manager import ConfigManager


class LogManager:
    """
    Enhanced Log Manager
    Supports flexible log rotation and consistent log level handling
    Implemented as a singleton
    Provides standard logging-like interface for easier integration
    """
    
    _instance = None
    
    # Standard log levels
    LOG_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    # Default formatters
    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DETAILED_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    
    def __new__(cls, config: Optional[ConfigManager] = None):
        """
        Create LogManager singleton
        
        Args:
            config: Log configuration settings
        """
        if cls._instance is None:
            if not config:
                raise ValueError("Config must be provided for initial initialization")
            cls._instance = super(LogManager, cls).__new__(cls)
            cls._instance._init_logger(config)
        return cls._instance
    
    def _init_logger(self, config):
        """
        Initialize logger (called only on first instance creation)
        """
        # Store config reference for later use
        self.config = config
        
        # Get settings from configuration
        self.base_dir = Path(config.get("logging", "base_path", default="./logs"))
        self.default_level = self._get_log_level(config.get("logging", "level", default="INFO"))
        self.log_to_console = config.get("logging", "handlers", "console", "enabled", default=True)
        self.log_to_file = config.get("logging", "handlers", "file", "enabled", default=True)
        
        # Log rotation configuration
        self.rotation_strategy = config.get("logging", "rotation", "strategy", default="size")
        self.max_file_size = int(config.get("logging", "rotation", "max_file_size", default=10)) * 1024 * 1024  # Default 10MB
        self.backup_count = int(config.get("logging", "rotation", "backup_count", default=5))
        
        # Format configuration
        use_detailed = config.get("logging", "detailed_format", default=False)
        format_from_config = config.get("logging", "format", default=None)
        
        if format_from_config:
            self.format_str = format_from_config
        else:
            self.format_str = self.DETAILED_FORMAT if use_detailed else self.DEFAULT_FORMAT
        
        # Initialize root logger
        self.configured_loggers = set()
        
        # Ensure log directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup root logger
        self.setup_root_logger()
    
    def setup_logger(self, name: str, base_dir: Optional[Union[str, Path]] = None, 
                   level: Optional[Union[str, int]] = None) -> logging.Logger:
        """
        Set up and configure a logger
        
        Args:
            name: Logger name
            base_dir: Optional new base directory
            level: Optional log level
            
        Returns:
            Configured logger
        """
        # Update base directory if provided
        if base_dir:
            self.base_dir = Path(base_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Get logger instance
        return self._get_logger_instance(name, level)
    
    def _get_log_level(self, level: Union[str, int]) -> int:
        """
        Convert string log level to numeric value
        
        Args:
            level: String or integer log level
            
        Returns:
            Numeric log level
        """
        if isinstance(level, int):
            return level
        
        level_upper = level.upper()
        if level_upper in self.LOG_LEVELS:
            return self.LOG_LEVELS[level_upper]
        
        raise ValueError(f"Invalid log level: {level}. Valid levels are: {', '.join(self.LOG_LEVELS.keys())}")
    
    def setup_root_logger(self):
        """Configure the root logger"""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.default_level)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Console handler
        if self.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(self.format_str))
            console_handler.setLevel(self.default_level)
            root_logger.addHandler(console_handler)

        # File handler
        if self.log_to_file:
            file_handler = self._create_file_handler("system")
            file_handler.setLevel(self.default_level)
            root_logger.addHandler(file_handler)
    
    def _create_file_handler(self, name: str) -> logging.Handler:
        """
        Create a file handler for logging
        
        Args:
            name: Logger name to create file for
            
        Returns:
            Configured log handler
        """
        # Determine log file path
        module_path = None

        # Check if we have module-specific path configuration
        modules_config = self.config.get("logging", "modules", default={})
        if modules_config and name in modules_config:
            module_path = modules_config[name].get("path", None)
        
        # Set the log file path
        if module_path:
            # Use the configured path
            log_file = self.base_dir / module_path
        else:
            # Use a more organized directory structure
            # Add year-month directory structure
            from datetime import datetime
            year_month = datetime.now().strftime('%Y-%m')
            
            # For system.log, put it in logs/system/YYYY-MM/system.log
            category = "system"
            if "." in name:
                parts = name.split(".")
                category = parts[0]
            
            log_file = self.base_dir / category / year_month / f"{name}.log"
        
        # Ensure parent directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create appropriate handler based on rotation strategy
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
    def get_logger(cls, name: str, level: Optional[Union[str, int]] = None) -> logging.Logger:
        """
        Get a configured logger
        
        Args:
            name: Logger name
            level: Optional log level
            
        Returns:
            Configured logger
        """
        instance = cls._instance
        if not instance:
            raise RuntimeError("LogManager not initialized. Call LogManager(config) first.")
            
        return instance._get_logger_instance(name, level)
    
    def _get_logger_instance(self, name: str, level: Optional[Union[str, int]] = None) -> logging.Logger:
        """
        Get a configured logger (instance method)
        
        Args:
            name: Logger name
            level: Optional log level
            
        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        
        # If already configured, just update level if needed
        if name in self.configured_loggers:
            if level is not None:
                logger.setLevel(self._get_log_level(level))
            return logger
        
        # Set log level
        logger_level = self._get_log_level(level) if level is not None else self.default_level
        logger.setLevel(logger_level)
        
        # Add file handler if enabled
        if self.log_to_file:
            file_handler = self._create_file_handler(name)
            logger.addHandler(file_handler)
        
        # Mark as configured
        self.configured_loggers.add(name)
        
        return logger
    
    @classmethod
    def set_level(cls, level: Union[str, int], logger_name: Optional[str] = None) -> None:
        """
        Set log level for one or all loggers (class method)
        
        Args:
            level: Log level to set
            logger_name: Specific logger to update (all if None)
        """
        instance = cls._instance
        if not instance:
            raise RuntimeError("LogManager not initialized. Call LogManager(config) first.")
            
        instance._set_level_instance(level, logger_name)
    
    def _set_level_instance(self, level: Union[str, int], logger_name: Optional[str] = None) -> None:
        """
        Set log level for one or all loggers (instance method)
        
        Args:
            level: Log level to set
            logger_name: Specific logger to update (all if None)
        """
        numeric_level = self._get_log_level(level)
        
        if logger_name is not None:
            # Update a single logger
            logger = logging.getLogger(logger_name)
            logger.setLevel(numeric_level)
            
            # Update all handlers for this logger
            for handler in logger.handlers:
                handler.setLevel(numeric_level)
        else:
            # Update all configured loggers
            for name in self.configured_loggers:
                logger = logging.getLogger(name if name != 'root' else '')
                logger.setLevel(numeric_level)
                
                # Update all handlers
                for handler in logger.handlers:
                    handler.setLevel(numeric_level)
            
            # Update default level
            self.default_level = numeric_level
    
    def add_json_handler(self, logger_name: str, json_file: Optional[str] = None):
        """
        Add JSON format log handler
        
        Args:
            logger_name: Logger name to add handler to
            json_file: Optional JSON file path (default: logger_name.json.log)
        """
        logger = logging.getLogger(logger_name)
        
        if json_file:
            json_path = Path(json_file)
        else:
            json_path = self.base_dir / f"{logger_name}.json.log"
        
        # Ensure parent directory exists
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
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
            json_path,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding="utf-8"
        )
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    
    def configure_from_dict(self, config: Dict[str, Any]) -> None:
        """
        Configure log manager from a dictionary
        
        Args:
            config: Dictionary with log configuration
        """
        # Update basic configuration
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
        
        # Reconfigure root logger
        self.setup_root_logger()
        
        # Configure specific loggers
        if 'loggers' in config and isinstance(config['loggers'], dict):
            for logger_name, logger_config in config['loggers'].items():
                level = logger_config.get('level', self.default_level)
                logger = self.get_logger(logger_name, level)
                
                # Add JSON handler if configured
                if logger_config.get('json_output'):
                    json_file = logger_config.get('json_file')
                    self.add_json_handler(logger_name, json_file)
    
    # ---- Standard logging-like interface methods ----
    
    def setLevel(self, level: Union[str, int], logger_name: Optional[str] = None) -> None:
        """
        Set log level for a logger (standard logging-like interface)
        
        Args:
            level: Log level to set
            logger_name: Specific logger to update (root if None)
        """
        self._set_level_instance(level, logger_name)
    
    @classmethod
    def addHandler(cls, handler: logging.Handler, logger_name: str = None) -> None:
        """
        Add a handler to a specific logger
        
        Args:
            handler: The handler to add
            logger_name: Name of the logger to add handler to (root if None)
        """
        instance = cls._instance
        if not instance:
            raise RuntimeError("LogManager not initialized. Call LogManager(config) first.")
            
        instance._add_handler_instance(handler, logger_name)
    
    def _add_handler_instance(self, handler: logging.Handler, logger_name: str = None) -> None:
        """
        Add a handler to a specific logger (instance method)
        
        Args:
            handler: The handler to add
            logger_name: Name of the logger to add handler to (root if None)
        """
        logger = logging.getLogger(logger_name if logger_name else '')
        logger.addHandler(handler)
        
        # If this is a new logger, track it
        if logger_name and logger_name not in self.configured_loggers:
            self.configured_loggers.add(logger_name)
    
    @classmethod
    def removeHandler(cls, handler: logging.Handler, logger_name: str = None) -> None:
        """
        Remove a handler from a specific logger
        
        Args:
            handler: The handler to remove
            logger_name: Name of the logger to remove handler from (root if None)
        """
        instance = cls._instance
        if not instance:
            raise RuntimeError("LogManager not initialized. Call LogManager(config) first.")
            
        instance._remove_handler_instance(handler, logger_name)
    
    def _remove_handler_instance(self, handler: logging.Handler, logger_name: str = None) -> None:
        """
        Remove a handler from a specific logger (instance method)
        
        Args:
            handler: The handler to remove
            logger_name: Name of the logger to remove handler from (root if None)
        """
        logger = logging.getLogger(logger_name if logger_name else '')
        if handler in logger.handlers:
            logger.removeHandler(handler)
    
    @classmethod
    def getHandlers(cls, logger_name: str = None) -> List[logging.Handler]:
        """
        Get all handlers for a specific logger
        
        Args:
            logger_name: Name of the logger to get handlers from (root if None)
            
        Returns:
            List of handlers attached to the logger
        """
        instance = cls._instance
        if not instance:
            raise RuntimeError("LogManager not initialized. Call LogManager(config) first.")
            
        return instance._get_handlers_instance(logger_name)
    
    def _get_handlers_instance(self, logger_name: str = None) -> List[logging.Handler]:
        """
        Get all handlers for a specific logger (instance method)
        
        Args:
            logger_name: Name of the logger to get handlers from (root if None)
            
        Returns:
            List of handlers attached to the logger
        """
        logger = logging.getLogger(logger_name if logger_name else '')
        return logger.handlers
    
    @classmethod
    def clearHandlers(cls, logger_name: str = None) -> None:
        """
        Remove all handlers from a specific logger
        
        Args:
            logger_name: Name of the logger to clear handlers from (root if None)
        """
        instance = cls._instance
        if not instance:
            raise RuntimeError("LogManager not initialized. Call LogManager(config) first.")
            
        instance._clear_handlers_instance(logger_name)
    
    def _clear_handlers_instance(self, logger_name: str = None) -> None:
        """
        Remove all handlers from a specific logger (instance method)
        
        Args:
            logger_name: Name of the logger to clear handlers from (root if None)
        """
        logger = logging.getLogger(logger_name if logger_name else '')
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    @classmethod
    def createConsoleHandler(cls, level: Union[str, int] = None, formatter: logging.Formatter = None) -> logging.Handler:
        """
        Create a console handler with optional level and formatter
        
        Args:
            level: Log level for the handler
            formatter: Formatter for the handler
            
        Returns:
            Configured console handler
        """
        instance = cls._instance
        if not instance:
            raise RuntimeError("LogManager not initialized. Call LogManager(config) first.")
            
        return instance._create_console_handler(level, formatter)
    
    def _create_console_handler(self, level: Union[str, int] = None, formatter: logging.Formatter = None) -> logging.Handler:
        """
        Create a console handler with optional level and formatter (instance method)
        
        Args:
            level: Log level for the handler
            formatter: Formatter for the handler
            
        Returns:
            Configured console handler
        """
        handler = logging.StreamHandler(sys.stdout)
        
        if level is not None:
            numeric_level = self._get_log_level(level)
            handler.setLevel(numeric_level)
        else:
            handler.setLevel(self.default_level)
            
        if formatter is not None:
            handler.setFormatter(formatter)
        else:
            handler.setFormatter(logging.Formatter(self.format_str))
            
        return handler
    
    @classmethod
    def createFileHandler(cls, filename: str, level: Union[str, int] = None, 
                         formatter: logging.Formatter = None, 
                         rotate: bool = True, max_bytes: int = None, 
                         backup_count: int = None) -> logging.Handler:
        """
        Create a file handler with optional level, formatter and rotation settings
        
        Args:
            filename: Path to log file
            level: Log level for the handler
            formatter: Formatter for the handler
            rotate: Whether to use rotating file handler
            max_bytes: Maximum file size before rotation (for RotatingFileHandler)
            backup_count: Number of backup files to keep
            
        Returns:
            Configured file handler
        """
        instance = cls._instance
        if not instance:
            raise RuntimeError("LogManager not initialized. Call LogManager(config) first.")
            
        return instance._create_custom_file_handler(filename, level, formatter, rotate, max_bytes, backup_count)
    
    def _create_custom_file_handler(self, filename: str, level: Union[str, int] = None, 
                                  formatter: logging.Formatter = None, 
                                  rotate: bool = True, max_bytes: int = None, 
                                  backup_count: int = None) -> logging.Handler:
        """
        Create a file handler with optional level, formatter and rotation settings (instance method)
        
        Args:
            filename: Path to log file
            level: Log level for the handler
            formatter: Formatter for the handler
            rotate: Whether to use rotating file handler
            max_bytes: Maximum file size before rotation (for RotatingFileHandler)
            backup_count: Number of backup files to keep
            
        Returns:
            Configured file handler
        """
        log_path = Path(filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure rotation settings
        _max_bytes = max_bytes if max_bytes is not None else self.max_file_size
        _backup_count = backup_count if backup_count is not None else self.backup_count
        
        # Create appropriate handler
        if rotate and self.rotation_strategy == 'size':
            handler = RotatingFileHandler(
                log_path,
                maxBytes=_max_bytes,
                backupCount=_backup_count,
                encoding="utf-8"
            )
        elif rotate and self.rotation_strategy == 'time':
            handler = TimedRotatingFileHandler(
                log_path,
                when="midnight",
                interval=1,
                backupCount=_backup_count,
                encoding="utf-8"
            )
        else:
            handler = logging.FileHandler(log_path, encoding="utf-8")
            
        # Set level
        if level is not None:
            numeric_level = self._get_log_level(level)
            handler.setLevel(numeric_level)
        else:
            handler.setLevel(self.default_level)
            
        # Set formatter
        if formatter is not None:
            handler.setFormatter(formatter)
        else:
            handler.setFormatter(logging.Formatter(self.format_str))
            
        return handler