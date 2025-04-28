# src/common/log_manager.py - Simplified logging handler

import logging
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import gzip
import shutil
from collections import deque


class SimpleBufferedHandler(logging.Handler):
    """Simplified buffered log handler without threading or locks"""
    
    def __init__(self, handler, buffer_size=1000, flush_interval=None):
        """Initialize the handler
        
        Args:
            handler: The actual handler that writes logs
            buffer_size: Maximum buffer record count
            flush_interval: Ignored, kept for compatibility
        """
        super().__init__()
        self.handler = handler
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)  # Use deque instead of list for auto size limit
        self.flush_count = 0
        
    def emit(self, record):
        """Buffer log records"""
        try:
            # Add record to buffer
            self.buffer.append(record)
            self.flush_count += 1
            
            # Flush when buffer reaches threshold
            if self.flush_count >= min(100, self.buffer_size // 10):
                self.flush()
                self.flush_count = 0
        except Exception:
            self.handleError(record)
    
    def flush(self):
        """Flush buffer to actual handler"""
        if not self.buffer:
            return
            
        # Process all records in buffer
        current_buffer = list(self.buffer)
        self.buffer.clear()
        
        # Direct write, no threading needed
        for record in current_buffer:
            try:
                self.handler.emit(record)
            except Exception:
                self.handleError(record)
    
    def close(self):
        """Close handler and flush remaining records"""
        self.flush()
        if hasattr(self.handler, 'close'):
            self.handler.close()
        super().close()


class SimpleCompressedRotatingFileHandler(RotatingFileHandler):
    """Simplified log rotating handler with compression"""
    
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, 
                 encoding=None, delay=False, compress=True):
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.compress = compress
        
    def doRollover(self):
        """Perform log rotation and compress old files"""
        # Perform standard rotation
        super().doRollover()
        
        # Compress old log files if enabled
        if self.compress:
            for i in range(1, self.backupCount + 1):
                log_file = f"{self.baseFilename}.{i}"
                compressed_file = f"{log_file}.gz"
                
                if os.path.exists(log_file):
                    try:
                        with open(log_file, 'rb') as f_in:
                            with gzip.open(compressed_file, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        # Delete original after compression
                        os.remove(log_file)
                    except Exception:
                        # If compression fails, continue execution
                        pass


class JsonFormatter(logging.Formatter):
    """Format logs as JSON"""
    
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


class LogManager:
    """Simplified log manager that avoids complex threading"""
    
    # Singleton instance
    _instance = None
    _initialized = False
    
    # Standard log levels
    LOG_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    # --- Singleton pattern methods ---
    
    @classmethod
    def initialize(cls, config=None):
        """Initialize the singleton log manager"""
        if cls._instance is None:
            cls._instance = LogManager()
            
        if not cls._initialized and config:
            cls._instance._setup_logging(config)
            cls._initialized = True
            
        return cls._instance
    
    @classmethod
    def is_initialized(cls):
        """Check if already initialized"""
        return cls._initialized
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance, creating basic instance if needed"""
        if cls._instance is None:
            cls._instance = LogManager()
            if not cls._initialized:
                cls._instance._setup_basic_logging()
        return cls._instance
    
    @classmethod
    def get_logger(cls, name):
        """Get a logger for the given name - main API for clients"""
        # Get the singleton and configure it if needed
        instance = cls.get_instance()
        # Call the private implementation method - no recursion
        return instance._configure_logger(name)
    
    @classmethod
    def set_level(cls, level, logger_name=None):
        """Set log level for a logger or all loggers"""
        instance = cls.get_instance()
        instance._set_level_impl(level, logger_name)
    
    # --- Implementation methods ---
    
    def __init__(self):
        """Private constructor, only called by get_instance()"""
        # Basic defaults - will be overridden by setup
        self.configured_loggers = set()
        self.base_path = Path(f"./logs")
        self.format_str = '%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s'
        self.date_format = '%Y-%m-%d %H:%M:%S'
        self.default_level = logging.INFO
        self.log_to_console = True
        self.log_to_file = False
        self.categories = {}
        self.module_category_map = {}
    
    def _setup_basic_logging(self):
        """Basic console logging when no config is available"""
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(self.format_str)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Set basic properties
        self.base_path.mkdir(exist_ok=True)
        self.categories = {
            "system": {"level": "INFO", "modules": []},
            "app": {"level": "INFO", "modules": []}
        }
        self.configured_loggers = set(["root"])
    
    def _setup_logging(self, config):
        """Configure logging from a config object"""
        try:
            from common.config import ConfigManager
            if not isinstance(config, ConfigManager):
                raise ValueError("Config must be a ConfigManager instance")
                
            self.config = config
            
            # Get current date
            if not self.date_str:
                self.date_str = datetime.now().strftime('%Y-%m-%d')
            self.base_path = Path(config.get("logging", "base_path", default=f"./logs"))
            
            # Ensure base directory exists
            self.base_path.mkdir(parents=True, exist_ok=True)
            
            # Parse categories and create module map
            self.categories = self._parse_categories(config)
            self.module_category_map = self._create_module_map()
            
            # Reset configured loggers
            self.configured_loggers = set()
            
            # Get formatting settings
            self.format_str = config.get("logging", "format", 
                default="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s")
            self.detailed_format = config.get("logging", "detailed_format", 
                default="%(asctime)s | %(levelname)-8s | %(name)-25s | [%(filename)s:%(lineno)d] | %(message)s")
            self.date_format = config.get("logging", "date_format", default='%Y-%m-%d %H:%M:%S')
            
            # Default level
            self.default_level = self._get_log_level(config.get("logging", "level", default="INFO"))
            
            # Log rotation settings
            self.rotation_strategy = config.get("logging", "rotation", "strategy", default="size")
            self.max_file_size = int(config.get("logging", "rotation", "max_file_size", default=10485760))
            self.backup_count = int(config.get("logging", "rotation", "backup_count", default=5))
            
            # Console logging settings
            self.log_to_console = config.get("logging", "handlers", "console", "enabled", default=True)
            self.console_colors = config.get("logging", "handlers", "console", "colors", default=True)
            
            # File logging settings
            self.log_to_file = config.get("logging", "handlers", "file", "enabled", default=True)
            self.use_compression = config.get("logging", "handlers", "file", "compression", default="gz") != ""
            
            # Whether to use buffering
            self.use_buffer = config.get("logging", "handlers", "file", "buffer", "enabled", default=True)
            self.buffer_size = int(config.get("logging", "handlers", "file", "buffer", "capacity", default=1000))
            
            # Initialize root logger
            self._setup_root_logger()
            
            # Set up combined log if enabled
            if config.get("logging", "combined_log", default=True):
                self._setup_combined_log()
                
        except Exception as e:
            # Fall back to basic setup on failure
            self._setup_basic_logging()
            print(f"Log initialization failed: {str(e)}")
            
    def _parse_categories(self, config):
        """Define consolidated log categories with module mappings"""
        # Default categories with detailed module mappings
        default_categories = {
            "system": {
                "level": config.get("logging", "level", default="INFO"),
                "modules": [
                    "core", "main", "launcher", "cli", "cli_runner", "common", "common.config", 
                    "common.helpers", "common.abstract_factory", "common.async_executor",
                    "utils"
                ]
            },
            "strategy": {
                "level": config.get("logging", "level", default="INFO"),
                "modules": [
                    "strategy", "strategy.base", "strategy.factor_lib", "strategy.registry",
                    "strategy.implementations", "strategy.implementations.dual_ma",
                    "backtest", "backtest.engine", "backtest.performance"
                ]
            },
            "data": {
                "level": config.get("logging", "level", default="INFO"),
                "modules": [
                    "datasource", "datasource.datasources", "datasource.downloader", 
                    "datasource.integrity", "datasource.manager", "datasource.processor"
                ]
            },
            "exchange": {
                "level": config.get("logging", "level", default="WARNING"),
                "modules": [
                    "exchange", "exchange.adapters", "exchange.adapters.base", 
                    "exchange.adapters.binance"
                ]
            },
            "portfolio": {
                "level": config.get("logging", "level", default="INFO"),
                "modules": [
                    "portfolio", "portfolio.manager", "portfolio.portfolio",
                    "portfolio.assets", "portfolio.assets.base", "portfolio.assets.bond",
                    "risk", "risk.manager", "risk.rules", "risk.rules.basic_rules"
                ]
            },
            "trading": {
                "level": config.get("logging", "level", default="INFO"),
                "modules": [
                    "trading", "trading.execution", "trading.execution.manager", 
                    "trading.execution.order", "trading.modes", "trading.modes.backtest"
                ]
            }
        }
        
        try:
            # Use categories from config to override if available
            config_categories = config.get("logging", "categories", default={})
            
            for category, settings in config_categories.items():
                if category in default_categories:
                    # Merge with default settings
                    level = settings.get("level", default_categories[category]["level"])
                    modules = settings.get("modules", default_categories[category]["modules"])
                    
                    default_categories[category]["level"] = level
                    default_categories[category]["modules"] = modules
                else:
                    # Add new category
                    default_categories[category] = {
                        "level": settings.get("level", config.get("logging", "level", default="INFO")),
                        "modules": settings.get("modules", [])
                    }
        except Exception:
            # Use defaults on parsing failure
            pass
        
        return default_categories
    
    def _create_module_map(self):
        """Create mapping from module names to log categories"""
        module_map = {}
        
        # Map modules to categories
        for category, settings in self.categories.items():
            for module_name in settings["modules"]:
                # Add module
                module_map[module_name] = category
                # Add fully qualified version
                module_map[f"src.{module_name}"] = category
                # Add versions with trailing dot (for submodules)
                module_map[f"{module_name}."] = category
                module_map[f"src.{module_name}."] = category
        
        return module_map

    def _get_category_for_module(self, module_name):
        """Determine the appropriate category for a module"""
        # Direct match
        if module_name in self.module_category_map:
            return self.module_category_map[module_name]
        
        # Check prefix match
        for prefix, category in self.module_category_map.items():
            if prefix.endswith('.') and module_name.startswith(prefix[:-1]):
                return category
        
        # Try matching by parts
        parts = module_name.split('.')
        while parts:
            # Try current prefix
            prefix = '.'.join(parts)
            if prefix in self.module_category_map:
                return self.module_category_map[prefix]
            # Remove last part and try again
            parts.pop()
        
        # First part may indicate general category
        if "." in module_name:
            first_part = module_name.split('.')[0]
            if first_part in self.categories:
                return first_part
                
        # Unknown modules default to system
        return "system"
    
    def _get_log_level(self, level: Union[str, int]) -> int:
        """Convert string log level to numeric value"""
        if isinstance(level, int):
            return level
        
        level_upper = level.upper()
        if level_upper in self.LOG_LEVELS:
            return self.LOG_LEVELS[level_upper]
        
        return logging.INFO
    
    def _setup_root_logger(self):
        """Configure the root logger"""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.default_level)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Console handler
        if self.log_to_console:
            console_level = self._get_log_level(
                self.config.get("logging", "handlers", "console", "level", default="INFO")
            )
            console_handler = logging.StreamHandler()
            console_handler.setLevel(console_level)
            
            # Use colored output if enabled
            if self.console_colors:
                formatter = self._get_colored_formatter()
            else:
                formatter = logging.Formatter(self.format_str, datefmt=self.date_format)
            
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # Add default file handlers for root logger
        if self.log_to_file:
            # Main system log
            main_log_path = self.base_path / "trading_system.log"
            file_handler = self._create_file_handler(main_log_path)
            file_handler.setLevel(self.default_level)
            root_logger.addHandler(file_handler)
            
            # Also create error log
            error_log_path = self.base_path / "errors.log"
            error_handler = self._create_file_handler(error_log_path)
            error_handler.setLevel(logging.ERROR)
            root_logger.addHandler(error_handler)
        
        # Mark root logger as configured
        self.configured_loggers.add("root")
    
    def _setup_combined_log(self):
        """Set up combined log file"""
        # Main combined log
        combined_log_path = self.base_path / "combined.log"
        handler = self._create_file_handler(combined_log_path)
        handler.setFormatter(logging.Formatter(self.format_str, datefmt=self.date_format))
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        
        # Also create daily log
        daily_log_path = self.base_path / f"day_{self.date_str}.log"
        daily_handler = logging.FileHandler(daily_log_path)
        daily_handler.setFormatter(logging.Formatter(self.format_str, datefmt=self.date_format))
        root_logger.addHandler(daily_handler)
    
    def _get_colored_formatter(self):
        """Get formatter with ANSI color codes"""
        # Define colors
        RESET = "\033[0m"
        BOLD = "\033[1m"
        RED = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        CYAN = "\033[36m"
        
        # Level colors
        COLORS = {
            'DEBUG': BLUE,
            'INFO': GREEN,
            'WARNING': YELLOW,
            'ERROR': RED,
            'CRITICAL': BOLD + RED
        }
        
        class ColoredFormatter(logging.Formatter):
            def format(self, record):
                levelname = record.levelname
                if levelname in COLORS:
                    record.levelname = f"{COLORS[levelname]}{levelname}{RESET}"
                    record.name = f"{CYAN}{record.name}{RESET}"
                    record.msg = f"{record.msg}{RESET}"
                return super().format(record)
        
        return ColoredFormatter(self.format_str, datefmt=self.date_format)
    
    def _create_file_handler(self, log_path, level=None):
        """Create appropriate handler based on rotation settings"""
        try:
            # Ensure directory exists
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create appropriate handler based on rotation settings
            if self.rotation_strategy == "size":
                if self.use_compression:
                    handler = SimpleCompressedRotatingFileHandler(
                        log_path,
                        maxBytes=self.max_file_size,
                        backupCount=self.backup_count,
                        encoding="utf-8",
                        compress=True
                    )
                else:
                    handler = logging.handlers.RotatingFileHandler(
                        log_path,
                        maxBytes=self.max_file_size,
                        backupCount=self.backup_count,
                        encoding="utf-8"
                    )
            elif self.rotation_strategy == "time":
                handler = logging.handlers.TimedRotatingFileHandler(
                    log_path,
                    when="midnight",
                    interval=1,
                    backupCount=self.backup_count,
                    encoding="utf-8"
                )
            else:
                handler = logging.FileHandler(log_path, encoding="utf-8")
            
            # Set log level
            if level:
                handler.setLevel(self._get_log_level(level))
            
            # Use consistent format
            formatter = logging.Formatter(self.format_str, datefmt=self.date_format)
            handler.setFormatter(formatter)
            
            # If buffering is enabled, wrap it with simple buffer handler
            if hasattr(self, 'use_buffer') and self.use_buffer:
                buffer_size = getattr(self, 'buffer_size', 1000)
                handler = SimpleBufferedHandler(
                    handler=handler,
                    buffer_size=buffer_size
                )
            
            return handler
        except Exception as e:
            print(f"Failed to create file handler {log_path}: {str(e)}")
            # Create fallback handler that only prints to console
            handler = logging.StreamHandler()
            if level:
                handler.setLevel(self._get_log_level(level))
            handler.setFormatter(logging.Formatter(self.format_str, datefmt=self.date_format))
            return handler
    
    def _configure_logger(self, name):
        """Internal method to configure a logger"""
        try:
            # Return existing logger if already created
            if name in self.configured_loggers:
                return logging.getLogger(name)
            
            # Create new logger
            logger = logging.getLogger(name)
            
            # Determine category
            category = self._get_category_for_module(name)
            
            # Get appropriate log level
            category_config = self.categories.get(category, {"level": "INFO"})
            level = category_config.get("level", self.default_level)
            logger.setLevel(self._get_log_level(level))
            
            # Don't add handlers if this is a non-root logger with propagate=True
            # This avoids duplicate log entries
            if not logger.handlers and not (name != "" and logger.propagate):
                # Get correct log file path - simplified to just category logs
                log_path = self.base_path / f"{category}.log"
                
                # Check if specific file mapping in config
                if hasattr(self, 'config'):
                    files_config = self.config.get("logging", "files", default={})
                    for logger_name, logger_config in files_config.items():
                        if logger_name == category:
                            config_path = logger_config.get("path")
                            if config_path:
                                log_path = self.base_path / config_path
                            break
                
                # If file logging enabled, create and add file handler
                if self.log_to_file:
                    file_handler = self._create_file_handler(log_path, level)
                    logger.addHandler(file_handler)
                
                # Error level also logs to error log file
                if self.log_to_file:
                    error_path = self.base_path / "errors.log"
                    error_handler = self._create_file_handler(error_path, logging.ERROR)
                    logger.addHandler(error_handler)
                
                # Add console handler if enabled and not inherited
                if self.log_to_console and name != "":
                    console_level = self._get_log_level(
                        self.config.get("logging", "handlers", "console", "level", default=level) 
                        if hasattr(self, 'config') else level
                    )
                    console_handler = logging.StreamHandler()
                    console_handler.setLevel(console_level)
                    
                    if hasattr(self, 'console_colors') and self.console_colors:
                        formatter = self._get_colored_formatter()
                    else:
                        formatter = logging.Formatter(self.format_str, datefmt=self.date_format)
                    
                    console_handler.setFormatter(formatter)
                    logger.addHandler(console_handler)
            
            # Remember this logger
            self.configured_loggers.add(name)
            
            return logger
            
        except Exception as e:
            # Create basic fallback logger that logs only to console
            print(f"Error creating logger '{name}': {str(e)}")
            fallback_logger = logging.getLogger(name)
            
            # Only add handler if none exist
            if not fallback_logger.handlers:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                formatter = logging.Formatter(self.format_str)
                console_handler.setFormatter(formatter)
                fallback_logger.addHandler(console_handler)
                
            return fallback_logger
    
    def add_json_handler(self, logger_name, path=None):
        """Add JSON format handler to a logger"""
        try:
            logger = logging.getLogger(logger_name)
            
            # Generate path if not provided
            if not path:
                category = self._get_category_for_module(logger_name)
                path = self.base_path / f"{category}.json"
            
            # Create handler
            handler = logging.handlers.RotatingFileHandler(
                path,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding="utf-8"
            )
            
            # Set formatter
            handler.setFormatter(JsonFormatter())
            
            # Add handler
            logger.addHandler(handler)
            
            return logger
        except Exception as e:
            print(f"Failed to add JSON handler to logger '{logger_name}': {str(e)}")
            return logging.getLogger(logger_name)
    
    def _set_level_impl(self, level, logger_name=None):
        """Implementation of set_level"""
        try:
            numeric_level = self._get_log_level(level)
            
            if logger_name:
                # Update single logger
                logger = logging.getLogger(logger_name)
                logger.setLevel(numeric_level)
                
                # Update its handlers
                for handler in logger.handlers:
                    handler.setLevel(numeric_level)
            else:
                # Update all loggers
                for name in self.configured_loggers:
                    logger_name_to_use = name if name != "root" else ""
                    logger = logging.getLogger(logger_name_to_use)
                    logger.setLevel(numeric_level)
                    
                    for handler in logger.handlers:
                        handler.setLevel(numeric_level)
                
                # Update default level
                self.default_level = numeric_level
        except Exception as e:
            print(f"Failed to set log level to {level}: {str(e)}")


class LogInitializer:
    """Simplified log initializer using LogManager"""
    
    def __init__(self, config):
        """Initialize with config"""
        self.config = config
        
        # Initialize log manager
        self.log_manager = LogManager.initialize(config)
    
    def initialize(self):
        """Initialize logging system with simplified structure"""
        # If already initialized, we're done
        if LogManager.is_initialized():
            return self
        
        # Log initialization start
        system_logger = LogManager.get_logger("system")
        system_logger.info("Log system initialization started")
        
        # Set up explicit loggers based on files.* config
        files_config = self.config.get("logging", "files", default={})
        for logger_name, logger_config in files_config.items():
            path = logger_config.get("path", None)
            level = logger_config.get("level", "INFO")
            
            if path:
                # Create logger
                logger = LogManager.get_logger(logger_name)
                
                # Set level
                level_value = self.log_manager._get_log_level(level)
                logger.setLevel(level_value)
                
                # Add specific handler for this file
                log_path = Path(self.log_manager.base_path) / path
                handler = self.log_manager._create_file_handler(log_path)
                handler.setLevel(level_value)
                logger.addHandler(handler)
        
        # Enable syslog if configured
        if self.config.get("logging", "syslog", "enabled", default=False):
            self._setup_syslog()
        
        # Enable summary logging if configured
        if self.config.get("logging", "summary", "enabled", default=False):
            self._setup_summary_logger()
        
        # Log initialization complete
        system_logger.info("Log system initialization complete")
        
        return self
    
    def _setup_syslog(self):
        """Set up syslog handler if configured"""
        try:
            from logging.handlers import SysLogHandler
            
            syslog_host = self.config.get("logging", "syslog", "host", default="localhost")
            syslog_port = int(self.config.get("logging", "syslog", "port", default=514))
            syslog_facility = self.config.get("logging", "syslog", "facility", default="local0")
            
            # Map facility string to SysLogHandler constants
            facility_map = {
                "auth": SysLogHandler.LOG_AUTH,
                "authpriv": SysLogHandler.LOG_AUTHPRIV,
                "cron": SysLogHandler.LOG_CRON,
                "daemon": SysLogHandler.LOG_DAEMON,
                "kern": SysLogHandler.LOG_KERN,
                "local0": SysLogHandler.LOG_LOCAL0,
                "local1": SysLogHandler.LOG_LOCAL1,
                "local2": SysLogHandler.LOG_LOCAL2,
                "local3": SysLogHandler.LOG_LOCAL3,
                "local4": SysLogHandler.LOG_LOCAL4,
                "local5": SysLogHandler.LOG_LOCAL5,
                "local6": SysLogHandler.LOG_LOCAL6,
                "local7": SysLogHandler.LOG_LOCAL7,
                "lpr": SysLogHandler.LOG_LPR,
                "mail": SysLogHandler.LOG_MAIL,
                "news": SysLogHandler.LOG_NEWS,
                "syslog": SysLogHandler.LOG_SYSLOG,
                "user": SysLogHandler.LOG_USER,
                "uucp": SysLogHandler.LOG_UUCP
            }
            
            facility = facility_map.get(syslog_facility.lower(), SysLogHandler.LOG_USER)
            
            # Create syslog handler
            syslog_handler = SysLogHandler(address=(syslog_host, syslog_port), facility=facility)
            
            # Format for syslog (no timestamps since syslog adds those)
            formatter = logging.Formatter('%(name)s[%(process)d]: %(levelname)s %(message)s')
            syslog_handler.setFormatter(formatter)
            
            # Add to root logger
            root_logger = logging.getLogger()
            root_logger.addHandler(syslog_handler)
            
            # Log successful setup
            root_logger.info(f"Syslog logging configured: {syslog_host}:{syslog_port} facility={syslog_facility}")
            
        except (ImportError, Exception) as e:
            # Log failure but continue
            logging.getLogger("system").warning(f"Failed to set up syslog handler: {str(e)}")
    
    def _setup_summary_logger(self):
        """Set up summary logging for important events"""
        try:
            summary_path = self.log_manager.base_path / "summary.log"
            
            # Create summary logger
            summary_logger = logging.getLogger("summary")
            summary_logger.setLevel(self.log_manager._get_log_level(
                self.config.get("logging", "summary", "level", default="WARNING")
            ))
            
            # Create summary handler
            summary_format = self.config.get("logging", "summary", "format", default="json")
            
            if summary_format.lower() == "json":
                handler = self.log_manager._create_file_handler(summary_path)
                handler.setFormatter(JsonFormatter())
            else:
                handler = self.log_manager._create_file_handler(summary_path)
                formatter = logging.Formatter(self.config.get("logging", "format", 
                    default="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"),
                    datefmt=self.config.get("logging", "date_format", default='%Y-%m-%d %H:%M:%S'))
                handler.setFormatter(formatter)
            
            summary_logger.addHandler(handler)
            
            # Mark as configured
            self.log_manager.configured_loggers.add("summary")
        except Exception as e:
            print(f"Failed to set up summary log: {str(e)}")
    
    def get_logger(self, module_name):
        """Get logger mapped to appropriate category"""
        return LogManager.get_logger(module_name)