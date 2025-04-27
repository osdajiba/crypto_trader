# src/common/log_manager.py

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
import threading
from concurrent.futures import ThreadPoolExecutor

from common.config import ConfigManager


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler that buffers records and processes them in a background thread"""
    
    def __init__(self, handler, buffer_size=1000, flush_interval=5, max_workers=2):
        """Initialize with the handler to wrap and buffer settings"""
        super().__init__()
        self.handler = handler
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.buffer = []
        self.lock = threading.Lock()
        self.timer = None
        self._is_closing = False
        self._schedule_flush()
    
    def emit(self, record):
        """Buffer a log record and flush if needed"""
        try:
            should_flush = False
            with self.lock:
                self.buffer.append(record)
                if len(self.buffer) >= self.buffer_size:
                    should_flush = True
            
            # Important: Call flush outside the lock to prevent deadlocks
            if should_flush:
                self.flush()
        except Exception:
            self.handleError(record)
    
    def _schedule_flush(self):
        """Schedule the next timed flush"""
        if not self._is_closing:
            self.timer = threading.Timer(self.flush_interval, self._timed_flush)
            self.timer.daemon = True
            self.timer.start()
    
    def _timed_flush(self):
        """Handle timed flush and reschedule"""
        try:
            self.flush()
        finally:
            # Always reschedule even if flush fails
            self._schedule_flush()
    
    def flush(self):
        """Flush the buffer to the actual handler"""
        buffer_copy = None
        with self.lock:
            if self.buffer:
                buffer_copy = self.buffer.copy()
                self.buffer = []
        
        if buffer_copy:
            self.executor.submit(self._process_buffer, buffer_copy)
    
    def _process_buffer(self, buffer):
        """Process a batch of records in background thread"""
        for record in buffer:
            try:
                self.handler.emit(record)
            except Exception:
                self.handleError(record)
    
    def close(self):
        """Clean up resources"""
        self._is_closing = True
        if self.timer:
            self.timer.cancel()
            self.timer = None
        self.flush()
        self.executor.shutdown(wait=False)  # Don't block on shutdown
        if hasattr(self.handler, 'close'):
            self.handler.close()
        super().close()


class CompressedRotatingFileHandler(RotatingFileHandler):
    """Enhanced rotating file handler that compresses rotated logs"""
    
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, 
                 encoding=None, delay=False, compress=True):
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.compress = compress
        self._lock = threading.Lock()  # Lock for thread safety during rotation
        
    def doRollover(self):
        with self._lock:
            # First perform normal rotation
            super().doRollover()
            
            # Then compress the rotated file if enabled
            if self.compress:
                for i in range(1, self.backupCount + 1):
                    log_file = f"{self.baseFilename}.{i}"
                    compressed_file = f"{log_file}.gz"
                    
                    if os.path.exists(log_file):
                        with open(log_file, 'rb') as f_in:
                            with gzip.open(compressed_file, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        # Remove original file after compression
                        os.remove(log_file)


class JsonFormatter(logging.Formatter):
    """Formatter for JSON log output with consistent structure"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
            "path": record.pathname,
            "function": record.funcName,
            "thread": record.thread,
            "thread_name": record.threadName
        }
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        if hasattr(record, 'extra') and record.extra:
            log_entry["extra"] = record.extra
            
        return json.dumps(log_entry)


class LogManager:
    """Simplified log manager with cleaner organization"""
    
    _instance = None
    
    # Standard log levels
    LOG_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    def __new__(cls, config: Optional[ConfigManager] = None):
        """Create LogManager singleton"""
        if cls._instance is None:
            if not config:
                raise ValueError("Config must be provided for initial initialization")
            cls._instance = super(LogManager, cls).__new__(cls)
            cls._instance._init_logger(config)
        return cls._instance
    
    def _init_logger(self, config):
        """Initialize logger (called only on first instance creation)"""
        self.config = config
        self.base_path = Path(config.get("logging", "base_path", default="./logs"))
        
        # Get current date for directory structure
        now = datetime.now()
        self.date_str = now.strftime('%Y-%m-%d')
        
        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create simplified directory structure
        self._setup_directory_structure()
        
        # Parse categories from config
        self.categories = self._parse_categories()
        
        # Module to category mapping
        self.module_category_map = self._create_module_map()
        
        # Track initialized loggers
        self.configured_loggers = set()
        
        # Set formatter settings
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
        
        # Should we use async logging
        self.async_logging = config.get("logging", "async_write", default=False)
        
        # Initialize root logger
        self.setup_root_logger()
        
        # Set up combined log if enabled
        if config.get("logging", "combined_log", default=True):
            self._setup_combined_log()
        
        # Set up async logging if enabled
        if self.async_logging:
            self._setup_async_logging()
    
    def _setup_directory_structure(self):
        """Create a minimal directory structure for logs"""
        # Just ensure the base directory exists
        # We'll create subdirectories as needed when creating log files
        self.base_path.mkdir(exist_ok=True)

    def _parse_categories(self):
        """Define consolidated log categories with comprehensive module mapping"""
        # Default categories with comprehensive module mapping
        default_categories = {
            "system": {
                "level": self.config.get("logging", "level", default="INFO"),
                "modules": [
                    "core", "main", "launcher", "cli", "cli_runner", "common", "common.config", 
                    "common.helpers", "common.abstract_factory", "common.async_executor",
                    "utils"
                ]
            },
            "strategy": {
                "level": self.config.get("logging", "level", default="INFO"),
                "modules": [
                    "strategy", "strategy.base", "strategy.factor_lib", "strategy.registry",
                    "strategy.implementations", "strategy.implementations.dual_ma",
                    "strategy.implementations.multi_factors", "strategy.implementations.nn",
                    "backtest", "backtest.engine", "backtest.performance"
                ]
            },
            "data": {
                "level": self.config.get("logging", "level", default="INFO"),
                "modules": [
                    "datasource", "datasource.datasources", "datasource.downloader", 
                    "datasource.integrity", "datasource.manager", "datasource.processor"
                ]
            },
            "exchange": {
                "level": self.config.get("logging", "level", default="WARNING"),
                "modules": [
                    "exchange", "exchange.adapters", "exchange.adapters.base", 
                    "exchange.adapters.binance"
                ]
            },
            "portfolio": {
                "level": self.config.get("logging", "level", default="INFO"),
                "modules": [
                    "portfolio", "portfolio.manager", "portfolio.portfolio",
                    "portfolio.assets", "portfolio.assets.base", "portfolio.assets.bond",
                    "portfolio.assets.fund", "portfolio.assets.future", "portfolio.assets.option",
                    "portfolio.assets.spot",
                    "risk", "risk.manager", "risk.rules", "risk.rules.basic_rules"
                ]
            },
            "trading": {
                "level": self.config.get("logging", "level", default="INFO"),
                "modules": [
                    "trading", "trading.execution", "trading.execution.manager", 
                    "trading.execution.order", "trading.modes", "trading.modes.backtest", 
                    "trading.modes.base", "trading.modes.live", "trading.modes.paper"
                ]
            }
        }
        
        # Override with configured categories if available
        config_categories = self.config.get("logging", "categories", default={})
        
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
                    "level": settings.get("level", self.config.get("logging", "level", default="INFO")),
                    "modules": settings.get("modules", [])
                }
        
        return default_categories
    
    def _create_module_map(self):
        """Create mapping from module names to log categories"""
        module_map = {}
        
        # Map modules to categories
        for category, settings in self.categories.items():
            for module_name in settings["modules"]:
                # Add the module
                module_map[module_name] = category
                # Also add qualified version
                module_map[f"src.{module_name}"] = category
                # Add with trailing dot (for submodules)
                module_map[f"{module_name}."] = category
                module_map[f"src.{module_name}."] = category
        
        return module_map

    def _get_category_for_module(self, module_name):
        """Determine the appropriate category for a module"""
        # Direct match
        if module_name in self.module_category_map:
            return self.module_category_map[module_name]
        
        # Check for prefix match
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
        
        # Default to app for unknown modules
        return "app"
    
    def _get_log_level(self, level: Union[str, int]) -> int:
        """Convert string log level to numeric value"""
        if isinstance(level, int):
            return level
        
        level_upper = level.upper()
        if level_upper in self.LOG_LEVELS:
            return self.LOG_LEVELS[level_upper]
        
        return logging.INFO
    
    def setup_root_logger(self):
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

        # Add a default file handler for root logger
        if self.log_to_file:
            # Main system log
            main_log_path = self.base_path / "trading_system.log"
            file_handler = self._create_file_handler(main_log_path)
            file_handler.setLevel(self.default_level)
            root_logger.addHandler(file_handler)
            
            # Also create an error log
            error_log_path = self.base_path / "errors.log"
            error_handler = self._create_file_handler(error_log_path)
            error_handler.setLevel(logging.ERROR)
            root_logger.addHandler(error_handler)
        
        # Mark root logger as configured
        self.configured_loggers.add("root")
    
    def _setup_combined_log(self):
        """Set up a combined log file"""
        # Main combined log
        combined_log_path = self.base_path / "combined.log"
        handler = self._create_file_handler(combined_log_path)
        handler.setFormatter(logging.Formatter(self.format_str, datefmt=self.date_format))
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        
        # Also create a daily log
        daily_log_path = self.base_path / f"{self.date_str}.log"
        daily_handler = logging.FileHandler(daily_log_path)
        daily_handler.setFormatter(logging.Formatter(self.format_str, datefmt=self.date_format))
        root_logger.addHandler(daily_handler)
    
    def _setup_async_logging(self):
        """Set up asynchronous logging"""
        try:
            # Get buffer settings from config
            buffer_size = int(self.config.get("logging", "handlers", "file", "buffer", "capacity", default=1000))
            flush_interval = int(self.config.get("logging", "handlers", "file", "buffer", "flush_interval", default=5))
            
            # Track modified handlers to avoid applying multiple times
            processed_handlers = set()
            
            # Apply async wrapper to file handlers
            for logger_name in list(self.configured_loggers):
                logger = logging.getLogger(logger_name if logger_name != 'root' else '')
                
                # Create new handler list to avoid modifying during iteration
                new_handlers = []
                
                for handler in logger.handlers:
                    # Skip handlers we've already processed
                    if id(handler) in processed_handlers:
                        new_handlers.append(handler)
                        continue
                    
                    # Only wrap file handlers
                    if isinstance(handler, (logging.FileHandler, RotatingFileHandler, TimedRotatingFileHandler)):
                        processed_handlers.add(id(handler))
                        
                        # Create async wrapper
                        async_handler = AsyncLogHandler(
                            handler=handler,
                            buffer_size=buffer_size,
                            flush_interval=flush_interval
                        )
                        async_handler.setLevel(handler.level)
                        async_handler.setFormatter(handler.formatter)
                        new_handlers.append(async_handler)
                    else:
                        new_handlers.append(handler)
                
                # Replace all handlers
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                
                for handler in new_handlers:
                    logger.addHandler(handler)
            
            # Log successful setup
            logging.getLogger("system").info("Async logging enabled")
            
        except Exception as e:
            print(f"Warning: Failed to set up async logging: {str(e)}")
    
    def _get_colored_formatter(self):
        """Get a formatter with ANSI color codes for console output"""
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
        """Create the appropriate handler based on rotation settings"""
        # Ensure directory exists
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create the appropriate handler based on rotation settings
        if self.rotation_strategy == "size":
            if self.use_compression:
                handler = CompressedRotatingFileHandler(
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
        
        return handler
    
    def get_logger(self, name):
        """Get a configured logger for a module"""
        # Return existing logger if already created
        if name in self.configured_loggers:
            return logging.getLogger(name)
        
        # Create a new logger
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
            # Get the correct log file path - simplify to just category logs
            log_path = self.base_path / f"{category}.log"
            
            # Check if there's a specific file mapping in the config
            files_config = self.config.get("logging", "files", default={})
            for logger_name, logger_config in files_config.items():
                if logger_name == category:
                    config_path = logger_config.get("path")
                    if config_path:
                        log_path = self.base_path / config_path
                    break
            
            # Create and add the file handler
            if self.log_to_file:
                file_handler = self._create_file_handler(log_path, level)
                logger.addHandler(file_handler)
            
            # For ERROR level, also log to the error log file
            if self.log_to_file:
                error_path = self.base_path / "errors.log"
                error_handler = self._create_file_handler(error_path, logging.ERROR)
                logger.addHandler(error_handler)
            
            # Add console handler if enabled and not already inherited
            if self.log_to_console and name != "":
                console_level = self._get_log_level(
                    self.config.get("logging", "handlers", "console", "level", default=level)
                )
                console_handler = logging.StreamHandler()
                console_handler.setLevel(console_level)
                
                if self.console_colors:
                    formatter = self._get_colored_formatter()
                else:
                    formatter = logging.Formatter(self.format_str, datefmt=self.date_format)
                
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
        
        # Remember this logger
        self.configured_loggers.add(name)
        
        return logger
    
    def add_json_handler(self, logger_name, path=None):
        """Add a JSON format handler to a logger"""
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
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance"""
        if cls._instance is None:
            raise RuntimeError("LogManager not initialized")
        return cls._instance
    
    @classmethod
    def get_logger(cls, name):
        """Class method to get a logger"""
        instance = cls.get_instance()
        return instance.get_logger(name)
    
    def set_level(self, level, logger_name=None):
        """Set log level for a logger or all loggers"""
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
    
    @classmethod
    def set_level(cls, level, logger_name=None):
        """Class method to set log level"""
        instance = cls.get_instance()
        instance.set_level(level, logger_name)


class LogInitializer:
    """
    Simplified Log Initializer that uses the new LogManager
    """
    
    def __init__(self, config: ConfigManager):
        """Initialize with configuration"""
        self.config = config
        
        # Create log manager
        self.log_manager = LogManager(config)
    
    def initialize(self):
        """Initialize the logging system with simplified structure"""
        # Log initialization started
        system_logger = self.log_manager.get_logger("system")
        system_logger.info("Logging system initialization started")
        
        # Set up explicit loggers for key components based on files.* config
        files_config = self.config.get("logging", "files", default={})
        for logger_name, logger_config in files_config.items():
            path = logger_config.get("path", None)
            level = logger_config.get("level", "INFO")
            
            if path:
                # Create the logger
                logger = self.log_manager.get_logger(logger_name)
                
                # Set the level
                logger.setLevel(self.log_manager._get_log_level(level))
                
                # Add a specific handler for this file
                log_path = Path(self.log_manager.base_path) / path
                handler = self.log_manager._create_file_handler(log_path)
                handler.setLevel(self.log_manager._get_log_level(level))
                logger.addHandler(handler)
        
        # Enable syslog if configured
        if self.config.get("logging", "syslog", "enabled", default=False):
            self._setup_syslog()
        
        # Enable summary logging if configured
        if self.config.get("logging", "summary", "enabled", default=False):
            self._setup_summary_logger()
        
        # Log initialization completed
        system_logger.info("Logging system initialized with simplified organization")
        
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
        summary_path = self.log_manager.base_path / "summary.log"
        
        # Create summary logger
        summary_logger = logging.getLogger("summary")
        summary_logger.setLevel(self.log_manager._get_log_level(
            self.config.get("logging", "summary", "level", default="WARNING")
        ))
        
        # Create handler for summary
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
    
    def get_logger(self, module_name):
        """Get a logger mapped to the appropriate category"""
        return self.log_manager.get_logger(module_name)