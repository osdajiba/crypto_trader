# src/common/logging.py

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
    """
    Enhanced rotating file handler that compresses rotated logs
    """
    
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, 
                 encoding=None, delay=False, compress=True):
        """Initialize with compression option"""
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.compress = compress
        self._lock = threading.Lock()  # Lock for thread safety during rotation
        
    def doRollover(self):
        """
        Override doRollover to add compression
        """
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
    """
    Formatter for JSON log output with consistent structure
    """
    
    def format(self, record):
        """Format log record as JSON"""
        # Basic log entry structure
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
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra attributes added through adapter
        if hasattr(record, 'extra') and record.extra:
            log_entry["extra"] = record.extra
            
        return json.dumps(log_entry)


class LogInitializer:
    """
    Enhanced Log Initializer with improved organization and aggregation
    """
    
    def __init__(self, config: ConfigManager):
        """Initialize with configuration"""
        self.config = config
        self.base_path = Path(config.get("logging", "base_path", default="./logs"))
        
        # Get current date components for directory structure
        now = datetime.now()
        self.year_month = now.strftime('%Y-%m')
        self.current_date = now.strftime('%Y-%m-%d')
        
        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize log manager
        self.log_manager = LogManager(config)
        
        # Track initialized loggers
        self.module_loggers = {}
        
        # Parse categories from config
        self.categories = self._parse_categories()
        
        # Module to category mapping for quick lookup
        self.module_category_map = self._create_module_map()
        
        # Setup directory structure
        self._setup_directory_structure()
    
    def _parse_categories(self):
        """Define consolidated log categories from configuration"""
        # Default categories
        default_categories = {
            "trading": {
                "level": self.config.get("logging", "level", default="INFO"),
                "modules": [
                    "core", "execution", "mode",
                    "core.", "execution.", "mode."
                ]
            },
            "data": {
                "level": self.config.get("logging", "level", default="INFO"),
                "modules": [
                    "datasource", "exchange",
                    "datasource.", "exchange."
                ]
            },
            "strategy": {
                "level": self.config.get("logging", "level", default="INFO"),
                "modules": [
                    "strategy", "backtest",
                    "strategy.", "backtest."
                ]
            },
            "portfolio": {
                "level": self.config.get("logging", "level", default="INFO"),
                "modules": [
                    "portfolio", "risk",
                    "portfolio.", "risk."
                ]
            },
            "system": {
                "level": self.config.get("logging", "level", default="INFO"),
                "modules": [
                    "main", "common", "utils",
                    "main.", "common.", "utils."
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
        
        # Map specific modules to categories
        for category, settings in self.categories.items():
            for module_pattern in settings["modules"]:
                # For exact module matches
                if not module_pattern.endswith('.'):
                    module_map[module_pattern] = category
                    # Also map qualified paths
                    module_map[f"src.{module_pattern}"] = category
                # For module prefixes (anything in that package)
                else:
                    prefix = module_pattern[:-1]  # Remove trailing dot
                    # Add the base module
                    module_map[prefix] = category
                    # Also add qualified version
                    module_map[f"src.{prefix}"] = category
        
        return module_map

    def get_logger(self, module_name):
        """Get a logger mapped to the appropriate category"""
        # Return existing logger if already created
        if module_name in self.module_loggers:
            return self.module_loggers[module_name]
        
        # Determine the category for this module
        category = self._get_category_for_module(module_name)
        
        # Create the module logger
        logger = logging.getLogger(module_name)
        
        # Set appropriate log level
        category_config = self.categories.get(category, {"level": "INFO"})
        level = category_config.get("level", self.config.get("logging", "level", default="INFO"))
        logger.setLevel(self._get_log_level(level))
        
        # Add handlers if not already present
        if not logger.handlers:
            # Add category log handler
            category_path = self.base_path / category / self.year_month / f"{category}.log"
            category_handler = self._create_file_handler(str(category_path), level)
            logger.addHandler(category_handler)
            
            # Add module-specific log handler
            module_path = self.base_path / category / self.year_month / f"{module_name}.log"
            module_handler = self._create_file_handler(str(module_path), level)
            logger.addHandler(module_handler)
            
            # Add to combined log if enabled
            if self.config.get("logging", "combined_log", default=True):
                combined_path = self.base_path / "combined" / self.year_month / "all.log"
                combined_handler = self._create_file_handler(str(combined_path), level)
                logger.addHandler(combined_handler)
            
            # Add console handler if enabled
            if self.config.get("logging", "handlers", "console", "enabled", default=True):
                console_level = self.config.get("logging", "handlers", "console", "level", default=level)
                console_handler = logging.StreamHandler()
                console_handler.setLevel(self._get_log_level(console_level))
                
                # Use colorized output if enabled
                if self.config.get("logging", "handlers", "console", "colors", default=True):
                    formatter = self._get_colored_formatter()
                else:
                    formatter = logging.Formatter(self.config.get(
                        "logging", "format", 
                        default="%(asctime)s | %(levelname)-8s | %(name)-25s | [%(filename)s:%(lineno)d] | %(message)s"
                    ))
                
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
        
        # Store the logger for future reference
        self.module_loggers[module_name] = logger
        
        return logger

    def _get_category_for_module(self, module_name):
        """Determine the appropriate category for a module"""
        # Direct match
        if module_name in self.module_category_map:
            return self.module_category_map[module_name]
        
        # Try matching by prefix (for submodules)
        parts = module_name.split('.')
        while parts:
            # Try current prefix
            prefix = '.'.join(parts)
            if prefix in self.module_category_map:
                return self.module_category_map[prefix]
            # Remove last part and try again
            parts.pop()
        
        # Default to system for unknown modules
        return "system"
    
    def _setup_directory_structure(self):
        """Create the organized directory structure"""
        # Create category directories
        for category in self.categories:
            category_dir = self.base_path / category / self.year_month
            category_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a directory for uncategorized modules
        uncategorized_dir = self.base_path / "other" / self.year_month
        uncategorized_dir.mkdir(parents=True, exist_ok=True)
        
        # Create combined logs directory
        if self.config.get("logging", "combined_log", default=True):
            combined_dir = self.base_path / "combined" / self.year_month
            combined_dir.mkdir(parents=True, exist_ok=True)
            
        # Create summary logs directory if enabled
        if self.config.get("logging", "summary", "enabled", default=False):
            summary_dir = self.base_path / "summary" / self.year_month
            summary_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_log_path(self, module_name):
        """Get the appropriate log path for a module"""
        # Find the category for this module
        category = self.module_category_map.get(module_name, "other")
        
        # Create the log path
        return self.base_path / category / self.year_month / f"{module_name}.log"
    
    def initialize(self):
        """Initialize the logging system with the enhanced structure"""
        # Configure log manager with base settings
        log_config = {
            'base_dir': str(self.base_path),
            'log_to_console': self.config.get("logging", "handlers", "console", "enabled", default=True),
            'console_level': self.config.get("logging", "handlers", "console", "level", default="INFO"),
            'log_to_file': self.config.get("logging", "handlers", "file", "enabled", default=True),
            'default_level': self.config.get("logging", "level", default="INFO"),
            'rotation_strategy': self.config.get("logging", "rotation", "strategy", default="size"),
            'max_file_size': int(self.config.get("logging", "rotation", "max_file_size", default=10485760)),
            'backup_count': int(self.config.get("logging", "rotation", "backup_count", default=5)),
            'format_str': self.config.get("logging", "format", 
                default='%(asctime)s | %(levelname)-8s | %(name)-25s | [%(filename)s:%(lineno)d] | %(message)s'),
            'date_format': self.config.get("logging", "date_format", 
                default='%Y-%m-%d %H:%M:%S%z'),
            'compression': self.config.get("logging", "handlers", "file", "compression", default="gz")
        }
        
        self.log_manager.configure_from_dict(log_config)
        
        # Setup root logger with reasonable defaults
        root_logger = logging.getLogger()
        root_logger.setLevel(self._get_log_level(log_config['default_level']))
        
        # Clear any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add console handler to root logger if enabled
        if log_config['log_to_console']:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self._get_log_level(log_config['console_level']))
            
            # Use colored output if enabled
            if self.config.get("logging", "handlers", "console", "colors", default=True):
                formatter = self._get_colored_formatter()
            else:
                formatter = logging.Formatter(log_config['format_str'], datefmt=log_config['date_format'])
                
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # Setup combined logger if enabled
        if self.config.get("logging", "combined_log", default=True):
            self._setup_combined_logger()
        
        # Setup category loggers
        for category, settings in self.categories.items():
            self._setup_category_logger(category, settings["level"])
        
        # Create a system logger as the main application logger
        system_logger = self.get_logger("system")
        system_logger.info(f"Logging system initialized with enhanced organization")
        
        # Setup syslog if enabled
        if self.config.get("logging", "syslog", "enabled", default=False):
            self._setup_syslog()
        
        # Setup summary logging if enabled
        if self.config.get("logging", "summary", "enabled", default=False):
            self._setup_summary_logger()
        
        return self
    
    def _setup_combined_logger(self):
        """Set up a combined log file that captures all log entries"""
        combined_path = self.base_path / "combined" / self.year_month / f"combined.log"
        
        # Create handler for combined log
        handler = self._create_file_handler(str(combined_path))
        
        # Add handler to the root logger to capture all logs
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        
        # Also create a daily combined log
        daily_path = self.base_path / "combined" / self.year_month / f"{self.current_date}.log"
        daily_handler = logging.FileHandler(str(daily_path))
        formatter = logging.Formatter(self.config.get("logging", "format", 
            default='%(asctime)s | %(levelname)-8s | %(name)-25s | [%(filename)s:%(lineno)d] | %(message)s'),
            datefmt=self.config.get("logging", "date_format", default='%Y-%m-%d %H:%M:%S%z'))
        daily_handler.setFormatter(formatter)
        root_logger.addHandler(daily_handler)
    
    def _setup_category_logger(self, category, level):
        """Set up a logger for a category that will capture all logs from that category"""
        category_path = self.base_path / category / self.year_month / f"{category}.log"
        
        # Create the category logger
        category_logger = logging.getLogger(f"category.{category}")
        category_logger.setLevel(self._get_log_level(level))
        
        # Create file handler for this category
        handler = self._create_file_handler(str(category_path), level)
        category_logger.addHandler(handler)
        
        # Store in module loggers map
        self.module_loggers[f"category.{category}"] = category_logger
    
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
        summary_path = self.base_path / "summary" / self.year_month / f"summary.log"
        
        # Create summary logger
        summary_logger = logging.getLogger("summary")
        summary_logger.setLevel(self._get_log_level(
            self.config.get("logging", "summary", "level", default="WARNING")
        ))
        
        # Create handler for summary
        summary_format = self.config.get("logging", "summary", "format", default="json")
        
        if summary_format.lower() == "json":
            handler = self._create_file_handler(str(summary_path), rotation=False)
            handler.setFormatter(JsonFormatter())
        else:
            handler = self._create_file_handler(str(summary_path), rotation=False)
            formatter = logging.Formatter(self.config.get("logging", "format", 
                default='%(asctime)s | %(levelname)-8s | %(name)-25s | [%(filename)s:%(lineno)d] | %(message)s'),
                datefmt=self.config.get("logging", "date_format", default='%Y-%m-%d %H:%M:%S%z'))
            handler.setFormatter(formatter)
        
        summary_logger.addHandler(handler)
        
        # Store in module loggers map
        self.module_loggers["summary"] = summary_logger
    
    def _get_log_level(self, level):
        """Convert string log level to numeric value"""
        if isinstance(level, int):
            return level
        
        log_levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        level_upper = level.upper()
        if level_upper in log_levels:
            return log_levels[level_upper]
        
        return logging.INFO

    def _get_colored_formatter(self):
        """Get a formatter with ANSI color codes for console output"""
        # Define colors
        RESET = "\033[0m"
        BOLD = "\033[1m"
        RED = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        MAGENTA = "\033[35m"
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
        
        # Use the configured log format with colors
        format_str = self.config.get("logging", "format", 
            default='%(asctime)s | %(levelname)-8s | %(name)-25s | [%(filename)s:%(lineno)d] | %(message)s')
        date_format = self.config.get("logging", "date_format", default='%Y-%m-%d %H:%M:%S%z')
        
        return ColoredFormatter(format_str, datefmt=date_format)

    def _create_file_handler(self, log_path, level=None, rotation=True):
        """Create the appropriate handler based on rotation settings"""
        # Ensure directory exists
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create the appropriate handler based on rotation settings
        rotation_strategy = self.config.get("logging", "rotation", "strategy", default="size")
        max_size = int(self.config.get("logging", "rotation", "max_file_size", default=10485760))
        backup_count = int(self.config.get("logging", "rotation", "backup_count", default=5))
        
        # Check if we should use compressed rotation
        use_compression = self.config.get("logging", "handlers", "file", "compression", default="gz") != ""
        
        if rotation and rotation_strategy == "size":
            if use_compression:
                handler = CompressedRotatingFileHandler(
                    log_path,
                    maxBytes=max_size,
                    backupCount=backup_count,
                    encoding="utf-8",
                    compress=True
                )
            else:
                handler = logging.handlers.RotatingFileHandler(
                    log_path,
                    maxBytes=max_size,
                    backupCount=backup_count,
                    encoding="utf-8"
                )
        elif rotation and rotation_strategy == "time":
            handler = logging.handlers.TimedRotatingFileHandler(
                log_path,
                when="midnight",
                interval=1,
                backupCount=backup_count,
                encoding="utf-8"
            )
        else:
            handler = logging.FileHandler(log_path, encoding="utf-8")
        
        # Set log level
        if level:
            handler.setLevel(self._get_log_level(level))
        
        # Use consistent format
        format_str = self.config.get("logging", "format", 
            default='%(asctime)s | %(levelname)-8s | %(name)-25s | [%(filename)s:%(lineno)d] | %(message)s')
        date_format = self.config.get("logging", "date_format", default='%Y-%m-%d %H:%M:%S%z')
        
        formatter = logging.Formatter(format_str, datefmt=date_format)
        handler.setFormatter(formatter)
        
        return handler
    
    def add_json_handler(self, logger_name, path=None):
        """Add a JSON format handler to a logger"""
        # Get the logger
        logger = self.get_logger(logger_name)
        
        # Generate path if not provided
        if not path:
            category = self.module_category_map.get(logger_name, "other")
            path = self.base_path / category / self.year_month / f"{logger_name}.json.log"
        
        # Create handler
        handler = logging.handlers.RotatingFileHandler(
            str(path),
            maxBytes=int(self.config.get("logging", "rotation", "max_file_size", default=10485760)),
            backupCount=int(self.config.get("logging", "rotation", "backup_count", default=5)),
            encoding="utf-8"
        )
        
        # Set formatter
        handler.setFormatter(JsonFormatter())
        
        # Add handler
        logger.addHandler(handler)
        
        return logger
    
    
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
        
        # Date format
        self.date_format = config.get("logging", "date_format", default='%Y-%m-%d %H:%M:%S%z')
        
        # Initialize root logger
        self.configured_loggers = set()
        
        # Ensure log directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup root logger
        self.setup_root_logger()
        
        # Check if we should enable async logging
        if config.get("logging", "async_write", default=False):
            self._setup_async_logging()
    
    # Then replace the _setup_async_logging method
    def _setup_async_logging(self):
        """Set up asynchronous logging if enabled"""
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
            
            # Use a separate console logger to avoid cycles
            console = logging.StreamHandler(sys.stdout)
            console.setFormatter(logging.Formatter('%(message)s'))
            
            temp_logger = logging.getLogger("async_setup")
            temp_logger.setLevel(logging.INFO)
            for handler in temp_logger.handlers[:]:
                temp_logger.removeHandler(handler)
            temp_logger.addHandler(console)
            temp_logger.propagate = False
            
            temp_logger.info("Async logging enabled")
            
        except ImportError:
            print("Warning: concurrent.futures not available, async logging disabled")
        except Exception as e:
            print(f"Warning: Failed to set up async logging: {str(e)}")
            
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
            console_level = self._get_log_level(
                self.config.get("logging", "handlers", "console", "level", default=self.default_level)
            )
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(self.format_str, datefmt=self.date_format))
            console_handler.setLevel(console_level)
            root_logger.addHandler(console_handler)

        # File handler
        if self.log_to_file:
            file_handler = self._create_file_handler("system")
            file_handler.setLevel(self.default_level)
            root_logger.addHandler(file_handler)
            
        # Mark root logger as configured
        self.configured_loggers.add("root")
    
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
        
        # Check if we should use compressed rotation
        use_compression = self.config.get("logging", "handlers", "file", "compression", default="gz") != ""
        
        # Create appropriate handler based on rotation strategy
        if self.rotation_strategy == "size":
            if use_compression:
                handler = CompressedRotatingFileHandler(
                    log_file,
                    maxBytes=self.max_file_size,
                    backupCount=self.backup_count,
                    encoding="utf-8",
                    compress=True
                )
            else:
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
                
        handler.setFormatter(logging.Formatter(self.format_str, datefmt=self.date_format))
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
    
    def add_json_handler(self, logger_name: str, json_file: Optional[str] = None) -> logging.Logger:
        """
        Add JSON format log handler
        
        Args:
            logger_name: Logger name to add handler to
            json_file: Optional JSON file path (default: logger_name.json.log)
            
        Returns:
            Logger with JSON handler added
        """
        logger = logging.getLogger(logger_name)
        
        if json_file:
            json_path = Path(json_file)
        else:
            json_path = self.base_dir / f"{logger_name}.json.log"
        
        # Ensure parent directory exists
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        handler = RotatingFileHandler(
            json_path,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding="utf-8"
        )
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        
        return logger
    
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
        
        if 'format_str' in config:
            self.format_str = config['format_str']
        
        if 'date_format' in config:
            self.date_format = config['date_format']
        
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
            handler.setFormatter(logging.Formatter(self.format_str, datefmt=self.date_format))
            
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
            max_bytes: Maximum file size before rotation
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
            max_bytes: Maximum file size before rotation
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
            handler.setFormatter(logging.Formatter(self.format_str, datefmt=self.date_format))
            
        return handler