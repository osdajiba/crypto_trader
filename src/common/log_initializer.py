# src/common/log_initializer.py

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
import logging.handlers

from src.common.log_manager import LogManager
from src.common.config_manager import ConfigManager


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
        """Define consolidated log categories matching project structure"""
        return {
            "trading": {
                "level": self.config.get("logging", "level", default="INFO"),
                "modules": [
                    # Core trading engine components
                    "core", "execution", "mode",
                    # Include all modules within these packages
                    "core.", "execution.", "mode."
                ]
            },
            "data": {
                "level": self.config.get("logging", "level", default="INFO"),
                "modules": [
                    # Data handling components
                    "datasource", "exchange",
                    # Include all modules within these packages
                    "datasource.", "exchange."
                ]
            },
            "strategy": {
                "level": self.config.get("logging", "level", default="INFO"),
                "modules": [
                    # Strategy and backtest components
                    "strategy", "backtest",
                    # Include all modules within these packages
                    "strategy.", "backtest."
                ]
            },
            "portfolio": {
                "level": self.config.get("logging", "level", default="INFO"),
                "modules": [
                    # Money management components
                    "portfolio", "risk",
                    # Include all modules within these packages
                    "portfolio.", "risk."
                ]
            },
            "system": {
                "level": self.config.get("logging", "level", default="INFO"),
                "modules": [
                    # Infrastructure components
                    "main", "common", "utils",
                    # Include all modules within these packages
                    "main.", "common.", "utils."
                ]
            }
        }
        
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
            category_path = self.base_path / "modules" / self.year_month / f"{category}.log"
            category_handler = self._create_file_handler(str(category_path), level)
            logger.addHandler(category_handler)
            
            # Add to combined log if enabled
            if self.config.get("logging", "combined_log", default=True):
                combined_path = self.base_path / "combined" / self.year_month / "all.log"
                combined_handler = self._create_file_handler(str(combined_path), level)
                logger.addHandler(combined_handler)
        
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
            'log_to_file': self.config.get("logging", "handlers", "file", "enabled", default=True),
            'default_level': self.config.get("logging", "level", default="INFO"),
            'rotation_strategy': self.config.get("logging", "rotation", "strategy", default="size"),
            'max_file_size': int(self.config.get("logging", "rotation", "max_file_size", default=10)) * 1024 * 1024,
            'backup_count': int(self.config.get("logging", "rotation", "backup_count", default=5)),
            'format_str': self.config.get("logging", "format", default=LogManager.DEFAULT_FORMAT)
        }
        
        self.log_manager.configure_from_dict(log_config)
        
        # Setup combined logger if enabled
        if self.config.get("logging", "combined_log", default=True):
            self._setup_combined_logger()
        
        # Setup category loggers
        for category, settings in self.categories.items():
            self._setup_category_logger(category, settings["level"])
        
        # Create a system logger
        system_logger = self.get_logger("system")
        system_logger.info(f"Logging system initialized with enhanced organization")
        
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
        formatter = logging.Formatter(self.config.get("logging", "format", default=LogManager.DEFAULT_FORMAT))
        daily_handler.setFormatter(formatter)
        root_logger.addHandler(daily_handler)
    
    def _setup_category_logger(self, category, level):
        """Set up a logger for a category that will capture all logs from that category"""
        category_path = self.base_path / category / self.year_month / f"{category}.log"
        
        # Create the category logger
        category_logger = logging.getLogger(f"category.{category}")
        category_logger.setLevel(self._get_log_level(level))
        
        # Create file handler for this category
        handler = logging.FileHandler(str(category_path))
        formatter = logging.Formatter(self.config.get("logging", "format", default=LogManager.DEFAULT_FORMAT))
        handler.setFormatter(formatter)
        category_logger.addHandler(handler)
        
        self.module_loggers[f"category.{category}"] = category_logger
    
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



    def _create_file_handler(self, log_path, level=None, rotation=True):
        """Create the appropriate handler based on rotation settings"""
        # Ensure directory exists
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create the appropriate handler based on rotation settings
        rotation_strategy = self.config.get("logging", "rotation", "strategy", default="size")
        max_size = int(self.config.get("logging", "rotation", "max_file_size", default=10485760))
        backup_count = int(self.config.get("logging", "rotation", "backup_count", default=5))
        
        if rotation and rotation_strategy == "size":
            handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=max_size,
                backupCount=backup_count
            )
        elif rotation and rotation_strategy == "time":
            handler = logging.handlers.TimedRotatingFileHandler(
                log_path,
                when="midnight",
                interval=1,
                backupCount=backup_count
            )
        else:
            handler = logging.FileHandler(log_path)
        
        # Set log level
        if level:
            handler.setLevel(self._get_log_level(level))
        
        # Use consistent pipe-separated format with module path info
        format_str = "%(asctime)s | %(levelname)-8s | %(name)-25s | [%(filename)s:%(lineno)d] | %(message)s"
        formatter = logging.Formatter(format_str)
        handler.setFormatter(formatter)
        
        return handler
    
    def add_json_handler(self, logger_name, path=None):
        """Add a JSON format handler to a logger"""
        import json
        
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
            backupCount=int(self.config.get("logging", "rotation", "backup_count", default=5))
        )
        
        # Set formatter
        handler.setFormatter(JsonFormatter())
        
        # Add handler
        logger.addHandler(handler)
        
        return logger