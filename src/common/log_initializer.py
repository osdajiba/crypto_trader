# src/common/log_initializer.py

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union

from src.common.log_manager import LogManager
from src.common.config_manager import ConfigManager


class LogInitializer:
    """
    Log Initializer
    Sets up loggers for different modules based on configuration
    and organizes logs in date-based directory structure
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the log initializer
        
        Args:
            config: ConfigManager instance
        """
        self.config = config
        
        # Set up the base directory with date-based subdirectory
        base_path = config.get("logging", "base_path", default="./logs")
        self.current_date = datetime.now().strftime('%Y-%m-%d')
        self.log_base_dir = Path(base_path) / self.current_date
        
        # Ensure log directory exists
        self.log_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for all modules in advance
        self._create_module_directories()
        
        # Initialize the log manager with the dated directory
        self.log_manager = LogManager(config)
        self._configure_log_manager()
        
        # Track initialized loggers
        self.module_loggers = {}
    
    def _create_module_directories(self):
        """
        Pre-create all subdirectories needed for module logs
        This ensures directories exist before any logging occurs
        """
        modules_config = self.config.get("logging", "modules", default={})
        
        for module_name, module_config in modules_config.items():
            path_str = module_config.get("path", f"{module_name}.log")
            log_path = Path(path_str)
            
            # Create subdirectory if path contains directory component
            if log_path.parent != Path("."):
                subdir = self.log_base_dir / log_path.parent
                subdir.mkdir(parents=True, exist_ok=True)
    
    def _configure_log_manager(self):
        """Configure log manager with the date-based directory"""
        # Create configuration dictionary
        log_config = {
            'base_dir': str(self.log_base_dir),
            'log_to_console': self.config.get("logging", "handlers", "console", "enabled", default=True),
            'log_to_file': self.config.get("logging", "handlers", "file", "enabled", default=True),
            'default_level': self.config.get("logging", "level", default="INFO"),
            'rotation_strategy': self.config.get("logging", "rotation", "strategy", default="size"),
            'max_file_size': int(self.config.get("logging", "rotation", "max_file_size", default=10)) * 1024 * 1024,
            'backup_count': int(self.config.get("logging", "rotation", "backup_count", default=5)),
            'detailed_format': self.config.get("logging", "detailed_format", default=True)
        }
        
        # Apply configuration to log manager
        self.log_manager.configure_from_dict(log_config)
    
    def initialize(self):
        """
        Initialize all loggers based on configuration
        Creates module-specific loggers with appropriate paths and levels
        """
        # Configure module-specific loggers
        modules_config = self.config.get("logging", "modules", default={})
        
        # Create system logger first
        system_level = self.config.get("logging", "level", default="INFO")
        system_logger = self.log_manager.setup_logger("system", level=system_level)
        self.module_loggers["system"] = system_logger
        
        # Set up module-specific loggers
        for module_name, module_config in modules_config.items():
            # Get logger level
            level = module_config.get("level", "INFO")
            
            # Setup the logger
            logger_name = f"{module_name}"
            logger = self.log_manager.setup_logger(logger_name, level=level)
            self.module_loggers[module_name] = logger
            
            # Log initialization
            logger.info(f"Logger for {module_name} initialized")
        
        # Configure syslog if enabled
        if self.config.get("logging", "syslog", "enabled", default=False):
            self._setup_syslog()
        
        # Log initialization complete
        system_logger.info(f"Logging system initialized. Log path: {self.log_base_dir}")
        return self
    
    def _setup_syslog(self):
        """Set up remote syslog if enabled in configuration"""
        import logging.handlers
        
        # Get syslog configuration
        syslog_host = self.config.get("logging", "syslog", "host", default="localhost")
        syslog_port = int(self.config.get("logging", "syslog", "port", default=514))
        
        # Create syslog handler
        syslog_handler = logging.handlers.SysLogHandler(
            address=(syslog_host, syslog_port)
        )
        
        # Add syslog handler to root logger
        root_logger = LogManager.get_logger()
        formatter = logging.Formatter(LogManager.DETAILED_FORMAT)
        syslog_handler.setFormatter(formatter)
        root_logger.addHandler(syslog_handler)
        
        # Log syslog setup
        system_logger = self.get_logger("system")
        system_logger.info(f"Syslog handler configured: {syslog_host}:{syslog_port}")
    
    def get_logger(self, module_name: str) -> logging.Logger:
        """
        Get a logger for a specific module
        
        Args:
            module_name: Module name (like 'core', 'strategy', 'data', etc.)
            
        Returns:
            Configured logger for the module
        """
        # Return existing logger if already created
        if module_name in self.module_loggers:
            return self.module_loggers[module_name]
        
        # Get module configuration if available
        modules_config = self.config.get("logging", "modules", default={})
        module_config = modules_config.get(module_name, {})
        
        # Get level from config or use default
        level = module_config.get("level", self.config.get("logging", "level", default="INFO"))
        
        # Create and set up logger
        logger = self.log_manager.setup_logger(module_name, level=level)
        self.module_loggers[module_name] = logger
        
        return logger
        
    @staticmethod
    def ensure_log_directory(log_path):
        """
        Utility method to ensure a log directory exists
        Can be called from other parts of the application
        
        Args:
            log_path: Path to the log file or directory
        """
        # Convert to Path object if string
        path = Path(log_path)
        
        # If path is a file, get its parent directory
        if path.suffix:
            directory = path.parent
        else:
            directory = path
            
        # Create directory if it doesn't exist
        directory.mkdir(parents=True, exist_ok=True)