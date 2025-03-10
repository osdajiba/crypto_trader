#!/usr/bin/env python3
"""
Utility for generating and validating the system configuration file.

This script checks if the configuration file exists and offers to create
a default configuration file if it doesn't.
"""

import os
import sys
import yaml
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
RESET = "\033[0m"
BRIGHT = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"

# Get project root directory
def get_project_root():
    """Get the absolute path to the project root directory."""
    # When script is in src/utils/, project root is two levels up
    return Path(__file__).resolve().parent.parent.parent

# Configuration paths
PROJECT_ROOT = get_project_root()
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
DEFAULT_CONFIG_TEMPLATE = PROJECT_ROOT / "config.yaml"  # Template in project root

def print_colored(message, color=BRIGHT + CYAN):
    """Print colored message to terminal."""
    print(f"{color}{message}{RESET}")

def check_config_file():
    """Check if the configuration file exists."""
    return CONFIG_FILE.exists()

def create_config_directory():
    """Create the configuration directory if it doesn't exist."""
    if not CONFIG_DIR.exists():
        logger.info(f"Creating configuration directory: {CONFIG_DIR}")
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

def copy_default_config():
    """Copy the default configuration file to the config directory."""
    if DEFAULT_CONFIG_TEMPLATE.exists():
        logger.info(f"Copying default configuration from: {DEFAULT_CONFIG_TEMPLATE}")
        shutil.copy(DEFAULT_CONFIG_TEMPLATE, CONFIG_FILE)
        return True
    else:
        logger.warning(f"Default configuration template not found at: {DEFAULT_CONFIG_TEMPLATE}")
        return False

def generate_default_config():
    """Generate a minimal default configuration file."""
    logger.info("Generating minimal default configuration")
    
    default_config = {
        "system": {
            "operational_mode": "backtest",
            "timezone": "UTC",
            "performance": {
                "max_threads": 4,
                "max_memory": 1073741824,  # 1GB
                "task_timeout": 300,
            }
        },
        "api": {
            "binance": {
                "apiKey": "your apiKey",
                "secret": "your secret",
                "options": {
                    "adjustForTimeDifference": True,
                    "defaultType": "spot"
                }
            },
            "timeout": 60000,
            "enableRateLimit": True
        },
        "data": {
            "source": {
                "live": False,
                "type": "local"
            },
            "storage": {
                "historical": "db/historical/",
                "orders": "db/order_history.csv",
                "trades": "db/trade_records.json"
            }
        },
        "database": {
            "url": "sqlite:///database/trading.sqlite",
            "pool_size": 5,
            "timeout": 30,
            "echo_sql": False
        },
        "logging": {
            "base_path": "./logs",
            "level": "INFO",
            "handlers": {
                "file": {"enabled": True},
                "console": {"enabled": True, "level": "WARNING"}
            }
        },
        "trading": {
            "capital": {
                "initial": 100000,
                "min_order": 0.001
            },
            "instruments": ["BTC/USDT", "ETH/USDT"],
            "execution": {
                "slippage": 0.001,
                "leverage": 1
            }
        },
        "strategy": {
            "active": "dual_ma",
            "parameters": {
                "fast_period": 50,
                "slow_period": 200,
                "threshold": 0.005
            }
        },
        "backtest": {
            "period": {
                "start": "2023-01-01",
                "end": "2023-12-31"
            }
        }
    }
    
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Default configuration generated at: {CONFIG_FILE}")
    return True

def validate_config():
    """Validate the configuration file."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        
        # Perform basic validation
        required_sections = ["system", "database", "trading"]
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required configuration section: {section}")
                return False
        
        logger.info("Configuration validation successful")
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        return False

def prompt_generate_config():
    """Prompt the user to generate a default configuration file."""
    print_colored("\n======================================================")
    print_colored("      CONFIGURATION FILE NOT FOUND", BRIGHT + YELLOW)
    print_colored("======================================================")
    print_colored(f"A configuration file is required at:", BRIGHT)
    print_colored(f"  {CONFIG_FILE}", BRIGHT + BLUE)
    print_colored("\nWould you like to generate a default configuration file?")
    response = input("(y/n): ").strip().lower()
    return response == 'y' or response == 'yes'

def main():
    """Main function to check and generate configuration."""
    print_colored("[*] Checking configuration file...", BRIGHT + BLUE)
    
    if check_config_file():
        print_colored(f"[✓] Configuration file found at: {CONFIG_FILE}", BRIGHT + GREEN)
        if validate_config():
            print_colored("[✓] Configuration file is valid.", BRIGHT + GREEN)
            return True
        else:
            print_colored("[!] Configuration file validation failed.", BRIGHT + YELLOW)
            return False
    else:
        print_colored("[!] Configuration file not found.", BRIGHT + YELLOW)
        
        if len(sys.argv) > 1 and sys.argv[1] == "--auto":
            should_generate = True
        else:
            should_generate = prompt_generate_config()
        
        if should_generate:
            create_config_directory()
            
            # Try to copy default template first
            if copy_default_config():
                print_colored("[✓] Default configuration copied successfully.", BRIGHT + GREEN)
                return True
            
            # If no template exists, generate a minimal config
            if generate_default_config():
                print_colored("[✓] Default configuration generated successfully.", BRIGHT + GREEN)
                return True
            else:
                print_colored("[✗] Failed to generate configuration file.", BRIGHT + RED)
                return False
        else:
            print_colored("Configuration generation skipped. The system may not function correctly.", BRIGHT + YELLOW)
            return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)