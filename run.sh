#!/bin/bash
# Trading System CLI - A consolidated script combining functionality from:
# - color_setup.sh
# - install.sh
# - network.sh
# - run.sh

# Get project root directory and set up paths
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
CONFIG_DIR="$PROJECT_ROOT/conf"
LOG_DIR="$PROJECT_ROOT/logs"
SESSION_ID=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/trading_${SESSION_ID}.log"
PID_FILE="/tmp/trading_system_${SESSION_ID}.pid"
SAVED_CONFIG_DIR="$CONFIG_DIR/saved_configs"

# Create necessary directories
mkdir -p "$LOG_DIR" "$CONFIG_DIR" "$SCRIPTS_DIR" "$SAVED_CONFIG_DIR"

#------------------------------------------
# Color and style definitions
#------------------------------------------
RESET="\033[0m"
BOLD="\033[1m"
DIM="\033[2m"
ITALIC="\033[3m"
UNDERLINE="\033[4m"

# Regular colors
BLACK="\033[30m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
MAGENTA="\033[35m"
CYAN="\033[36m"
WHITE="\033[37m"

# Bright colors
BRIGHT_RED="\033[91m"
BRIGHT_GREEN="\033[92m"
BRIGHT_YELLOW="\033[93m"
BRIGHT_BLUE="\033[94m"
BRIGHT_MAGENTA="\033[95m"
BRIGHT_CYAN="\033[96m"
BRIGHT_WHITE="\033[97m"

# Background colors
BG_RED="\033[41m"
BG_GREEN="\033[42m"
BG_YELLOW="\033[43m"
BG_BLUE="\033[44m"

#------------------------------------------
# Color support
#------------------------------------------
COLOR_ENABLED=true
check_color_support() {
    # Check if terminal supports colors
    if [[ -t 1 && "$TERM" != "dumb" ]]; then
        # Check if color is disabled via flags
        for arg in "$@"; do
            if [[ "$arg" == "--no-color" ]]; then
                COLOR_ENABLED=false
                break
            fi
        done
        
        # Check if running in pipeline
        if [[ ! -t 1 ]]; then
            COLOR_ENABLED=false
        fi
        
        # Check if NO_COLOR environment variable is set
        if [[ -n "$NO_COLOR" ]]; then
            COLOR_ENABLED=false
        fi
    else
        COLOR_ENABLED=false
    fi
    
    # Override if FORCE_COLOR is set
    if [[ -n "$FORCE_COLOR" ]]; then
        COLOR_ENABLED=true
    fi
    
    # If colors are disabled, reset all color variables
    if ! $COLOR_ENABLED; then
        RESET=""
        BOLD=""
        DIM=""
        ITALIC=""
        UNDERLINE=""
        BLACK=""
        RED=""
        GREEN=""
        YELLOW=""
        BLUE=""
        MAGENTA=""
        CYAN=""
        WHITE=""
        BRIGHT_RED=""
        BRIGHT_GREEN=""
        BRIGHT_YELLOW=""
        BRIGHT_BLUE=""
        BRIGHT_MAGENTA=""
        BRIGHT_CYAN=""
        BRIGHT_WHITE=""
        BG_RED=""
        BG_GREEN=""
        BG_YELLOW=""
        BG_BLUE=""
    fi
}

#------------------------------------------
# Style functions
#------------------------------------------
style_header() {
    echo -e "${BOLD}${BRIGHT_CYAN}$1${RESET}"
}

style_subheader() {
    echo -e "${CYAN}$1${RESET}"
}

style_success() {
    echo -e "${BRIGHT_GREEN}$1${RESET}"
}

style_error() {
    echo -e "${BRIGHT_RED}$1${RESET}"
}

style_warning() {
    echo -e "${BRIGHT_YELLOW}$1${RESET}"
}

style_info() {
    echo -e "${BRIGHT_BLUE}$1${RESET}"
}

style_menu_item() {
    echo -e "  ${BRIGHT_GREEN}$1${RESET}) $2"
}

style_menu_section() {
    echo -e "\n${BOLD}${BRIGHT_WHITE}$1:${RESET}"
}

style_parameter() {
    echo -e "${BOLD}${BRIGHT_WHITE}$1:${RESET} $2"
}

style_command() {
    echo -e "${BRIGHT_YELLOW}$1${RESET}"
}

#------------------------------------------
# Signal handling
#------------------------------------------
trap_handler() {
    echo -e "\n$(style_warning "Received termination signal. Cleaning up...")"
    
    # Kill the trading process if running
    if [ -f "$PID_FILE" ]; then
        TRADING_PID=$(cat "$PID_FILE")
        if ps -p $TRADING_PID > /dev/null; then
            echo -e "$(style_warning "Stopping trading process (PID: $TRADING_PID)...")"
            kill -SIGTERM $TRADING_PID 2>/dev/null
            sleep 2
            if ps -p $TRADING_PID > /dev/null; then
                echo -e "$(style_error "Process did not terminate gracefully, forcing...")"
                kill -SIGKILL $TRADING_PID 2>/dev/null
            fi
        fi
        rm -f "$PID_FILE"
    fi
    
    # Deactivate virtual environment if active
    deactivate 2>/dev/null || true
    
    echo -e "$(style_success "Cleanup complete.")"
    exit 0
}

# Set up trap for various signals
trap trap_handler SIGINT SIGTERM SIGHUP

#------------------------------------------
# UI Functions
#------------------------------------------
show_header() {
    clear
    style_header "Trading System CLI"
    style_subheader "=================================="
    echo -e "Session ID: $SESSION_ID"
    echo -e "Project root: $PROJECT_ROOT"
    echo -e ""
}

#------------------------------------------
# Log Functions
#------------------------------------------
colorize_log_output() {
    # Process stdin and add colors to log output
    sed --unbuffered \
        -e "s/\(DEBUG\)/${BLUE}\1${RESET}/g" \
        -e "s/\(INFO\)/${GREEN}\1${RESET}/g" \
        -e "s/\(WARNING\)/${BRIGHT_YELLOW}\1${RESET}/g" \
        -e "s/\(ERROR\)/${BRIGHT_RED}\1${RESET}/g" \
        -e "s/\(CRITICAL\)/${BOLD}${BG_RED}${WHITE}\1${RESET}/g" \
        -e "s/\(exception\|Exception\|error\|Error\|ERROR\|failed\|Failed\|FAILED\)/${BRIGHT_RED}\1${RESET}/g" \
        -e "s/\(warning\|Warning\|WARNING\)/${BRIGHT_YELLOW}\1${RESET}/g" \
        -e "s/\(success\|Success\|SUCCESS\|completed\|Completed\)/${BRIGHT_GREEN}\1${RESET}/g" \
        -e "s/\([0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}[[:space:]][0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}\)/${CYAN}\1${RESET}/g"
}

create_log_viewer() {
    echo -e "$(style_info "Creating colored log viewer script...")"
    
    # Create log viewer script in scripts directory
    cat > "$SCRIPTS_DIR/view_logs.sh" << 'EOF'
#!/bin/bash
# view_logs.sh - Script to view log files with color support

# ANSI color codes
RESET="\033[0m"
BOLD="\033[1m"
BLACK="\033[30m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
MAGENTA="\033[35m"
CYAN="\033[36m"
WHITE="\033[37m"

# Get project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
LOG_DIR="$PROJECT_ROOT/logs"

# Check if log directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo -e "${RED}Error: Log directory not found at $LOG_DIR${RESET}"
    exit 1
fi

# Function to display usage
show_usage() {
    echo -e "${BOLD}Usage:${RESET}"
    echo -e "  $0 [options] <log_file>"
    echo -e ""
    echo -e "${BOLD}Options:${RESET}"
    echo -e "  -l, --list         List available log files"
    echo -e "  -f, --follow       Follow log file updates (like tail -f)"
    echo -e "  -n, --lines N      Show last N lines (default: 100)"
    echo -e "  -c, --category C   Show logs from category C (system, strategy, etc.)"
    echo -e "  -h, --help         Show this help message"
}

# Function to list available log files
list_logs() {
    echo -e "${BOLD}${CYAN}Available log files:${RESET}"
    
    # Get all log files and sort them by modification time
    find "$LOG_DIR" -type f -name "*.log" | sort -t / -k 2 | while read -r log_file; do
        filename=$(basename "$log_file")
        filesize=$(du -h "$log_file" | cut -f1)
        last_modified=$(stat -c "%y" "$log_file" | cut -d"." -f1)
        
        # Determine color based on log type
        color=$GREEN
        if [[ "$filename" == *error* ]]; then
            color=$RED
        elif [[ "$filename" == *warning* ]]; then
            color=$YELLOW
        elif [[ "$filename" == *system* ]]; then
            color=$BLUE
        elif [[ "$filename" == *trading* ]]; then
            color=$CYAN
        elif [[ "$filename" == *combined* ]]; then
            color=$MAGENTA
        fi
        
        echo -e "$color$filename${RESET} (${filesize}, last modified: $last_modified)"
    done
}

# Function to add colors to log output
colorize_log() {
    # Colorize log levels, timestamps, and logger names
    sed \
        -e "s/\(|[[:space:]]*DEBUG[[:space:]]*|[[:space:]]*\)/${BLUE}\1${RESET}/g" \
        -e "s/\(|[[:space:]]*INFO[[:space:]]*|[[:space:]]*\)/${GREEN}\1${RESET}/g" \
        -e "s/\(|[[:space:]]*WARNING[[:space:]]*|[[:space:]]*\)/${YELLOW}\1${RESET}/g" \
        -e "s/\(|[[:space:]]*ERROR[[:space:]]*|[[:space:]]*\)/${RED}\1${RESET}/g" \
        -e "s/\(|[[:space:]]*CRITICAL[[:space:]]*|[[:space:]]*\)/${BOLD}${RED}\1${RESET}/g" \
        -e "s/\([0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}[[:space:]][0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}\)/${CYAN}\1${RESET}/g" \
        -e "s/\(|[[:space:]]*[^|]*|[[:space:]]*\)\([^|]*\)/\1${MAGENTA}\2${RESET}/"
}

# Default values
FOLLOW=false
LINES=100
LOG_FILE=""
CATEGORY=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -l|--list)
            list_logs
            exit 0
            ;;
        -f|--follow)
            FOLLOW=true
            shift
            ;;
        -n|--lines)
            LINES="$2"
            shift 2
            ;;
        -c|--category)
            CATEGORY="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            LOG_FILE="$1"
            shift
            ;;
    esac
done

# If a category is specified, use that log file
if [ -n "$CATEGORY" ]; then
    LOG_FILE="${CATEGORY}.log"
fi

# If no log file specified, show usage and exit
if [ -z "$LOG_FILE" ]; then
    show_usage
    exit 1
fi

# If not a full path, prepend log directory
if [[ "$LOG_FILE" != /* ]]; then
    LOG_FILE="$LOG_DIR/$LOG_FILE"
fi

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo -e "${RED}Error: Log file not found at $LOG_FILE${RESET}"
    echo -e "${YELLOW}Use --list to see available log files${RESET}"
    exit 1
fi

# View log file with colors
if $FOLLOW; then
    echo -e "${BOLD}${CYAN}Viewing $LOG_FILE (press Ctrl+C to stop):${RESET}\n"
    tail -n "$LINES" -f "$LOG_FILE" | colorize_log
else
    echo -e "${BOLD}${CYAN}Viewing $LOG_FILE:${RESET}\n"
    tail -n "$LINES" "$LOG_FILE" | colorize_log
fi
EOF

    # Make it executable
    chmod +x "$SCRIPTS_DIR/view_logs.sh"
    
    echo -e "$(style_success "Created log viewer script at $SCRIPTS_DIR/view_logs.sh")"
}

#------------------------------------------
# Color Configuration Functions
#------------------------------------------
create_color_config() {
    echo -e "$(style_info "Creating color configuration...")"
    
    # Create color config file
    cat > "$CONFIG_DIR/color_config.yaml" << EOF
# Color configuration for Trading System
# This file configures color settings for logs and shell scripts

# Console color settings
console:
  enabled: true
  colors:
    debug: "blue"
    info: "green"
    warning: "yellow"
    error: "red"
    critical: "bold_red"
    timestamp: "cyan"
    logger_name: "magenta"

# Log file color settings
log_files:
  enabled: true
  include_colors: true
  
# Terminal color scheme
terminal:
  prompt: "bold_cyan"
  headers: "bold_cyan"
  success: "green"
  warning: "yellow"
  error: "red"
  info: "blue"
  
# Specific component color settings
components:
  strategy: "green"
  data: "blue"
  exchange: "yellow"
  portfolio: "magenta"
  trading: "cyan"
  system: "white"
  
# Color definitions (ANSI escape sequences)
color_codes:
  reset: "\033[0m"
  bold: "\033[1m"
  black: "\033[30m"
  red: "\033[31m"
  green: "\033[32m"
  yellow: "\033[33m"
  blue: "\033[34m"
  magenta: "\033[35m"
  cyan: "\033[36m"
  white: "\033[37m"
  bold_red: "\033[1;31m"
  bold_green: "\033[1;32m"
  bold_yellow: "\033[1;33m"
  bold_blue: "\033[1;34m"
  bold_magenta: "\033[1;35m"
  bold_cyan: "\033[1;36m"
  bold_white: "\033[1;37m"
EOF
    
    echo -e "$(style_success "Color configuration created at $CONFIG_DIR/color_config.yaml")"
}

update_main_config() {
    echo -e "$(style_info "Updating main configuration...")"
    
    # Check if main config exists
    if [ ! -f "$CONFIG_DIR/config.yaml" ]; then
        echo -e "$(style_warning "Main config not found. Creating minimal config.")"
        
        # Create minimal config file
        cat > "$CONFIG_DIR/config.yaml" << EOF
# Main Trading System Configuration

# Logging settings
logging:
  base_path: "./logs"
  level: "INFO"
  format: "%(asctime)s | %(levelname)s | %(module)-18s | [%(filename)s:%(lineno)d] | %(message)s"
  date_format: "%Y-%m-%d %H:%M:%S%z"
  
  # Console logging settings
  handlers:
    console:
      enabled: true
      colors: true
      level: "INFO"
    
    file:
      enabled: true
      compression: "gz"
      buffer:
        enabled: true
        capacity: 1000
  
  # Log categories
  categories:
    system:
      level: "INFO"
    strategy:
      level: "INFO"
    data:
      level: "INFO"
    exchange:
      level: "WARNING"
    portfolio:
      level: "INFO"
    trading:
      level: "INFO"
EOF
        echo -e "$(style_success "Created minimal config with color logging enabled")"
    else
        # Check if colors are already enabled
        if grep -q "colors: *true" "$CONFIG_DIR/config.yaml"; then
            echo -e "$(style_success "Color logging already enabled in main config")"
        else
            # Try to add color configuration
            if grep -q "handlers:" "$CONFIG_DIR/config.yaml" && grep -q "console:" "$CONFIG_DIR/config.yaml"; then
                # Add colors: true under console section
                sed -i '/console:/,/enabled:/s/enabled:.*$/enabled: true\n      colors: true/' "$CONFIG_DIR/config.yaml"
                echo -e "$(style_success "Enabled color logging in main config")"
            else
                echo -e "$(style_warning "Could not automatically update config.yaml")"
                echo -e "$(style_warning "Please manually ensure 'colors: true' is set under logging.handlers.console")"
            fi
        fi
    fi
}

#------------------------------------------
# Setup Functions
#------------------------------------------
setup_virtual_env() {
    show_header
    style_header "Setting up virtual environment"
    
    # Check Python version
    echo -e "$(style_info "Checking Python installation...")"
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version)
        echo -e "$(style_success "Found $PYTHON_VERSION")"
    else
        echo -e "$(style_error "Python 3 not found")"
        echo -e "Please install Python 3.8 or newer:"
        echo -e "  sudo apt update"
        echo -e "  sudo apt install python3 python3-pip python3-venv"
        read -p "Press Enter to continue..." dummy
        return 1
    fi
    
    # Create virtual environment in the root directory
    echo -e "$(style_info "Setting up virtual environment...")"
    if [ ! -d "$PROJECT_ROOT/venv" ]; then
        cd "$PROJECT_ROOT" && python3 -m venv venv
        echo -e "$(style_success "Virtual environment created")"
    else
        echo -e "$(style_warning "Virtual environment already exists")"
    fi
    
    # Activate virtual environment
    source "$PROJECT_ROOT/venv/bin/activate"
    
    # Install requirements
    echo -e "$(style_info "Installing dependencies...")"
    REQUIREMENTS=(
        "pandas"
        "numpy"
        "pyyaml"
        "tqdm"
    )
    
    for pkg in "${REQUIREMENTS[@]}"; do
        echo -n "Installing $pkg... "
        pip install $pkg > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo -e "$(style_success "✓")"
        else
            echo -e "$(style_error "✗")"
        fi
    done
    
    # Make scripts executable
    if [ -f "$PROJECT_ROOT/src/main.py" ]; then
        chmod +x "$PROJECT_ROOT/src/main.py"
    fi
    if [ -f "$PROJECT_ROOT/src/common/cli.py" ]; then
        chmod +x "$PROJECT_ROOT/src/common/cli.py"
    fi
    
    # Create symlink in user's bin directory if it exists
    if [ -d "$HOME/bin" ]; then
        echo -e "$(style_info "Creating shortcut in ~/bin...")"
        
        # Create trading_cli.sh if it doesn't exist
        cat > "$SCRIPTS_DIR/trading_cli.sh" << EOF
#!/bin/bash
# Simple wrapper for the main run.sh script
"$PROJECT_ROOT/run.sh" "\$@"
EOF
        chmod +x "$SCRIPTS_DIR/trading_cli.sh"
        
        ln -sf "$SCRIPTS_DIR/trading_cli.sh" "$HOME/bin/trading_cli"
        echo -e "$(style_success "Shortcut created: ~/bin/trading_cli")"
        
        # Check if PATH includes ~/bin
        if [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
            echo -e "$(style_warning "Your PATH doesn't include ~/bin")"
            echo -e "  Add this to your ~/.bashrc or ~/.profile:"
            echo -e "  export PATH=\$HOME/bin:\$PATH"
        fi
    fi
    
    # Deactivate virtual environment
    deactivate
    
    echo -e "$(style_success "Setup complete!")"
    read -p "Press Enter to continue..." dummy
}

setup_colors() {
    show_header
    style_header "Color Setup"
    
    # Create color configuration
    create_color_config
    
    # Update main configuration to enable colors
    update_main_config
    
    # Create log viewer script
    create_log_viewer
    
    # Create log color patch for Python
    create_log_color_patch
    
    echo -e "\n$(style_success "Color setup complete!")"
    echo -e "\nTo use colored logs:"
    echo -e "  1. $(style_info "View logs with color:")"
    echo -e "     $SCRIPTS_DIR/view_logs.sh --list            # List available logs"
    echo -e "     $SCRIPTS_DIR/view_logs.sh system.log        # View system log with colors"
    echo -e "     $SCRIPTS_DIR/view_logs.sh -f trading.log    # Follow trading log with colors"
    echo -e ""
    echo -e "  2. $(style_info "Check color configuration:")"
    echo -e "     The file is at $CONFIG_DIR/color_config.yaml"
    
    read -p "Press Enter to continue..." dummy
}

create_log_color_patch() {
    echo -e "$(style_info "Creating log color patch...")"
    
    # Create patch file for log_manager.py
    PATCH_FILE="$SCRIPTS_DIR/log_color_patch.py"
    
    cat > "$PATCH_FILE" << 'EOF'
# log_color_patch.py - Patch to enhance color logging in LogManager

# Import the original module
from src.common.log_manager import LogManager

# Store the original _get_colored_formatter method
original_get_colored_formatter = LogManager._get_colored_formatter

# Enhanced colored formatter with more colors and better visibility
def enhanced_get_colored_formatter(self):
    """Get enhanced formatter with better ANSI color codes"""
    # Define colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    
    # Define bright/bold colors for better visibility
    BRIGHT_RED = "\033[1;31m"
    BRIGHT_GREEN = "\033[1;32m"
    BRIGHT_YELLOW = "\033[1;33m"
    BRIGHT_BLUE = "\033[1;34m"
    BRIGHT_MAGENTA = "\033[1;35m"
    BRIGHT_CYAN = "\033[1;36m"
    
    # Level colors with improved visibility
    COLORS = {
        'DEBUG': BLUE,
        'INFO': GREEN,
        'WARNING': BRIGHT_YELLOW,
        'ERROR': BRIGHT_RED,
        'CRITICAL': BOLD + BRIGHT_RED
    }
    
    # Background highlights for critical and error messages
    BG_RED = "\033[41m"
    
    class EnhancedColoredFormatter(logging.Formatter):
        def format(self, record):
            # Format the timestamp
            timestamp_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}+d{4}'
            
            # Save original message
            original_msg = record.msg
            
            # Apply colors to the log message based on level
            if record.levelname == 'CRITICAL':
                # For critical errors, highlight with background
                record.msg = f"{BOLD}{BG_RED}{record.msg}{RESET}"
            elif record.levelname == 'ERROR':
                # For errors, make them stand out
                record.msg = f"{BRIGHT_RED}{record.msg}{RESET}"
            elif 'error' in record.msg.lower() or 'exception' in record.msg.lower():
                # Highlight error keywords
                record.msg = record.msg.replace('error', f"{BRIGHT_RED}error{RESET}")
                record.msg = record.msg.replace('Error', f"{BRIGHT_RED}Error{RESET}")
                record.msg = record.msg.replace('ERROR', f"{BRIGHT_RED}ERROR{RESET}")
                record.msg = record.msg.replace('exception', f"{BRIGHT_RED}exception{RESET}")
                record.msg = record.msg.replace('Exception', f"{BRIGHT_RED}Exception{RESET}")
            elif 'warning' in record.msg.lower():
                # Highlight warning keywords
                record.msg = record.msg.replace('warning', f"{BRIGHT_YELLOW}warning{RESET}")
                record.msg = record.msg.replace('Warning', f"{BRIGHT_YELLOW}Warning{RESET}")
                record.msg = record.msg.replace('WARNING', f"{BRIGHT_YELLOW}WARNING{RESET}")
            elif 'success' in record.msg.lower():
                # Highlight success keywords
                record.msg = record.msg.replace('success', f"{BRIGHT_GREEN}success{RESET}")
                record.msg = record.msg.replace('Success', f"{BRIGHT_GREEN}Success{RESET}")
                record.msg = record.msg.replace('SUCCESS', f"{BRIGHT_GREEN}SUCCESS{RESET}")
            
            # Format the record
            levelname = record.levelname
            if levelname in COLORS:
                record.levelname = f"{COLORS[levelname]}{levelname}{RESET}"
                record.name = f"{MAGENTA}{record.name}{RESET}"
            
            # Apply default formatting
            formatted_msg = super().format(record)
            
            # Color timestamps
            formatted_msg = re.sub(timestamp_pattern, lambda m: f"{CYAN}{m.group(0)}{RESET}", formatted_msg)
            
            # Restore original message
            record.msg = original_msg
            
            return formatted_msg
    
    return EnhancedColoredFormatter(self.format_str, datefmt=self.date_format)

# Replace the original method with our enhanced version
import re
import logging
LogManager._get_colored_formatter = enhanced_get_colored_formatter

# Function to apply the patch
def apply_color_patch():
    """Apply the color patch to the logging system"""
    log_manager = LogManager.get_instance()
    
    # Force color enabled
    log_manager.console_colors = True
    
    # Reconfigure all existing loggers
    for name in log_manager.configured_loggers:
        if name == "root":
            logger = logging.getLogger()
        else:
            logger = logging.getLogger(name)
            
        # Update console handlers with new formatter
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setFormatter(log_manager._get_colored_formatter())
    
    # Log that we've applied the patch
    logger = logging.getLogger("system")
    logger.info("Enhanced color logging patch applied")

# Export the apply function
__all__ = ["apply_color_patch"]
EOF
    
    echo -e "$(style_success "Created log color patch at $PATCH_FILE")"
    
    # Create an example script to demonstrate how to use the patch
    EXAMPLE_FILE="$SCRIPTS_DIR/enable_colors.py"
    
    cat > "$EXAMPLE_FILE" << 'EOF'
#!/usr/bin/env python3
# enable_colors.py - Example of how to enable enhanced color logging

# Add this to your main.py or other entry point

def enable_enhanced_logging():
    """Enable enhanced color logging throughout the application"""
    try:
        # Import and apply the color patch
        from scripts.log_color_patch import apply_color_patch
        apply_color_patch()
        print("Enhanced color logging enabled")
    except ImportError:
        print("Color patch not found - using standard logging")
    except Exception as e:
        print(f"Failed to apply color patch: {str(e)}")

# Example usage in main.py:
if __name__ == "__main__":
    # Enable enhanced color logging
    enable_enhanced_logging()
    
    # Continue with normal application initialization
    # ...
EOF
    
    echo -e "$(style_success "Created example script at $EXAMPLE_FILE")"
}

#------------------------------------------
# Network Test Function
#------------------------------------------
run_network_test() {
    show_header
    style_header "Network Test"
    
    # Create logs directory if it doesn't exist
    mkdir -p "$LOG_DIR"
    NETWORK_LOG="$LOG_DIR/network_test.log"
    TEST_SCRIPT="$PROJECT_ROOT/src/exchange/network.py"
    
    # Check if test script exists
    if [ ! -f "$TEST_SCRIPT" ]; then
        echo -e "$(style_error "Network test script not found at $TEST_SCRIPT")"
        echo -e "$(style_info "Creating a basic network test script...")"
        
        # Create directory for test script if it doesn't exist
        mkdir -p "$(dirname "$TEST_SCRIPT")"
        
        # Create a basic network test script
        cat > "$TEST_SCRIPT" << 'EOF'
#!/usr/bin/env python3
# Basic network test script for Trading System

import os
import sys
import socket
import urllib.request
import urllib.error
import ssl
import time
import json
from datetime import datetime

def test_internet_connection():
    """Test basic internet connectivity"""
    print("\n=== Testing Internet Connection ===")
    hosts = ["google.com", "github.com", "amazon.com", "cloudflare.com"]
    
    for host in hosts:
        try:
            socket.create_connection((host, 80), timeout=5)
            print(f"✓ Connection to {host} successful")
        except (socket.timeout, socket.error) as e:
            print(f"✗ Connection to {host} failed: {e}")
    
def test_api_connectivity():
    """Test connectivity to common financial APIs"""
    print("\n=== Testing API Connectivity ===")
    
    apis = [
        {"name": "CoinGecko", "url": "https://api.coingecko.com/api/v3/ping"},
        {"name": "Binance", "url": "https://api.binance.com/api/v3/time"},
        {"name": "Alpha Vantage", "url": "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo"}
    ]
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    for api in apis:
        try:
            start = time.time()
            with urllib.request.urlopen(api["url"], timeout=10, context=ctx) as response:
                data = response.read()
                end = time.time()
                latency = (end - start) * 1000  # Convert to ms
                print(f"✓ {api['name']} API: Status {response.status} ({latency:.1f}ms)")
        except Exception as e:
            print(f"✗ {api['name']} API failed: {str(e)}")

def test_dns_resolution():
    """Test DNS resolution for trading domains"""
    print("\n=== Testing DNS Resolution ===")
    
    domains = [
        "binance.com",
        "coinbase.com",
        "kraken.com",
        "ftx.com",
        "bitfinex.com"
    ]
    
    for domain in domains:
        try:
            ip = socket.gethostbyname(domain)
            print(f"✓ {domain} resolved to {ip}")
        except socket.gaierror as e:
            print(f"✗ {domain} resolution failed: {e}")

def run_tests():
    """Run all network tests"""
    print(f"=== Trading System Network Test ===")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S%z')}")
    print(f"Hostname: {socket.gethostname()}")
    
    test_internet_connection()
    test_dns_resolution()
    test_api_connectivity()
    
    print("\n=== Test Summary ===")
    print("Network test completed")

if __name__ == "__main__":
    run_tests()
EOF
        
        # Make the script executable
        chmod +x "$TEST_SCRIPT"
        echo -e "$(style_success "Created basic network test script")"
    fi
    
    echo -e "$(style_info "Running network tests...")"
    echo "=== Starting Network Tests ===" | tee -a "$NETWORK_LOG"
    echo "Timestamp: $(date)" | tee -a "$NETWORK_LOG"
    echo "" | tee -a "$NETWORK_LOG"
    
    # Check Python version
    echo "Python Version:" | tee -a "$NETWORK_LOG"
    python3 --version | tee -a "$NETWORK_LOG"
    echo "" | tee -a "$NETWORK_LOG"
    
    # Run the test script
    echo "Running network tests..." | tee -a "$NETWORK_LOG"
    
    # Activate virtual environment
    if [ -d "$PROJECT_ROOT/venv" ]; then
        source "$PROJECT_ROOT/venv/bin/activate"
        
        # Run tests with appropriate Python path
        PYTHONPATH="$PROJECT_ROOT" python3 "$TEST_SCRIPT" 2>&1 | tee -a "$NETWORK_LOG"
        TEST_EXIT_CODE=${PIPESTATUS[0]}
        
        # Deactivate virtual environment
        deactivate
    else
        echo -e "$(style_warning "Virtual environment not found. Running with system Python...")" | tee -a "$NETWORK_LOG"
        PYTHONPATH="$PROJECT_ROOT" python3 "$TEST_SCRIPT" 2>&1 | tee -a "$NETWORK_LOG"
        TEST_EXIT_CODE=${PIPESTATUS[0]}
    fi
    
    # Check exit status
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo "" | tee -a "$NETWORK_LOG"
        echo -e "$(style_success "Network tests completed successfully")" | tee -a "$NETWORK_LOG"
    else
        echo "" | tee -a "$NETWORK_LOG"
        echo -e "$(style_error "Network tests failed with errors")" | tee -a "$NETWORK_LOG"
    fi
    
    echo "Log saved to: $NETWORK_LOG" | tee -a "$NETWORK_LOG"
    read -p "Press Enter to continue..." dummy
}

#------------------------------------------
# Configuration Management Functions
#------------------------------------------
save_config() {
    local config_name="$1"
    if [ -z "$config_name" ]; then
        config_name="config_${SESSION_ID}"
    fi
    
    local save_file="${SAVED_CONFIG_DIR}/${config_name}.conf"
    
    echo "# Trading configuration saved on $(date)" > "$save_file"
    echo "MODE=\"$MODE\"" >> "$save_file"
    [ "$MODE" == "backtest" ] && echo "BACKTEST_ENGINE=\"$BACKTEST_ENGINE\"" >> "$save_file"
    echo "STRATEGY=\"$STRATEGY\"" >> "$save_file"
    echo "SYMBOL=\"$SYMBOL\"" >> "$save_file"
    echo "TIMEFRAME=\"$TIMEFRAME\"" >> "$save_file"
    [ "$MODE" == "backtest" ] && echo "START_DATE=\"$START_DATE\"" >> "$save_file"
    [ "$MODE" == "backtest" ] && echo "END_DATE=\"$END_DATE\"" >> "$save_file"
    echo "CONFIG=\"$CONFIG\"" >> "$save_file"
    echo "DEBUG=\"$DEBUG\"" >> "$save_file"
    echo "ADDITIONAL_PARAMS=\"$ADDITIONAL_PARAMS\"" >> "$save_file"
    
    echo -e "$(style_success "Configuration saved to: $save_file")"
}

load_config() {
    local configs=()
    local i=0
    
    style_menu_section "Available saved configurations"
    
    # List all available configuration files
    for conf_file in "$SAVED_CONFIG_DIR"/*.conf; do
        if [ -f "$conf_file" ]; then
            local conf_name=$(basename "$conf_file" .conf)
            local conf_date=$(grep "# Trading configuration saved on" "$conf_file" | cut -d' ' -f5-)
            
            echo -e "  ${BRIGHT_GREEN}$i${RESET}) $conf_name ${DIM}($conf_date)${RESET}"
            configs[$i]="$conf_file"
            ((i++))
        fi
    done
    
    if [ $i -eq 0 ]; then
        echo -e "$(style_warning "No saved configurations found.")"
        return 1
    fi
    
    read -p "Select configuration to load [0-$((i-1)), r=return]: " conf_choice
    
    if [ "$conf_choice" == "r" ]; then
        return 1
    elif [[ "$conf_choice" =~ ^[0-9]+$ ]] && [ "$conf_choice" -lt "$i" ]; then
        echo -e "$(style_info "Loading configuration: ${configs[$conf_choice]}")"
        source "${configs[$conf_choice]}"
        return 0
    else
        echo -e "$(style_error "Invalid selection.")"
        return 1
    fi
}

#------------------------------------------
# Logs Management Functions
#------------------------------------------
show_logs_menu() {
    local logs=()
    local i=0
    
    show_header
    style_header "Log Management"
    
    # Find all log files
    for log_file in "$LOG_DIR"/*.log; do
        if [ -f "$log_file" ]; then
            local log_name=$(basename "$log_file")
            local log_date=$(stat -c %y "$log_file" | cut -d'.' -f1)
            local log_size=$(du -h "$log_file" | cut -f1)
            
            # Color-code different log types
            local name_style="${RESET}"
            if [[ "$log_name" == *error* ]]; then
                name_style="${BRIGHT_RED}"
            elif [[ "$log_name" == *warning* ]]; then
                name_style="${BRIGHT_YELLOW}"
            elif [[ "$log_name" == *system* ]]; then
                name_style="${BRIGHT_BLUE}"
            elif [[ "$log_name" == *trading* ]]; then
                name_style="${BRIGHT_CYAN}"
            elif [[ "$log_name" == *network* ]]; then
                name_style="${BRIGHT_MAGENTA}"
            fi
            
            echo -e "  ${BRIGHT_GREEN}$i${RESET}) ${name_style}$log_name${RESET} (${BRIGHT_CYAN}$log_size${RESET}) ${DIM}$log_date${RESET}"
            logs[$i]="$log_file"
            ((i++))
        fi
    done
    
    if [ $i -eq 0 ]; then
        echo -e "$(style_warning "No log files found.")"
        read -p "Press Enter to return to main menu..." dummy
        return
    fi
    
    echo -e ""
    style_menu_item "v" "View a log file"
    style_menu_item "c" "Clean old logs"
    style_menu_item "r" "Return to main menu"
    
    read -p "Enter your choice: " log_choice
    
    case $log_choice in
        v)
            read -p "Enter log number to view [0-$((i-1))]: " log_number
            if [[ "$log_number" =~ ^[0-9]+$ ]] && [ "$log_number" -lt "$i" ]; then
                echo -e "$(style_info "Viewing log: ${logs[$log_number]}")"
                echo -e "Press q to exit, arrow keys to navigate"
                sleep 1
                
                if $COLOR_ENABLED; then
                    # Use less with raw control chars for colors
                    cat "${logs[$log_number]}" | colorize_log_output | less -R
                else
                    # Use regular less
                    less "${logs[$log_number]}"
                fi
            else
                echo -e "$(style_error "Invalid selection.")"
                sleep 1
            fi
            show_logs_menu
            ;;
        c)
            clean_old_logs
            show_logs_menu
            ;;
        r)
            return
            ;;
        *)
            echo -e "$(style_error "Invalid selection.")"
            sleep 1
            show_logs_menu
            ;;
    esac
}

clean_old_logs() {
    show_header
    style_header "Clean Old Logs"
    
    # Count total logs
    local total_logs=$(find "$LOG_DIR" -name "*.log" | wc -l)
    local total_size=$(du -sh "$LOG_DIR" | cut -f1)
    
    echo -e "Log directory: $LOG_DIR"
    echo -e "Total logs: $total_logs"
    echo -e "Total size: $total_size"
    echo -e ""
    
    style_menu_section "Cleaning Options"
    style_menu_item "1" "Delete logs older than 7 days"
    style_menu_item "2" "Delete logs older than 30 days"
    style_menu_item "3" "Delete all logs except current session"
    style_menu_item "4" "Compress old logs (keeps files but reduces size)"
    style_menu_item "5" "Return to log menu"
    
    read -p "Enter choice [1-5]: " clean_choice
    
    case $clean_choice in
        1)
            echo -e "$(style_warning "Deleting logs older than 7 days...")"
            find "$LOG_DIR" -name "*.log" -type f -mtime +7 -delete
            echo -e "$(style_success "Old logs deleted.")"
            ;;
        2)
            echo -e "$(style_warning "Deleting logs older than 30 days...")"
            find "$LOG_DIR" -name "*.log" -type f -mtime +30 -delete
            echo -e "$(style_success "Old logs deleted.")"
            ;;
        3)
            echo -e "$(style_warning "Deleting all logs except current session...")"
            find "$LOG_DIR" -name "*.log" -type f -not -name "*${SESSION_ID}*" -delete
            echo -e "$(style_success "Logs deleted.")"
            ;;
        4)
            echo -e "$(style_info "Compressing old logs...")"
            for log_file in $(find "$LOG_DIR" -name "*.log" -type f -mtime +1 -not -name "*.gz"); do
                gzip -f "$log_file"
                echo -e "Compressed: $log_file"
            done
            echo -e "$(style_success "Log compression complete.")"
            ;;
        5|*)
            return
            ;;
    esac
    
    # Show new total
    local new_total_logs=$(find "$LOG_DIR" -name "*.log" | wc -l)
    local new_total_size=$(du -sh "$LOG_DIR" | cut -f1)
    
    echo -e "New total logs: $new_total_logs"
    echo -e "New total size: $new_total_size"
    
    read -p "Press Enter to continue..." dummy
}

#------------------------------------------
# System Diagnostic Functions
#------------------------------------------
run_diagnostics() {
    show_header
    style_header "System Diagnostics"
    
    # Python version
    style_menu_section "Python Environment"
    local py_version=$(python3 --version 2>&1)
    local pip_version=$(pip --version 2>&1 | cut -d" " -f1,2)
    
    echo -e "Python Version: ${BRIGHT_GREEN}$py_version${RESET}"
    echo -e "Pip Version: ${BRIGHT_GREEN}$pip_version${RESET}"
    
    # Virtual env status
    style_menu_section "Virtual Environment"
    if [ -d "$PROJECT_ROOT/venv" ]; then
        echo -e "Status: ${BRIGHT_GREEN}Installed${RESET}"
        echo -e "Path: $PROJECT_ROOT/venv"
    else
        echo -e "Status: ${BRIGHT_RED}Not found${RESET}"
    fi
    
    # Installed packages
    style_menu_section "Installed Packages"
    if [ -d "$PROJECT_ROOT/venv" ]; then
        source "$PROJECT_ROOT/venv/bin/activate"
        for pkg in pandas numpy pyyaml tqdm; do
            if python3 -c "import $pkg" &> /dev/null; then
                ver=$(python3 -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "Unknown")
                echo -e "$pkg: ${BRIGHT_GREEN}Installed${RESET} (Version: $ver)"
            else
                echo -e "$pkg: ${BRIGHT_RED}Missing${RESET}"
            fi
        done
        deactivate
    else
        echo -e "${BRIGHT_YELLOW}Virtual environment not found, checking system packages${RESET}"
        for pkg in pandas numpy pyyaml tqdm; do
            if python3 -c "import $pkg" &> /dev/null; then
                echo -e "$pkg: ${BRIGHT_GREEN}Installed${RESET} (system)"
            else
                echo -e "$pkg: ${BRIGHT_RED}Missing${RESET}"
            fi
        done
    fi
    
    # System info
    style_menu_section "System Information"
    echo -e "Operating System: $(uname -a)"
    echo -e "CPU: $(grep "model name" /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs || echo "Unknown")"
    
    # Color code memory usage
    local mem_total=$(free -m | grep Mem | awk '{print $2}')
    local mem_used=$(free -m | grep Mem | awk '{print $3}')
    local mem_percent=$((mem_used * 100 / mem_total))
    local mem_color=$GREEN
    
    if [ $mem_percent -gt 80 ]; then
        mem_color=$BRIGHT_RED
    elif [ $mem_percent -gt 60 ]; then
        mem_color=$BRIGHT_YELLOW
    fi
    
    echo -e "Memory: ${mem_color}${mem_used}MB${RESET} / ${mem_total}MB (${mem_color}${mem_percent}%${RESET})"
    
    # Color code disk usage
    local disk_avail=$(df -h . | tail -1 | awk '{print $4}')
    local disk_percent=$(df -h . | tail -1 | awk '{print $5}' | tr -d '%')
    local disk_color=$GREEN
    
    if [ $disk_percent -gt 90 ]; then
        disk_color=$BRIGHT_RED
    elif [ $disk_percent -gt 70 ]; then
        disk_color=$BRIGHT_YELLOW
    fi
    
    echo -e "Disk Space: ${disk_color}${disk_avail}${RESET} available (${disk_color}${disk_percent}%${RESET} used)"
    
    # Project structure check
    style_menu_section "Project Structure"
    local structure_ok=true
    
    check_dir() {
        if [ -d "$1" ]; then
            echo -e "$2: ${BRIGHT_GREEN}OK${RESET}"
        else
            echo -e "$2: ${BRIGHT_RED}Missing${RESET}"
            structure_ok=false
        fi
    }
    
    check_file() {
        if [ -f "$1" ]; then
            echo -e "$2: ${BRIGHT_GREEN}OK${RESET}"
        else
            echo -e "$2: ${BRIGHT_RED}Missing${RESET}"
            structure_ok=false
        fi
    }
    
    check_dir "$PROJECT_ROOT/src" "Source Directory"
    check_dir "$PROJECT_ROOT/conf" "Config Directory"
    check_dir "$PROJECT_ROOT/scripts" "Scripts Directory"
    
    # Network connectivity check
    style_menu_section "Network Connectivity"
    if ping -c 1 google.com &> /dev/null; then
        echo -e "Internet Connection: ${BRIGHT_GREEN}OK${RESET}"
    else
        echo -e "Internet Connection: ${BRIGHT_RED}Failed${RESET}"
    fi
    
    # Color log configuration check
    style_menu_section "Color Configuration"
    if $COLOR_ENABLED; then
        echo -e "Color Support: ${BRIGHT_GREEN}Enabled${RESET}"
    else
        echo -e "Color Support: ${BRIGHT_YELLOW}Disabled${RESET}"
    fi
    
    if [ -f "$CONFIG_DIR/color_config.yaml" ]; then
        echo -e "Color Config File: ${BRIGHT_GREEN}Present${RESET}"
    else
        echo -e "Color Config File: ${BRIGHT_YELLOW}Not found${RESET}"
        echo -e "  Run the color setup option from the main menu to create color configuration"
    fi
    
    read -p "Press Enter to return to main menu..." dummy
}

#------------------------------------------
# Trading System Functions
#------------------------------------------
build_command() {
    CMD="python3 -m src.main --mode $MODE"
    [ "$MODE" == "backtest" ] && CMD="$CMD --backtest-engine $BACKTEST_ENGINE"
    CMD="$CMD --strategy $STRATEGY --symbol \"$SYMBOL\" --timeframe $TIMEFRAME"
    [ "$MODE" == "backtest" ] && CMD="$CMD --start-date $START_DATE --end-date $END_DATE"
    CMD="$CMD --config $CONFIG $DEBUG $ADDITIONAL_PARAMS"
}

run_trading_system() {
    # Log the command
    echo "=== Trading System Started ===" >> "$LOG_FILE"
    echo "Date: $(date)" >> "$LOG_FILE"
    echo "Command: $CMD" >> "$LOG_FILE"
    
    # Activate virtual environment if it exists
    if [ -d "$PROJECT_ROOT/venv" ]; then
        source "$PROJECT_ROOT/venv/bin/activate"
    else
        echo -e "$(style_warning "Virtual environment not found. Using system Python...")"
    fi
    
    # Start the trading system in background with colorized output
    if $COLOR_ENABLED; then
        # Use colored output
        eval "$CMD" 2>&1 | colorize_log_output | tee -a "$LOG_FILE" &
    else
        # No color processing
        eval "$CMD" 2>&1 | tee -a "$LOG_FILE" &
    fi
    
    TRADING_PID=$!
    echo $TRADING_PID > "$PID_FILE"
    
    echo -e "\n$(style_info "Trading system running with PID: $TRADING_PID")"
    echo -e "Press [${BOLD}C${RESET}] to view configuration"
    echo -e "Press [${BOLD}L${RESET}] to view live log"
    echo -e "Press [${BOLD}S${RESET}] to save current configuration"
    echo -e "Press [${BOLD}Q${RESET}] to stop trading and return to menu"
    
    # Wait for user input or process completion
    while ps -p $TRADING_PID > /dev/null; do
        if read -t 1 -n 1 user_input; then
            case $user_input in
                C|c)
                    echo -e "\n$(style_header "Current Configuration:")"
                    style_parameter "Mode" "$MODE"
                    [ "$MODE" == "backtest" ] && style_parameter "Engine" "$BACKTEST_ENGINE"
                    style_parameter "Strategy" "$STRATEGY"
                    style_parameter "Symbol(s)" "$SYMBOL"
                    style_parameter "Timeframe" "$TIMEFRAME"
                    [ "$MODE" == "backtest" ] && style_parameter "Date Range" "$START_DATE to $END_DATE"
                    style_parameter "Config" "$CONFIG"
                    [ -n "$DEBUG" ] && style_parameter "Debug" "Enabled"
                    ;;
                L|l)
                    echo -e "\n$(style_header "Opening log file...") (Press Ctrl+C to return)"
                    sleep 1
                    if $COLOR_ENABLED; then
                        # Use the colorization function to tail the log
                        tail -f "$LOG_FILE" | colorize_log_output || true
                    else
                        # No color processing
                        tail -f "$LOG_FILE" || true
                    fi
                    echo -e "\n$(style_info "Returned from log view")"
                    ;;
                S|s)
                    echo -e "\n$(style_header "Save current configuration")"
                    read -p "Enter a name for this configuration: " config_name
                    save_config "$config_name"
                    ;;
                Q|q)
                    echo -e "\n$(style_warning "Stopping trading system...")"
                    kill -SIGTERM $TRADING_PID 2>/dev/null
                    wait $TRADING_PID 2>/dev/null
                    echo -e "$(style_success "Trading system stopped.")"
                    rm -f "$PID_FILE"
                    return 0
                    ;;
            esac
            echo -e "\nPress [${BOLD}C${RESET}] Config, [${BOLD}L${RESET}] Log, [${BOLD}S${RESET}] Save, [${BOLD}Q${RESET}] Quit"
        fi
    done
    
    # Process completed on its own
    wait $TRADING_PID
    EXIT_CODE=$?
    rm -f "$PID_FILE"
    
    echo -e "\n=== Trading System Finished ===" >> "$LOG_FILE"
    echo "Exit Code: $EXIT_CODE" >> "$LOG_FILE"
    echo "End Time: $(date)" >> "$LOG_FILE"
    
    # Deactivate virtual environment if it was activated
    deactivate 2>/dev/null || true
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "\n$(style_success "Trading system completed successfully")"
    else
        echo -e "\n$(style_error "Trading system encountered errors (Exit code: $EXIT_CODE)")"
    fi
    
    read -p "Press Enter to continue..." dummy
    return $EXIT_CODE
}

ask_rerun() {
    style_menu_section "Run Options"
    style_menu_item "1" "Run again with same configuration"
    style_menu_item "2" "Configure new run"
    style_menu_item "3" "Return to main menu"
    
    read -p "Enter choice [1-3]: " rerun_choice
    
    case $rerun_choice in
        1)
            run_trading_system
            ask_rerun
            ;;
        2)
            configure_and_run
            ;;
        3)
            # Return to main menu
            return
            ;;
        *)
            echo -e "$(style_error "Invalid choice.")"
            ask_rerun
            ;;
    esac
}

configure_and_run() {
    # Default values
    MODE="backtest"
    BACKTEST_ENGINE="ohlcv"
    STRATEGY="dual_ma"
    SYMBOL="BTC/USDT"
    TIMEFRAME="1h"
    START_DATE="2025-01-01"
    END_DATE="2025-04-28"  # Updated to current date
    CONFIG="$PROJECT_ROOT/conf/config.yaml"
    DEBUG=""
    ADDITIONAL_PARAMS=""
    
    show_header
    
    # Mode selection with interactive menu
    style_menu_section "Select trading mode"
    style_menu_item "1" "Backtest  - Historical data simulation"
    style_menu_item "2" "Paper     - Real-time simulation (no real orders)"
    style_menu_item "3" "Live      - ${BRIGHT_RED}Real-money trading${RESET}"
    read -p "Enter choice [1-3] (default: 1): " mode_choice
    
    case $mode_choice in
        2) MODE="paper" ;;
        3) 
            MODE="live"
            echo -e "\n${BOLD}${BG_RED}⚠️  WARNING: LIVE TRADING MODE SELECTED ⚠️${RESET}"
            echo -e "${BRIGHT_RED}This will use REAL FUNDS from your exchange account.${RESET}"
            read -p "Are you sure you want to continue? (y/n): " confirm
            if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
                echo "Exiting live mode"
                configure_and_run
                return
            fi
            ;;
        *) MODE="backtest" ;;
    esac
    
    # Backtest engine selection (only for backtest mode)
    if [ "$MODE" == "backtest" ]; then
        style_menu_section "Select backtest engine"
        style_menu_item "1" "OHLCV (Basic)       - Standard candlestick backtesting"
        style_menu_item "2" "Market Replay       - Advanced order book simulation"
        read -p "Enter choice [1-2] (default: 1): " engine_choice
        
        case $engine_choice in
            2) BACKTEST_ENGINE="market_replay" ;;
            *) BACKTEST_ENGINE="ohlcv" ;;
        esac
    fi
    
    # Strategy selection
    style_menu_section "Select strategy"
    style_menu_item "1" "dual_ma         - Dual Moving Average crossover"
    style_menu_item "2" "neural_network  - Neural Network prediction"
    style_menu_item "3" "Custom strategy"
    read -p "Enter choice [1-3] (default: 1): " strategy_choice
    
    case $strategy_choice in
        2) STRATEGY="neural_network" ;;
        3) 
            read -p "Enter custom strategy name: " custom_strategy
            STRATEGY="$custom_strategy"
            ;;
        *) STRATEGY="dual_ma" ;;
    esac
    
    # Symbol
    read -p "Enter trading symbol(s) (comma-separated, default: BTC/USDT): " symbol_input
    [ -n "$symbol_input" ] && SYMBOL="$symbol_input"
    
    # Timeframe
    style_menu_section "Select timeframe"
    echo -e "  ${BRIGHT_GREEN}1${RESET}) 1m    ${BRIGHT_GREEN}2${RESET}) 5m    ${BRIGHT_GREEN}3${RESET}) 15m   ${BRIGHT_GREEN}4${RESET}) 30m"
    echo -e "  ${BRIGHT_GREEN}5${RESET}) 1h    ${BRIGHT_GREEN}6${RESET}) 4h    ${BRIGHT_GREEN}7${RESET}) 1d    ${BRIGHT_GREEN}8${RESET}) 1w"
    read -p "Enter choice [1-8] (default: 5): " timeframe_choice
    
    case $timeframe_choice in
        1) TIMEFRAME="1m" ;;
        2) TIMEFRAME="5m" ;;
        3) TIMEFRAME="15m" ;;
        4) TIMEFRAME="30m" ;;
        6) TIMEFRAME="4h" ;;
        7) TIMEFRAME="1d" ;;
        8) TIMEFRAME="1w" ;;
        *) TIMEFRAME="1h" ;;
    esac
    
    # Date range (for backtest mode)
    if [ "$MODE" == "backtest" ]; then
        read -p "Start date (YYYY-MM-DD, default: $START_DATE): " start_input
        read -p "End date (YYYY-MM-DD, default: $END_DATE): " end_input
        
        [ -n "$start_input" ] && START_DATE="$start_input"
        [ -n "$end_input" ] && END_DATE="$end_input"
    fi
    
    # Config file
    read -p "Config file path (default: $CONFIG): " config_input
    [ -n "$config_input" ] && CONFIG="$config_input"
    
    # Debug mode
    read -p "Enable debug logging? (y/n, default: n): " debug_choice
    if [[ "$debug_choice" == "y" || "$debug_choice" == "Y" ]]; then
        DEBUG="--debug"
    fi
    
    # Additional parameters
    read -p "Additional parameters: " ADDITIONAL_PARAMS
    
    # Build command
    build_command
    
    # Display summary
    style_header "Configuration Summary:"
    style_parameter "Mode" "$MODE"
    [ "$MODE" == "backtest" ] && style_parameter "Engine" "$BACKTEST_ENGINE"
    style_parameter "Strategy" "$STRATEGY"
    style_parameter "Symbol(s)" "$SYMBOL"
    style_parameter "Timeframe" "$TIMEFRAME"
    [ "$MODE" == "backtest" ] && style_parameter "Date Range" "$START_DATE to $END_DATE"
    style_parameter "Config" "$CONFIG"
    [ -n "$DEBUG" ] && style_parameter "Debug" "Enabled"
    [ -n "$ADDITIONAL_PARAMS" ] && style_parameter "Additional" "$ADDITIONAL_PARAMS"
    
    # Confirm execution
    echo -e "\n$(style_command "Command: $CMD")"
    read -p "Execute trading system? (y/n): " run_choice
    
    if [[ "$run_choice" == "y" || "$run_choice" == "Y" ]]; then
        echo -e "\n$(style_info "Starting trading system...")\n"
        run_trading_system
        ask_rerun
    else
        echo -e "\n$(style_warning "Operation cancelled")"
        sleep 1
    fi
}

#------------------------------------------
# Main Menu Functions
#------------------------------------------
show_main_menu() {
    local choice
    
    show_header
    
    style_menu_section "Main Menu"
    style_menu_item "1" "Start new trading session"
    style_menu_item "2" "Load saved configuration"
    style_menu_item "3" "View logs"
    style_menu_item "4" "System diagnostics"
    style_menu_item "5" "Network test"
    style_menu_item "6" "Setup & configuration"
    style_menu_item "7" "Exit"
    
    read -p "Enter choice [1-7]: " choice
    
    case $choice in
        1)
            configure_and_run
            ;;
        2)
            if load_config; then
                read -p "Start trading with this configuration? (y/n): " start_choice
                if [[ "$start_choice" == "y" || "$start_choice" == "Y" ]]; then
                    build_command
                    run_trading_system
                    ask_rerun
                fi
            fi
            ;;
        3)
            show_logs_menu
            ;;
        4)
            run_diagnostics
            ;;
        5)
            run_network_test
            ;;
        6)
            show_setup_menu
            ;;
        7)
            echo -e "$(style_success "Exiting...")"
            deactivate 2>/dev/null || true
            exit 0
            ;;
        *)
            echo -e "$(style_error "Invalid choice.")"
            sleep 1
            ;;
    esac
}

show_setup_menu() {
    local setup_choice
    
    show_header
    style_header "Setup & Configuration"
    
    style_menu_section "Options"
    style_menu_item "1" "Setup virtual environment"
    style_menu_item "2" "Configure color settings"
    style_menu_item "3" "Clean project directories"
    style_menu_item "4" "Return to main menu"
    
    read -p "Enter choice [1-4]: " setup_choice
    
    case $setup_choice in
        1)
            setup_virtual_env
            ;;
        2)
            setup_colors
            ;;
        3)
            clean_project
            ;;
        4)
            return
            ;;
        *)
            echo -e "$(style_error "Invalid choice.")"
            sleep 1
            show_setup_menu
            ;;
    esac
}

clean_project() {
    show_header
    style_header "Clean Project Directories"
    
    # Define target directories to clean
    TARGET_DIRS=("logs" "reports" "cached" "__pycache__")
    
    echo -e "This will delete these directories recursively from your project:"
    for dir in "${TARGET_DIRS[@]}"; do
        echo -e "  - ${BRIGHT_YELLOW}$dir/${RESET}"
    done
    
    # Count items that would be deleted
    echo -e "\nSearching for directories to clean..."
    total_count=0
    for dir_name in "${TARGET_DIRS[@]}"; do
        count=$(find "$PROJECT_ROOT" -type d -name "$dir_name" | wc -l)
        total_count=$((total_count + count))
        if [ $count -gt 0 ]; then
            echo -e "Found ${BRIGHT_YELLOW}$count${RESET} $dir_name/ directories"
        fi
    done
    
    if [ $total_count -eq 0 ]; then
        echo -e "$(style_info "No directories found to clean.")"
        read -p "Press Enter to return..." dummy
        return
    fi
    
    # Confirmation prompt
    read -p "Proceed with deletion? [y/N] " confirm
    if [[ "${confirm,,}" != "y" ]]; then
        echo -e "$(style_info "Operation cancelled")"
        read -p "Press Enter to return..." dummy
        return
    fi
    
    # Find and delete target directories
    deleted_count=0
    for dir_name in "${TARGET_DIRS[@]}"; do
        echo -e "Searching for $dir_name/ directories..."
        
        # Use find to locate all matching directories
        for dir_path in $(find "$PROJECT_ROOT" -type d -name "$dir_name"); do
            echo -e "Deleting: $dir_path"
            rm -rf "$dir_path"
            deleted_count=$((deleted_count + 1))
        done
    done
    
    if [ $deleted_count -gt 0 ]; then
        echo -e "$(style_success "Cleaned $deleted_count directories successfully")"
    else
        echo -e "$(style_warning "No directories were deleted")"
    fi
    
    read -p "Press Enter to continue..." dummy
}

#------------------------------------------
# Main Execution
#------------------------------------------
# Check color support
check_color_support "$@"

# Main program loop
clear
style_header "Trading System CLI"
style_subheader "=================================="
echo -e "$(style_info "Starting...")"

# Check if script is run with arguments
if [ $# -gt 0 ]; then
    # Process command line arguments
    case "$1" in
        --help|-h)
            echo "Trading System CLI"
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --setup, -s      Run setup for virtual environment and dependencies"
            echo "  --colors, -c     Setup color configuration"
            echo "  --network, -n    Run network tests"
            echo "  --logs, -l       Show logs menu"
            echo "  --run, -r        Run trading system with saved configuration"
            echo "  --clean          Clean project directories"
            echo "  --no-color       Disable color output"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
        --setup|-s)
            setup_virtual_env
            ;;
        --colors|-c)
            setup_colors
            ;;
        --network|-n)
            run_network_test
            ;;
        --logs|-l)
            show_logs_menu
            ;;
        --run|-r)
            if load_config; then
                build_command
                run_trading_system
            else
                echo -e "$(style_error "Could not load configuration")"
                exit 1
            fi
            ;;
        --clean)
            clean_project
            ;;
        *)
            echo -e "$(style_error "Unknown option: $1")"
            echo "Try '$0 --help' for more information."
            exit 1
            ;;
    esac
    exit 0
fi

# Main loop for interactive mode
while true; do
    show_main_menu
done