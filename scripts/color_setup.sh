#!/bin/bash
# /scripts/color_setup.sh
# Setup script to enable consistent colors for logs and shell output

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
BG_BLACK="\033[40m"
BG_RED="\033[41m"
BG_GREEN="\033[42m"
BG_YELLOW="\033[43m"
BG_BLUE="\033[44m"
BG_MAGENTA="\033[45m"
BG_CYAN="\033[46m"
BG_WHITE="\033[47m"

# Get project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
CONFIG_DIR="$PROJECT_ROOT/conf"

# Create a color configuration file for the project
create_color_config() {
    echo -e "${BOLD}${CYAN}Creating color configuration...${RESET}"
    
    # Create configuration directory if it doesn't exist
    mkdir -p "$CONFIG_DIR"
    
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

# Log file color settings (only affects viewing through less -R or cat)
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
    
    echo -e "${GREEN}✓ Color configuration created at $CONFIG_DIR/color_config.yaml${RESET}"
}

# Update the config.yaml file to ensure color logging is enabled
update_main_config() {
    echo -e "${BOLD}${CYAN}Updating main configuration...${RESET}"
    
    # Check if main config exists
    if [ ! -f "$CONFIG_DIR/config.yaml" ]; then
        echo -e "${YELLOW}! Main config not found. Creating minimal config.${RESET}"
        
        # Create minimal config file
        cat > "$CONFIG_DIR/config.yaml" << EOF
# Main Trading System Configuration

# Logging settings
logging:
  base_path: "./logs"
  level: "INFO"
  format: "%(asctime)s | %(levelname)-8s | %(module)-18s | [%(filename)s:%(lineno)d] | %(message)s"
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
        echo -e "${GREEN}✓ Created minimal config with color logging enabled${RESET}"
    else
        # Check if colors are already enabled
        if grep -q "colors: *true" "$CONFIG_DIR/config.yaml"; then
            echo -e "${GREEN}✓ Color logging already enabled in main config${RESET}"
        else
            # Try to add color configuration
            if grep -q "handlers:" "$CONFIG_DIR/config.yaml" && grep -q "console:" "$CONFIG_DIR/config.yaml"; then
                # Add colors: true under console section
                sed -i '/console:/,/enabled:/s/enabled:.*$/enabled: true\n      colors: true/' "$CONFIG_DIR/config.yaml"
                echo -e "${GREEN}✓ Enabled color logging in main config${RESET}"
            else
                echo -e "${YELLOW}! Could not automatically update config.yaml${RESET}"
                echo -e "${YELLOW}! Please manually ensure 'colors: true' is set under logging.handlers.console${RESET}"
            fi
        fi
    fi
}

# Create a script to view logs with colors
create_log_viewer() {
    echo -e "${BOLD}${CYAN}Creating colored log viewer script...${RESET}"
    
    # Create log viewer script
    cat > "$SCRIPT_DIR/view_logs.sh" << 'EOF'
#!/bin/bash
# /scripts/view_logs.sh - Script to view log files with color support

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
    echo -e ""
    echo -e "${BOLD}Examples:${RESET}"
    echo -e "  $0 --list                    # List all available logs"
    echo -e "  $0 trading_system.log        # View the main system log"
    echo -e "  $0 -f system.log             # Follow the system log for updates"
    echo -e "  $0 -n 50 errors.log          # Show the last 50 lines of errors.log"
    echo -e "  $0 -c strategy               # Show the strategy category log"
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
    chmod +x "$SCRIPT_DIR/view_logs.sh"
    
    echo -e "${GREEN}✓ Created log viewer script at $SCRIPT_DIR/view_logs.sh${RESET}"
}

# Update logging configuration in Python to ensure colors work
create_log_color_patch() {
    echo -e "${BOLD}${CYAN}Creating log color patch...${RESET}"
    
    # Create patch file for log_manager.py
    PATCH_FILE="$SCRIPT_DIR/log_color_patch.py"
    
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
            timestamp_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
            
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
    
    echo -e "${GREEN}✓ Created log color patch at $PATCH_FILE${RESET}"
    
    # Create an example script to demonstrate how to use the patch
    EXAMPLE_FILE="$SCRIPT_DIR/enable_colors.py"
    
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
    
    echo -e "${GREEN}✓ Created example script at $EXAMPLE_FILE${RESET}"
}

# Main execution
echo -e "${BOLD}${CYAN}Trading System Color Setup${RESET}"
echo -e "${CYAN}===========================${RESET}\n"

# Create color configuration file
create_color_config

# Update main configuration to enable colors
update_main_config

# Create log viewer script
create_log_viewer

# Create log color patch for Python
create_log_color_patch

echo -e "\n${BOLD}${GREEN}Color setup complete!${RESET}"
echo -e "\nTo use colored logs:"
echo -e "  1. ${BOLD}View logs with color:${RESET}"
echo -e "     ./scripts/view_logs.sh --list            # List available logs"
echo -e "     ./scripts/view_logs.sh system.log        # View system log with colors"
echo -e "     ./scripts/view_logs.sh -f trading.log    # Follow trading log with colors"
echo -e ""
echo -e "  2. ${BOLD}Apply enhanced color patch in Python:${RESET}"
echo -e "     Add this to your main.py:"
echo -e "     from scripts.log_color_patch import apply_color_patch"
echo -e "     apply_color_patch()"
echo -e ""
echo -e "  3. ${BOLD}Check color configuration:${RESET}"
echo -e "     The file is at ./conf/color_config.yaml"