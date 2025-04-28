#!/bin/bash
# /scripts/run.sh
# Enhanced Trading System CLI Launcher with improved color rendering

# Enhanced ANSI color codes with better visibility
RESET="\033[0m"
BOLD="\033[1m"
DIM="\033[2m"
ITALIC="\033[3m"
UNDERLINE="\033[4m"
BLINK="\033[5m"
REVERSE="\033[7m"
HIDDEN="\033[8m"
STRIKETHROUGH="\033[9m"

# Regular colors
BLACK="\033[30m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
MAGENTA="\033[35m"
CYAN="\033[36m"
WHITE="\033[37m"

# Bright/bold colors for better visibility
BRIGHT_BLACK="\033[90m"
BRIGHT_RED="\033[91m"
BRIGHT_GREEN="\033[92m"
BRIGHT_YELLOW="\033[93m"
BRIGHT_BLUE="\033[94m"
BRIGHT_MAGENTA="\033[95m"
BRIGHT_CYAN="\033[96m"
BRIGHT_WHITE="\033[97m"

# Background colors
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
cd "$PROJECT_ROOT"

# Global variables
SESSION_ID=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/trading_${SESSION_ID}.log"
CONFIG_DIR="$PROJECT_ROOT/conf"
SAVED_CONFIG_DIR="$PROJECT_ROOT/conf/saved_configs"
PID_FILE="/tmp/trading_system_${SESSION_ID}.pid"
RERUN_MODE=false
COLOR_ENABLED=true  # Default to color enabled

# Create necessary directories
mkdir -p "$LOG_DIR"
mkdir -p "$SAVED_CONFIG_DIR"

# Function to check if color is supported and enabled
check_color_support() {
    # Check if terminal supports colors
    if [[ -t 1 && "$TERM" != "dumb" ]]; then
        # Check if color is disabled via --no-color flag
        for arg in "$@"; do
            if [[ "$arg" == "--no-color" ]]; then
                COLOR_ENABLED=false
                break
            fi
        done
        
        # Check if running in pipeline (no TTY)
        if [[ ! -t 1 ]]; then
            COLOR_ENABLED=false
        fi
        
        # Check if NO_COLOR environment variable is set (industry standard)
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
        BLINK=""
        REVERSE=""
        HIDDEN=""
        STRIKETHROUGH=""
        BLACK=""
        RED=""
        GREEN=""
        YELLOW=""
        BLUE=""
        MAGENTA=""
        CYAN=""
        WHITE=""
        BRIGHT_BLACK=""
        BRIGHT_RED=""
        BRIGHT_GREEN=""
        BRIGHT_YELLOW=""
        BRIGHT_BLUE=""
        BRIGHT_MAGENTA=""
        BRIGHT_CYAN=""
        BRIGHT_WHITE=""
        BG_BLACK=""
        BG_RED=""
        BG_GREEN=""
        BG_YELLOW=""
        BG_BLUE=""
        BG_MAGENTA=""
        BG_CYAN=""
        BG_WHITE=""
    fi
}

# Style functions for consistent UI elements
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

style_highlight() {
    echo -e "${BOLD}${BRIGHT_WHITE}$1${RESET}"
}

style_dim() {
    echo -e "${DIM}$1${RESET}"
}

style_menu_item() {
    echo -e "  ${BRIGHT_GREEN}$1${RESET}) $2"
}

style_menu_highlight() {
    echo -e "  ${BG_GREEN}${BLACK} $1 ${RESET} $2"
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

# Trap for graceful exit
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
    
    echo -e "$(style_success "Cleanup complete.")"
    exit 0
}

# Set up trap for various signals
trap trap_handler SIGINT SIGTERM SIGHUP

# Function to save current configuration
save_config() {
    local config_name="$1"
    if [ -z "$config_name" ]; then
        config_name="config_${SESSION_ID}"
    fi
    
    local save_file="${SAVED_CONFIG_DIR}/${config_name}.conf"
    
    echo "# Trading configuration saved on $(date)" > "$save_file"
    echo "MODE=\"$MODE\"" >> "$save_file"
    echo "BACKTEST_ENGINE=\"$BACKTEST_ENGINE\"" >> "$save_file"
    echo "STRATEGY=\"$STRATEGY\"" >> "$save_file"
    echo "SYMBOL=\"$SYMBOL\"" >> "$save_file"
    echo "TIMEFRAME=\"$TIMEFRAME\"" >> "$save_file"
    echo "START_DATE=\"$START_DATE\"" >> "$save_file"
    echo "END_DATE=\"$END_DATE\"" >> "$save_file"
    echo "CONFIG=\"$CONFIG\"" >> "$save_file"
    echo "DEBUG=\"$DEBUG\"" >> "$save_file"
    echo "ADDITIONAL_PARAMS=\"$ADDITIONAL_PARAMS\"" >> "$save_file"
    
    echo -e "$(style_success "Configuration saved to: $save_file")"
}

# Function to load a saved configuration
load_config() {
    local configs=()
    local i=0
    
    style_menu_section "Available saved configurations"
    
    # List all available configuration files
    for conf_file in "$SAVED_CONFIG_DIR"/*.conf; do
        if [ -f "$conf_file" ]; then
            local conf_name=$(basename "$conf_file" .conf)
            local conf_date=$(grep "# Trading configuration saved on" "$conf_file" | cut -d' ' -f5-)
            
            echo -e "  ${BRIGHT_GREEN}$i${RESET}) $conf_name $(style_dim "($conf_date)")"
            configs[$i]="$conf_file"
            ((i++))
        fi
    done
    
    if [ $i -eq 0 ]; then
        echo -e "$(style_warning "No saved configurations found.")"
        return 1
    fi
    
    read -p "Select configuration to load [0-$((i-1))]: " conf_choice
    
    if [[ "$conf_choice" =~ ^[0-9]+$ ]] && [ "$conf_choice" -lt "$i" ]; then
        echo -e "$(style_info "Loading configuration: ${configs[$conf_choice]}")"
        source "${configs[$conf_choice]}"
        return 0
    else
        echo -e "$(style_error "Invalid selection.")"
        return 1
    fi
}

# Function to display header
show_header() {
    clear
    style_header "Trading System CLI - Enhanced Edition"
    style_subheader "==========================================="
    
    if $RERUN_MODE; then
        echo -e "${BRIGHT_YELLOW}[RERUN MODE]${RESET} Session ID: $SESSION_ID"
    else
        echo -e "Session ID: $SESSION_ID"
    fi
    echo -e "Log file: $LOG_FILE"
    echo -e ""
}

# Function to colorize log output in real-time
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

# Function to run the trading system
run_trading_system() {
    # Log the command
    echo "=== Trading System Started ===" >> "$LOG_FILE"
    echo "Date: $(date)" >> "$LOG_FILE"
    echo "Command: $CMD" >> "$LOG_FILE"
    
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
    
    echo -e "\n$(style_highlight "Trading system running with PID: $TRADING_PID")"
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
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "\n$(style_success "Trading system completed successfully")"
    else
        echo -e "\n$(style_error "Trading system encountered errors (Exit code: $EXIT_CODE)")"
    fi
    
    return $EXIT_CODE
}

# Function to select virtual environment
select_venv() {
    local venvs=(
        "$PROJECT_ROOT/venv"
    )
    
    # Check for additional venvs in the project directory
    for dir in "$PROJECT_ROOT"/*env*/; do
        if [ -d "$dir" ] && [ -f "$dir/bin/activate" ] && [ "$dir" != "$PROJECT_ROOT/venv/" ]; then
            venvs+=($(basename "$dir"))
        fi
    done
    
    if [ ${#venvs[@]} -eq 1 ]; then
        # Just use the default venv
        VENV_PATH="$PROJECT_ROOT/venv"
    else
        style_menu_section "Select virtual environment"
        for ((i=0; i<${#venvs[@]}; i++)); do
            echo -e "  ${BRIGHT_GREEN}$i${RESET}) ${venvs[$i]}"
        done
        
        read -p "Enter choice [0-$((${#venvs[@]}-1))] (default: 0): " venv_choice
        
        if [[ "$venv_choice" =~ ^[0-9]+$ ]] && [ "$venv_choice" -lt "${#venvs[@]}" ]; then
            VENV_PATH="$PROJECT_ROOT/${venvs[$venv_choice]}"
        else
            VENV_PATH="$PROJECT_ROOT/venv"
        fi
    fi
    
    echo -e "$(style_success "Using virtual environment: $VENV_PATH")"
    source "$VENV_PATH/bin/activate"
}

# Main menu function
show_main_menu() {
    local choice
    
    show_header
    
    style_menu_section "Main Menu"
    style_menu_item "1" "Start new trading session"
    style_menu_item "2" "Load saved configuration"
    style_menu_item "3" "View logs"
    style_menu_item "4" "System diagnostics"
    style_menu_item "5" "Exit"
    
    read -p "Enter choice [1-5]: " choice
    
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

# Logs menu function with color support
show_logs_menu() {
    local logs=()
    local i=0
    
    show_header
    style_header "Available Logs:"
    
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
            elif [[ "$log_name" == *combined* ]]; then
                name_style="${BRIGHT_MAGENTA}"
            fi
            
            echo -e "  ${BRIGHT_GREEN}$i${RESET}) ${name_style}$log_name${RESET} (${BRIGHT_CYAN}$log_size${RESET}) $(style_dim "$log_date")"
            logs[$i]="$log_file"
            ((i++))
        fi
    done
    
    if [ $i -eq 0 ]; then
        echo -e "$(style_warning "No log files found.")"
        read -p "Press Enter to return to main menu..." dummy
        return
    fi
    
    echo -e "  ${BRIGHT_YELLOW}r${RESET}) Return to main menu"
    echo -e "  ${BRIGHT_YELLOW}c${RESET}) Clean old logs"
    
    read -p "Select log to view [0-$((i-1)), r, c]: " log_choice
    
    if [ "$log_choice" == "r" ]; then
        return
    elif [ "$log_choice" == "c" ]; then
        clean_old_logs
    elif [[ "$log_choice" =~ ^[0-9]+$ ]] && [ "$log_choice" -lt "$i" ]; then
        echo -e "$(style_info "Viewing log: ${logs[$log_choice]}")"
        echo -e "Press Ctrl+C to return to menu"
        sleep 1
        
        if $COLOR_ENABLED; then
            # Use less with raw control chars for colors
            cat "${logs[$log_choice]}" | colorize_log_output | less -R
        else
            # Use regular less
            less "${logs[$log_choice]}"
        fi
    else
        echo -e "$(style_error "Invalid selection.")"
        sleep 1
    fi
}

# Function to clean old logs
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

# System diagnostics function with color-coded output
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
    for pkg in pandas numpy pyyaml tqdm; do
        if python3 -c "import $pkg" &> /dev/null; then
            ver=$(python3 -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "Unknown")
            echo -e "$pkg: ${BRIGHT_GREEN}Installed${RESET} (Version: $ver)"
        else
            echo -e "$pkg: ${BRIGHT_RED}Missing${RESET}"
        fi
    done
    
    # System info
    style_menu_section "System Information"
    echo -e "Operating System: $(uname -a)"
    echo -e "CPU: $(grep "model name" /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)"
    
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
    check_file "$PROJECT_ROOT/src/main.py" "Main Script"
    
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
        echo -e "  Run ${BRIGHT_CYAN}./scripts/color_setup.sh${RESET} to create color configuration"
    fi
    
    read -p "Press Enter to return to main menu..." dummy
}

# Ask if user wants to run trading system again
ask_rerun() {
    style_menu_section "Run Options"
    style_menu_item "1" "Run again with same configuration"
    style_menu_item "2" "Configure new run"
    style_menu_item "3" "Return to main menu"
    style_menu_item "4" "Exit"
    
    read -p "Enter choice [1-4]: " rerun_choice
    
    case $rerun_choice in
        1)
            RERUN_MODE=true
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
        4)
            echo -e "$(style_success "Exiting...")"
            deactivate 2>/dev/null || true
            exit 0
            ;;
        *)
            echo -e "$(style_error "Invalid choice.")"
            ask_rerun
            ;;
    esac
}

# Function to build command based on current parameters
build_command() {
    CMD="python3 -m src.main --mode $MODE"
    [ "$MODE" == "backtest" ] && CMD="$CMD --backtest-engine $BACKTEST_ENGINE"
    CMD="$CMD --strategy $STRATEGY --symbol \"$SYMBOL\" --timeframe $TIMEFRAME"
    [ "$MODE" == "backtest" ] && CMD="$CMD --start-date $START_DATE --end-date $END_DATE"
    CMD="$CMD --config $CONFIG $DEBUG $ADDITIONAL_PARAMS"
}

# Configure and run trading system
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
                echo "Exiting..."
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

# Main execution
clear
style_header "Trading System CLI - Enhanced Edition"
style_subheader "=================================="
echo -e "$(style_info "Initializing...")"

# Check color support
check_color_support "$@"

# Select and activate virtual environment
select_venv

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo -e "$(style_error "Error: Python 3 is required but not installed.")"
    deactivate 2>/dev/null || true
    exit 1
fi

# Check for required packages
for pkg in tqdm; do
    if ! python3 -c "import $pkg" &> /dev/null; then
        echo -e "$(style_warning "Installing required package: $pkg")"
        pip install $pkg
    fi
done

# Main loop
while true; do
    show_main_menu
done