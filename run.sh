#!/bin/bash
# Trading System CLI

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CONFIG_DIR="$PROJECT_ROOT/conf"
LOG_DIR="$PROJECT_ROOT/logs"
SESSION_ID=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/trading_${SESSION_ID}.log"
PID_FILE="/tmp/trading_system_${SESSION_ID}.pid"
SAVED_CONFIG_DIR="$CONFIG_DIR/saved_configs"

mkdir -p "$LOG_DIR" "$CONFIG_DIR" "$SAVED_CONFIG_DIR"

trap_handler() {
    echo "Received termination signal. Cleaning up..."
    
    if [ -f "$PID_FILE" ]; then
        TRADING_PID=$(cat "$PID_FILE")
        if ps -p $TRADING_PID > /dev/null; then
            echo "Stopping trading process (PID: $TRADING_PID)..."
            kill -SIGTERM $TRADING_PID 2>/dev/null
            sleep 2
            [ $(ps -p $TRADING_PID > /dev/null) ] && kill -SIGKILL $TRADING_PID 2>/dev/null
        fi
        rm -f "$PID_FILE"
    fi
    
    deactivate 2>/dev/null || true
    echo "Cleanup complete."
    exit 0
}

trap trap_handler SIGINT SIGTERM SIGHUP

show_header() {
    clear
    echo "Trading System CLI"
    echo "=================="
    echo "Session ID: $SESSION_ID"
    echo "Project root: $PROJECT_ROOT"
    echo
}

save_config() {
    local config_name="${1:-config_${SESSION_ID}}"
    local save_file="${SAVED_CONFIG_DIR}/${config_name}.conf"
    
    cat > "$save_file" << EOF
# Trading configuration saved on $(date)
MODE="$MODE"
STRATEGY="$STRATEGY"
SYMBOL="$SYMBOL"
TIMEFRAME="$TIMEFRAME"
CONFIG="$CONFIG"
DEBUG="$DEBUG"
ADDITIONAL_PARAMS="$ADDITIONAL_PARAMS"
EOF
    [ "$MODE" == "backtest" ] && cat >> "$save_file" << EOF
BACKTEST_ENGINE="$BACKTEST_ENGINE"
START_DATE="$START_DATE"
END_DATE="$END_DATE"
EOF
    
    echo "Configuration saved to: $save_file"
}

load_config() {
    local configs=()
    local i=0
    
    echo "Available saved configurations"
    for conf_file in "$SAVED_CONFIG_DIR"/*.conf; do
        if [ -f "$conf_file" ]; then
            local conf_name=$(basename "$conf_file" .conf)
            local conf_date=$(grep "# Trading configuration saved on" "$conf_file" | cut -d' ' -f5-)
            echo "  $i) $conf_name ($conf_date)"
            configs[$i]="$conf_file"
            ((i++))
        fi
    done
    
    if [ $i -eq 0 ]; then
        echo "No saved configurations found."
        return 1
    fi
    
    read -p "Select configuration to load [0-$((i-1)), r=return]: " conf_choice
    
    if [ "$conf_choice" == "r" ]; then
        return 1
    elif [[ "$conf_choice" =~ ^[0-9]+$ ]] && [ "$conf_choice" -lt "$i" ]; then
        echo "Loading configuration: ${configs[$conf_choice]}"
        source "${configs[$conf_choice]}"
        return 0
    else
        echo "Invalid selection."
        return 1
    fi
}

setup_virtual_env() {
    show_header
    echo "Setting up virtual environment"
    
    if command -v python3 &> /dev/null; then
        echo "Found $(python3 --version)"
    else
        echo "Python 3 not found"
        echo "Please install Python 3.8 or newer:"
        echo "  sudo apt update"
        echo "  sudo apt install python3 python3-pip python3-venv"
        read -p "Press Enter to continue..." dummy
        return 1
    fi
    
    if [ ! -d "$PROJECT_ROOT/venv" ]; then
        cd "$PROJECT_ROOT" && python3 -m venv venv
        echo "Virtual environment created"
    else
        echo "Virtual environment already exists"
    fi
    
    source "$PROJECT_ROOT/venv/bin/activate"
    
    echo "Installing dependencies..."
    REQUIREMENTS=(pandas numpy pyyaml tqdm)
    
    for pkg in "${REQUIREMENTS[@]}"; do
        echo -n "Installing $pkg... "
        pip install $pkg > /dev/null 2>&1 && echo "✓" || echo "✗"
    done
    
    [ -f "$PROJECT_ROOT/src/main.py" ] && chmod +x "$PROJECT_ROOT/src/main.py"
    [ -f "$PROJECT_ROOT/src/common/cli.py" ] && chmod +x "$PROJECT_ROOT/src/common/cli.py"
    
    deactivate
    echo "Setup complete!"
    read -p "Press Enter to continue..." dummy
}

run_network_test() {
    show_header
    echo "Network Test"
    
    NETWORK_LOG="$LOG_DIR/network_test.log"
    TEST_SCRIPT="$PROJECT_ROOT/tests/network.py"
    
    echo "Running network tests..."
    echo "=== Starting Network Tests ===" | tee -a "$NETWORK_LOG"
    echo "Timestamp: $(date)" | tee -a "$NETWORK_LOG"
    echo | tee -a "$NETWORK_LOG"
    
    if [ -d "$PROJECT_ROOT/venv" ]; then
        source "$PROJECT_ROOT/venv/bin/activate"
        PYTHONPATH="$PROJECT_ROOT" python3 "$TEST_SCRIPT" 2>&1 | tee -a "$NETWORK_LOG"
        TEST_EXIT_CODE=${PIPESTATUS[0]}
        deactivate
    else
        echo "Virtual environment not found. Running with system Python..." | tee -a "$NETWORK_LOG"
        PYTHONPATH="$PROJECT_ROOT" python3 "$TEST_SCRIPT" 2>&1 | tee -a "$NETWORK_LOG"
        TEST_EXIT_CODE=${PIPESTATUS[0]}
    fi
    
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo "Network tests completed successfully" | tee -a "$NETWORK_LOG"
    else
        echo "Network tests failed with errors" | tee -a "$NETWORK_LOG"
    fi
    
    echo "Log saved to: $NETWORK_LOG" | tee -a "$NETWORK_LOG"
    read -p "Press Enter to continue..." dummy
}

show_logs_menu() {
    local logs=()
    local i=0
    
    show_header
    echo "Log Management"
    
    for log_file in "$LOG_DIR"/*.log; do
        if [ -f "$log_file" ]; then
            local log_name=$(basename "$log_file")
            local log_date=$(stat -c %y "$log_file" | cut -d'.' -f1)
            local log_size=$(du -h "$log_file" | cut -f1)
            echo "  $i) $log_name ($log_size) $log_date"
            logs[$i]="$log_file"
            ((i++))
        fi
    done
    
    if [ $i -eq 0 ]; then
        echo "No log files found."
        read -p "Press Enter to return to main menu..." dummy
        return
    fi
    
    echo
    echo "  v) View a log file"
    echo "  c) Clean old logs"
    echo "  r) Return to main menu"
    
    read -p "Enter your choice: " log_choice
    
    case $log_choice in
        v)
            read -p "Enter log number to view [0-$((i-1))]: " log_number
            if [[ "$log_number" =~ ^[0-9]+$ ]] && [ "$log_number" -lt "$i" ]; then
                echo "Viewing log: ${logs[$log_number]}"
                echo "Press q to exit, arrow keys to navigate"
                sleep 1
                less "${logs[$log_number]}"
            else
                echo "Invalid selection."
                sleep 1
            fi
            show_logs_menu
            ;;
        c)
            clean_old_logs
            show_logs_menu
            ;;
        r|*)
            return
            ;;
    esac
}

clean_old_logs() {
    show_header
    echo "Clean Old Logs"
    
    local total_logs=$(find "$LOG_DIR" -name "*.log" | wc -l)
    local total_size=$(du -sh "$LOG_DIR" | cut -f1)
    
    echo "Log directory: $LOG_DIR"
    echo "Total logs: $total_logs"
    echo "Total size: $total_size"
    echo
    
    echo "Cleaning Options"
    echo "  1) Delete logs older than 7 days"
    echo "  2) Delete logs older than 30 days"
    echo "  3) Delete all logs except current session"
    echo "  4) Compress old logs (keeps files but reduces size)"
    echo "  5) Return to log menu"
    
    read -p "Enter choice [1-5]: " clean_choice
    
    case $clean_choice in
        1)
            echo "Deleting logs older than 7 days..."
            find "$LOG_DIR" -name "*.log" -type f -mtime +7 -delete
            echo "Old logs deleted."
            ;;
        2)
            echo "Deleting logs older than 30 days..."
            find "$LOG_DIR" -name "*.log" -type f -mtime +30 -delete
            echo "Old logs deleted."
            ;;
        3)
            echo "Deleting all logs except current session..."
            find "$LOG_DIR" -name "*.log" -type f -not -name "*${SESSION_ID}*" -delete
            echo "Logs deleted."
            ;;
        4)
            echo "Compressing old logs..."
            for log_file in $(find "$LOG_DIR" -name "*.log" -type f -mtime +1 -not -name "*.gz"); do
                gzip -f "$log_file"
                echo "Compressed: $log_file"
            done
            echo "Log compression complete."
            ;;
        5|*)
            return
            ;;
    esac
    
    local new_total_logs=$(find "$LOG_DIR" -name "*.log" | wc -l)
    local new_total_size=$(du -sh "$LOG_DIR" | cut -f1)
    
    echo "New total logs: $new_total_logs"
    echo "New total size: $new_total_size"
    
    read -p "Press Enter to continue..." dummy
}

run_diagnostics() {
    show_header
    echo "System Diagnostics"
    
    echo "Python Environment"
    local py_version=$(python3 --version 2>&1)
    local pip_version=$(pip --version 2>&1 | cut -d" " -f1,2)
    echo "Python Version: $py_version"
    echo "Pip Version: $pip_version"
    
    echo -e "\nVirtual Environment"
    if [ -d "$PROJECT_ROOT/venv" ]; then
        echo "Status: Installed"
        echo "Path: $PROJECT_ROOT/venv"
    else
        echo "Status: Not found"
    fi
    
    echo -e "\nInstalled Packages"
    if [ -d "$PROJECT_ROOT/venv" ]; then
        source "$PROJECT_ROOT/venv/bin/activate"
        for pkg in pandas numpy pyyaml tqdm; do
            if python3 -c "import $pkg" &> /dev/null; then
                ver=$(python3 -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "Unknown")
                echo "$pkg: Installed (Version: $ver)"
            else
                echo "$pkg: Missing"
            fi
        done
        deactivate
    else
        echo "Virtual environment not found, checking system packages"
        for pkg in pandas numpy pyyaml tqdm; do
            if python3 -c "import $pkg" &> /dev/null; then
                echo "$pkg: Installed (system)"
            else
                echo "$pkg: Missing"
            fi
        done
    fi
    
    echo -e "\nSystem Information"
    echo "Operating System: $(uname -a)"
    echo "CPU: $(grep "model name" /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs || echo "Unknown")"
    
    local mem_total=$(free -m | grep Mem | awk '{print $2}')
    local mem_used=$(free -m | grep Mem | awk '{print $3}')
    local mem_percent=$((mem_used * 100 / mem_total))
    echo "Memory: ${mem_used}MB / ${mem_total}MB (${mem_percent}%)"
    
    local disk_avail=$(df -h . | tail -1 | awk '{print $4}')
    local disk_percent=$(df -h . | tail -1 | awk '{print $5}' | tr -d '%')
    echo "Disk Space: ${disk_avail} available (${disk_percent}% used)"
    
    echo -e "\nProject Structure"
    [ -d "$PROJECT_ROOT/src" ] && echo "Source Directory: OK" || echo "Source Directory: Missing"
    [ -d "$PROJECT_ROOT/conf" ] && echo "Config Directory: OK" || echo "Config Directory: Missing"
    
    echo -e "\nNetwork Connectivity"
    ping -c 1 google.com &> /dev/null && echo "Internet Connection: OK" || echo "Internet Connection: Failed"
    
    read -p "Press Enter to return to main menu..." dummy
}

build_command() {
    CMD="python3 -m src.main --mode $MODE"
    [ "$MODE" == "backtest" ] && CMD="$CMD --backtest-engine $BACKTEST_ENGINE"
    CMD="$CMD --strategy $STRATEGY --symbol \"$SYMBOL\" --timeframe $TIMEFRAME"
    [ "$MODE" == "backtest" ] && CMD="$CMD --start-date $START_DATE --end-date $END_DATE"
    CMD="$CMD --config $CONFIG $DEBUG $ADDITIONAL_PARAMS"
}

run_trading_system() {
    echo "=== Trading System Started ===" >> "$LOG_FILE"
    echo "Date: $(date)" >> "$LOG_FILE"
    echo "Command: $CMD" >> "$LOG_FILE"
    
    if [ -d "$PROJECT_ROOT/venv" ]; then
        source "$PROJECT_ROOT/venv/bin/activate"
    else
        echo "Virtual environment not found. Using system Python..."
    fi
    
    eval "$CMD" 2>&1 | tee -a "$LOG_FILE" &
    TRADING_PID=$!
    echo $TRADING_PID > "$PID_FILE"
    
    echo -e "\nTrading system running with PID: $TRADING_PID"
    echo "Press [C] to view configuration"
    echo "Press [L] to view live log"
    echo "Press [S] to save current configuration"
    echo "Press [Q] to stop trading and return to menu"
    
    while ps -p $TRADING_PID > /dev/null; do
        if read -t 1 -n 1 user_input; then
            case $user_input in
                C|c)
                    echo -e "\nCurrent Configuration:"
                    echo "Mode: $MODE"
                    [ "$MODE" == "backtest" ] && echo "Engine: $BACKTEST_ENGINE"
                    echo "Strategy: $STRATEGY"
                    echo "Symbol(s): $SYMBOL"
                    echo "Timeframe: $TIMEFRAME"
                    [ "$MODE" == "backtest" ] && echo "Date Range: $START_DATE to $END_DATE"
                    echo "Config: $CONFIG"
                    [ -n "$DEBUG" ] && echo "Debug: Enabled"
                    ;;
                L|l)
                    echo -e "\nOpening log file... (Press Ctrl+C to return)"
                    sleep 1
                    tail -f "$LOG_FILE" || true
                    echo -e "\nReturned from log view"
                    ;;
                S|s)
                    echo -e "\nSave current configuration"
                    read -p "Enter a name for this configuration: " config_name
                    save_config "$config_name"
                    ;;
                Q|q)
                    echo -e "\nStopping trading system..."
                    kill -SIGTERM $TRADING_PID 2>/dev/null
                    wait $TRADING_PID 2>/dev/null
                    echo "Trading system stopped."
                    rm -f "$PID_FILE"
                    return 0
                    ;;
            esac
            echo -e "\nPress [C] Config, [L] Log, [S] Save, [Q] Quit"
        fi
    done
    
    wait $TRADING_PID
    EXIT_CODE=$?
    rm -f "$PID_FILE"
    
    echo -e "\n=== Trading System Finished ===" >> "$LOG_FILE"
    echo "Exit Code: $EXIT_CODE" >> "$LOG_FILE"
    echo "End Time: $(date)" >> "$LOG_FILE"
    
    deactivate 2>/dev/null || true
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "\nTrading system completed successfully"
    else
        echo -e "\nTrading system encountered errors (Exit code: $EXIT_CODE)"
    fi
    
    read -p "Press Enter to continue..." dummy
    return $EXIT_CODE
}

ask_rerun() {
    echo "Run Options"
    echo "  1) Run again with same configuration"
    echo "  2) Configure new run"
    echo "  3) Return to main menu"
    
    read -p "Enter choice [1-3]: " rerun_choice
    
    case $rerun_choice in
        1)
            run_trading_system
            ask_rerun
            ;;
        2)
            configure_and_run
            ;;
        3|*)
            return
            ;;
    esac
}

configure_and_run() {
    MODE="backtest"
    BACKTEST_ENGINE="ohlcv"
    STRATEGY="dual_ma"
    SYMBOL="BTC/USDT"
    TIMEFRAME="1h"
    START_DATE="2025-01-01"
    END_DATE="2025-04-28"
    CONFIG="$PROJECT_ROOT/conf/config.yaml"
    DEBUG=""
    ADDITIONAL_PARAMS=""
    
    show_header
    
    echo "Select trading mode"
    echo "  1) Backtest  - Historical data simulation"
    echo "  2) Paper     - Real-time simulation (no real orders)"
    echo "  3) Live      - Real-money trading"
    read -p "Enter choice [1-3] (default: 1): " mode_choice
    
    case $mode_choice in
        2) MODE="paper" ;;
        3) 
            MODE="live"
            echo -e "\n⚠️  WARNING: LIVE TRADING MODE SELECTED ⚠️"
            echo "This will use REAL FUNDS from your exchange account."
            read -p "Are you sure you want to continue? (y/n): " confirm
            if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
                echo "Exiting live mode"
                configure_and_run
                return
            fi
            ;;
        *) MODE="backtest" ;;
    esac
    
    if [ "$MODE" == "backtest" ]; then
        echo "Select backtest engine"
        echo "  1) OHLCV (Basic)       - Standard candlestick backtesting"
        echo "  2) Market Replay       - Advanced order book simulation"
        read -p "Enter choice [1-2] (default: 1): " engine_choice
        [ "$engine_choice" == "2" ] && BACKTEST_ENGINE="market_replay" || BACKTEST_ENGINE="ohlcv"
    fi
    
    echo "Select strategy"
    echo "  1) dual_ma         - Dual Moving Average crossover"
    echo "  2) neural_network  - Neural Network prediction"
    echo "  3) Custom strategy"
    read -p "Enter choice [1-3] (default: 1): " strategy_choice
    
    case $strategy_choice in
        2) STRATEGY="neural_network" ;;
        3) 
            read -p "Enter custom strategy name: " custom_strategy
            STRATEGY="$custom_strategy"
            ;;
        *) STRATEGY="dual_ma" ;;
    esac
    
    read -p "Enter trading symbol(s) (comma-separated, default: BTC/USDT): " symbol_input
    [ -n "$symbol_input" ] && SYMBOL="$symbol_input"
    
    echo "Select timeframe"
    echo "  1) 1m    2) 5m    3) 15m   4) 30m"
    echo "  5) 1h    6) 4h    7) 1d    8) 1w"
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
    
    if [ "$MODE" == "backtest" ]; then
        read -p "Start date (YYYY-MM-DD, default: $START_DATE): " start_input
        read -p "End date (YYYY-MM-DD, default: $END_DATE): " end_input
        [ -n "$start_input" ] && START_DATE="$start_input"
        [ -n "$end_input" ] && END_DATE="$end_input"
    fi
    
    read -p "Config file path (default: $CONFIG): " config_input
    [ -n "$config_input" ] && CONFIG="$config_input"
    
    read -p "Enable debug logging? (y/n, default: n): " debug_choice
    [[ "$debug_choice" == "y" || "$debug_choice" == "Y" ]] && DEBUG="--debug"
    
    read -p "Additional parameters: " ADDITIONAL_PARAMS
    
    build_command
    
    echo "Configuration Summary:"
    echo "Mode: $MODE"
    [ "$MODE" == "backtest" ] && echo "Engine: $BACKTEST_ENGINE"
    echo "Strategy: $STRATEGY"
    echo "Symbol(s): $SYMBOL"
    echo "Timeframe: $TIMEFRAME"
    [ "$MODE" == "backtest" ] && echo "Date Range: $START_DATE to $END_DATE"
    echo "Config: $CONFIG"
    [ -n "$DEBUG" ] && echo "Debug: Enabled"
    [ -n "$ADDITIONAL_PARAMS" ] && echo "Additional: $ADDITIONAL_PARAMS"
    
    echo -e "\nCommand: $CMD"
    read -p "Execute trading system? (y/n): " run_choice
    
    if [[ "$run_choice" == "y" || "$run_choice" == "Y" ]]; then
        echo -e "\nStarting trading system...\n"
        run_trading_system
        ask_rerun
    else
        echo -e "\nOperation cancelled"
        sleep 1
    fi
}

show_main_menu() {
    local choice
    
    show_header
    
    echo "Main Menu"
    echo "  1) Start new trading session"
    echo "  2) Load saved configuration"
    echo "  3) View logs"
    echo "  4) System diagnostics"
    echo "  5) Network test"
    echo "  6) Setup & configuration"
    echo "  7) Exit"
    
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
            echo "Exiting..."
            deactivate 2>/dev/null || true
            exit 0
            ;;
        *)
            echo "Invalid choice."
            sleep 1
            ;;
    esac
}

show_setup_menu() {
    local setup_choice
    
    show_header
    echo "Setup & Configuration"
    
    echo "Options"
    echo "  1) Setup virtual environment"
    echo "  2) Clean project directories"
    echo "  3) Return to main menu"
    
    read -p "Enter choice [1-3]: " setup_choice
    
    case $setup_choice in
        1)
            setup_virtual_env
            ;;
        2)
            clean_project
            ;;
        3|*)
            return
            ;;
    esac
}

clean_project() {
    show_header
    echo "Clean Project Directories"
    
    TARGET_DIRS=("logs" "reports" "cached" "__pycache__")
    
    echo "This will delete these directories recursively from your project:"
    for dir in "${TARGET_DIRS[@]}"; do
        echo "  - $dir/"
    done
    
    echo -e "\nSearching for directories to clean..."
    total_count=0
    for dir_name in "${TARGET_DIRS[@]}"; do
        count=$(find "$PROJECT_ROOT" -type d -name "$dir_name" | wc -l)
        total_count=$((total_count + count))
        [ $count -gt 0 ] && echo "Found $count $dir_name/ directories"
    done
    
    if [ $total_count -eq 0 ]; then
        echo "No directories found to clean."
        read -p "Press Enter to return..." dummy
        return
    fi
    
    read -p "Proceed with deletion? [y/N] " confirm
    if [[ "${confirm,,}" != "y" ]]; then
        echo "Operation cancelled"
        read -p "Press Enter to return..." dummy
        return
    fi
    
    deleted_count=0
    for dir_name in "${TARGET_DIRS[@]}"; do
        echo "Searching for $dir_name/ directories..."
        
        for dir_path in $(find "$PROJECT_ROOT" -type d -name "$dir_name"); do
            echo "Deleting: $dir_path"
            rm -rf "$dir_path"
            deleted_count=$((deleted_count + 1))
        done
    done
    
    if [ $deleted_count -gt 0 ]; then
        echo "Cleaned $deleted_count directories successfully"
    else
        echo "No directories were deleted"
    fi
    
    read -p "Press Enter to continue..." dummy
}

clear
echo "Trading System CLI"
echo "=================="
echo "Starting..."

if [ $# -gt 0 ]; then
    case "$1" in
        --help|-h)
            echo "Trading System CLI"
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --setup, -s      Run setup for virtual environment and dependencies"
            echo "  --network, -n    Run network tests"
            echo "  --logs, -l       Show logs menu"
            echo "  --run, -r        Run trading system with saved configuration"
            echo "  --clean          Clean project directories"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
        --setup|-s)
            setup_virtual_env
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
                echo "Could not load configuration"
                exit 1
            fi
            ;;
        --clean)
            clean_project
            ;;
        *)
            echo "Unknown option: $1"
            echo "Try '$0 --help' for more information."
            exit 1
            ;;
    esac
    exit 0
fi

while true; do
    show_main_menu
done