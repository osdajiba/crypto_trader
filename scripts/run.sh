#!/bin/bash
<<<<<<< HEAD
# Script Name: scripts/run.sh
# Description: Enhanced trading system launcher with CLI visualization

# --------------------------
# Terminal Colors
# --------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
UNDERLINE='\033[4m'
NC='\033[0m' # No Color

# --------------------------
# Environment Configuration
# --------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MAIN_SCRIPT="$PROJECT_ROOT/src/bin/main.py"
LOG_DIR="$PROJECT_ROOT/logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# --------------------------
# ASCII Art Header
# --------------------------
echo -e "\n${BOLD}${CYAN}"
echo "  ==============================================="
echo "   _____               _ _                "
echo "  |_   _|             | (_)               "
echo "    | |_ __ __ _  __| |_ _ _ __   __ _ "
echo "    | | '__/ _\` |/ _\` | | | '_ \\ / _\` |"
echo "    | | | | (_| | (_| | | | | | | (_| |"
echo "    \\_/_|  \\__,_|\\__,_|_|_|_| |_|\\__, |"
echo "                                   __/ |"
echo "                                  |___/ "
echo "   _____           _                 "
echo "  / ____|         | |                "
echo "  | (___  _   _ __| |_ ___ _ __ ___  "
echo "   \\___ \\| | | / _\` | __/ _ \\ '_ \` _ \\ "
echo "   ____) | |_| | (_| | ||  __/ | | | | |"
echo "  |_____/ \\__, |\\__,_|\\__\\___/_| |_| |_|"
echo "           __/ |                      "
echo "          |___/                       "
echo "  ==============================================="
echo -e "${NC}\n"

# --------------------------
# Dependency Checks
# --------------------------
check_python() {
    echo -e "${BLUE}[*] Checking dependencies...${NC}"
    
    # Check for Python 3.8+ 
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}[ERROR] Python 3.8+ is required but not found.${NC}"
        exit 1
    fi
    
    # Verify Python version
    PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo -e "${BLUE}[*] Python version: ${PY_VERSION}${NC}"
    
    if [[ $(echo "$PY_VERSION < 3.8" | bc -l) -eq 1 ]]; then
        echo -e "${RED}[ERROR] Python 3.8+ required. Found version ${PY_VERSION}${NC}"
        exit 1
    fi
}

# --------------------------
# Path Validation
# --------------------------
validate_paths() {
    if [ ! -f "$MAIN_SCRIPT" ]; then
        echo -e "${RED}[ERROR] Main script not found at $MAIN_SCRIPT${NC}"
        exit 1
    fi
}

# --------------------------
# Mode Selection Prompt
# --------------------------
prompt_mode_selection() {
    echo -e "\n${BOLD}Trading Mode Selection${NC}"
    echo -e "${CYAN}===============================================${NC}"
    echo -e "${BOLD}${BLUE} 1.${NC} ${BOLD}Backtest${NC}         Historical data simulation"
    echo -e "${BOLD}${GREEN} 2.${NC} ${BOLD}Paper${NC}            Real-time simulation with mock execution"
    echo -e "${BOLD}${RED} 3.${NC} ${BOLD}Live${NC}             Real order execution (CAUTION)"
    echo -e "${CYAN}===============================================${NC}"
    echo -e "${YELLOW}⚠️  Warning: Live trading requires proper configuration${NC}\n"

    read -p "Select trading mode (1-3): " MODE_CHOICE
    
    case $MODE_CHOICE in
        1) TRADING_MODE="--mode backtest" ;;
        2) TRADING_MODE="--mode paper" ;;
        3) TRADING_MODE="--mode live" ;;
        *)
            echo -e "${YELLOW}Invalid selection. Defaulting to backtest mode.${NC}"
            TRADING_MODE="--mode backtest"
            ;;
    esac

    echo -e "You selected: ${BOLD}${TRADING_MODE}${NC}\n"
}

# --------------------------
# Parse Command Line Args
# --------------------------
parse_args() {
    # Initialize variables
    GUI_MODE=""
    TRADING_MODE=""
    CONFIG_FILE=""
    STRATEGY=""
    SYMBOL=""
    TIMEFRAME=""
    START_DATE=""
    END_DATE=""
    DEBUG=""
    VERBOSE=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --gui)
                GUI_MODE="--gui"
                shift
                ;;
            --mode)
                TRADING_MODE="--mode $2"
                shift 2
                ;;
            --config)
                CONFIG_FILE="--config $2"
                shift 2
                ;;
            --strategy)
                STRATEGY="--strategy $2"
                shift 2
                ;;
            --symbol)
                SYMBOL="--symbol $2"
                shift 2
                ;;
            --timeframe)
                TIMEFRAME="--timeframe $2"
                shift 2
                ;;
            --start-date)
                START_DATE="--start-date $2"
                shift 2
                ;;
            --end-date)
                END_DATE="--end-date $2"
                shift 2
                ;;
            --debug)
                DEBUG="--debug"
                shift
                ;;
            --verbose|-v)
                VERBOSE="-v"
                shift
                ;;
            *)
                echo -e "${YELLOW}[WARNING] Unknown argument: $1${NC}"
                shift
                ;;
        esac
    done
    
    # If no mode specified and not GUI, prompt for mode
    if [ -z "$GUI_MODE" ] && [ -z "$TRADING_MODE" ]; then
        prompt_mode_selection
    fi
}

# --------------------------
# Main Execution
# --------------------------
main() {
    # Parse command line arguments
    parse_args "$@"
    
    # Check environment
    check_python
    validate_paths
    
    # Construct command
    COMMAND="python3 '$MAIN_SCRIPT' $GUI_MODE $TRADING_MODE $CONFIG_FILE $STRATEGY $SYMBOL $TIMEFRAME $START_DATE $END_DATE $DEBUG $VERBOSE"
    
    # Execute
    echo -e "\n${BLUE}[*] Launching trading system...${NC}"
    echo -e "${BLUE}[*] Command: ${COMMAND}${NC}"
    echo -e "${BLUE}[*] Log file: ${LOG_FILE}${NC}\n"
    echo -e "${CYAN}===============================================${NC}"
    
    # Execute with simultaneous logging to file
    EXIT_CODE=${PIPESTATUS[0]}
    
    echo -e "${CYAN}===============================================${NC}\n"
    
    # Handle exit status
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}[SUCCESS] System exited successfully${NC}"
    else
        echo -e "${RED}[ERROR] System exited with error code: ${EXIT_CODE}${NC}"
    fi
    
    exit $EXIT_CODE
}

# Start main process
cd "$PROJECT_ROOT"  # Change to project root directory
main "$@"
=======
# trading_system.sh - Main launcher script for Linux/Mac

# ANSI color codes
RESET="\033[0m"
BRIGHT="\033[1m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
MAGENTA="\033[35m"
CYAN="\033[36m"
WHITE="\033[37m"

echo -e "${BRIGHT}${CYAN}Trading System Launcher${RESET}"
echo -e "${BRIGHT}${CYAN}==========================${RESET}"
echo
echo -e "${BRIGHT}${GREEN}[1]${RESET} Start in GUI mode"
echo
echo -e "${BRIGHT}${GREEN}[2]${RESET} Start in CLI mode"
echo
echo -e "${BRIGHT}${GREEN}[3]${RESET} Exit"
echo
echo -e "${BRIGHT}${CYAN}==========================${RESET}"
echo

read -p "Enter option (1-3): " option

case $option in
    1)
        echo -e "${BRIGHT}${CYAN}[*] Launching trading system...${RESET}"
        echo -e "${BRIGHT}${CYAN}[*] Launching in --gui mode...${RESET}"
        python3 src/bin/main.py --gui
        ;;
    2)
        echo
        echo -e "${BRIGHT}${CYAN}CLI Mode Configuration${RESET}"
        echo -e "${BRIGHT}${CYAN}==========================${RESET}"
        echo -e "${BRIGHT}${CYAN}Select trading mode:${RESET}"
        echo
        echo -e "${BRIGHT}${GREEN}[1]${RESET} Backtest - Historical data simulation"
        echo
        echo -e "${BRIGHT}${GREEN}[2]${RESET} Paper - Real-time simulation with virtual orders"
        echo
        echo -e "${BRIGHT}${GREEN}[3]${RESET} Live - Real order execution (${BRIGHT}${RED}USE WITH CAUTION${RESET})"
        echo
        echo -e "${BRIGHT}${CYAN}==========================${RESET}"
        echo

        read -p "Enter trading mode (1-3): " trade_mode

        case $trade_mode in
            1) mode_param="backtest" ;;
            2) mode_param="paper" ;;
            3) mode_param="live" ;;
            *) 
                echo -e "${BRIGHT}${YELLOW}Invalid selection, defaulting to backtest mode.${RESET}"
                mode_param="backtest" 
                ;;
        esac

        echo
        echo -e "Selected trading mode: ${BRIGHT}${MAGENTA}$mode_param${RESET}"
        echo

        echo -e "${BRIGHT}${CYAN}Select strategy:${RESET}"
        echo
        echo -e "${BRIGHT}${GREEN}[1]${RESET} dual_ma (Dual Moving Average)"
        echo
        echo -e "${BRIGHT}${GREEN}[2]${RESET} neural_network"
        echo
        echo -e "${BRIGHT}${GREEN}[3]${RESET} Custom (specify name)"
        echo

        read -p "Enter strategy choice (1-3): " strat_choice

        case $strat_choice in
            1) strategy_param="dual_ma" ;;
            2) strategy_param="neural_network" ;;
            3) 
                read -p "Enter custom strategy name: " strategy_param
                ;;
            *) 
                echo -e "${BRIGHT}${YELLOW}Invalid selection, defaulting to dual_ma strategy.${RESET}"
                strategy_param="dual_ma" 
                ;;
        esac

        echo
        echo -e "Selected strategy: ${BRIGHT}${MAGENTA}$strategy_param${RESET}"
        echo

        read -p "Enter trading symbols (comma-separated, default=BTC/USDT): " symbol_param
        if [ -z "$symbol_param" ]; then
            symbol_param="BTC/USDT"
        fi

        echo
        echo -e "${BRIGHT}${CYAN}Select timeframe:${RESET}"
        echo
        echo -e "${BRIGHT}${GREEN}[1]${RESET} 1m  ${BRIGHT}${GREEN}[2]${RESET} 5m  ${BRIGHT}${GREEN}[3]${RESET} 15m  ${BRIGHT}${GREEN}[4]${RESET} 30m  ${BRIGHT}${GREEN}[5]${RESET} 1h  ${BRIGHT}${GREEN}[6]${RESET} 4h  ${BRIGHT}${GREEN}[7]${RESET} 1d  ${BRIGHT}${GREEN}[8]${RESET} 1w"
        echo

        read -p "Enter timeframe choice (1-8): " tf_choice

        case $tf_choice in
            1) timeframe_param="1m" ;;
            2) timeframe_param="5m" ;;
            3) timeframe_param="15m" ;;
            4) timeframe_param="30m" ;;
            5) timeframe_param="1h" ;;
            6) timeframe_param="4h" ;;
            7) timeframe_param="1d" ;;
            8) timeframe_param="1w" ;;
            *) 
                echo -e "${BRIGHT}${YELLOW}Invalid selection, defaulting to 1h timeframe.${RESET}"
                timeframe_param="1h" 
                ;;
        esac

        echo
        echo -e "Selected timeframe: ${BRIGHT}${MAGENTA}$timeframe_param${RESET}"
        echo

        # Date parameters only for backtest mode
        if [ "$mode_param" == "backtest" ]; then
            read -p "Enter backtest start date (YYYY-MM-DD, default=2023-01-01): " start_date
            if [ -z "$start_date" ]; then
                start_date="2023-01-01"
            fi
            
            read -p "Enter backtest end date (YYYY-MM-DD, default=2023-12-31): " end_date
            if [ -z "$end_date" ]; then
                end_date="2023-12-31"
            fi
            
            date_params="--start-date $start_date --end-date $end_date"
        else
            date_params=""
        fi

        echo
        read -p "Enable debug mode? (y/n, default=n): " debug_mode
        if [[ "$debug_mode" == "y" || "$debug_mode" == "Y" ]]; then
            debug_param="--debug"
        else
            debug_param=""
        fi

        echo
        read -p "Enter any additional parameters (or press Enter to skip): " additional_params

        echo
        echo -e "${BRIGHT}${CYAN}[*] Launching trading system...${RESET}"
        echo -e "${BRIGHT}${CYAN}[*] Launching in CLI mode with the following configuration:${RESET}"
        echo -e "    - Mode: ${BRIGHT}${MAGENTA}$mode_param${RESET}"
        echo -e "    - Strategy: ${BRIGHT}${MAGENTA}$strategy_param${RESET}"
        echo -e "    - Symbols: ${BRIGHT}${MAGENTA}$symbol_param${RESET}"
        echo -e "    - Timeframe: ${BRIGHT}${MAGENTA}$timeframe_param${RESET}"
        
        if [ -n "$date_params" ]; then
            echo -e "    - Start date: ${BRIGHT}${MAGENTA}$start_date${RESET}"
            echo -e "    - End date: ${BRIGHT}${MAGENTA}$end_date${RESET}"
        fi
        
        if [ -n "$debug_param" ]; then
            echo -e "    - Debug mode: ${BRIGHT}${MAGENTA}Enabled${RESET}"
        fi
        
        if [ -n "$additional_params" ]; then
            echo -e "    - Additional parameters: ${BRIGHT}${MAGENTA}$additional_params${RESET}"
        fi
        
        echo

        cmd="python3 src/bin/main.py --cli --mode $mode_param --strategy $strategy_param --symbol \"$symbol_param\" --timeframe $timeframe_param $date_params $debug_param $additional_params"
        
        echo -e "${BRIGHT}${YELLOW}Command:${RESET} $cmd"
        echo
        eval "$cmd"
        ;;
    3)
        echo -e "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${BRIGHT}${YELLOW}Invalid option selected!${RESET}"
        exit 1
        ;;
esac

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo -e "${BRIGHT}${YELLOW}[ERROR] System exited with error code: $exit_code${RESET}"
else
    echo -e "${BRIGHT}${GREEN}[SUCCESS] System completed successfully.${RESET}"
fi

read -p "Press any key to continue... " -n1 -s
echo
exit 0
>>>>>>> dev
