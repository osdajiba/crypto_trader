#!/bin/bash
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