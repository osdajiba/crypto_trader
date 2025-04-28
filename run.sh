#!/bin/bash
# trading_cli.sh - Trading System CLI Launcher

# ANSI color codes
RESET="\033[0m"
BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
CYAN="\033[36m"
MAGENTA="\033[35m"
RED="\033[31m"

# Get project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

# Header
echo -e "${BOLD}${CYAN}Trading System CLI${RESET} - Optimized for Linux"
echo -e "${CYAN}===========================================${RESET}"

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required but not installed.${RESET}"
    exit 1
fi

# Check for required packages
for pkg in tqdm; do
    if ! python3 -c "import $pkg" &> /dev/null; then
        echo -e "${YELLOW}Installing required package: $pkg${RESET}"
        pip install $pkg
    fi
done

# Default values
MODE="backtest"
BACKTEST_ENGINE="ohlcv"
STRATEGY="dual_ma"
SYMBOL="BTC/USDT"
TIMEFRAME="1h"
START_DATE="2023-01-01"
END_DATE="2023-12-31"
CONFIG="$PROJECT_ROOT/conf/config.yaml"
DEBUG=""

# Mode selection with interactive menu
echo -e "\n${BOLD}Select trading mode:${RESET}"
echo -e "  ${GREEN}1${RESET}) Backtest  - Historical data simulation"
echo -e "  ${GREEN}2${RESET}) Paper     - Real-time simulation (no real orders)"
echo -e "  ${GREEN}3${RESET}) Live      - ${RED}Real-money trading${RESET}"
read -p "Enter choice [1-3] (default: 1): " mode_choice

case $mode_choice in
    2) MODE="paper" ;;
    3) 
        MODE="live"
        echo -e "\n${BOLD}${RED}⚠️  WARNING: LIVE TRADING MODE SELECTED ⚠️${RESET}"
        echo -e "${RED}This will use REAL FUNDS from your exchange account.${RESET}"
        read -p "Are you sure you want to continue? (y/n): " confirm
        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            echo "Exiting..."
            exit 0
        fi
        ;;
    *) MODE="backtest" ;;
esac

# Backtest engine selection (only for backtest mode)
if [ "$MODE" == "backtest" ]; then
    echo -e "\n${BOLD}Select backtest engine:${RESET}"
    echo -e "  ${GREEN}1${RESET}) OHLCV (Basic)       - Standard candlestick backtesting"
    echo -e "  ${GREEN}2${RESET}) Market Replay       - Advanced order book simulation"
    read -p "Enter choice [1-2] (default: 1): " engine_choice
    
    case $engine_choice in
        2) BACKTEST_ENGINE="market_replay" ;;
        *) BACKTEST_ENGINE="ohlcv" ;;
    esac
fi

# Strategy selection
echo -e "\n${BOLD}Select strategy:${RESET}"
echo -e "  ${GREEN}1${RESET}) dual_ma         - Dual Moving Average crossover"
echo -e "  ${GREEN}2${RESET}) neural_network  - Neural Network prediction"
echo -e "  ${GREEN}3${RESET}) Custom strategy"
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
echo -e "\n${BOLD}Select timeframe:${RESET}"
echo -e "  ${GREEN}1${RESET}) 1m    ${GREEN}2${RESET}) 5m    ${GREEN}3${RESET}) 15m   ${GREEN}4${RESET}) 30m"
echo -e "  ${GREEN}5${RESET}) 1h    ${GREEN}6${RESET}) 4h    ${GREEN}7${RESET}) 1d    ${GREEN}8${RESET}) 1w"
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
CMD="python3 -m src.ui.cli_runner --mode $MODE"
[ "$MODE" == "backtest" ] && CMD="$CMD --backtest-engine $BACKTEST_ENGINE"
CMD="$CMD --strategy $STRATEGY --symbol \"$SYMBOL\" --timeframe $TIMEFRAME"
[ "$MODE" == "backtest" ] && CMD="$CMD --start-date $START_DATE --end-date $END_DATE"
CMD="$CMD --config $CONFIG $DEBUG $ADDITIONAL_PARAMS"

# Display summary
echo -e "\n${BOLD}${CYAN}Configuration Summary:${RESET}"
echo -e "${BOLD}Mode:${RESET}         $MODE"
[ "$MODE" == "backtest" ] && echo -e "${BOLD}Engine:${RESET}       $BACKTEST_ENGINE"
echo -e "${BOLD}Strategy:${RESET}     $STRATEGY"
echo -e "${BOLD}Symbol(s):${RESET}    $SYMBOL"
echo -e "${BOLD}Timeframe:${RESET}    $TIMEFRAME"
[ "$MODE" == "backtest" ] && echo -e "${BOLD}Date Range:${RESET}   $START_DATE to $END_DATE"
echo -e "${BOLD}Config:${RESET}       $CONFIG"
[ -n "$DEBUG" ] && echo -e "${BOLD}Debug:${RESET}        Enabled"
[ -n "$ADDITIONAL_PARAMS" ] && echo -e "${BOLD}Additional:${RESET}   $ADDITIONAL_PARAMS"

# Confirm execution
echo -e "\n${YELLOW}Command:${RESET} $CMD"
read -p "Execute trading system? (y/n): " run_choice

if [[ "$run_choice" == "y" || "$run_choice" == "Y" ]]; then
    echo -e "\n${CYAN}Starting trading system...${RESET}\n"
    eval "$CMD"
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}Trading system completed successfully${RESET}"
    else
        echo -e "\n${RED}Trading system encountered errors${RESET}"
    fi
else
    echo -e "\n${YELLOW}Operation cancelled${RESET}"
fi