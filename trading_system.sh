#!/bin/bash
# simplified_trading_system.sh - Streamlined CLI launcher for Linux

# ANSI color codes
RESET="\033[0m"
BRIGHT="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
CYAN="\033[36m"
MAGENTA="\033[35m"
RED="\033[31m"

# Get project root directory
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$PROJECT_ROOT"

# Default values
MODE="backtest"
STRATEGY="dual_ma"
SYMBOL="BTC/USDT"
TIMEFRAME="1h"
START_DATE="2023-01-01"
END_DATE="2023-12-31"
DEBUG=""

# Header
echo -e "${BRIGHT}${CYAN}Trading System CLI Launcher${RESET}"
echo -e "${BRIGHT}${CYAN}=========================${RESET}"

# Trading mode selection
echo -e "\n${BRIGHT}${CYAN}Select trading mode:${RESET}"
echo -e "${GREEN}[1]${RESET} Backtest - Historical data simulation"
echo -e "${GREEN}[2]${RESET} Paper - Real-time simulation with virtual orders"
echo -e "${GREEN}[3]${RESET} Live - Real order execution (${BRIGHT}${RED}USE WITH CAUTION${RESET})"
read -p "Choice (1-3, default=1): " mode_choice

case $mode_choice in
    2) MODE="paper" ;;
    3) MODE="live" ;;
    *) MODE="backtest" ;;
esac

# Strategy selection
echo -e "\n${BRIGHT}${CYAN}Select strategy:${RESET}"
echo -e "${GREEN}[1]${RESET} dual_ma (Dual Moving Average)"
echo -e "${GREEN}[2]${RESET} neural_network"
echo -e "${GREEN}[3]${RESET} Custom"
read -p "Choice (1-3, default=1): " strat_choice

case $strat_choice in
    2) STRATEGY="neural_network" ;;
    3) read -p "Enter custom strategy name: " STRATEGY ;;
    *) STRATEGY="dual_ma" ;;
esac

# Symbol
read -p "Trading symbols (comma-separated, default=BTC/USDT): " symbol_input
[ -n "$symbol_input" ] && SYMBOL="$symbol_input"

# Timeframe
echo -e "\n${BRIGHT}${CYAN}Select timeframe:${RESET}"
echo -e "${GREEN}[1]${RESET} 1m  ${GREEN}[2]${RESET} 5m  ${GREEN}[3]${RESET} 15m  ${GREEN}[4]${RESET} 30m"
echo -e "${GREEN}[5]${RESET} 1h  ${GREEN}[6]${RESET} 4h  ${GREEN}[7]${RESET} 1d  ${GREEN}[8]${RESET} 1w"
read -p "Choice (1-8, default=5): " tf_choice

case $tf_choice in
    1) TIMEFRAME="1m" ;;
    2) TIMEFRAME="5m" ;;
    3) TIMEFRAME="15m" ;;
    4) TIMEFRAME="30m" ;;
    6) TIMEFRAME="4h" ;;
    7) TIMEFRAME="1d" ;;
    8) TIMEFRAME="1w" ;;
    *) TIMEFRAME="1h" ;;
esac

# Backtest dates
if [ "$MODE" == "backtest" ]; then
    read -p "Start date (YYYY-MM-DD, default=2023-01-01): " start_input
    read -p "End date (YYYY-MM-DD, default=2023-12-31): " end_input
    
    [ -n "$start_input" ] && START_DATE="$start_input"
    [ -n "$end_input" ] && END_DATE="$end_input"
    
    DATE_PARAMS="--start-date $START_DATE --end-date $END_DATE"
else
    DATE_PARAMS=""
fi

# Debug mode
read -p "Enable debug? (y/n, default=n): " debug_choice
[[ "$debug_choice" == "y" || "$debug_choice" == "Y" ]] && DEBUG="--debug"

# Additional parameters
read -p "Additional parameters: " ADDITIONAL

# Command execution
echo -e "\n${BRIGHT}${CYAN}Configuration Summary:${RESET}"
echo -e "- Mode: ${MAGENTA}$MODE${RESET}"
echo -e "- Strategy: ${MAGENTA}$STRATEGY${RESET}"
echo -e "- Symbols: ${MAGENTA}$SYMBOL${RESET}"
echo -e "- Timeframe: ${MAGENTA}$TIMEFRAME${RESET}"
[ -n "$DATE_PARAMS" ] && echo -e "- Date range: ${MAGENTA}$START_DATE to $END_DATE${RESET}"
[ -n "$DEBUG" ] && echo -e "- Debug: ${MAGENTA}Enabled${RESET}"
[ -n "$ADDITIONAL" ] && echo -e "- Additional: ${MAGENTA}$ADDITIONAL${RESET}"

# Build and execute command
CMD="python3 $PROJECT_ROOT/src/main.py --cli --mode $MODE --strategy $STRATEGY --symbol \"$SYMBOL\" --timeframe $TIMEFRAME $DATE_PARAMS $DEBUG $ADDITIONAL"
echo -e "\n${YELLOW}Executing:${RESET} $CMD\n"
eval "$CMD"

# Check exit status
if [ $? -ne 0 ]; then
    echo -e "\n${YELLOW}[ERROR] System exited with errors${RESET}"
else
    echo -e "\n${GREEN}[SUCCESS] System completed successfully${RESET}"
fi

read -p "Press any key to continue... " -n1 -s
echo