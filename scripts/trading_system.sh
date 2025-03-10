#!/bin/bash
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

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Get the project root directory (one level up from scripts)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check the environment and set the working directory to project root
echo -e "${BRIGHT}${BLUE}[*] Checking environment and required files...${RESET}"
python3 "$PROJECT_ROOT/src/utils/check_env.py"
if [ $? -ne 0 ]; then
    echo -e "${BRIGHT}${RED}[*] Environment check reported issues that need to be resolved.${RESET}"
    read -p "Press any key to continue or Ctrl+C to abort... " -n1 -s
    echo
    echo -e "${BRIGHT}${YELLOW}[WARNING] Continuing despite environment check issues.${RESET}"
fi

# Change to project root directory
cd "$PROJECT_ROOT"

# Display menu
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

# Read user option
read -p "Enter option (1-3): " option

case "$option" in
    1)
        echo -e "${BRIGHT}${CYAN}[*] Launching trading system...${RESET}"
        echo -e "${BRIGHT}${CYAN}[*] Launching in --gui mode...${RESET}"
        python3 src/main/main.py --gui
        ;;
    2)
        echo -e "${BRIGHT}${CYAN}[*] Launching in CLI mode...${RESET}"
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

        cmd="python3 src/main/main.py --cli --mode $mode_param --strategy $strategy_param --symbol \"$symbol_param\" --timeframe $timeframe_param $date_params $debug_param $additional_params"
        
        echo -e "${BRIGHT}${YELLOW}Command:${RESET} $cmd"
        echo
        eval "$cmd"
        ;;
    3)
        echo -e "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${BRIGHT}${YELLOW}Invalid option selected! Please enter 1, 2, or 3.${RESET}"
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