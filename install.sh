#!/bin/bash
# install.sh - Quick setup script for Trading System on Linux

# ANSI color codes
RESET="\033[0m"
BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
CYAN="\033[36m"
RED="\033[31m"

echo -e "${BOLD}${CYAN}Trading System Installation${RESET}"
echo -e "${CYAN}===========================${RESET}\n"

# Check Python version
echo -e "${BOLD}Checking Python installation...${RESET}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ Found $PYTHON_VERSION${RESET}"
else
    echo -e "${RED}✗ Python 3 not found${RESET}"
    echo -e "Please install Python 3.8 or newer:${RESET}"
    echo -e "  sudo apt update"
    echo -e "  sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

# Create virtual environment
echo -e "\n${BOLD}Setting up virtual environment...${RESET}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${RESET}"
else
    echo -e "${YELLOW}! Virtual environment already exists${RESET}"
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo -e "\n${BOLD}Installing dependencies...${RESET}"
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
        echo -e "${GREEN}✓${RESET}"
    else
        echo -e "${RED}✗${RESET}"
    fi
done

# Make scripts executable
chmod +x trading_cli.sh
chmod +x src/main.py
chmod +x src/ui/cli_runner.py

# Create symlink in user's bin directory if it exists
if [ -d "$HOME/bin" ]; then
    echo -e "\n${BOLD}Creating shortcut in ~/bin...${RESET}"
    ln -sf "$(pwd)/trading_cli.sh" "$HOME/bin/trading_cli"
    echo -e "${GREEN}✓ Shortcut created: ~/bin/trading_cli${RESET}"
    
    # Check if PATH includes ~/bin
    if [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
        echo -e "${YELLOW}! Your PATH doesn't include ~/bin${RESET}"
        echo -e "  Add this to your ~/.bashrc or ~/.profile:${RESET}"
        echo -e "  export PATH=\$HOME/bin:\$PATH"
    fi
fi

# Done
echo -e "\n${BOLD}${GREEN}Installation complete!${RESET}"
echo -e "\nTo use the trading system:"
echo -e "  1. ${BOLD}Activate the environment:${RESET}"
echo -e "     source venv/bin/activate"
echo -e "  2. ${BOLD}Run the CLI interface:${RESET}"
echo -e "     ./trading_cli.sh"
echo -e "\nOr use the shortcut if created:"
echo -e "  trading_cli"

# Deactivate virtual environment
deactivate