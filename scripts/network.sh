#!/bin/bash
# /scripts/network.sh
# network.sh - Test script for network functionality


# Set the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
LOG_FILE="$PROJECT_ROOT/logs/network_test.log"
TEST_SCRIPT="$PROJECT_ROOT/src/exchange/network.py"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/logs"

# Function to run the test with logging
run_test() {
    echo "=== Starting Network Tests ===" | tee -a "$LOG_FILE"
    echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # Check Python version
    echo "Python Version:" | tee -a "$LOG_FILE"
    python3 --version | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # Run the test script
    echo "Running network tests..." | tee -a "$LOG_FILE"
    cd "$PROJECT_ROOT" || exit 1
    
    # Activate virtual environment
    source "$PROJECT_ROOT/venv/bin/activate"
    
    # Run tests with appropriate Python path
    PYTHONPATH="$PROJECT_ROOT" python3 "$TEST_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
    TEST_EXIT_CODE=${PIPESTATUS[0]}
    
    # Deactivate virtual environment
    deactivate
    
    # Check exit status
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "=== Tests Completed Successfully ===" | tee -a "$LOG_FILE"
    else
        echo "" | tee -a "$LOG_FILE"
        echo "!!! Tests Failed !!!" | tee -a "$LOG_FILE"
    fi
    
    echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
}

# Main execution
run_test

exit 0