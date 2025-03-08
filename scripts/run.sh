#!/bin/bash

CONFIG_FILE="../conf/config.json"
BUILD_SCRIPT="../conf/build_config.py"
MAIN_SCRIPT="./start.py"

# Check if config.example.json exists, and run build_config.py if not
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Configuration file not found. Generating it using $BUILD_SCRIPT..."
    if python "$BUILD_SCRIPT"; then
        echo "Configuration file generated successfully."
    else
        echo "Error: Failed to generate configuration file."
        exit 1
    fi
else
    echo "Configuration file found."
fi

# Run start.py
echo "Running $MAIN_SCRIPT..."
python "$MAIN_SCRIPT"
