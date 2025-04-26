#!/usr/bin/env python
# src/bin/main.py

import sys
import os
from src.launcher import launch

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

def main():
    """
    Main entry point - processes command line arguments and
    delegates to the appropriate module
    """
    try:
        return launch()
    except Exception as e:
        print(f"Critical error: {str(e)}")
        return {"error": f"Critical error: {str(e)}"}

if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0 if not result or "error" not in result else 1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)
    except Exception as e:
        print(f"Unhandled error: {str(e)}")
        sys.exit(1)