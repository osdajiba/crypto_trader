#!/usr/bin/env python3
"""
Environment checker for the trading system.

This script:
1. Sets the working directory to the project root
2. Checks if required configuration and requirements files exist
3. Offers to generate missing files

It should be called from run scripts in the scripts/ directory.
"""

import os
import sys
import subprocess
from pathlib import Path

# ANSI color codes for terminal output
RESET = "\033[0m"
BRIGHT = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"

def print_colored(message, color=BRIGHT + CYAN):
    """Print colored message to terminal."""
    print(f"{color}{message}{RESET}")

def set_project_root_as_working_dir():
    """Set the working directory to the project root."""
    # Get the absolute path of this script
    script_path = Path(__file__).resolve()
    
    # Project root is two levels up from src/utils/ directory
    project_root = script_path.parent.parent.parent
    
    # Set the working directory to project root
    os.chdir(project_root)
    
    print_colored(f"[*] Working directory set to: {project_root}", BRIGHT + BLUE)
    return project_root

def check_file_exists(file_path):
    """Check if a file exists."""
    return os.path.exists(file_path)

def run_script(script_path, auto=False):
    """Run a Python script."""
    cmd = [sys.executable, script_path]
    if auto:
        cmd.append("--auto")
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False

def prompt_yes_no(message):
    """Prompt the user for a yes/no response."""
    response = input(f"{message} (y/n): ").strip().lower()
    return response == 'y' or response == 'yes'

def main():
    """Main function to check files and run generators if needed."""
    # First set the working directory to project root
    project_root = set_project_root_as_working_dir()
    
    # Define paths
    config_file = project_root / "conf" / "config.yaml"
    requirements_file = project_root / "requirements.txt"
    config_generator = project_root / "src" / "utils" / "config_generator.py"
    requirement_generator = project_root / "src" / "utils" / "requirement_generator.py"
    
    print_colored("\n======================================================")
    print_colored("            TRADING SYSTEM ENVIRONMENT CHECK", BRIGHT + MAGENTA)
    print_colored("======================================================")
    
    # Check for requirements.txt
    print_colored("\n[*] Checking for requirements.txt...", BRIGHT + BLUE)
    if check_file_exists(requirements_file):
        print_colored(f"[✓] Requirements file found at: {requirements_file}", BRIGHT + GREEN)
    else:
        print_colored(f"[!] Requirements file not found at: {requirements_file}", BRIGHT + YELLOW)
        
        if check_file_exists(requirement_generator):
            if prompt_yes_no("Would you like to generate requirements.txt now?"):
                print_colored("\n[*] Running requirement generator...", BRIGHT + BLUE)
                if run_script(requirement_generator):
                    print_colored("[✓] Requirements file generated successfully.", BRIGHT + GREEN)
                else:
                    print_colored("[✗] Failed to generate requirements file.", BRIGHT + RED)
                    if not prompt_yes_no("Continue anyway?"):
                        print_colored("Exiting.", BRIGHT + RED)
                        return 1
        else:
            print_colored(f"[!] Requirement generator not found at {requirement_generator}", BRIGHT + YELLOW)
            if not prompt_yes_no("Continue without requirements.txt?"):
                print_colored("Exiting.", BRIGHT + RED)
                return 1
    
    # Check for config.yaml
    print_colored("\n[*] Checking for config.yaml...", BRIGHT + BLUE)
    if check_file_exists(config_file):
        print_colored(f"[✓] Configuration file found at: {config_file}", BRIGHT + GREEN)
    else:
        print_colored(f"[!] Configuration file not found at: {config_file}", BRIGHT + YELLOW)
        
        if check_file_exists(config_generator):
            if prompt_yes_no("Would you like to generate a default configuration file now?"):
                print_colored("\n[*] Running configuration generator...", BRIGHT + BLUE)
                if run_script(config_generator):
                    print_colored("[✓] Configuration file generated successfully.", BRIGHT + GREEN)
                else:
                    print_colored("[✗] Failed to generate configuration file.", BRIGHT + RED)
                    if not prompt_yes_no("Continue anyway?"):
                        print_colored("Exiting.", BRIGHT + RED)
                        return 1
        else:
            print_colored(f"[!] Configuration generator not found at {config_generator}", BRIGHT + YELLOW)
            if not prompt_yes_no("Continue without configuration?"):
                print_colored("Exiting.", BRIGHT + RED)
                return 1
    
    print_colored("\n[✓] Environment check completed successfully.", BRIGHT + GREEN)
    print_colored("    Ready to launch the trading system.", BRIGHT + GREEN)
    print_colored("======================================================\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())