#!/usr/bin/env python3

import os
import sys
import subprocess
import yaml
import shutil
from pathlib import Path
import re
import logging
import ast
import importlib.util


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(module)-15s | [%(filename)s:%(lineno)d] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

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

def get_project_root():
    """Get the absolute path to the project root directory."""
    # When script is in src/utils/, project root is two levels up
    return Path(__file__).resolve().parent.parent.parent

def prompt_yes_no(message):
    """Prompt the user for a yes/no response."""
    response = input(f"{message} (y/n): ").strip().lower()
    return response == 'y' or response == 'yes'

# Set up common paths
PROJECT_ROOT = get_project_root()
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
DEFAULT_CONFIG_TEMPLATE = PROJECT_ROOT / "config.yaml"  # Template in project root
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"

#==============================================================================
# Configuration File Generation
#==============================================================================

def check_config_file():
    """Check if the configuration file exists."""
    return CONFIG_FILE.exists()

def create_config_directory():
    """Create the configuration directory if it doesn't exist."""
    if not CONFIG_DIR.exists():
        logger.info(f"Creating configuration directory: {CONFIG_DIR}")
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

def copy_default_config():
    """Copy the default configuration file to the config directory."""
    if DEFAULT_CONFIG_TEMPLATE.exists():
        logger.info(f"Copying default configuration from: {DEFAULT_CONFIG_TEMPLATE}")
        shutil.copy(DEFAULT_CONFIG_TEMPLATE, CONFIG_FILE)
        return True
    else:
        logger.warning(f"Default configuration template not found at: {DEFAULT_CONFIG_TEMPLATE}")
        return False

def generate_default_config():
    """Generate a minimal default configuration file."""
    logger.info("Generating minimal default configuration")
    
    default_config = {
        "system": {
            "operational_mode": "backtest",
            "timezone": "UTC",
            "performance": {
                "max_threads": 4,
                "max_memory": 1073741824,  # 1GB
                "task_timeout": 300,
            }
        },
        "api": {
            "binance": {
                "apiKey": "your apiKey",
                "secret": "your secret",
                "options": {
                    "adjustForTimeDifference": True,
                    "defaultType": "spot"
                }
            },
            "timeout": 60000,
            "enableRateLimit": True
        },
        "data": {
            "source": {
                "live": False,
                "type": "local"
            },
            "storage": {
                "historical": "db/historical/",
                "orders": "db/order_history.csv",
                "trades": "db/trade_records.json"
            }
        },
        "database": {
            "url": "sqlite:///database/trading.sqlite",
            "pool_size": 5,
            "timeout": 30,
            "echo_sql": False
        },
        "logging": {
            "base_path": "./logs",
            "level": "INFO",
            "handlers": {
                "file": {"enabled": True},
                "console": {"enabled": True, "level": "WARNING"}
            }
        },
        "trading": {
            "capital": {
                "initial": 100000,
                "min_order": 0.001
            },
            "instruments": ["BTC/USDT", "ETH/USDT"],
            "execution": {
                "slippage": 0.001,
                "leverage": 1
            }
        },
        "strategy": {
            "active": "dual_ma",
            "parameters": {
                "fast_period": 50,
                "slow_period": 200,
                "threshold": 0.005
            }
        },
        "backtest": {
            "period": {
                "start": "2023-01-01",
                "end": "2023-12-31"
            }
        }
    }
    
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Default configuration generated at: {CONFIG_FILE}")
    return True

def validate_config():
    """Validate the configuration file."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        
        # Perform basic validation
        required_sections = ["system", "database", "trading"]
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required configuration section: {section}")
                return False
        
        logger.info("Configuration validation successful")
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        return False

def prompt_generate_config():
    """Prompt the user to generate a default configuration file."""
    print_colored("\n======================================================")
    print_colored("      CONFIGURATION FILE NOT FOUND", BRIGHT + YELLOW)
    print_colored("======================================================")
    print_colored(f"A configuration file is required at:", BRIGHT)
    print_colored(f"  {CONFIG_FILE}", BRIGHT + BLUE)
    print_colored("\nWould you like to generate a default configuration file?")
    return prompt_yes_no("(y/n)")

def config_generator_main():
    """Main function to check and generate configuration."""
    print_colored("[*] Checking configuration file...", BRIGHT + BLUE)
    
    if check_config_file():
        print_colored(f"[✓] Configuration file found at: {CONFIG_FILE}", BRIGHT + GREEN)
        if validate_config():
            print_colored("[✓] Configuration file is valid.", BRIGHT + GREEN)
            return True
        else:
            print_colored("[!] Configuration file validation failed.", BRIGHT + YELLOW)
            return False
    else:
        print_colored("[!] Configuration file not found.", BRIGHT + YELLOW)
        
        if len(sys.argv) > 1 and sys.argv[1] == "--auto":
            should_generate = True
        else:
            should_generate = prompt_generate_config()
        
        if should_generate:
            create_config_directory()
            
            # Try to copy default template first
            if copy_default_config():
                print_colored("[✓] Default configuration copied successfully.", BRIGHT + GREEN)
                return True
            
            # If no template exists, generate a minimal config
            if generate_default_config():
                print_colored("[✓] Default configuration generated successfully.", BRIGHT + GREEN)
                return True
            else:
                print_colored("[✗] Failed to generate configuration file.", BRIGHT + RED)
                return False
        else:
            print_colored("Configuration generation skipped. The system may not function correctly.", BRIGHT + YELLOW)
            return False

#==============================================================================
# Requirements.txt Generation
#==============================================================================

# Common standard library modules to exclude
STDLIB_MODULES = {
    "os", "sys", "re", "datetime", "time", "json", "math", "random",
    "collections", "itertools", "functools", "pathlib", "shutil",
    "argparse", "logging", "traceback", "copy", "tempfile", "typing",
    "unittest", "abc", "io", "contextlib", "glob", "hashlib", "uuid",
    "csv", "pickle", "subprocess", "threading", "multiprocessing", "queue",
    "socket", "ssl", "email", "urllib", "http", "ftplib", "xml", "html",
    "webbrowser", "platform", "inspect", "ast", "importlib", "tkinter"
}

# Known third-party packages and their common imports
KNOWN_PACKAGES = {
    "numpy": ["numpy", "np"],
    "pandas": ["pandas", "pd"],
    "matplotlib": ["matplotlib", "mpl", "plt"],
    "ccxt": ["ccxt"],
    "scipy": ["scipy"],
    "scikit-learn": ["sklearn"],
    "tensorflow": ["tensorflow", "tf"],
    "torch": ["torch"],
    "sqlalchemy": ["sqlalchemy"],
    "flask": ["flask"],
    "django": ["django"],
    "requests": ["requests"],
    "aiohttp": ["aiohttp"],
    "pytest": ["pytest"],
    "dateutil": ["dateutil"],
    "python-dateutil": ["dateutil"],
    "pyyaml": ["yaml"],
    "aiofiles": ["aiofiles"],
    "asyncio": ["asyncio"],
    "polars": ["polars", "pl"],
    "ta-lib": ["talib"],
    "pandas-ta": ["pandas_ta"],
    "loguru": ["loguru"],
    "plotly": ["plotly"],
    "numba": ["numba"],
    "dask": ["dask"],
    "ray": ["ray"],
    "pyodbc": ["pyodbc"],
    "pymysql": ["pymysql"],
    "statsmodels": ["statsmodels"],
    "tqdm": ["tqdm"],
    "fastapi": ["fastapi"],
    "uvicorn": ["uvicorn"],
    "concurrent-futures": ["concurrent.futures"],
}

def check_requirements_file():
    """Check if the requirements.txt file exists."""
    return REQUIREMENTS_FILE.exists()

class ImportCollector(ast.NodeVisitor):
    """AST visitor to collect imports from Python files."""
    
    def __init__(self):
        self.imports = set()
        
    def visit_Import(self, node):
        """Extract import statements."""
        for name in node.names:
            # Get the module name (without 'as' aliases)
            module_name = name.name.split('.')[0]
            self.imports.add(module_name)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        """Extract from-import statements."""
        if node.module:
            # Skip relative imports
            if node.level == 0:
                module_name = node.module.split('.')[0]
                self.imports.add(module_name)
        self.generic_visit(node)

def get_imports_from_file(file_path):
    """Extract import statements from a Python file using AST."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        try:
            tree = ast.parse(source)
            collector = ImportCollector()
            collector.visit(tree)
            return collector.imports
        except SyntaxError:
            logger.warning(f"Failed to parse {file_path} (syntax error)")
            # Fallback to regex method for syntax errors
            return get_imports_with_regex(source)
    except Exception as e:
        logger.warning(f"Failed to process {file_path}: {str(e)}")
        return set()

def get_imports_with_regex(source):
    """Fallback method to extract imports using regex."""
    imports = set()
    
    # Match import statements
    import_pattern = re.compile(r'^import\s+([\w.]+)', re.MULTILINE)
    for match in import_pattern.finditer(source):
        module_name = match.group(1).split('.')[0]
        imports.add(module_name)
    
    # Match from-import statements
    from_pattern = re.compile(r'^from\s+([\w.]+)\s+import', re.MULTILINE)
    for match in from_pattern.finditer(source):
        if not match.group(1).startswith('.'):  # Skip relative imports
            module_name = match.group(1).split('.')[0]
            imports.add(module_name)
            
    return imports

def is_third_party_module(module_name):
    """Check if a module is a third-party package (not part of stdlib)."""
    if module_name in STDLIB_MODULES:
        return False
    
    # Check if it's a built-in or standard library module
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return True  # Not found, likely a third-party that's not installed
            
        # Check if it's a stdlib module
        if spec.origin is not None and 'site-packages' not in spec.origin:
            # Standard library modules are typically not in site-packages
            return False
            
        return True  # Likely a third-party module
    except (ImportError, ValueError, AttributeError):
        # If we can't determine, assume it might be a third-party module
        return True

def scan_project_for_imports():
    """Scan all Python files in the project and collect imports."""
    all_imports = set()
    
    total_files = 0
    processed_files = 0
    
    print_colored("\n[*] Scanning project for Python files and imports...", BRIGHT + BLUE)
    
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Skip virtual environments and hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', 'env', '__pycache__']]
        
        for file in files:
            if file.endswith('.py'):
                total_files += 1
                file_path = os.path.join(root, file)
                imports = get_imports_from_file(file_path)
                if imports:
                    all_imports.update(imports)
                    processed_files += 1
    
    print_colored(f"[*] Scanned {total_files} Python files, found imports in {processed_files} files", BRIGHT + BLUE)
    
    # Filter out standard library modules
    third_party_imports = {module for module in all_imports if is_third_party_module(module)}
    
    # Map to actual package names
    packages = set()
    for module in third_party_imports:
        # Check if it's a known module alias
        added = False
        for package, aliases in KNOWN_PACKAGES.items():
            if module in aliases:
                packages.add(package)
                added = True
                break
        
        # If not an alias, use the module name
        if not added:
            packages.add(module)
    
    return packages

def generate_requirements_file(packages):
    """Generate the requirements.txt file."""
    if not packages:
        print_colored("No third-party packages found to add to requirements.txt", BRIGHT + YELLOW)
        return False
    
    # Sort packages alphabetically
    sorted_packages = sorted(packages)
    
    print_colored(f"[*] Writing {len(sorted_packages)} packages to requirements.txt", BRIGHT + BLUE)
    
    try:
        with open(REQUIREMENTS_FILE, 'w') as f:
            for package in sorted_packages:
                f.write(f"{package}\n")
        
        print_colored(f"[✓] Requirements file generated at: {REQUIREMENTS_FILE}", BRIGHT + GREEN)
        return True
    except Exception as e:
        print_colored(f"[✗] Failed to write requirements file: {str(e)}", BRIGHT + RED)
        return False

def copy_existing_requirements():
    """Check if there's an existing requirements file to copy from template."""
    template_req = PROJECT_ROOT / "requirements.txt.template"
    if template_req.exists():
        print_colored(f"[*] Found template requirements file at {template_req}", BRIGHT + BLUE)
        try:
            with open(template_req, 'r') as src, open(REQUIREMENTS_FILE, 'w') as dst:
                dst.write(src.read())
            print_colored(f"[✓] Template requirements copied to {REQUIREMENTS_FILE}", BRIGHT + GREEN)
            return True
        except Exception as e:
            print_colored(f"[✗] Failed to copy template requirements: {str(e)}", BRIGHT + RED)
            
    return False

def prompt_generate_requirements():
    """Prompt the user to generate a requirements.txt file."""
    print_colored("\n======================================================")
    print_colored("      REQUIREMENTS.TXT FILE NOT FOUND", BRIGHT + YELLOW)
    print_colored("======================================================")
    print_colored(f"A requirements.txt file is recommended at:", BRIGHT)
    print_colored(f"  {REQUIREMENTS_FILE}", BRIGHT + BLUE)
    print_colored("\nWould you like to generate requirements.txt by scanning project files?")
    print_colored("This will analyze all Python files to find external dependencies.", BRIGHT)
    return prompt_yes_no("(y/n)")

def requirements_generator_main():
    """Main function to check and generate requirements."""
    print_colored("[*] Checking requirements.txt file...", BRIGHT + BLUE)
    
    if check_requirements_file():
        print_colored(f"[✓] Requirements file found at: {REQUIREMENTS_FILE}", BRIGHT + GREEN)
        return True
    else:
        print_colored("[!] Requirements file not found.", BRIGHT + YELLOW)
        
        if len(sys.argv) > 1 and sys.argv[1] == "--auto":
            should_generate = True
        else:
            should_generate = prompt_generate_requirements()
        
        if should_generate:
            # First try to copy from template
            if copy_existing_requirements():
                return True
            
            # If no template, scan project
            print_colored("[*] Scanning project for Python imports...", BRIGHT + BLUE)
            packages = scan_project_for_imports()
            
            if generate_requirements_file(packages):
                print_colored("[✓] Requirements file generated successfully.", BRIGHT + GREEN)
                print_colored("\nReview the file and install dependencies with:")
                print_colored("  pip install -r requirements.txt", BRIGHT + BLUE)
                return True
            else:
                print_colored("[✗] Failed to generate requirements file.", BRIGHT + RED)
                return False
        else:
            print_colored("Requirements generation skipped.", BRIGHT + YELLOW)
            return False

#==============================================================================
# Environment Checker
#==============================================================================

def set_project_root_as_working_dir():
    """Set the working directory to the project root."""
    # Project root is already determined at the beginning of the script
    os.chdir(PROJECT_ROOT)
    print_colored(f"[*] Working directory set to: {PROJECT_ROOT}", BRIGHT + BLUE)
    return PROJECT_ROOT

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

def environment_checker_main():
    """Main function to check files and run generators if needed."""
    # First set the working directory to project root
    set_project_root_as_working_dir()
    
    # Define paths for scripts
    config_generator_path = PROJECT_ROOT / "src" / "utils" / "config_generator.py"
    requirement_generator_path = PROJECT_ROOT / "src" / "utils" / "requirement_generator.py"
    
    # Define paths for files
    config_file = PROJECT_ROOT / "conf" / "config.yaml"
    
    print_colored("\n======================================================")
    print_colored("            TRADING SYSTEM ENVIRONMENT CHECK", BRIGHT + MAGENTA)
    print_colored("======================================================")
    
    # Check for requirements.txt
    print_colored("\n[*] Checking for requirements.txt...", BRIGHT + BLUE)
    if check_requirements_file():
        print_colored(f"[✓] Requirements file found at: {REQUIREMENTS_FILE}", BRIGHT + GREEN)
    else:
        print_colored(f"[!] Requirements file not found at: {REQUIREMENTS_FILE}", BRIGHT + YELLOW)
        
        if requirement_generator_path.exists():
            if prompt_yes_no("Would you like to generate requirements.txt now?"):
                print_colored("\n[*] Running requirement generator...", BRIGHT + BLUE)
                if run_script(requirement_generator_path):
                    print_colored("[✓] Requirements file generated successfully.", BRIGHT + GREEN)
                else:
                    print_colored("[✗] Failed to generate requirements file.", BRIGHT + RED)
                    if not prompt_yes_no("Continue anyway?"):
                        print_colored("Exiting.", BRIGHT + RED)
                        return 1
        else:
            print_colored(f"[!] Requirement generator not found at {requirement_generator_path}", BRIGHT + YELLOW)
            if not prompt_yes_no("Continue without requirements.txt?"):
                print_colored("Exiting.", BRIGHT + RED)
                return 1
    
    # Check for config.yaml
    print_colored("\n[*] Checking for config.yaml...", BRIGHT + BLUE)
    if check_file_exists(config_file):
        print_colored(f"[✓] Configuration file found at: {config_file}", BRIGHT + GREEN)
    else:
        print_colored(f"[!] Configuration file not found at: {config_file}", BRIGHT + YELLOW)
        
        if config_generator_path.exists():
            if prompt_yes_no("Would you like to generate a default configuration file now?"):
                print_colored("\n[*] Running configuration generator...", BRIGHT + BLUE)
                if run_script(config_generator_path):
                    print_colored("[✓] Configuration file generated successfully.", BRIGHT + GREEN)
                else:
                    print_colored("[✗] Failed to generate configuration file.", BRIGHT + RED)
                    if not prompt_yes_no("Continue anyway?"):
                        print_colored("Exiting.", BRIGHT + RED)
                        return 1
        else:
            print_colored(f"[!] Configuration generator not found at {config_generator_path}", BRIGHT + YELLOW)
            if not prompt_yes_no("Continue without configuration?"):
                print_colored("Exiting.", BRIGHT + RED)
                return 1
    
    print_colored("\n[✓] Environment check completed successfully.", BRIGHT + GREEN)
    print_colored("    Ready to launch the trading system.", BRIGHT + GREEN)
    print_colored("======================================================\n")
    return 0

def check_file_exists(file_path):
    """Check if a file exists."""
    return os.path.exists(file_path)

#==============================================================================
# Main Entry Point
#==============================================================================

if __name__ == "__main__":
    # Determine which utility to run based on the script name or argument
    script_name = os.path.basename(__file__)
    
    if script_name == "config_generator.py" or (len(sys.argv) > 1 and sys.argv[1] == "config"):
        sys.exit(config_generator_main())
    elif script_name == "requirement_generator.py" or (len(sys.argv) > 1 and sys.argv[1] == "requirements"):
        sys.exit(requirements_generator_main())
    else:
        # Default to environment checker
        sys.exit(environment_checker_main())