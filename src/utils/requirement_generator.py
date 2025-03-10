#!/usr/bin/env python3
"""
Utility for generating requirements.txt by scanning Python files for imports.

This script scans all Python files in the project directory and extracts
imported modules to create a requirements.txt file.
"""

import os
import re
import sys
import logging
from pathlib import Path
from collections import defaultdict
import ast
import importlib.util

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
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

# Get project root directory
def get_project_root():
    """Get the absolute path to the project root directory."""
    # When script is in src/utils/, project root is two levels up
    return Path(__file__).resolve().parent.parent.parent

def print_colored(message, color=BRIGHT + CYAN):
    """Print colored message to terminal."""
    print(f"{color}{message}{RESET}")

PROJECT_ROOT = get_project_root()
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"

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
    response = input("(y/n): ").strip().lower()
    return response == 'y' or response == 'yes'

def main():
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

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)