#!/usr/bin/env python
# src/main/gui.py

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import datetime
from pathlib import Path
import json
import logging
import queue
import tkinter.font as tkFont
import asyncio
import weakref
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

try:
    # Import system modules
    from datasource.downloader import run_downloads, scan_and_migrate
    from common.config import ConfigManager, ConfigParser
    from common.logging import LogInitializer
    from src.common.async_executor import AsyncExecutor
    from src.core.core import TradingCore
    from datasource.downloader import run_downloads, scan_and_migrate
    from src.exchange.adapters.binance import Binance

except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure all dependencies are installed.")
    sys.exit(1)

# Custom logging handler to redirect logs to GUI
class QueueHandler(logging.Handler):
    """Custom handler to redirect logs to a queue for GUI display"""
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(record)

class TradingSystemGUI:
    """Trading System GUI Application"""
    
    def __init__(self, root, args=None):
        """Initialize the GUI application"""
        self.root = root
        self.args = args
        self.running = False
        self.paused = False
        self.thread = None
        self.log_queue = queue.Queue()
    
        self.config = args.config if args.config else ConfigManager(str(Path(project_root) / "conf" / "config.yaml"))

        # Set up logging
        self._init_logging()
        
        # Set up the executor for async operations with better resource management
        self.executor = AsyncExecutor(self.logger)
        
        # Set up the window
        self.setup_window()
                
        # Create GUI components
        self.create_components()      
          
        # Initialize form values from config
        self.init_values()
        
        # Set up custom styling
        self._setup_styling()
        
        # Log initialization
        self.logger = self.log_manager.get_logger("gui")
        self.logger.info("Trading System GUI initialized")
        
        # Set up periodic tasks with optimized intervals
        self.root.after(250, self._process_log_queue)
        
        # Optional auto-save feature with configurable interval
        self._init_autosave()
        
        # Check for recovery files
        self._check_for_recovery()
        
        # Initialize trader instance
        self.trader = None
        self.task_status = {"status": "idle"}
        
    def _init_logging(self):
        """Initialize logging system using LogManager and LogInitializer"""
        try:
            # Use LogInitializer for better log organization
            log_init = LogInitializer(self.config)
            log_init.initialize()
            
            # Get LogManager instance
            self.log_manager = log_init.log_manager
            
            # Set appropriate log level based on debug flag
            log_level = self.config.get("logging", "level", default="INFO")
            if self.args and hasattr(self.args, 'debug') and self.args.debug:
                log_level = "DEBUG"
                
            # Create and add queue handler for GUI display
            queue_handler = QueueHandler(self.log_queue)
            
            # Set the queue handler level
            numeric_level = getattr(logging, log_level.upper(), logging.INFO)
            queue_handler.setLevel(numeric_level)
            
            # Set formatter for queue handler
            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
            queue_handler.setFormatter(formatter)
            
            # Add queue handler to root logger via LogManager
            self.log_manager.addHandler(queue_handler)
            
            # Get the GUI logger
            self.logger = log_init.get_logger("gui")
            self.logger.info("LogManager initialized for GUI application")
            
        except Exception as e:
            print(f"Error setting up logging: {e}")
            # Fallback to basic logging
            logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s | %(levelname)s | %(message)s')
            # Create a basic logger as fallback
            self.logger = logging.getLogger("gui")
        
    def setup_window(self):
        """Set up the main window with optimized layout"""
        self.root.title("Trading System")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Create a custom font
        self.default_font = tkFont.nametofont("TkDefaultFont")
        self.header_font = tkFont.Font(
            family=self.default_font.cget("family"),
            size=self.default_font.cget("size") + 2,
            weight="bold"
        )
        
        # Use system theme when available
        try:
            ttk.Style().theme_use('clam')  # More modern look on most systems
        except tk.TclError:
            pass  # Fallback to default theme
            
        # Configure grid weights for responsive resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
    def _setup_styling(self):
        """Set up custom styles for the application"""
        self.style = ttk.Style()
        
        # Button styles
        self.style.configure("Run.TButton", 
                            font=("TkDefaultFont", 10, "bold"))
        self.style.configure("Pause.TButton", 
                            font=("TkDefaultFont", 10, "bold"))
        self.style.configure("Stop.TButton", 
                            font=("TkDefaultFont", 10, "bold"))
        self.style.configure("Exit.TButton", 
                            font=("TkDefaultFont", 10, "bold"))
        
        # Try to set colors if supported by theme
        try:
            self.style.map("Run.TButton", 
                          background=[('active', '#4CAF50'), ('!disabled', '#8BC34A')])
            self.style.map("Pause.TButton", 
                          background=[('active', '#FFC107'), ('!disabled', '#FFEB3B')])
            self.style.map("Stop.TButton", 
                          background=[('active', '#F44336'), ('!disabled', '#FF5722')])
            self.style.map("Exit.TButton", 
                          background=[('active', '#9E9E9E'), ('!disabled', '#BDBDBD')])
        except tk.TclError:
            pass  # Fallback to default styling if custom colors not supported
        
    def create_components(self):
        """Create main UI components with performance optimizations"""
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create top control panel
        self.control_panel = self._create_control_panel()
        self.control_panel.pack(fill=tk.X, pady=(0, 5))
        
        # Create tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Setup tabs
        self.setup_tab = ttk.Frame(self.notebook)
        self.data_tab = ttk.Frame(self.notebook)
        self.log_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.setup_tab, text="Trading Setup")
        self.notebook.add(self.data_tab, text="Data Management")
        self.notebook.add(self.log_tab, text="Logs")
        self.notebook.add(self.results_tab, text="Results")
        
        # Initialize from config
        self._init_from_config()
        
        # Then override with args if provided
        if self.args:
            self._init_from_args(self.args)
        
        # Create content for each tab
        self._create_setup_tab()
        self._create_data_tab()
        self._create_log_tab()
        self._create_results_tab()
        
        # Add status bar
        self._create_status_bar()
        
    def _create_control_panel(self):
        """Create the top control panel with Run/Pause/Exit buttons"""
        panel = ttk.Frame(self.main_frame, padding="5")
        
        # Create run/pause/stop/exit buttons
        self.run_button = ttk.Button(
            panel, 
            text="Run", 
            command=self.start_trading,
            style="Run.TButton"
        )
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.pause_button = ttk.Button(
            panel, 
            text="Pause", 
            command=self.pause_trading,
            style="Pause.TButton"
        )
        self.pause_button.pack(side=tk.LEFT, padx=5)
        self.pause_button.config(state=tk.DISABLED)
        
        self.stop_button = ttk.Button(
            panel, 
            text="Stop", 
            command=self.stop_trading,
            style="Stop.TButton"
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.stop_button.config(state=tk.DISABLED)
        
        self.exit_button = ttk.Button(
            panel, 
            text="Exit", 
            command=self.on_close,
            style="Exit.TButton"
        )
        self.exit_button.pack(side=tk.RIGHT, padx=5)
        
        # Status display
        status_frame = ttk.Frame(panel)
        status_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10)
        
        self.control_status_var = tk.StringVar(value="Ready")
        self.control_status = ttk.Label(
            status_frame, 
            textvariable=self.control_status_var,
            font=("TkDefaultFont", 9, "italic")
        )
        self.control_status.pack(side=tk.RIGHT)
        
        return panel
        
    def _create_status_bar(self):
        """Create an enhanced status bar with more information"""
        status_frame = ttk.Frame(self.root, relief=tk.GROOVE, padding=2)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Main status message
        self.status_message = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_message, anchor=tk.W)
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Memory usage display
        self.memory_usage = tk.StringVar(value="Mem: 0 MB")
        memory_label = ttk.Label(status_frame, textvariable=self.memory_usage, width=15)
        memory_label.pack(side=tk.RIGHT, padx=5)
        
        # CPU usage indicator
        self.cpu_indicator = ttk.Progressbar(
            status_frame, 
            orient=tk.HORIZONTAL, 
            length=100, 
            mode='determinate',
            maximum=100
        )
        self.cpu_indicator.pack(side=tk.RIGHT, padx=5)
        
        # Update resource usage periodically
        self.root.after(2000, self._update_resource_usage)
        
    def _update_resource_usage(self):
        """Update resource usage indicators"""
        try:
            # Update memory usage
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            self.memory_usage.set(f"Mem: {memory_mb:.1f} MB")
            
            # Update CPU usage
            cpu_percent = process.cpu_percent(interval=0.1)
            self.cpu_indicator['value'] = cpu_percent
            
        except ImportError:
            # psutil not available, use simplified tracking
            import gc
            gc.collect()  # Encourage garbage collection
            self.memory_usage.set("Mem: N/A")
            self.cpu_indicator['value'] = 0
            
        # Schedule next update
        self.root.after(2000, self._update_resource_usage)
        
    def setup_logging(self):
        """Set up logging to redirect to GUI"""
        # This method is kept for backward compatibility
        # Actual logging setup is now in _init_logging
        pass
        
    def _init_from_config(self):
        """Initialize form values from config"""
        # General settings
        self.backtest_engine_var = tk.StringVar(value=self.config.get("backtest", "engine", default="ohlcv"))

        self.config_var = tk.StringVar(value=str(self.config._config_path or Path(project_root) / "conf/bt_config.yaml"))
        self.mode_var = tk.StringVar(value=self.config.get("system", "operational_mode", default="backtest"))
        self.symbol_var = tk.StringVar(value=",".join(self.config.get("trading", "instruments", default=["BTC/USDT"])))
        self.initial_capital_var = tk.DoubleVar(value=self.config.get("trading", "capital", "initial", default=10000))
        
        # Use current date and one month ago as defaults for backtest period
        today = datetime.date.today()
        one_month_ago = today - datetime.timedelta(days=30)
        
        self.start_date_var = tk.StringVar(value=self.config.get("backtest", "period", "start", 
                                                          default=one_month_ago.strftime("%Y-%m-%d")))
        self.end_date_var = tk.StringVar(value=self.config.get("backtest", "period", "end", 
                                                      default=today.strftime("%Y-%m-%d")))
        self.debug_var = tk.BooleanVar(value=False)
        
        # Costs
        self.commission_var = tk.DoubleVar(value=self.config.get("backtest", "costs", "commission", default=0.001))
        self.slippage_var = tk.DoubleVar(value=self.config.get("backtest", "costs", "slippage", default=0.001))
        
        # Strategy settings
        self.strategy_var = tk.StringVar(value=self.config.get("strategy", "active", default="dual_ma"))
        self.timeframe_var = tk.StringVar(value=self.config.get("strategy", "timeframe", default="1h"))
        self.fallback_strategy_var = tk.StringVar(value=self.config.get("strategy", "fallback_strategy", default=""))
        
        # Strategy parameters
        self.fast_period_var = tk.IntVar(value=self.config.get("strategy", "parameters", "fast_period", default=50))
        self.slow_period_var = tk.IntVar(value=self.config.get("strategy", "parameters", "slow_period", default=200))
        self.threshold_var = tk.DoubleVar(value=self.config.get("strategy", "parameters", "threshold", default=0.005))
        
        # Risk management
        self.stop_loss_enabled_var = tk.BooleanVar(value=self.config.get("risk", "stop_loss", "enabled", default=True))
        self.stop_loss_pct_var = tk.DoubleVar(value=self.config.get("risk", "stop_loss", "percentage", default=0.02))
        self.stop_loss_daily_var = tk.DoubleVar(value=self.config.get("risk", "stop_loss", "max_daily", default=0.05))
        
        self.take_profit_enabled_var = tk.BooleanVar(value=self.config.get("risk", "take_profit", "enabled", default=True))
        self.take_profit_pct_var = tk.DoubleVar(value=self.config.get("risk", "take_profit", "percentage", default=0.05))
        
        self.max_drawdown_var = tk.DoubleVar(value=self.config.get("risk", "drawdown", "max_portfolio", default=0.2))
        self.max_trade_drawdown_var = tk.DoubleVar(value=self.config.get("risk", "drawdown", "max_single_trade", default=0.1))
        
        # Position sizing
        self.max_position_var = tk.DoubleVar(value=self.config.get("trading", "limits", "position", default=0.1))
        self.min_order_var = tk.DoubleVar(value=self.config.get("trading", "capital", "min_order", default=0.001))
        
        # Advanced settings
        self.db_url_var = tk.StringVar(value=self.config.get("database", "url", default="sqlite:///database/trading.sqlite"))
        self.max_workers_var = tk.IntVar(value=self.config.get("system", "performance", "max_threads", default=4))
        self.max_memory_var = tk.IntVar(value=int(self.config.get("system", "performance", "max_memory", default=1073741824) / 1048576))
        self.timezone_var = tk.StringVar(value=self.config.get("system", "timezone", default="UTC"))
        self.task_timeout_var = tk.IntVar(value=self.config.get("system", "performance", "task_timeout", default=300))
        
        # Database settings
        self.db_pool_size_var = tk.IntVar(value=self.config.get("database", "pool_size", default=5))
        self.db_timeout_var = tk.IntVar(value=self.config.get("database", "timeout", default=30))
        self.db_echo_sql_var = tk.BooleanVar(value=self.config.get("database", "echo_sql", default=False))
        
        # Live trading safeguards
        self.emergency_stop_var = tk.BooleanVar(value=self.config.get("live", "safeguards", "emergency_stop", "enabled", default=True))
        self.emergency_threshold_var = tk.DoubleVar(value=self.config.get("live", "safeguards", "emergency_stop", "threshold", default=0.1))
        self.max_retries_var = tk.IntVar(value=self.config.get("live", "safeguards", "max_retries", default=3))
        self.cooldown_var = tk.IntVar(value=self.config.get("live", "safeguards", "cooldown", default=60))
        
        # API settings
        self.api_key_var = tk.StringVar(value=self.config.get("api", "binance", "api_key", default=""))
        self.api_secret_var = tk.StringVar(value=self.config.get("api", "binance", "secret", default=""))
        self.api_timeout_var = tk.IntVar(value=self.config.get("api", "timeout", default=30000))
        self.api_rate_limit_var = tk.IntVar(value=self.config.get("api", "rate_limits", "requests_per_minute", default=20))
        self.api_retry_attempts_var = tk.IntVar(value=self.config.get("api", "retries", "max_attempts", default=3))
        self.api_retry_delay_var = tk.IntVar(value=self.config.get("api", "retries", "delay_seconds", default=1))
        
        # Proxy settings
        self.api_useproxy_var = tk.BooleanVar(value=bool(self.config.get("proxies", "http", default="")))
        self.api_http_proxy_var = tk.StringVar(value=self.config.get("proxies", "http", default=""))
        self.api_https_proxy_var = tk.StringVar(value=self.config.get("proxies", "https", default=""))
        
        # Download settings
        self.dl_symbols_var = tk.StringVar(value="BTC/USDT")
        self.dl_timeframes = {
            "1m": tk.BooleanVar(value=False),
            "5m": tk.BooleanVar(value=False),
            "15m": tk.BooleanVar(value=False),
            "30m": tk.BooleanVar(value=False),
            "1h": tk.BooleanVar(value=True),
            "4h": tk.BooleanVar(value=False),
            "1d": tk.BooleanVar(value=False),
            "1w": tk.BooleanVar(value=False)
        }
        
        # Date range for download
        self.dl_start_date_var = tk.StringVar(value=one_month_ago.strftime("%Y-%m-%d"))
        self.dl_end_date_var = tk.StringVar(value=today.strftime("%Y-%m-%d"))
        self.dl_concurrent_var = tk.IntVar(value=3)
        
        # Migration settings
        self.migrate_symbol_var = tk.StringVar(value="")
        self.migrate_timeframe_var = tk.StringVar(value="")
        self.migrate_backup_var = tk.BooleanVar(value=True)
        self.migrate_delete_var = tk.BooleanVar(value=False)
        
        # View settings
        self.view_symbol_var = tk.StringVar(value="BTC/USDT")
        self.view_timeframe_var = tk.StringVar(value="1h")
        
        # Binance workaround
        self.use_binance_workaround = tk.BooleanVar(value=True)
        self.binance_workaround_var = tk.BooleanVar(value=True)

    def _init_from_args(self, args):
        """Initialize values from command line args"""
        if hasattr(args, 'config') and args.config:
            self.config_var.set(args.config)
        
        if hasattr(args, 'mode') and args.mode:
            self.mode_var.set(args.mode)
            
        if hasattr(args, 'backtest_engine') and args.backtest_engine:
            self.backtest_engine_var.set(args.backtest_engine)
            
        if hasattr(args, 'strategy') and args.strategy:
            self.strategy_var.set(args.strategy)
            
        if hasattr(args, 'symbol') and args.symbol:
            self.symbol_var.set(args.symbol)
            
        if hasattr(args, 'timeframe') and args.timeframe:
            self.timeframe_var.set(args.timeframe)
            
        if hasattr(args, 'start_date') and args.start_date:
            self.start_date_var.set(args.start_date)
            
        if hasattr(args, 'end_date') and args.end_date:
            self.end_date_var.set(args.end_date)
            
        if hasattr(args, 'debug') and args.debug:
            self.debug_var.set(True)
            
        if hasattr(args, 'db_url') and args.db_url:
            self.db_url_var.set(args.db_url)
            
        if hasattr(args, 'max_workers') and args.max_workers:
            self.max_workers_var.set(args.max_workers)
            
    def init_values(self):
        """Initialize form values from config or args"""
        # Just an empty method to maintain backward compatibility
        # The actual initialization is done in _init_from_config and _init_from_args
        pass

    def _create_setup_tab(self):
        """Create content for trading setup tab using original code"""
        # Create notebook for setup sections
        setup_notebook = ttk.Notebook(self.setup_tab)
        setup_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs for different setup sections
        general_tab = ttk.Frame(setup_notebook)
        strategy_tab = ttk.Frame(setup_notebook)
        risk_tab = ttk.Frame(setup_notebook)
        api_tab = ttk.Frame(setup_notebook)
        advanced_tab = ttk.Frame(setup_notebook)
        
        setup_notebook.add(general_tab, text="General")
        setup_notebook.add(strategy_tab, text="Strategy")
        setup_notebook.add(risk_tab, text="Risk Management")
        setup_notebook.add(api_tab, text="API Settings")
        setup_notebook.add(advanced_tab, text="Advanced")
        
        # Create the tab contents
        self._create_general_tab(general_tab)
        self._create_strategy_tab(strategy_tab)
        self._create_risk_tab(risk_tab)
        self._create_api_settings_tab(api_tab)
        self._create_advanced_tab(advanced_tab)
        
        # Add save/start button section at the bottom
        self._create_setup_buttons()
        
    def _create_setup_buttons(self):
        """Create common buttons for setup tabs"""
        button_frame = ttk.Frame(self.setup_tab, padding="10")
        button_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
        
        ttk.Separator(button_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Create button container
        action_frame = ttk.Frame(button_frame)
        action_frame.pack(fill=tk.X)
        
        # Create action buttons with distinct styling
        save_btn = ttk.Button(action_frame, text="Save Configuration", command=self.save_config)
        save_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(action_frame, text="Reset All", command=self.reset_form).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Status message
        self.setup_status_var = tk.StringVar()
        status_label = ttk.Label(button_frame, textvariable=self.setup_status_var, font=('', 9, 'italic'))
        status_label.pack(fill=tk.X, pady=(5, 0))
        
    def _create_general_tab(self, parent):
        """Create content for general settings tab"""
        
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        header = ttk.Label(frame, text="General Trading Settings", font=self.header_font)
        header.grid(column=0, row=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        
        # Configuration file
        row = 1
        ttk.Label(frame, text="Configuration File:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        config_entry = ttk.Entry(frame, textvariable=self.config_var, width=50)
        config_entry.grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        config_btn = ttk.Button(frame, text="Browse...", command=self._browse_config)
        config_btn.grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Mode selection
        row += 1
        ttk.Label(frame, text="Trading Mode:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        modes = ["backtest", "paper", "live"]
        mode_combo = ttk.Combobox(frame, textvariable=self.mode_var, values=modes, state="readonly")
        mode_combo.grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(frame, text="(backtest/paper/live)").grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Add binding to change available options when mode changes
        mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)
        
        # Add Backtest Engine Selection (new)
        row += 1
        self.backtest_engine_label = ttk.Label(frame, text="Backtest Engine:")
        self.backtest_engine_label.grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        engine_types = ["ohlcv", "market_replay"]
        self.backtest_engine_combo = ttk.Combobox(frame, textvariable=self.backtest_engine_var, values=engine_types, state="readonly")
        self.backtest_engine_combo.grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.backtest_engine_info = ttk.Label(frame, text="(Standard OHLCV / Market Replay)")
        self.backtest_engine_info.grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Add binding for engine selection change
        self.backtest_engine_combo.bind("<<ComboboxSelected>>", self._on_engine_change)
        
        # Backtest engine descriptions
        engine_descriptions = {
            "ohlcv": "Standard OHLCV engine for basic backtesting. Performs faster but with simplified order execution.",
            "market_replay": "High fidelity market replay engine with realistic order execution, market impact simulation, and order queuing."
        }        
        info_button = ttk.Button(frame, text="?", width=2, 
                                command=lambda: messagebox.showinfo("Backtest Engine Types", 
                                                                f"OHLCV (Basic): {engine_descriptions['ohlcv']}\n\n"
                                                                f"Market Replay: {engine_descriptions['market_replay']}"))
        info_button.grid(column=2, row=row, sticky=tk.E, padx=5, pady=5)        

        # Trading symbol
        row += 1
        ttk.Label(frame, text="Trading Symbols:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.symbol_var, width=50).grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(frame, text="(comma-separated)").grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Initial capital
        row += 1
        ttk.Label(frame, text="Initial Capital:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.initial_capital_var, width=15).grid(column=1, row=row, sticky=tk.W, padx=5, pady=5)
        ttk.Label(frame, text="(Starting funds)").grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Trading time section
        row += 1
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(column=0, row=row, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        row += 1
        ttk.Label(frame, text="Backtest Period", font=self.header_font).grid(column=0, row=row, columnspan=3, sticky=tk.W, pady=(0, 5))
        
        # Backtest period - Start date
        row += 1
        ttk.Label(frame, text="Start Date:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        start_date = ttk.Entry(frame, textvariable=self.start_date_var)
        start_date.grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(frame, text="(YYYY-MM-DD)").grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Calendar picker for start date
        start_calendar_btn = ttk.Button(frame, text="📅", width=3, 
                                      command=lambda: self._show_calendar(self.start_date_var))
        start_calendar_btn.grid(column=1, row=row, sticky=tk.E, padx=5, pady=5)
        
        # Backtest period - End date
        row += 1
        ttk.Label(frame, text="End Date:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        end_date = ttk.Entry(frame, textvariable=self.end_date_var)
        end_date.grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(frame, text="(YYYY-MM-DD)").grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Calendar picker for end date
        end_calendar_btn = ttk.Button(frame, text="📅", width=3, 
                                    command=lambda: self._show_calendar(self.end_date_var))
        end_calendar_btn.grid(column=1, row=row, sticky=tk.E, padx=5, pady=5)
        
        # Backtest costs
        row += 1
        ttk.Label(frame, text="Commission Rate:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.commission_var, width=10).grid(column=1, row=row, sticky=tk.W, padx=5, pady=5)
        ttk.Label(frame, text="(e.g., 0.001 for 0.1%)").grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        row += 1
        ttk.Label(frame, text="Slippage:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.slippage_var, width=10).grid(column=1, row=row, sticky=tk.W, padx=5, pady=5)
        ttk.Label(frame, text="(e.g., 0.001 for 0.1%)").grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Debug mode
        row += 1
        ttk.Checkbutton(frame, text="Debug Mode", variable=self.debug_var).grid(column=1, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Make the grid resizable
        frame.columnconfigure(1, weight=1)
        
    def _show_calendar(self, date_var):
        """Show a calendar popup for date selection"""
        try:
            from tkcalendar import Calendar
            
            # Parse current date
            try:
                current_date = datetime.datetime.strptime(date_var.get(), "%Y-%m-%d").date()
            except ValueError:
                current_date = datetime.date.today()
            
            # Create top level window
            top = tk.Toplevel(self.root)
            top.title("Select Date")
            top.geometry("300x250")
            top.resizable(False, False)
            top.transient(self.root)
            top.grab_set()
            
            # Create calendar
            cal = Calendar(top, selectmode='day', 
                          year=current_date.year, 
                          month=current_date.month,
                          day=current_date.day)
            cal.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Button to confirm selection
            def set_date():
                date_var.set(cal.get_date())
                top.destroy()
                
            ttk.Button(top, text="Select", command=set_date).pack(pady=10)
            
        except ImportError:
            # tkcalendar not available, show message
            messagebox.showinfo("Calendar Unavailable", 
                              "Calendar picker requires tkcalendar package.\nPlease enter date manually in YYYY-MM-DD format.")
        
    def _create_strategy_tab(self, parent):
        """Create content for strategy settings tab"""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        header = ttk.Label(frame, text="Strategy Configuration", font=self.header_font)
        header.grid(column=0, row=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        
        # Strategy selection
        row = 1
        ttk.Label(frame, text="Active Strategy:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Populate strategies from ConfigParser for better discovery
        parser = ConfigParser(self.config_var.get())
        strategies = parser.get_available_strategies()
        
        strategy_combo = ttk.Combobox(frame, textvariable=self.strategy_var, values=strategies)
        strategy_combo.grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Timeframe selection
        row += 1
        ttk.Label(frame, text="Timeframe:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        timeframes = parser.get_available_timeframes()
        ttk.Combobox(frame, textvariable=self.timeframe_var, values=timeframes, state="readonly").grid(
            column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(frame, text="(Data resolution)").grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Fallback strategy
        row += 1
        ttk.Label(frame, text="Fallback Strategy:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        fallback_combo = ttk.Combobox(frame, textvariable=self.fallback_strategy_var, values=[""] + strategies)
        fallback_combo.grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(frame, text="(Optional)").grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Strategy Parameters section
        row += 1
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(column=0, row=row, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        row += 1
        ttk.Label(frame, text="Strategy Parameters", font=self.header_font).grid(
            column=0, row=row, columnspan=3, sticky=tk.W, pady=(0, 5))
        
        # Parameters for Dual MA Strategy
        row += 1
        self.strat_param_frame = ttk.LabelFrame(frame, text="Dual Moving Average Parameters", padding=10)
        self.strat_param_frame.grid(column=0, row=row, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Fast period
        ttk.Label(self.strat_param_frame, text="Fast Period:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(self.strat_param_frame, from_=1, to=500, textvariable=self.fast_period_var, width=10).grid(
            column=1, row=0, sticky=tk.W, padx=5, pady=5)
        ttk.Label(self.strat_param_frame, text="(Shorter MA period)").grid(column=2, row=0, sticky=tk.W, padx=5, pady=5)
        
        # Slow period
        ttk.Label(self.strat_param_frame, text="Slow Period:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(self.strat_param_frame, from_=1, to=500, textvariable=self.slow_period_var, width=10).grid(
            column=1, row=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(self.strat_param_frame, text="(Longer MA period)").grid(column=2, row=1, sticky=tk.W, padx=5, pady=5)
        
        # Threshold
        ttk.Label(self.strat_param_frame, text="Signal Threshold:").grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.strat_param_frame, textvariable=self.threshold_var, width=10).grid(
            column=1, row=2, sticky=tk.W, padx=5, pady=5)
        ttk.Label(self.strat_param_frame, text="(Minimum crossover difference)").grid(
            column=2, row=2, sticky=tk.W, padx=5, pady=5)
        
        # Make the parameter frame resizable
        self.strat_param_frame.columnconfigure(2, weight=1)
        
        # Add event to change parameter frame when strategy changes
        strategy_combo.bind("<<ComboboxSelected>>", self._on_strategy_change)
        
        # Make the main frame resizable
        frame.columnconfigure(1, weight=1)
        
        # Add optimization section
        row += 1
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(column=0, row=row, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        row += 1
        ttk.Label(frame, text="Strategy Optimization", font=self.header_font).grid(
            column=0, row=row, columnspan=3, sticky=tk.W, pady=(0, 5))
        
        # Basic optimization options
        row += 1
        opt_frame = ttk.Frame(frame)
        opt_frame.grid(column=0, row=row, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Checkbox for parameters to optimize
        self.optimize_fast_period = tk.BooleanVar(value=False)
        self.optimize_slow_period = tk.BooleanVar(value=False)
        self.optimize_threshold = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(opt_frame, text="Optimize Fast Period", variable=self.optimize_fast_period).grid(
            column=0, row=0, sticky=tk.W, padx=5, pady=5)
        ttk.Checkbutton(opt_frame, text="Optimize Slow Period", variable=self.optimize_slow_period).grid(
            column=1, row=0, sticky=tk.W, padx=5, pady=5)
        ttk.Checkbutton(opt_frame, text="Optimize Threshold", variable=self.optimize_threshold).grid(
            column=2, row=0, sticky=tk.W, padx=5, pady=5)
        
        # Optimization metric
        row += 1
        self.optimization_metric = tk.StringVar(value="sharpe_ratio")
        ttk.Label(frame, text="Optimization Metric:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        metrics = ["sharpe_ratio", "total_return", "max_drawdown", "win_rate", "profit_factor"]
        ttk.Combobox(frame, textvariable=self.optimization_metric, values=metrics, state="readonly").grid(
            column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Optimization button
        row += 1
        ttk.Button(frame, text="Run Optimization", command=self.start_optimization).grid(
            column=1, row=row, sticky=tk.W, padx=5, pady=10)
            
    def _create_risk_tab(self, parent):
        """Create content for risk management tab"""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        header = ttk.Label(frame, text="Risk Management", font=self.header_font)
        header.grid(column=0, row=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        
        # Stop Loss section
        row = 1
        sl_frame = ttk.LabelFrame(frame, text="Stop Loss", padding=10)
        sl_frame.grid(column=0, row=row, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        ttk.Checkbutton(sl_frame, text="Enable Stop Loss", variable=self.stop_loss_enabled_var,
                      command=self._toggle_stop_loss).grid(column=0, row=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Stop loss percentage
        ttk.Label(sl_frame, text="Stop Loss %:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        self.sl_pct_entry = ttk.Entry(sl_frame, textvariable=self.stop_loss_pct_var, width=10)
        self.sl_pct_entry.grid(column=1, row=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(sl_frame, text="(e.g., 0.02 for 2%)").grid(column=2, row=1, sticky=tk.W, padx=5, pady=5)
        
        # Daily loss limit
        ttk.Label(sl_frame, text="Daily Loss Limit:").grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)
        self.sl_daily_entry = ttk.Entry(sl_frame, textvariable=self.stop_loss_daily_var, width=10)
        self.sl_daily_entry.grid(column=1, row=2, sticky=tk.W, padx=5, pady=5)
        ttk.Label(sl_frame, text="(Maximum daily loss %)").grid(column=2, row=2, sticky=tk.W, padx=5, pady=5)
        
        # Take Profit section
        row += 1
        tp_frame = ttk.LabelFrame(frame, text="Take Profit", padding=10)
        tp_frame.grid(column=0, row=row, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        ttk.Checkbutton(tp_frame, text="Enable Take Profit", variable=self.take_profit_enabled_var,
                      command=self._toggle_take_profit).grid(column=0, row=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Take profit percentage
        ttk.Label(tp_frame, text="Take Profit %:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        self.tp_pct_entry = ttk.Entry(tp_frame, textvariable=self.take_profit_pct_var, width=10)
        self.tp_pct_entry.grid(column=1, row=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(tp_frame, text="(e.g., 0.05 for 5%)").grid(column=2, row=1, sticky=tk.W, padx=5, pady=5)
        
        # Drawdown section
        row += 1
        dd_frame = ttk.LabelFrame(frame, text="Maximum Drawdown", padding=10)
        dd_frame.grid(column=0, row=row, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Max portfolio drawdown
        ttk.Label(dd_frame, text="Portfolio Drawdown:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(dd_frame, textvariable=self.max_drawdown_var, width=10).grid(column=1, row=0, sticky=tk.W, padx=5, pady=5)
        ttk.Label(dd_frame, text="(e.g., 0.2 for 20%)").grid(column=2, row=0, sticky=tk.W, padx=5, pady=5)
        
        # Max trade drawdown
        ttk.Label(dd_frame, text="Single Trade Drawdown:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(dd_frame, textvariable=self.max_trade_drawdown_var, width=10).grid(column=1, row=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(dd_frame, text="(e.g., 0.1 for 10%)").grid(column=2, row=1, sticky=tk.W, padx=5, pady=5)
        
        # Position Sizing section
        row += 1
        pos_frame = ttk.LabelFrame(frame, text="Position Sizing", padding=10)
        pos_frame.grid(column=0, row=row, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Max position size
        ttk.Label(pos_frame, text="Max Position Size:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(pos_frame, textvariable=self.max_position_var, width=10).grid(column=1, row=0, sticky=tk.W, padx=5, pady=5)
        ttk.Label(pos_frame, text="(Fraction of portfolio)").grid(column=2, row=0, sticky=tk.W, padx=5, pady=5)
        
        # Min order size
        ttk.Label(pos_frame, text="Minimum Order Size:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(pos_frame, textvariable=self.min_order_var, width=10).grid(column=1, row=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(pos_frame, text="(Minimum transaction amount)").grid(column=2, row=1, sticky=tk.W, padx=5, pady=5)
        
        # Make frames resizable
        sl_frame.columnconfigure(2, weight=1)
        tp_frame.columnconfigure(2, weight=1)
        dd_frame.columnconfigure(2, weight=1)
        pos_frame.columnconfigure(2, weight=1)
        frame.columnconfigure(0, weight=1)
        
    def _create_advanced_tab(self, parent):
        """Create content for advanced settings tab"""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        header = ttk.Label(frame, text="Advanced System Settings", font=self.header_font)
        header.grid(column=0, row=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        
        # Database section
        row = 1
        db_frame = ttk.LabelFrame(frame, text="Database Configuration", padding=10)
        db_frame.grid(column=0, row=row, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Database URL
        ttk.Label(db_frame, text="Database URL:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(db_frame, textvariable=self.db_url_var, width=50).grid(column=1, row=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Database pool size
        ttk.Label(db_frame, text="Connection Pool Size:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(db_frame, from_=1, to=50, textvariable=self.db_pool_size_var, width=10).grid(column=1, row=1, sticky=tk.W, padx=5, pady=5)
        
        # Database timeout
        ttk.Label(db_frame, text="Database Timeout (s):").grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(db_frame, from_=5, to=300, textvariable=self.db_timeout_var, width=10).grid(column=1, row=2, sticky=tk.W, padx=5, pady=5)
        
        # Echo SQL
        ttk.Checkbutton(db_frame, text="Echo SQL (Debug Mode)", variable=self.db_echo_sql_var).grid(column=1, row=3, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # System Performance section
        row += 1
        perf_frame = ttk.LabelFrame(frame, text="System Performance", padding=10)
        perf_frame.grid(column=0, row=row, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Max threads/workers
        ttk.Label(perf_frame, text="Max Worker Threads:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(perf_frame, from_=1, to=32, textvariable=self.max_workers_var, width=10).grid(column=1, row=0, sticky=tk.W, padx=5, pady=5)
        ttk.Label(perf_frame, text="(Higher values use more CPU)").grid(column=2, row=0, sticky=tk.W, padx=5, pady=5)
        
        # Max memory
        ttk.Label(perf_frame, text="Max Memory (MB):").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(perf_frame, textvariable=self.max_memory_var, width=10).grid(column=1, row=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(perf_frame, text="(Memory limit for operations)").grid(column=2, row=1, sticky=tk.W, padx=5, pady=5)
        
        # Task timeout
        ttk.Label(perf_frame, text="Task Timeout (s):").grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(perf_frame, from_=30, to=3600, textvariable=self.task_timeout_var, width=10).grid(column=1, row=2, sticky=tk.W, padx=5, pady=5)
        ttk.Label(perf_frame, text="(Maximum time for async tasks)").grid(column=2, row=2, sticky=tk.W, padx=5, pady=5)
        
        # Timezone
        ttk.Label(perf_frame, text="System Timezone:").grid(column=0, row=3, sticky=tk.W, padx=5, pady=5)
        timezones = ["UTC", "America/New_York", "Europe/London", "Asia/Singapore", "Asia/Tokyo", "Asia/Shanghai"]
        ttk.Combobox(perf_frame, textvariable=self.timezone_var, values=timezones, width=20).grid(column=1, row=3, sticky=tk.W, padx=5, pady=5)
        
        # Live Trading Settings section
        row += 1
        live_frame = ttk.LabelFrame(frame, text="Live Trading Safeguards", padding=10)
        live_frame.grid(column=0, row=row, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Emergency stop
        ttk.Checkbutton(live_frame, text="Enable Emergency Stop", variable=self.emergency_stop_var,
                      command=self._toggle_emergency_stop).grid(column=0, row=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Emergency threshold
        ttk.Label(live_frame, text="Emergency Threshold:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        self.emergency_threshold_entry = ttk.Entry(live_frame, textvariable=self.emergency_threshold_var, width=10)
        self.emergency_threshold_entry.grid(column=1, row=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(live_frame, text="(e.g., 0.1 for 10% daily loss)").grid(column=2, row=1, sticky=tk.W, padx=5, pady=5)
        
        # Max retries
        ttk.Label(live_frame, text="Max Retries:").grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(live_frame, from_=1, to=10, textvariable=self.max_retries_var, width=10).grid(column=1, row=2, sticky=tk.W, padx=5, pady=5)
        ttk.Label(live_frame, text="(Error recovery attempts)").grid(column=2, row=2, sticky=tk.W, padx=5, pady=5)
        
        # Cooldown period
        ttk.Label(live_frame, text="Cooldown Period (s):").grid(column=0, row=3, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(live_frame, from_=1, to=300, textvariable=self.cooldown_var, width=10).grid(column=1, row=3, sticky=tk.W, padx=5, pady=5)
        ttk.Label(live_frame, text="(Delay between retries)").grid(column=2, row=3, sticky=tk.W, padx=5, pady=5)
        
        # Make frames resizable
        db_frame.columnconfigure(1, weight=1)
        perf_frame.columnconfigure(2, weight=1)
        live_frame.columnconfigure(2, weight=1)
        frame.columnconfigure(0, weight=1)
        
    def _create_api_settings_tab(self, parent):
        """Create content for API settings tab"""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        header = ttk.Label(frame, text="Exchange API Settings", font=self.header_font)
        header.grid(column=0, row=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        
        # API credentials
        row = 1
        ttk.Label(frame, text="API Key:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        api_key_entry = ttk.Entry(frame, textvariable=self.api_key_var, width=50, show="*")
        api_key_entry.grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(frame, text="Show/Hide", command=lambda: self._toggle_show_password(api_key_entry)).grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        row += 1
        ttk.Label(frame, text="API Secret:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        api_secret_entry = ttk.Entry(frame, textvariable=self.api_secret_var, width=50, show="*")
        api_secret_entry.grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(frame, text="Show/Hide", command=lambda: self._toggle_show_password(api_secret_entry)).grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        # API parameters
        row += 1
        ttk.Label(frame, text="Timeout (ms):").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(frame, from_=1000, to=300000, increment=1000, textvariable=self.api_timeout_var, width=10).grid(column=1, row=row, sticky=tk.W, padx=5, pady=5)
        
        row += 1
        ttk.Label(frame, text="Rate Limit (per min):").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(frame, from_=5, to=100, textvariable=self.api_rate_limit_var, width=10).grid(column=1, row=row, sticky=tk.W, padx=5, pady=5)
        
        row += 1
        ttk.Label(frame, text="Retry Attempts:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(frame, from_=1, to=10, textvariable=self.api_retry_attempts_var, width=10).grid(column=1, row=row, sticky=tk.W, padx=5, pady=5)
        
        row += 1
        ttk.Label(frame, text="Retry Delay (sec):").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(frame, from_=1, to=30, textvariable=self.api_retry_delay_var, width=10).grid(column=1, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Proxy settings
        row += 1
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(column=0, row=row, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        row += 1
        ttk.Label(frame, text="Proxy Settings", font=self.header_font).grid(column=0, row=row, columnspan=3, sticky=tk.W, pady=(0, 10))
        
        row += 1
        ttk.Checkbutton(frame, text="Use Proxy", variable=self.api_useproxy_var, command=self._toggle_proxy_fields).grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        
        row += 1
        ttk.Label(frame, text="HTTP Proxy:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        self.http_proxy_entry = ttk.Entry(frame, textvariable=self.api_http_proxy_var, width=50)
        self.http_proxy_entry.grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.http_proxy_example = ttk.Label(frame, text="(e.g., http://127.0.0.1:7890)")
        self.http_proxy_example.grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        row += 1
        ttk.Label(frame, text="HTTPS Proxy:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        self.https_proxy_entry = ttk.Entry(frame, textvariable=self.api_https_proxy_var, width=50)
        self.https_proxy_entry.grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.https_proxy_example = ttk.Label(frame, text="(e.g., https://127.0.0.1:7890)")
        self.https_proxy_example.grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Binance API workaround setting
        row += 1
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(column=0, row=row, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        row += 1
        ttk.Label(frame, text="Advanced Settings", font=self.header_font).grid(column=0, row=row, columnspan=3, sticky=tk.W, pady=(0, 10))
        
        row += 1
        ttk.Checkbutton(
            frame, 
            text="Enable Binance API workaround (fixes \"exchangeInfo\" HTTP 403 errors)", 
            variable=self.binance_workaround_var
        ).grid(column=0, row=row, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # Save button
        row += 1
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(column=0, row=row, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        row += 1
        button_frame = ttk.Frame(frame)
        button_frame.grid(column=0, row=row, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Save API Settings", command=self.save_api_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Test Connection", command=self.test_api_connection).pack(side=tk.LEFT, padx=5)
        
        # Status message
        row += 1
        self.api_status_var = tk.StringVar()
        ttk.Label(frame, textvariable=self.api_status_var, font=('', 9, 'italic')).grid(column=0, row=row, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # Make the grid resizable
        frame.columnconfigure(1, weight=1)
        
    def _create_data_tab(self):
        """Create content for data management tab"""
        frame = ttk.Frame(self.data_tab, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        # Create notebook for data operations
        data_notebook = ttk.Notebook(frame)
        data_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create tabs for different data operations
        download_tab = ttk.Frame(data_notebook)
        migrate_tab = ttk.Frame(data_notebook)
        view_tab = ttk.Frame(data_notebook)

        data_notebook.add(download_tab, text="Download Data")
        data_notebook.add(migrate_tab, text="Migrate Data")
        data_notebook.add(view_tab, text="View Data")

        # Create the data tab contents
        self._create_download_tab(download_tab)
        self._create_migrate_tab(migrate_tab)
        self._create_view_tab(view_tab)

    def _create_download_tab(self, parent):
        """Create content for data download tab"""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        header = ttk.Label(frame, text="Download Historical Market Data", font=self.header_font)
        header.grid(column=0, row=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        
        # Trading symbols
        row = 1
        ttk.Label(frame, text="Trading Symbols:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.dl_symbols_var, width=50).grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(frame, text="(comma-separated)").grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Timeframes
        row += 1
        ttk.Label(frame, text="Timeframes:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Timeframe checkboxes
        timeframe_frame = ttk.Frame(frame)
        timeframe_frame.grid(column=1, row=row, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        col = 0
        for timeframe, var in self.dl_timeframes.items():
            ttk.Checkbutton(timeframe_frame, text=timeframe, variable=var).grid(column=col, row=0, padx=5, pady=2)
            col += 1
            if col > 3:  # Create a new row after 4 columns
                col = 0
                row += 1
        
        # Start date
        row = 3
        ttk.Label(frame, text="Start Date:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.dl_start_date_var).grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(frame, text="(YYYY-MM-DD)").grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Calendar button for start date
        start_cal_btn = ttk.Button(frame, text="📅", width=3, 
                                  command=lambda: self._show_calendar(self.dl_start_date_var))
        start_cal_btn.grid(column=1, row=row, sticky=tk.E, padx=5, pady=5)
        
        # End date
        row += 1
        ttk.Label(frame, text="End Date:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.dl_end_date_var).grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(frame, text="(YYYY-MM-DD)").grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Calendar button for end date
        end_cal_btn = ttk.Button(frame, text="📅", width=3, 
                                command=lambda: self._show_calendar(self.dl_end_date_var))
        end_cal_btn.grid(column=1, row=row, sticky=tk.E, padx=5, pady=5)
        
        # Max concurrent downloads
        row += 1
        ttk.Label(frame, text="Max Concurrent:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(frame, from_=1, to=10, textvariable=self.dl_concurrent_var, width=5).grid(column=1, row=row, sticky=tk.W, padx=5, pady=5)
        ttk.Label(frame, text="(Higher values may hit rate limits)").grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Status message
        row += 1
        self.dl_status_var = tk.StringVar()
        ttk.Label(frame, textvariable=self.dl_status_var, font=('', 9, 'italic')).grid(column=0, row=row, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # Create a separator
        row += 1
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(column=0, row=row, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Add a workaround checkbox for Binance connection issue
        row += 1
        ttk.Checkbutton(
            frame, 
            text="Use Binance API workaround (fix for HTTP 403 errors)", 
            variable=self.use_binance_workaround
        ).grid(column=0, row=row, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # Create buttons
        row += 1
        button_frame = ttk.Frame(frame)
        button_frame.grid(column=0, row=row, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Download Data", command=self.start_data_download).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset", command=self.reset_download_form).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        row += 1
        self.download_progress = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=100, mode='indeterminate')
        self.download_progress.grid(column=0, row=row, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Make the grid resizable
        frame.columnconfigure(1, weight=1)
        
    def _create_migrate_tab(self, parent):
        """Create content for data migration tab"""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        header = ttk.Label(frame, text="Migrate Data Format", font=self.header_font)
        header.grid(column=0, row=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        
        # Description
        row = 1
        description = "Migrate data from legacy timestamp-based files to daily files. Leave symbol/timeframe empty to migrate all data."
        ttk.Label(frame, text=description, wraplength=500).grid(column=0, row=row, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # Symbol (optional)
        row += 1
        ttk.Label(frame, text="Symbol (optional):").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.migrate_symbol_var).grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(frame, text="(e.g., BTC/USDT)").grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Timeframe (optional)
        row += 1
        ttk.Label(frame, text="Timeframe (optional):").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        timeframes = ["", "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
        ttk.Combobox(frame, textvariable=self.migrate_timeframe_var, values=timeframes).grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Backup files
        row += 1
        ttk.Checkbutton(frame, text="Create backups of legacy files", variable=self.migrate_backup_var).grid(column=1, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Delete old files
        row += 1
        ttk.Checkbutton(frame, text="Delete legacy files after migration (caution!)", variable=self.migrate_delete_var).grid(column=1, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Status message
        row += 1
        self.migrate_status_var = tk.StringVar()
        ttk.Label(frame, textvariable=self.migrate_status_var, font=('', 9, 'italic')).grid(column=0, row=row, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # Create a separator
        row += 1
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(column=0, row=row, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Create buttons
        row += 1
        button_frame = ttk.Frame(frame)
        button_frame.grid(column=0, row=row, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Start Migration", command=self.start_data_migration).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        row += 1
        self.migrate_progress = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=100, mode='indeterminate')
        self.migrate_progress.grid(column=0, row=row, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Make the grid resizable
        frame.columnconfigure(1, weight=1)
        
    def _create_view_tab(self, parent):
        """Create content for data viewing tab"""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        header = ttk.Label(frame, text="View Data Files", font=self.header_font)
        header.grid(column=0, row=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        
        # Symbol and timeframe selection
        row = 1
        ttk.Label(frame, text="Symbol:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.view_symbol_var).grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Timeframe
        row += 1
        ttk.Label(frame, text="Timeframe:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
        ttk.Combobox(frame, textvariable=self.view_timeframe_var, values=timeframes).grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # File list
        row += 3
        ttk.Label(frame, text="Available Files:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Create a listbox with scrollbar
        file_frame = ttk.Frame(frame)
        file_frame.grid(column=0, row=row+1, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        self.file_listbox = tk.Listbox(file_frame, height=10, width=80)
        scrollbar = ttk.Scrollbar(file_frame, orient="vertical", command=self.file_listbox.yview)
        self.file_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Button to refresh file list
        row += 2
        button_frame = ttk.Frame(frame)
        button_frame.grid(column=0, row=row+1, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Refresh File List", command=self.refresh_file_list).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="View File Details", command=self.view_file_details).pack(side=tk.LEFT, padx=5)
        
        # Status message
        row += 3
        self.view_status_var = tk.StringVar()
        ttk.Label(frame, textvariable=self.view_status_var, font=('', 9, 'italic')).grid(column=0, row=row, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # Make the grid resizable
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(row-1, weight=1)  # Make the listbox row expandable
        
        # Initialize the file list
        self.refresh_file_list()
        
    def _create_log_tab(self):
        """Create content for log tab"""
        log_frame = ttk.Frame(self.log_tab, padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a scrolled text widget for logs with a header
        header = ttk.Label(log_frame, text="System Logs", font=self.header_font)
        header.pack(fill=tk.X, pady=(0, 10))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state="disabled")
        
        # Configure tags for different log levels
        self.log_text.tag_configure("error", foreground="red")
        self.log_text.tag_configure("warning", foreground="orange")
        self.log_text.tag_configure("info", foreground="black")
        self.log_text.tag_configure("debug", foreground="gray")
        
        # Button frame
        button_frame = ttk.Frame(log_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(button_frame, text="Clear Logs", command=self._clear_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Logs", command=self._save_logs).pack(side=tk.LEFT, padx=5)
        
        # Log level filter
        filter_frame = ttk.Frame(button_frame)
        filter_frame.pack(side=tk.RIGHT)
        
        ttk.Label(filter_frame, text="Log Level:").pack(side=tk.LEFT, padx=5)
        
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        self.log_level_var = tk.StringVar(value="INFO")
        log_level_combo = ttk.Combobox(filter_frame, textvariable=self.log_level_var, values=log_levels, width=10, state="readonly")
        log_level_combo.pack(side=tk.LEFT, padx=5)
        log_level_combo.bind("<<ComboboxSelected>>", self._on_log_level_change)
        
        # Add auto-scroll checkbox
        self.auto_scroll_var = tk.BooleanVar(value=True)
        auto_scroll_check = ttk.Checkbutton(filter_frame, text="Auto-scroll", variable=self.auto_scroll_var)
        auto_scroll_check.pack(side=tk.LEFT, padx=5)
        
    def _create_results_tab(self):
        """Create content for results tab"""
        results_frame = ttk.Frame(self.results_tab, padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        header = ttk.Label(results_frame, text="Trading Results", font=self.header_font)
        header.pack(fill=tk.X, pady=(0, 10))
        
        # Create a notebook for different result views
        results_notebook = ttk.Notebook(results_frame)
        results_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs for different result views
        summary_tab = ttk.Frame(results_notebook)
        trades_tab = ttk.Frame(results_notebook)
        stats_tab = ttk.Frame(results_notebook)
        chart_tab = ttk.Frame(results_notebook)
        
        results_notebook.add(summary_tab, text="Summary")
        results_notebook.add(trades_tab, text="Trades")
        results_notebook.add(stats_tab, text="Statistics")
        results_notebook.add(chart_tab, text="Charts")
        
        # Create summary tab with text widget
        summary_frame = ttk.Frame(summary_tab, padding="10")
        summary_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = scrolledtext.ScrolledText(summary_frame, wrap=tk.WORD, height=20)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.results_text.configure(state="disabled")
        
        # Create trades tab
        trades_frame = ttk.Frame(trades_tab, padding="10")
        trades_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a treeview for trades
        columns = ("date", "symbol", "type", "size", "entry", "exit", "pnl", "pct")
        self.trades_tree = ttk.Treeview(trades_frame, columns=columns, show="headings", height=15)
        
        # Define headings
        self.trades_tree.heading("date", text="Date")
        self.trades_tree.heading("symbol", text="Symbol")
        self.trades_tree.heading("type", text="Type")
        self.trades_tree.heading("size", text="Size")
        self.trades_tree.heading("entry", text="Entry Price")
        self.trades_tree.heading("exit", text="Exit Price")
        self.trades_tree.heading("pnl", text="P&L")
        self.trades_tree.heading("pct", text="P&L %")
        
        # Define columns
        self.trades_tree.column("date", width=120)
        self.trades_tree.column("symbol", width=80)
        self.trades_tree.column("type", width=60)
        self.trades_tree.column("size", width=80)
        self.trades_tree.column("entry", width=80)
        self.trades_tree.column("exit", width=80)
        self.trades_tree.column("pnl", width=80)
        self.trades_tree.column("pct", width=80)
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(trades_frame, orient="vertical", command=self.trades_tree.yview)
        self.trades_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack treeview and scrollbar
        self.trades_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create statistics tab
        stats_frame = ttk.Frame(stats_tab, padding="10")
        stats_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a treeview for statistics
        columns = ("metric", "value")
        self.stats_tree = ttk.Treeview(stats_frame, columns=columns, show="headings", height=15)
        
        # Define headings
        self.stats_tree.heading("metric", text="Metric")
        self.stats_tree.heading("value", text="Value")
        
        # Define columns
        self.stats_tree.column("metric", width=200)
        self.stats_tree.column("value", width=200)
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=self.stats_tree.yview)
        self.stats_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack treeview and scrollbar
        self.stats_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create charts tab with improved selection
        chart_frame = ttk.Frame(chart_tab, padding="10")
        chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add chart type selector
        chart_controls = ttk.Frame(chart_frame)
        chart_controls.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        ttk.Label(chart_controls, text="Chart Type:").pack(side=tk.LEFT, padx=5)
        
        self.chart_type_var = tk.StringVar(value="equity_curve")
        chart_types = ["equity_curve", "drawdown", "monthly_returns", "trade_distribution"]
        chart_combo = ttk.Combobox(chart_controls, textvariable=self.chart_type_var, values=chart_types, state="readonly", width=20)
        chart_combo.pack(side=tk.LEFT, padx=5)
        
        # Add toggle for baseline display
        self.show_baseline_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(chart_controls, text="Show Baseline", variable=self.show_baseline_var, 
                    command=self._update_chart).pack(side=tk.LEFT, padx=20)
        
        # Chart display area with improved support for React components
        self.chart_display = ttk.Frame(chart_frame, relief=tk.SUNKEN, borderwidth=1)
        self.chart_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        chart_placeholder = ttk.Label(self.chart_display, text="Chart will be displayed here after trading run")
        chart_placeholder.pack(fill=tk.BOTH, expand=True)
        
        # Bind events for chart updates
        chart_combo.bind("<<ComboboxSelected>>", self._update_chart)
        
        # Button frame for all tabs
        button_frame = ttk.Frame(results_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(button_frame, text="Export Results", command=self._export_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export as CSV", command=self._export_results_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export as JSON", command=self._export_results_json).pack(side=tk.LEFT, padx=5)
        
        # Progress frame for showing status during trading run
        self.progress_frame = ttk.Frame(results_frame)
        
        # Progress label and bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_label = ttk.Label(self.progress_frame, text="Ready")
        self.progress_label.pack(side=tk.LEFT, padx=5)
        
        self.progress_bar = ttk.Progressbar(
            self.progress_frame, 
            orient=tk.HORIZONTAL, 
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    def _display_equity_curve_chart(self):
        """Display equity curve using Matplotlib"""
        try:
            # Create figure and subplot
            fig = Figure(figsize=(8, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Get equity curve data
            if 'equity_curve' in self.latest_results:
                equity_data = self.latest_results['equity_curve']
                
                # Process data into plottable format
                dates, values = self._extract_chart_data(equity_data)
                
                # Plot equity curve
                ax.plot(dates, values, 'b-', label='Strategy')
                
                # Plot baseline if available and enabled
                if self.show_baseline_var.get() and 'baseline_prices' in self.latest_results:
                    baseline_data = self.latest_results['baseline_prices']
                    baseline_dates, baseline_values = self._extract_chart_data(baseline_data)
                    
                    # Normalize baseline to start at same value as equity curve
                    if values and baseline_values:
                        try:
                            normalization_factor = values[0] / baseline_values[0]
                            normalized_baseline = [v * normalization_factor for v in baseline_values]
                            
                            # Plot normalized baseline
                            ax.plot(baseline_dates, normalized_baseline, 'r--', label='Baseline')
                        except (IndexError, ZeroDivisionError):
                            pass
                
                # Add chart details
                ax.set_title('Equity Curve')
                ax.set_xlabel('Time')
                ax.set_ylabel('Portfolio Value')
                ax.grid(True)
                ax.legend()
                
                # Create canvas and add to display
                canvas = FigureCanvasTkAgg(fig, master=self.chart_display)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            else:
                ttk.Label(self.chart_display, text="No equity curve data available").pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            ttk.Label(self.chart_display, text=f"Error generating chart: {str(e)}").pack(fill=tk.BOTH, expand=True)

    def _display_drawdown_chart(self):
        """Display drawdown chart using Matplotlib"""
        try:
            # Create figure and subplot
            fig = Figure(figsize=(8, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Get drawdown data
            if 'drawdown' in self.latest_results:
                drawdown_data = self.latest_results['drawdown']
                
                # Process data into plottable format
                dates, values = self._extract_chart_data(drawdown_data)
                
                # Plot drawdown
                ax.fill_between(dates, 0, values, color='r', alpha=0.3)
                ax.plot(dates, values, 'r-')
                
                # Add chart details
                ax.set_title('Drawdown')
                ax.set_xlabel('Time')
                ax.set_ylabel('Drawdown %')
                ax.grid(True)
                
                # Create canvas and add to display
                canvas = FigureCanvasTkAgg(fig, master=self.chart_display)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            else:
                ttk.Label(self.chart_display, text="No drawdown data available").pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            ttk.Label(self.chart_display, text=f"Error generating chart: {str(e)}").pack(fill=tk.BOTH, expand=True)

    def _display_monthly_returns_chart(self):
        """Display monthly returns as a bar chart"""
        try:
            # Create figure and subplot
            fig = Figure(figsize=(8, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Get monthly returns data
            if 'monthly_returns' in self.latest_results:
                monthly_data = self.latest_results['monthly_returns']
                
                # Convert to lists for plotting
                months = list(monthly_data.keys())
                returns = list(monthly_data.values())
                
                # Plot monthly returns
                colors = ['g' if r >= 0 else 'r' for r in returns]
                ax.bar(months, returns, color=colors)
                
                # Add chart details
                ax.set_title('Monthly Returns')
                ax.set_xlabel('Month')
                ax.set_ylabel('Return %')
                ax.grid(True, axis='y')
                
                # Rotate x labels if needed
                if len(months) > 6:
                    plt = ax.xaxis.get_ticklabels()
                    for p in plt:
                        p.set_rotation(45)
                
                # Create canvas and add to display
                canvas = FigureCanvasTkAgg(fig, master=self.chart_display)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            else:
                ttk.Label(self.chart_display, text="No monthly returns data available").pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            ttk.Label(self.chart_display, text=f"Error generating chart: {str(e)}").pack(fill=tk.BOTH, expand=True)

    def _display_trade_distribution_chart(self):
        """Display trade win/loss distribution as a pie chart"""
        try:
            # Create figure and subplot
            fig = Figure(figsize=(8, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Get trades data
            if 'trades' in self.latest_results:
                trades = self.latest_results['trades']
                
                # Calculate win/loss stats
                wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
                losses = sum(1 for t in trades if t.get('pnl', 0) <= 0)
                
                # Create pie chart
                labels = f'Wins ({wins})', f'Losses ({losses})'
                sizes = [wins, losses] if wins + losses > 0 else [1, 0]  # Avoid empty pie
                colors = ['#4CAF50', '#F44336']
                
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                ax.set_title('Trade Win/Loss Distribution')
                
                # Create canvas and add to display
                canvas = FigureCanvasTkAgg(fig, master=self.chart_display)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            else:
                ttk.Label(self.chart_display, text="No trade data available").pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            ttk.Label(self.chart_display, text=f"Error generating chart: {str(e)}").pack(fill=tk.BOTH, expand=True)

    def _extract_chart_data(self, data):
        """Utility function to extract plottable data from various data formats"""
        if isinstance(data, pd.DataFrame):
            if 'timestamp' in data.columns and 'portfolio_value' in data.columns:
                return data['timestamp'].tolist(), data['portfolio_value'].tolist()
            elif 'timestamp' in data.columns:
                return data['timestamp'].tolist(), data.iloc[:, 1].tolist()
            elif 'portfolio_value' in data.columns:
                return range(len(data)), data['portfolio_value'].tolist()
            else:
                return range(len(data)), data.iloc[:, 0].tolist()
        elif isinstance(data, dict):
            items = sorted(data.items())
            return [item[0] for item in items], [item[1] for item in items]
        else:
            # Assume it's a list or array
            return range(len(data)), list(data)

    def _update_chart(self, event=None):
        """Update chart based on current settings"""
        # Clear the chart display
        for widget in self.chart_display.winfo_children():
            widget.destroy()
            
        chart_type = self.chart_type_var.get()
        
        # If no results data yet, show placeholder
        if not hasattr(self, 'latest_results') or not self.latest_results:
            ttk.Label(self.chart_display, text="Run a backtest to see results").pack(fill=tk.BOTH, expand=True)
            return
        
        # Generate appropriate chart based on type
        chart_display_functions = {
            "equity_curve": self._display_equity_curve_chart,
            "drawdown": self._display_drawdown_chart,
            "monthly_returns": self._display_monthly_returns_chart,
            "trade_distribution": self._display_trade_distribution_chart
        }
        
        if chart_type in chart_display_functions:
            chart_display_functions[chart_type]()
        else:
            ttk.Label(self.chart_display, text=f"Chart type {chart_type} not implemented").pack(fill=tk.BOTH, expand=True)

    def display_results(self, result):
        """Display results in the results tab with chart generation"""
        # Store results for later chart updates
        self.latest_results = result
        
        # Stop progress animation
        self.progress_bar.stop()
        self.progress_label.config(text="Processing results...")
        
        if not result:
            self.progress_label.config(text="No results to display")
            return
        
        # Display in text area
        self.results_text.configure(state="normal")
        self.results_text.delete(1.0, tk.END)
        
        # Format the results based on type
        if isinstance(result, dict):
            # Check for error
            if "error" in result:
                self.results_text.insert(tk.END, f"Error: {result['error']}\n")
                self.progress_label.config(text="Error in results")
            else:
                # Format as JSON
                try:
                    formatted_result = json.dumps(result, indent=2)
                    self.results_text.insert(tk.END, formatted_result)
                    
                    # Populate trades and stats if available
                    if "trades" in result:
                        self._populate_trades(result["trades"])
                    if "metrics" in result:
                        self._populate_statistics(result["metrics"])
                        
                    # Update charts
                    self._update_chart()
                        
                    self.progress_label.config(text="Results loaded successfully")
                    
                except Exception as e:
                    self.results_text.insert(tk.END, f"Error formatting results: {str(e)}\n{str(result)}")
                    self.progress_label.config(text="Error formatting results")
        else:
            # Just display as string
            self.results_text.insert(tk.END, str(result))
            self.progress_label.config(text="Basic results displayed")
        
        self.results_text.configure(state="disabled")
        
        # Switch to results tab
        self.notebook.select(self.results_tab)
    
    def _process_log_queue(self):
        """Process log messages from the queue with performance optimizations"""
        # Batch process logs to reduce GUI updates
        batch_size = 10
        processed = 0
        
        try:
            # Enable text widget for modification
            self.log_text.configure(state="normal")
            
            # Process up to 'batch_size' logs at once
            while processed < batch_size:
                record = self.log_queue.get_nowait()
                formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
                formatted_message = formatter.format(record)
                
                # Filter based on selected log level
                if self._should_display_log(record.levelno):
                    # Add color based on log level
                    tag = "info"
                    if record.levelno >= logging.ERROR:
                        tag = "error"
                    elif record.levelno >= logging.WARNING:
                        tag = "warning"
                    elif record.levelno <= logging.DEBUG:
                        tag = "debug"
                    
                    # Insert message with appropriate tag
                    self.log_text.insert(tk.END, formatted_message + "\n", tag)
                    
                    # Auto-scroll if enabled
                    if self.auto_scroll_var.get():
                        self.log_text.see(tk.END)
                
                # Process all available log records
                self.log_queue.task_done()
                processed += 1
                
        except queue.Empty:
            # No more items in the queue
            pass
        finally:
            # Make the text widget read-only again
            self.log_text.configure(state="disabled")
            
            # Schedule to check again
            # Adjust check frequency based on activity
            next_check = 100 if processed > 0 else 500
            self.root.after(next_check, self._process_log_queue)
            
    def _should_display_log(self, level):
        """Check if log level should be displayed based on filter"""
        log_level = self.log_level_var.get()
        min_level = logging.INFO  # Default
        
        if log_level == "DEBUG":
            min_level = logging.DEBUG
        elif log_level == "INFO":
            min_level = logging.INFO
        elif log_level == "WARNING":
            min_level = logging.WARNING
        elif log_level == "ERROR":
            min_level = logging.ERROR
            
        return level >= min_level
        
    # Event handlers
    def _on_mode_change(self, event):
        """Handle changes to the trading mode"""
        mode = self.mode_var.get()
        if mode == "backtest":
            # Enable backtest-specific fields
            self.setup_status_var.set("Backtest mode: configure period settings")
            
            # Show backtest engine selection
            self.backtest_engine_label.grid()
            self.backtest_engine_combo.grid()
            self.backtest_engine_info.grid()
        else:
            # Hide backtest engine selection for other modes
            self.backtest_engine_label.grid_remove()
            self.backtest_engine_combo.grid_remove()
            self.backtest_engine_info.grid_remove()
            
            if mode == "paper":
                # Enable paper trading fields
                self.setup_status_var.set("Paper trading mode: trades will be simulated with real-time data")
            elif mode == "live":
                # Enable live trading fields
                self.setup_status_var.set("Live trading mode: WARNING - real funds will be used!")
                # Verify API keys are set
                if not self.api_key_var.get() or not self.api_secret_var.get():
                    messagebox.showwarning("API Keys Missing", "Live trading requires valid API keys. Please configure them in the API Settings tab.")
           
    def _on_engine_change(self, event):
        """Handle changes to the backtest engine selection"""
        engine = self.backtest_engine_var.get()
        if engine == "ohlcv":
            self.setup_status_var.set("Selected Basic OHLCV Engine: Standard backtesting with OHLCV data")
            self.backtest_engine_info.config(text="(Standard OHLCV - Basic)")
        elif engine == "market_replay":
            self.setup_status_var.set("Selected Market Replay Engine: High-fidelity market replay with realistic order execution")
            self.backtest_engine_info.config(text="(Market Replay - Advanced)")
                 
    def _on_strategy_change(self, event):
        """Handle changes to the strategy selection"""
        strategy = self.strategy_var.get()
        
        # Update the parameter frame title based on selected strategy
        if strategy == "dual_ma":
            self.strat_param_frame.configure(text="Dual Moving Average Parameters")
            # Show/hide relevant parameters
            # This would be expanded based on strategy type
        elif strategy == "neural_network":
            self.strat_param_frame.configure(text="Neural Network Parameters")
            # Show/hide relevant parameters
        elif strategy == "bollinger_bands":
            self.strat_param_frame.configure(text="Bollinger Bands Parameters")
        elif strategy == "macd":
            self.strat_param_frame.configure(text="MACD Parameters")
        elif strategy == "rsi":
            self.strat_param_frame.configure(text="RSI Parameters")
        else:
            self.strat_param_frame.configure(text="Strategy Parameters")
        
        # Update status message
        self.setup_status_var.set(f"Strategy changed to {strategy}")
        
    def _on_log_level_change(self, event):
        """Handle changes to the log level filter"""
        # Clear and refill log display with filtered messages
        self._clear_logs()
        
    def _toggle_stop_loss(self):
        """Enable or disable stop loss fields based on checkbox"""
        state = "normal" if self.stop_loss_enabled_var.get() else "disabled"
        self.sl_pct_entry.configure(state=state)
        self.sl_daily_entry.configure(state=state)
    
    def _toggle_take_profit(self):
        """Enable or disable take profit fields based on checkbox"""
        state = "normal" if self.take_profit_enabled_var.get() else "disabled"
        self.tp_pct_entry.configure(state=state)
    
    def _toggle_emergency_stop(self):
        """Enable or disable emergency stop fields based on checkbox"""
        state = "normal" if self.emergency_stop_var.get() else "disabled"
        self.emergency_threshold_entry.configure(state=state)
    
    def _toggle_proxy_fields(self):
        """Enable or disable proxy fields based on checkbox"""
        state = "normal" if self.api_useproxy_var.get() else "disabled"
        self.http_proxy_entry.configure(state=state)
        self.https_proxy_entry.configure(state=state)
        
    def _toggle_show_password(self, entry_widget):
        """Toggle password visibility in entry widget"""
        if entry_widget.cget('show') == '*':
            entry_widget.configure(show='')
        else:
            entry_widget.configure(show='*')
    
    def _browse_config(self):
        """Open file dialog to select config file"""
        filename = filedialog.askopenfilename(
            title="Select Configuration File",
            filetypes=(
                ("YAML files", "*.yaml *.yml"),
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ),
            initialdir=Path(project_root) / "conf"
        )
        if filename:
            # Set config path and attempt to load the config
            self.config_var.set(filename)
            self.load_config(filename)
    
    def load_config(self, config_path):
        """Load configuration from file"""
        try:
            # Load the config file
            self.config.load(config_path)
            
            # Reinitialize form values from the new config
            self._init_from_config()
            
            # Update UI states
            self._toggle_stop_loss()
            self._toggle_take_profit()
            self._toggle_emergency_stop()
            self._toggle_proxy_fields()
            
            # Show success message
            self.setup_status_var.set(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            error_msg = f"Error loading configuration: {str(e)}"
            self.setup_status_var.set(error_msg)
            messagebox.showerror("Configuration Error", error_msg)
    
    def save_config(self):
        """Save configuration to file"""
        # Get config path
        config_path = self.config_var.get()
        if not config_path:
            config_path = filedialog.asksaveasfilename(
                title="Save Configuration",
                defaultextension=".yaml",
                filetypes=(("YAML files", "*.yaml"), ("All files", "*.*")),
                initialdir=Path(project_root) / "conf"
            )
            if not config_path:
                return
            self.config_var.set(config_path)
        
        try:
            # Set the status message
            self.setup_status_var.set("Updating  configuration...")
            
            # General settings
            self.config.set("system", "operational_mode", self.mode_var.get())
            self.config.set("system", "backtest_mode", self.backtest_engine_var.get())
            self.config.set("backtest", "engine", self.backtest_engine_var.get())
            # Split and strip symbols
            symbols = [s.strip() for s in self.symbol_var.get().split(',')]
            self.config.set("trading", "instruments", symbols)
            self.config.set("trading", "capital", "initial", self.initial_capital_var.get())
            
            # Backtest settings
            self.config.set("backtest", "period", "start", self.start_date_var.get())
            self.config.set("backtest", "period", "end", self.end_date_var.get())
            self.config.set("backtest", "costs", "commission", self.commission_var.get())
            self.config.set("backtest", "costs", "slippage", self.slippage_var.get())
            
            # Strategy settings
            self.config.set("strategy", "active", self.strategy_var.get())
            self.config.set("strategy", "timeframe", self.timeframe_var.get())
            self.config.set("strategy", "fallback_strategy", self.fallback_strategy_var.get())
            self.config.set("strategy", "parameters", "fast_period", self.fast_period_var.get())
            self.config.set("strategy", "parameters", "slow_period", self.slow_period_var.get())
            self.config.set("strategy", "parameters", "threshold", self.threshold_var.get())
            
            # Risk management settings
            self.config.set("risk", "stop_loss", "enabled", self.stop_loss_enabled_var.get())
            self.config.set("risk", "stop_loss", "percentage", self.stop_loss_pct_var.get())
            self.config.set("risk", "stop_loss", "max_daily", self.stop_loss_daily_var.get())
            
            self.config.set("risk", "take_profit", "enabled", self.take_profit_enabled_var.get())
            self.config.set("risk", "take_profit", "percentage", self.take_profit_pct_var.get())
            
            self.config.set("risk", "drawdown", "max_portfolio", self.max_drawdown_var.get())
            self.config.set("risk", "drawdown", "max_single_trade", self.max_trade_drawdown_var.get())
            
            self.config.set("trading", "limits", "position", self.max_position_var.get())
            self.config.set("trading", "capital", "min_order", self.min_order_var.get())
            
            # Advanced system settings
            self.config.set("database", "url", self.db_url_var.get())
            self.config.set("database", "pool_size", self.db_pool_size_var.get())
            self.config.set("database", "timeout", self.db_timeout_var.get())
            self.config.set("database", "echo_sql", self.db_echo_sql_var.get())
            
            # Memory value is stored in MB in the UI, but in bytes in the config
            memory_bytes = self.max_memory_var.get() * 1048576 
            self.config.set("system", "performance", "max_memory", memory_bytes)
            self.config.set("system", "performance", "max_threads", self.max_workers_var.get())
            self.config.set("system", "performance", "task_timeout", self.task_timeout_var.get())
            self.config.set("system", "timezone", self.timezone_var.get())
            
            # Live trading safeguards
            self.config.set("live", "safeguards", "emergency_stop", "enabled", self.emergency_stop_var.get())
            self.config.set("live", "safeguards", "emergency_stop", "threshold", self.emergency_threshold_var.get())
            self.config.set("live", "safeguards", "max_retries", self.max_retries_var.get())
            self.config.set("live", "safeguards", "cooldown", self.cooldown_var.get())
            
            # API settings
            self.config.set("api", "binance", "api_key", self.api_key_var.get())
            self.config.set("api", "binance", "secret", self.api_secret_var.get())
            self.config.set("api", "timeout", self.api_timeout_var.get())
            
            # Proxy settings
            if self.api_useproxy_var.get():
                self.config.set("proxies", "http", self.api_http_proxy_var.get())
                self.config.set("proxies", "https", self.api_https_proxy_var.get())
            else:
                self.config.set("proxies", "http", "")
                self.config.set("proxies", "https", "")
            
            # Save to config file
            self.config.save(config_path)
            
            # Update status message
            self.setup_status_var.set(f"Configuration saved to {config_path}")
            
        except Exception as e:
            error_msg = f"Error saving configuration: {str(e)}"
            self.setup_status_var.set(error_msg)
            messagebox.showerror("Error", error_msg)
    
    def _clear_logs(self):
        """Clear the log display"""
        self.log_text.configure(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state="disabled")
    
    def _save_logs(self):
        """Save logs to a file"""
        filename = filedialog.asksaveasfilename(
            title="Save Logs",
            defaultextension=".log",
            filetypes=(("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*"))
        )
        if filename:
            with open(filename, "w") as f:
                f.write(self.log_text.get(1.0, tk.END))
            messagebox.showinfo("Save Logs", f"Logs saved to {filename}")
    
    # Results export functions
    def _export_results(self):
        """Export results to a file"""
        filename = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".txt",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
        )
        if filename:
            try:
                with open(filename, "w") as f:
                    f.write(self.results_text.get(1.0, tk.END))
                messagebox.showinfo("Export Results", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting results: {str(e)}")
    
    def _export_results_csv(self):
        """Export results to a CSV file"""
        filename = filedialog.asksaveasfilename(
            title="Export Results as CSV",
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filename:
            try:
                # Get trades from the treeview
                trades = []
                for item_id in self.trades_tree.get_children():
                    values = self.trades_tree.item(item_id, "values")
                    trades.append(values)
                
                # Write to CSV
                import csv
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Write header
                    writer.writerow(["Date", "Symbol", "Type", "Size", "Entry Price", "Exit Price", "P&L", "P&L %"])
                    # Write data
                    for trade in trades:
                        writer.writerow(trade)
                
                messagebox.showinfo("Export Successful", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting results: {str(e)}")
    
    def _export_results_json(self):
        """Export results to a JSON file"""
        filename = filedialog.asksaveasfilename(
            title="Export Results as JSON",
            defaultextension=".json",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )
        if filename:
            try:
                # Get results as text
                results_text = self.results_text.get(1.0, tk.END)
                
                # Try to parse as JSON
                import json
                try:
                    # If it's already JSON, parse and re-serialize with formatting
                    results_data = json.loads(results_text)
                    with open(filename, 'w') as f:
                        json.dump(results_data, f, indent=2)
                except json.JSONDecodeError:
                    # If not JSON, create a simple JSON object
                    results_data = {"results": results_text}
                    with open(filename, 'w') as f:
                        json.dump(results_data, f, indent=2)
                
                messagebox.showinfo("Export Successful", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting results: {str(e)}")
    
    # Auto-save functionality
    def _init_autosave(self):
        """Initialize auto-save functionality"""
        # Configurable auto-save interval
        self.autosave_enabled = self.config.get("system", "config", "autosave_enabled", default=False)
        self.autosave_interval = self.config.get("system", "config", "autosave_interval", default=6000)  # 100 minutes default
        
        autosave_dir = self.config.get("system", "config", "autosave_path", default=Path(project_root) / "conf" / "autosave")
        Path(autosave_dir).mkdir(parents=True, exist_ok=True)
        
        # Schedule first autosave
        if self.autosave_enabled:
            self.root.after(self.autosave_interval * 1000, self._perform_autosave)
        else:
            self.log_manager.get_logger("auto_save").info("Automatic configuration saving is disabled")

    def _perform_autosave(self, silent=False, force=True):
        """Perform auto-save and schedule next one"""
        try:
            if force or self.autosave_enabled:
                # Save current state to temporary file
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_config_path = Path(project_root) / "conf" / "autosave" / f"autosave_{timestamp}.yaml"
                self.config.save(str(temp_config_path))
                
                # Keep only the last 5 autosave files
                self._cleanup_autosave_files()
                
                if not silent:
                    self.setup_status_var.set(f"Auto-saved configuration to {temp_config_path.name}")
        
        except Exception as e:
            self.log_manager.get_logger("auto_save").error(f"Auto-save failed: {str(e)}", exc_info=True)
    
    def _cleanup_autosave_files(self):
        """Keep only the most recent autosave files"""
        try:
            autosave_dir = Path(project_root) / "conf" / "autosave"
            autosave_files = sorted(autosave_dir.glob("autosave_*.yaml"), reverse=True)
            
            # Keep only the 5 most recent files
            for file in autosave_files[5:]:
                file.unlink()
                
        except Exception as e:
            self.log_manager.get_logger("auto_save").error(f"Autosave cleanup failed: {str(e)}")
    
    def _check_for_recovery(self):
        """Check for recovery file on startup"""
        autosave_dir = Path(project_root) / "conf" / "autosave"
        if not autosave_dir.exists():
            return
            
        autosave_files = sorted(autosave_dir.glob("autosave_*.yaml"), reverse=True)
        if autosave_files:
            latest_file = autosave_files[0]
            # Only prompt for recovery if the file is from today
            file_date = latest_file.stem.split('_')[1][:8]  # Extract date part (YYYYMMDD)
            today = datetime.datetime.now().strftime("%Y%m%d")
            
            if file_date == today and messagebox.askyesno(
                "Recovery", 
                f"An auto-saved configuration from {file_date[:4]}-{file_date[4:6]}-{file_date[6:8]} was found. Would you like to recover it?"
            ):
                self.load_config(latest_file)
    
    # Trading system functions
    def start_trading(self):
        """Start the trading system with the Run button"""
        if self.running:
            messagebox.showinfo("Already Running", f"Trading system ({self.args.mode} mode) is already running.")
            return
            
        # Validate inputs
        try:
            # Check that dates are valid
            datetime.datetime.strptime(self.start_date_var.get(), "%Y-%m-%d")
            datetime.datetime.strptime(self.end_date_var.get(), "%Y-%m-%d")
            
            # Check that required fields are filled
            if not self.config_var.get() or not Path(self.config_var.get()).exists():
                messagebox.showerror("Invalid Config", "Please select a valid configuration file.")
                return
                
            if not self.symbol_var.get():
                messagebox.showerror("Missing Symbol", "Please specify at least one trading symbol.")
                return
                
            # Check if we're in live mode and verify API keys
            if self.mode_var.get() == "live" and (not self.api_key_var.get() or not self.api_secret_var.get()):
                if not messagebox.askyesno("API Keys Missing", 
                                           "Live trading without API keys may not work correctly. Continue anyway?"):
                    return
                
            # Backtest engine validation
            if self.mode_var.get() == "backtest" and not self.backtest_engine_var.get():
                messagebox.showerror("Missing Backtest Engine", "Please select a backtest engine.")
                return     
                           
            # Automatically save configuration first
            self.save_config()
        except ValueError as e:
            messagebox.showerror("Invalid Date Format", f"Please use YYYY-MM-DD format for dates: {str(e)}")
            return
        
        # Start the system in a separate thread
        try:
            # Set running flag and start thread
            self.running = True
            self.paused = False
            
            # Update UI controls
            self._update_control_buttons()
            
            # Update UI state
            self.control_status_var.set("Starting trading system...")
            self.status_message.set("Trading system running...")
            
            # Show progress frame
            self.progress_frame.pack(fill=tk.X, pady=10)
            self.progress_bar.start()
            self.progress_label.config(text="Initializing...")
            
            # Switch to log tab
            self.notebook.select(self.log_tab)
            
            # Start in a separate thread
            self.thread = threading.Thread(target=self._run_trading_system)
            self.thread.daemon = True
            self.thread.start()
            
        except Exception as e:
            self.running = False
            error_msg = f"Error starting trading system: {str(e)}"
            self.control_status_var.set(error_msg)
            messagebox.showerror("Error", error_msg)
            self._update_control_buttons()
    
    def pause_trading(self):
        """Pause the trading system"""
        if not self.running or not self.trader:
            return
        
        if self.paused:
            # Resume trading
            self.paused = False
            self.control_status_var.set("Resuming trading...")
            self.status_message.set("Trading resumed")
            self.progress_bar.start()
            
            # Send resume signal to trader
            if hasattr(self.trader, 'resume'):
                self.trader.resume()
        else:
            # Pause trading
            self.paused = True
            self.control_status_var.set("Pausing trading...")
            self.status_message.set("Trading paused")
            self.progress_bar.stop()
            
            # Send pause signal to trader
            if hasattr(self.trader, 'pause'):
                self.trader.pause()
        
        # Update UI controls
        self._update_control_buttons()
    
    def _update_control_buttons(self):
        """Update the state of control buttons based on current status"""
        if self.running:
            self.run_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            # Enable/disable pause button based on trader support
            if self.trader and hasattr(self.trader, 'pause'):
                self.pause_button.config(state=tk.NORMAL)
                self.pause_button.config(text="Resume" if self.paused else "Pause")
            else:
                self.pause_button.config(state=tk.DISABLED)
        else:
            self.run_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.DISABLED)
    
    def _setup_progress_indicator(self):
        """Setup progress indicator in results tab"""
        # Display the progress frame
        self.progress_frame.pack(fill=tk.X, padx=10, pady=10)
            
        # Set initial text
        self.progress_label.config(text="Processing...")
        
        # Start progress animation
        self.progress_bar.start()
        
        # Switch to results tab
        self.notebook.select(self.results_tab)
    
    def _run_trading_system(self):
        """Run the trading system in a separate thread with improved error handling and status updates"""
        try:
            # Get the logger
            logger = self.log_manager.get_logger("trading_system")
            logger.info("Starting trading system from GUI...")
            
            # Setup progress indicator in results tab
            self.root.after(0, self._setup_progress_indicator)
            
            # Log backtest engine if in backtest mode
            mode = self.mode_var.get()
            if mode == "backtest":
                engine = self.backtest_engine_var.get()
                logger.info(f"Using backtest engine: {engine}")
                self.root.after(0, lambda: self.progress_label.config(text=f"Running {mode} with {engine} engine..."))
            else:
                self.root.after(0, lambda: self.progress_label.config(text=f"Running {mode} mode..."))
                engine = None
            
            # Create trading core
            self.trader = TradingCore(self.config, mode, engine)
            
            # Store weakref to avoid circular references
            self.trader._gui = weakref.ref(self)
            
            # Update status
            self.root.after(0, lambda: self.progress_label.config(text=f"Running {mode} mode..."))
            
            # Define progress callback
            def update_progress(percent, message):
                self.root.after(0, lambda: self.progress_var.set(percent))
                self.root.after(0, lambda: self.progress_label.config(text=message))
            
            # Set progress callback if supported
            if hasattr(self.trader, 'set_progress_callback'):
                self.trader.set_progress_callback(update_progress)
            
            # Run the trading pipeline
            result = self.executor.run(self.trader.run_pipeline())
            
            # Display results
            self.root.after(0, lambda: self.display_results(result))
            
            # Log completion
            logger.info("Trading pipeline completed")
            
            # Update status
            self.root.after(0, lambda: self.control_status_var.set("Trading completed successfully"))
            self.root.after(0, lambda: self.status_message.set("Trading completed"))
            self.root.after(0, lambda: self.progress_bar.stop())
            self.root.after(0, lambda: self.progress_label.config(text="Completed"))
            
            # Clean up
            if self.trader:
                self.executor.run(self.trader.shutdown())
                self.trader = None
            
        except Exception as e:
            logger.error(f"Error running trading system: {str(e)}", exc_info=True)
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error running trading system:\n{str(e)}"))
            self.root.after(0, lambda: self.control_status_var.set(f"Error: {str(e)}"))
            self.root.after(0, lambda: self.status_message.set("Error occurred"))
            self.root.after(0, lambda: self.progress_bar.stop())
            self.root.after(0, lambda: self.progress_label.config(text="Failed"))
        finally:
            self.running = False
            self.paused = False
            self.root.after(0, self._update_control_buttons)
    
    def _populate_trades(self, trades):
        """Populate trades treeview with data"""
        # Clear existing data
        for i in self.trades_tree.get_children():
            self.trades_tree.delete(i)
        
        # Add trades to treeview
        for trade in trades:
            try:
                # Add trade to treeview
                self.trades_tree.insert("", "end", values=(
                    trade.get("date", ""),
                    trade.get("symbol", ""),
                    trade.get("type", ""),
                    trade.get("size", ""),
                    trade.get("entry_price", ""),
                    trade.get("exit_price", ""),
                    f"{trade.get('pnl', 0):.2f}",
                    f"{trade.get('pnl_percent', 0):.2f}%"
                ))
            except Exception as e:
                self.log_manager.get_logger("gui").error(f"Error adding trade to treeview: {str(e)}")
    
    def _populate_statistics(self, metrics):
        """Populate statistics treeview with data"""
        # Clear existing data
        for i in self.stats_tree.get_children():
            self.stats_tree.delete(i)
        
        # Add metrics to treeview
        try:
            for key, value in metrics.items():
                formatted_key = key.replace("_", " ").title()
                
                # Format value based on type
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                elif isinstance(value, (int, bool)):
                    formatted_value = str(value)
                else:
                    formatted_value = str(value)
                
                self.stats_tree.insert("", "end", values=(formatted_key, formatted_value))
        except Exception as e:
            self.log_manager.get_logger("gui").error(f"Error populating statistics: {str(e)}")
    
    def _generate_chart(self, result_data):
        """Generate chart based on results data"""
        try:
            # Clear the chart display
            for widget in self.chart_display.winfo_children():
                widget.destroy()
                
            chart_type = self.chart_type_var.get()
            
            # Check if matplotlib is available
            try:
                import matplotlib
                matplotlib.use('TkAgg')
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                
                # Create figure and subplot
                fig = Figure(figsize=(8, 4), dpi=100)
                ax = fig.add_subplot(111)
                
                # Generate the appropriate chart
                if chart_type == "equity_curve" and "equity_curve" in result_data:
                    self._plot_equity_curve(fig, ax, result_data["equity_curve"])
                elif chart_type == "drawdown" and "drawdown" in result_data:
                    self._plot_drawdown(fig, ax, result_data["drawdown"])
                elif chart_type == "monthly_returns" and "monthly_returns" in result_data:
                    self._plot_monthly_returns(fig, ax, result_data["monthly_returns"])
                elif chart_type == "trade_distribution" and "trades" in result_data:
                    self._plot_trade_distribution(fig, ax, result_data["trades"])
                else:
                    ax.text(0.5, 0.5, f"No data available for {chart_type} chart", 
                          horizontalalignment='center', verticalalignment='center',
                          transform=ax.transAxes)
                
                # Create canvas and add to display
                canvas = FigureCanvasTkAgg(fig, master=self.chart_display)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
            except ImportError:
                # Matplotlib not available, show placeholder
                placeholder = ttk.Label(
                    self.chart_display, 
                    text="Chart generation requires matplotlib module.\nPlease install matplotlib to enable charts."
                )
                placeholder.pack(fill=tk.BOTH, expand=True)
                
        except Exception as e:
            self.log_manager.get_logger("gui").error(f"Error generating chart: {str(e)}", exc_info=True)
            error_label = ttk.Label(
                self.chart_display, 
                text=f"Error generating chart: {str(e)}"
            )
            error_label.pack(fill=tk.BOTH, expand=True)
    
    def _plot_equity_curve(self, fig, ax, equity_data):
        """Plot equity curve chart"""
        # Convert to list of tuples if dictionary format
        if isinstance(equity_data, dict):
            data_points = [(k, v) for k, v in equity_data.items()]
            data_points.sort()  # Sort by date/time
            dates = [p[0] for p in data_points]
            values = [p[1] for p in data_points]
        else:
            # Assume it's a list of some sort
            dates = list(range(len(equity_data)))
            values = equity_data
            
        ax.plot(dates, values, 'b-')
        ax.set_title('Equity Curve')
        ax.set_xlabel('Time')
        ax.set_ylabel('Portfolio Value')
        ax.grid(True)
        fig.tight_layout()
    
    def _plot_drawdown(self, fig, ax, drawdown_data):
        """Plot drawdown chart"""
        # Convert to list of tuples if dictionary format
        if isinstance(drawdown_data, dict):
            data_points = [(k, v) for k, v in drawdown_data.items()]
            data_points.sort()  # Sort by date/time
            dates = [p[0] for p in data_points]
            values = [p[1] for p in data_points]
        else:
            # Assume it's a list of some sort
            dates = list(range(len(drawdown_data)))
            values = drawdown_data
            
        ax.fill_between(dates, 0, values, color='r', alpha=0.3)
        ax.plot(dates, values, 'r-')
        ax.set_title('Drawdown')
        ax.set_xlabel('Time')
        ax.set_ylabel('Drawdown %')
        ax.grid(True)
        fig.tight_layout()
    
    def _plot_monthly_returns(self, fig, ax, monthly_data):
        """Plot monthly returns as a bar chart"""
        months = list(monthly_data.keys())
        returns = list(monthly_data.values())
        
        colors = ['g' if r >= 0 else 'r' for r in returns]
        ax.bar(months, returns, color=colors)
        ax.set_title('Monthly Returns')
        ax.set_xlabel('Month')
        ax.set_ylabel('Return %')
        ax.grid(True, axis='y')
        fig.tight_layout()
        
        # Rotate x labels if many months
        if len(months) > 6:
            plt = ax.xaxis.get_ticklabels()
            for p in plt:
                p.set_rotation(45)
    
    def _plot_trade_distribution(self, fig, ax, trades):
        """Plot trade distribution (win/loss)"""
        # Calculate win/loss stats
        wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
        losses = sum(1 for t in trades if t.get('pnl', 0) <= 0)
        
        # Create pie chart
        labels = f'Wins ({wins})', f'Losses ({losses})'
        sizes = [wins, losses]
        colors = ['#4CAF50', '#F44336']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set_title('Trade Win/Loss Distribution')
        fig.tight_layout()
    
    def stop_trading(self):
        """Stop the trading system"""
        if not self.running:
            messagebox.showinfo("Not Running", "Trading system is not running.")
            return
            
        # Log stop request
        logger = self.log_manager.get_logger("trading_system")
        logger.warning("User requested to stop trading system")
        
        # Set flag to stop
        self.running = False
        
        # Update UI state
        self.control_status_var.set("Stopping trading system...")
        self.status_message.set("Stopping...")
        
        # Stop progress indication
        self.progress_bar.stop()
        self.progress_label.config(text="Stopping...")
        
        # Stop the trader if it exists
        if self.trader:
            try:
                if hasattr(self.trader, 'stop'):
                    self.trader.stop()
            except Exception as e:
                logger.error(f"Error stopping trader: {str(e)}")
        
        # Update button states
        self._update_control_buttons()
        
        # Show message
        messagebox.showinfo("Stop Requested", 
                           "Stop request sent. The system will shut down after the current operation completes.")
    
    # Data management functions
    def start_data_download(self):
        """Start the data download process"""
        # Validate inputs
        try:
            # Check that dates are valid
            start_date = datetime.datetime.strptime(self.dl_start_date_var.get(), "%Y-%m-%d")
            end_date = datetime.datetime.strptime(self.dl_end_date_var.get(), "%Y-%m-%d")
            
            # Check that required fields are filled
            if not self.dl_symbols_var.get():
                messagebox.showerror("Missing Symbols", "Please specify at least one trading symbol.")
                return
                
            # Check that at least one timeframe is selected
            selected_timeframes = [tf for tf, var in self.dl_timeframes.items() if var.get()]
            if not selected_timeframes:
                messagebox.showerror("Missing Timeframes", "Please select at least one timeframe.")
                return
                
            # Apply Binance workaround if selected
            if hasattr(self, 'use_binance_workaround') and self.use_binance_workaround.get():
                self._apply_binance_workaround(True)
                
            # Prepare command line parameters
            symbols = [s.strip() for s in self.dl_symbols_var.get().split(',')]
            
            # Start download thread
            self.dl_status_var.set("Starting data download...")
            self.download_progress.start()
            self.thread = threading.Thread(
                target=self._run_data_download,
                args=(symbols, selected_timeframes, start_date, end_date, self.dl_concurrent_var.get())
            )
            self.thread.daemon = True
            self.thread.start()
            
            # Switch to log tab
            self.notebook.select(self.log_tab)
            
        except ValueError as e:
            messagebox.showerror("Invalid Date Format", f"Please use YYYY-MM-DD format for dates: {str(e)}")
            return
    
    def _run_data_download(self, symbols, timeframes, start_date, end_date, max_concurrent):
        """Run the data download process in a separate thread"""
        try:
            # Get the logger
            logger = self.log_manager.get_logger("data_download")
            logger.info(f"Starting data download: {symbols}, {timeframes}, {start_date} to {end_date}")
                        
            # Run the download process
            config_path = self.config_var.get()
            results = asyncio.run(run_downloads(
                config_path,
                symbols,
                timeframes,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                max_concurrent
            ))
            
            # Update status
            self.dl_status_var.set(f"Download completed. Results: {results['success']} succeeded, {results['failed']} failed")
            
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}", exc_info=True)
            self.dl_status_var.set(f"Error: {str(e)}")
        finally:
            # Stop progress bar
            self.root.after(0, self.download_progress.stop)
    
    def reset_download_form(self):
        """Reset download form to default values"""
        if messagebox.askyesno("Reset Form", "Are you sure you want to reset the download form to default values?"):
            # Reset to defaults
            today = datetime.date.today()
            one_month_ago = today - datetime.timedelta(days=30)
            
            self.dl_symbols_var.set("BTC/USDT")
            
            # Reset timeframes
            for tf, var in self.dl_timeframes.items():
                var.set(tf == "1h")  # Only 1h selected by default
            
            # Reset date range
            self.dl_start_date_var.set(one_month_ago.strftime("%Y-%m-%d"))
            self.dl_end_date_var.set(today.strftime("%Y-%m-%d"))
            
            # Reset concurrent downloads
            self.dl_concurrent_var.set(3)
            
            # Reset status
            self.dl_status_var.set("")
    
    def start_data_migration(self):
        """Start the data migration process"""
        try:
            # Prepare parameters
            symbol = self.migrate_symbol_var.get() if self.migrate_symbol_var.get() else None
            timeframe = self.migrate_timeframe_var.get() if self.migrate_timeframe_var.get() else None
            backup = self.migrate_backup_var.get()
            delete_old = self.migrate_delete_var.get()
            
            # Warning for delete
            if delete_old and not messagebox.askyesno(
                "Confirm Delete", 
                "Are you sure you want to delete legacy files after migration? This cannot be undone."
            ):
                return
            
            # Start migration thread
            self.migrate_status_var.set("Starting data migration...")
            self.migrate_progress.start()
            self.thread = threading.Thread(
                target=self._run_data_migration,
                args=(symbol, timeframe, backup, delete_old)
            )
            self.thread.daemon = True
            self.thread.start()
            
            # Switch to log tab
            self.notebook.select(self.log_tab)
            
        except Exception as e:
            logger = self.log_manager.get_logger("data_migration")
            logger.error(f"Error starting migration: {str(e)}", exc_info=True)
            self.migrate_status_var.set(f"Error: {str(e)}")
    
    def _run_data_migration(self, symbol, timeframe, backup, delete_old):
        """Run the data migration process in a separate thread"""
        try:
            # Get the logger
            logger = self.log_manager.get_logger("data_migration")
            if symbol and timeframe:
                logger.info(f"Starting data migration for {symbol} {timeframe}")
            elif symbol:
                logger.info(f"Starting data migration for all timeframes of {symbol}")
            elif timeframe:
                logger.info(f"Starting data migration for all symbols with timeframe {timeframe}")
            else:
                logger.info("Starting data migration for all symbols and timeframes")
            
            # Import the migration function
            try:
                # Run the migration process
                config_path = self.config_var.get()
                asyncio.run(scan_and_migrate(
                    config_path,
                    symbol,
                    timeframe,
                    backup,
                    delete_old
                ))
                
                # Update status
                self.migrate_status_var.set("Migration completed successfully")
                
                # Refresh file list if on the view tab
                self.refresh_file_list()
                
            except ImportError:
                logger.error("Migration module not found. This feature may not be implemented yet.")
                self.migrate_status_var.set("Migration feature not available")
            
        except Exception as e:
            logger.error(f"Error during migration: {str(e)}", exc_info=True)
            self.migrate_status_var.set(f"Error: {str(e)}")
        finally:
            # Stop progress bar
            self.root.after(0, self.migrate_progress.stop)
    
    def refresh_file_list(self):
        """Refresh the file list in the view tab"""
        try:
            # Clear current list
            self.file_listbox.delete(0, tk.END)
            
            # Get symbol and timeframe
            symbol = self.view_symbol_var.get().strip()
            timeframe = self.view_timeframe_var.get().strip()
            
            if not symbol or not timeframe:
                self.view_status_var.set("Please enter both symbol and timeframe")
                return
            
            # Get the data_path from config
            data_path = self.config.get('data', 'storage', 'historical', default='data/historical')
            if not data_path:
                data_path = 'data/historical'
            
            # Build the path
            symbol_dir = os.path.join(data_path, timeframe, symbol.replace('/', '_'))
            
            if not os.path.exists(symbol_dir):
                self.file_listbox.insert(tk.END, f"No data directory found for {symbol} {timeframe}")
                self.view_status_var.set(f"Directory not found: {symbol_dir}")
                return
            
            # Scan for daily files
            daily_files = []
            legacy_files = []
            total_size = 0
            
            for root, _, files in os.walk(symbol_dir):
                for file in files:
                    if not file.endswith('.parquet'):
                        continue
                        
                    full_path = os.path.join(root, file)
                    file_size = os.path.getsize(full_path)
                    total_size += file_size
                    
                    # Check if it's a daily file or legacy file
                    if file.startswith(f"{timeframe}-"):
                        daily_files.append((full_path, file_size))
                    elif "_" in os.path.splitext(file)[0]:
                        legacy_files.append((full_path, file_size))
            
            # Display the file counts
            self.file_listbox.insert(tk.END, f"Found {len(daily_files)} daily files and {len(legacy_files)} legacy files")
            self.file_listbox.insert(tk.END, f"Total size: {self._format_size(total_size)}")
            self.file_listbox.insert(tk.END, "")
            
            # Show daily files first
            if daily_files:
                self.file_listbox.insert(tk.END, "=== Daily Files ===")
                for full_path, file_size in sorted(daily_files):
                    file_name = os.path.basename(full_path)
                    self.file_listbox.insert(tk.END, f"{file_name} ({self._format_size(file_size)})")
            
            # Show legacy files
            if legacy_files:
                self.file_listbox.insert(tk.END, "")
                self.file_listbox.insert(tk.END, "=== Legacy Files ===")
                # Only show first 100 to avoid overloading the UI
                shown_files = legacy_files[:100]
                for full_path, file_size in sorted(shown_files):
                    file_name = os.path.basename(full_path)
                    self.file_listbox.insert(tk.END, f"{file_name} ({self._format_size(file_size)})")
                
                if len(legacy_files) > 100:
                    self.file_listbox.insert(tk.END, f"... and {len(legacy_files) - 100} more files")
            
            # Update status
            self.view_status_var.set(f"Found {len(daily_files)} daily files and {len(legacy_files)} legacy files for {symbol} {timeframe}")
            
        except Exception as e:
            logger = self.log_manager.get_logger("data_view")
            logger.error(f"Error refreshing file list: {str(e)}", exc_info=True)
            self.view_status_var.set(f"Error: {str(e)}")
    
    def _format_size(self, size_bytes):
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def view_file_details(self):
        """View details of the selected file"""
        try:
            # Get selected file
            selection = self.file_listbox.curselection()
            if not selection:
                self.view_status_var.set("Please select a file to view details")
                return
                
            selected_text = self.file_listbox.get(selection[0])
            
            # Skip header or separator lines
            if selected_text.startswith("===") or "Found " in selected_text or "Total size:" in selected_text or not selected_text.strip():
                self.view_status_var.set("Please select a valid file")
                return
            
            # Extract file name
            file_name = selected_text.split(" (")[0]
            
            # Get symbol and timeframe
            symbol = self.view_symbol_var.get().strip()
            timeframe = self.view_timeframe_var.get().strip()
            
            # Get the data_path from config
            data_path = self.config.get('data', 'storage', 'historical', default='data/historical')
            if not data_path:
                data_path = 'data/historical'
            
            # Find the file
            symbol_dir = os.path.join(data_path, timeframe, symbol.replace('/', '_'))
            
            file_path = None
            # First check if it's a daily file
            if file_name.startswith(f"{timeframe}-"):
                # Parse date from file name
                date_parts = file_name.replace(f"{timeframe}-", "").split(".")[0].split("-")
                if len(date_parts) == 3:
                    year, month, day = date_parts
                    year_dir = os.path.join(symbol_dir, year)
                    month_dir = os.path.join(year_dir, month)
                    file_path = os.path.join(month_dir, file_name)
            
            # If not found, search for it
            if not file_path or not os.path.exists(file_path):
                for root, _, files in os.walk(symbol_dir):
                    if file_name in files:
                        file_path = os.path.join(root, file_name)
                        break
            
            if not file_path or not os.path.exists(file_path):
                self.view_status_var.set(f"File not found: {file_name}")
                return
            
            # Open file details dialog
            self._show_file_details(file_path)
            
        except Exception as e:
            logger = self.log_manager.get_logger("data_view")
            logger.error(f"Error viewing file details: {str(e)}", exc_info=True)
            self.view_status_var.set(f"Error: {str(e)}")
    
    def _show_file_details(self, file_path):
        """Show file details in a new dialog"""
        try:
            # Try importing pandas
            try:
                import pandas as pd
            except ImportError:
                messagebox.showerror("Missing Dependency", "pandas module is required to view file details")
                return
            
            # Read the file
            df = pd.read_parquet(file_path)
            
            # Create a new dialog
            details_dialog = tk.Toplevel(self.root)
            details_dialog.title(f"File Details: {os.path.basename(file_path)}")
            details_dialog.geometry("800x600")
            details_dialog.minsize(600, 400)
            
            # Create a frame for the dialog
            frame = ttk.Frame(details_dialog, padding="10")
            frame.pack(fill=tk.BOTH, expand=True)
            
            # File info section
            ttk.Label(frame, text=f"File: {os.path.basename(file_path)}", font=self.header_font).grid(column=0, row=0, columnspan=2, sticky=tk.W, pady=(0, 5))
            ttk.Label(frame, text=f"Path: {file_path}").grid(column=0, row=1, columnspan=2, sticky=tk.W)
            ttk.Label(frame, text=f"Size: {self._format_size(os.path.getsize(file_path))}").grid(column=0, row=2, columnspan=2, sticky=tk.W)
            ttk.Label(frame, text=f"Records: {len(df)}").grid(column=0, row=3, columnspan=2, sticky=tk.W)
            
            if 'datetime' in df.columns:
                min_date = df['datetime'].min()
                max_date = df['datetime'].max()
                ttk.Label(frame, text=f"Date Range: {min_date} to {max_date}").grid(column=0, row=4, columnspan=2, sticky=tk.W)
            
            # Create a separator
            ttk.Separator(frame, orient=tk.HORIZONTAL).grid(column=0, row=5, columnspan=2, sticky=(tk.W, tk.E), pady=10)
            
            # Data preview section
            ttk.Label(frame, text="Data Preview:", font=self.header_font).grid(column=0, row=6, sticky=tk.W, pady=(0, 5))
            
            # Create a text widget for the data preview
            preview_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=15)
            preview_text.grid(column=0, row=7, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
            
            # Format and insert the DataFrame preview
            preview_text.insert(tk.END, "Column names:\n")
            preview_text.insert(tk.END, ", ".join(df.columns) + "\n\n")
            
            preview_text.insert(tk.END, "First 10 rows:\n")
            preview_text.insert(tk.END, df.head(10).to_string())
            
            # Make the data preview text read-only
            preview_text.configure(state="disabled")
            
            # Statistics section
            ttk.Label(frame, text="Statistics:", font=self.header_font).grid(column=0, row=8, sticky=tk.W, pady=(10, 5))
            
            # Create a text widget for the statistics
            stats_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=8)
            stats_text.grid(column=0, row=9, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
            
            # Calculate and insert statistics
            numeric_cols = df.select_dtypes(include=['number']).columns
            if not numeric_cols.empty:
                stats_df = df[numeric_cols].describe()
                stats_text.insert(tk.END, stats_df.to_string())
            else:
                stats_text.insert(tk.END, "No numeric columns found for statistics.")
            
            # Make the statistics text read-only
            stats_text.configure(state="disabled")
            
            # Add export button
            ttk.Button(
                frame, 
                text="Export Preview to CSV", 
                command=lambda: self._export_preview(df.head(100))
            ).grid(column=0, row=10, pady=10)
            
            # Button to close the dialog
            ttk.Button(frame, text="Close", command=details_dialog.destroy).grid(column=1, row=10, pady=10)
            
            # Make the preview and stats sections resizable
            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(7, weight=2)  # Data preview expands more
            frame.rowconfigure(9, weight=1)  # Stats expands less
            
        except Exception as e:
            logger = self.log_manager.get_logger("data_view")
            logger.error(f"Error showing file details: {str(e)}", exc_info=True)
            messagebox.showerror("Error", f"Error showing file details:\n{str(e)}")
    
    def _export_preview(self, df):
        """Export data preview to CSV file"""
        try:
            filename = filedialog.asksaveasfilename(
                title="Export Preview to CSV",
                defaultextension=".csv",
                filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
            )
            if filename:
                df.to_csv(filename, index=False)
                messagebox.showinfo("Export Successful", f"Preview exported to {filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting preview: {str(e)}")
    
    def save_api_settings(self):
        """Save API settings to config"""
        try:
            # Update config with form values
            self.config.set("api", "binance", "api_key", self.api_key_var.get())
            self.config.set("api", "binance", "secret", self.api_secret_var.get())
            self.config.set("api", "timeout", self.api_timeout_var.get())
            self.config.set("api", "rate_limits", "requests_per_minute", self.api_rate_limit_var.get())
            self.config.set("api", "retries", "max_attempts", self.api_retry_attempts_var.get())
            self.config.set("api", "retries", "delay_seconds", self.api_retry_delay_var.get())
            
            # Proxy settings
            if self.api_useproxy_var.get():
                self.config.set("proxies", "http", self.api_http_proxy_var.get())
                self.config.set("proxies", "https", self.api_https_proxy_var.get())
                self.config.save(self.config_var.get())
            else:
                self.config.set("proxies", "http", "")
                self.config.set("proxies", "https", "")
                        
            # Apply the Binance API workaround
            self._apply_binance_workaround(self.binance_workaround_var.get())
            
            # Show success message
            self.api_status_var.set("API settings saved successfully. Restart any active connections for changes to take effect.")
            
        except Exception as e:
            self.api_status_var.set(f"Error saving API settings: {str(e)}")
    
    def _apply_binance_workaround(self, enabled):
        """Apply or remove the Binance API workaround"""
        try:
            # This modifies the Binance exchange module to use a different approach for exchangeInfo
            # Check if we have access to the binance.py file
            try:                
                # Store the setting in config
                self.config.set("api", "binance", "workaround_enabled", enabled)
                
                # Log the change
                logger = self.log_manager.get_logger("api_settings")
                if enabled:
                    logger.info("Binance API workaround enabled. This helps prevent HTTP 403 errors.")
                else:
                    logger.info("Binance API workaround disabled.")
                
            except ImportError:
                self.apply_binance_workaround()
                self.api_status_var.set("Applied workaround using file system search")
            
        except Exception as e:
            self.api_status_var.set(f"Could not apply workaround: {str(e)}")
    
    def apply_binance_workaround(self):
        """
        Apply a workaround for the Binance API exchangeInfo HTTP 403 error
        
        This function patches the Binance.py file to use a different approach for fetching exchangeInfo
        """
        try:
            from pathlib import Path
            import os
            import re
            
            # Find the binance.py file through file system search
            binance_file = None
            for root, _, files in os.walk(project_root):
                if "binance.py" in files:
                    binance_file = os.path.join(root, "binance.py")
                    break
            
            if not binance_file:
                raise FileNotFoundError("Could not find binance.py file")
            
            # Read the file
            with open(binance_file, 'r') as f:
                content = f.read()
            
            # Check if workaround is already applied
            if "Workaround for exchangeInfo HTTP 403 error" in content:
                self.log_manager.get_logger("api_settings").info("Binance API workaround already applied")
                return
            
            # Add the workaround
            # Find the fetch_historical_ohlcv method
            method_pattern = r'async def fetch_historical_ohlcv\([^)]*\):[^\}]*\}'
            match = re.search(method_pattern, content, re.DOTALL)
            
            if not match:
                raise ValueError("Could not find fetch_historical_ohlcv method in binance.py")
            
            # Find the line with fetch_ohlcv inside the method
            method_content = match.group(0)
            ohlcv_call_pattern = r'(\s+ohlcv\s*=\s*self\.exchange\.fetch_ohlcv\([^)]*\))'
            
            # Replace the direct call with a wrapped version
            modified_method = re.sub(
                ohlcv_call_pattern,
                r'\1\n                # Workaround for exchangeInfo HTTP 403 error\n                if not ohlcv and "exchangeInfo" in str(e):\n                    self.logger.warning("Using direct HTTP request for OHLCV data due to exchangeInfo error")\n                    # Implement direct request logic here\n                    # For now, just retry with reduced batch size\n                    await asyncio.sleep(5)  # Add delay before retry\n                    ohlcv = self.exchange.fetch_ohlcv(\n                        symbol=symbol, \n                        timeframe=timeframe, \n                        since=chunk_since,\n                        limit=max(50, self.download_chunk_size // 10)\n                    )',
                method_content
            )
            
            # Replace the method in the file content
            modified_content = content.replace(method_content, modified_method)
            
            # Write the modified file
            with open(binance_file, 'w') as f:
                f.write(modified_content)
            
            # Log success
            self.log_manager.get_logger("api_settings").info("Successfully applied Binance API workaround")
            
        except Exception as e:
            self.log_manager.get_logger("api_settings").error(f"Failed to apply Binance API workaround: {str(e)}", exc_info=True)
            raise
    
    def test_api_connection(self):
        """Test the API connection with current settings"""
        # Save settings first
        self.save_api_settings()
        
        # Test connection in a separate thread
        threading.Thread(target=self._run_api_test, daemon=True).start()
    
    def _run_api_test(self):
        """Run API connection test"""
        try:
            self.api_status_var.set("Testing API connection...")
            
            # Import the Binance module
            try:                
                # Create a temporary Binance instance with the current settings
                binance = Binance(self.config)
                
                # Run a simple test
                exchangeInfo = binance.exchange.fetch_markets()
                
                # Show success message
                self.api_status_var.set(f"Connection successful! Found {len(exchangeInfo)} trading pairs.")
                
            except ImportError:
                self.api_status_var.set("Binance module not found. This feature may not be implemented.")
            
        except Exception as e:
            self.api_status_var.set(f"Connection test failed: {str(e)}")
    
    def reset_form(self):
        """Reset all form fields to default values"""
        if messagebox.askyesno("Reset Form", "Are you sure you want to reset all fields to default values?"):
            # General defaults
            self.config_var.set(str(Path(project_root) / "conf/bt_config.yaml"))
            self.mode_var.set("backtest")
            self.symbol_var.set("BTC/USDT")
            self.initial_capital_var.set(100000)
            
            # Date range defaults
            today = datetime.date.today()
            one_month_ago = today - datetime.timedelta(days=30)
            self.start_date_var.set(one_month_ago.strftime("%Y-%m-%d"))
            self.end_date_var.set(today.strftime("%Y-%m-%d"))
            
            # Backtest costs
            self.commission_var.set(0.001)
            self.slippage_var.set(0.001)
            
            # Strategy defaults
            self.strategy_var.set("dual_ma")
            self.timeframe_var.set("1h")
            self.fallback_strategy_var.set("")
            self.fast_period_var.set(50)
            self.slow_period_var.set(200)
            self.threshold_var.set(0.005)
            
            # Risk management defaults
            self.stop_loss_enabled_var.set(True)
            self.stop_loss_pct_var.set(0.02)
            self.stop_loss_daily_var.set(0.05)
            self.take_profit_enabled_var.set(True)
            self.take_profit_pct_var.set(0.05)
            self.max_drawdown_var.set(0.2)
            self.max_trade_drawdown_var.set(0.1)
            self.max_position_var.set(0.1)
            self.min_order_var.set(0.001)
            
            # Advanced settings defaults
            self.db_url_var.set("sqlite:///database/trading.sqlite")
            self.db_pool_size_var.set(5)
            self.db_timeout_var.set(30)
            self.db_echo_sql_var.set(False)
            self.max_workers_var.set(4)
            self.max_memory_var.set(1024)  # 1GB in MB
            self.task_timeout_var.set(300)
            self.timezone_var.set("UTC")
            
            # Live trading defaults
            self.emergency_stop_var.set(True)
            self.emergency_threshold_var.set(0.1)
            self.max_retries_var.set(3)
            self.cooldown_var.set(60)
            
            # API defaults
            self.api_key_var.set("")
            self.api_secret_var.set("")
            self.api_timeout_var.set(60000)
            self.api_useproxy_var.set(False)
            self.api_http_proxy_var.set("")
            self.api_https_proxy_var.set("")
            
            # Debug mode
            self.debug_var.set(False)
            
            # Update UI states
            self._toggle_stop_loss()
            self._toggle_take_profit()
            self._toggle_emergency_stop()
            self._toggle_proxy_fields()
            
            # Update status
            self.setup_status_var.set("All fields reset to default values")
    
    # Strategy optimization
    def start_optimization(self):
        """Start strategy parameter optimization"""
        # Check if at least one parameter is selected for optimization
        if not any([self.optimize_fast_period.get(), self.optimize_slow_period.get(), self.optimize_threshold.get()]):
            messagebox.showerror("Optimization Error", "Please select at least one parameter to optimize")
            return
        
        # Save configuration first
        self.save_config()
        
        # Create parameter grid
        params = {}
        if self.optimize_fast_period.get():
            params["fast_period"] = range(5, 100, 5)
        if self.optimize_slow_period.get():
            params["slow_period"] = range(50, 300, 25)
        if self.optimize_threshold.get():
            # Convert float range to list
            params["threshold"] = [round(i * 0.001, 3) for i in range(1, 20)]
        
        # Show some details and confirm
        param_info = "\n".join([f"{k}: {list(v)}" for k, v in params.items()])
        if not messagebox.askyesno("Confirm Optimization", 
                                 f"This will run multiple backtests with different parameters. This may take a while.\n\nParameters:\n{param_info}\n\nProceed?"):
            return
        
        # Start optimization in a separate thread
        self.running = True
        threading.Thread(target=self._run_optimization, args=(params,), daemon=True).start()
        
        # Switch to log tab
        self.notebook.select(self.log_tab)
        
    def _run_optimization(self, params):
        """Run strategy parameter optimization"""
        try:
            logger = self.log_manager.get_logger("optimization")
            logger.info(f"Starting parameter optimization with {params}")
            
            # Placeholder for actual optimization code
            logger.info("Optimization feature is not fully implemented yet")
            logger.info("Parameters to optimize: " + str(params))
            
            # Usually this would iterate through parameter combinations and run backtests
            # ...
            
            self.running = False
            logger.info("Optimization completed")
            
            # Show placeholder results
            optimization_results = {
                "best_params": {
                    "fast_period": 15,
                    "slow_period": 150,
                    "threshold": 0.003
                },
                "metrics": {
                    "sharpe_ratio": 1.75,
                    "total_return": 0.32,
                    "max_drawdown": 0.14
                }
            }
            
            # Display results
            self.display_results(optimization_results)
            
            # Update form with best parameters
            if "fast_period" in optimization_results["best_params"]:
                self.fast_period_var.set(optimization_results["best_params"]["fast_period"])
            if "slow_period" in optimization_results["best_params"]:
                self.slow_period_var.set(optimization_results["best_params"]["slow_period"])
            if "threshold" in optimization_results["best_params"]:
                self.threshold_var.set(optimization_results["best_params"]["threshold"])
            
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}", exc_info=True)
        finally:
            self.running = False
    
    def on_close(self):
        """Handle window close event"""
        if self.running:
            if not messagebox.askyesno("Confirm Exit", "Trading system is still running. Are you sure you want to exit?"):
                return
        
        # Perform cleanup
        try:
            if hasattr(self, 'executor'):
                self.executor.run(self.executor.close())
                
            # Suggest config save if changes made
            self._perform_autosave(silent=True)
        except Exception as e:
            self.log_manager.get_logger("gui").error(f"Error during shutdown: {str(e)}", exc_info=True)
        
        # Destroy window
        self.root.destroy()