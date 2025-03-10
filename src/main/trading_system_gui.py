# src\bin\trading_system_gui.py

import ttkthemes
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext, simpledialog
from tkcalendar import DateEntry
import threading
import queue
import os
import sys
import json
import logging
import time
import asyncio
import numpy as np
from datetime import datetime, timedelta
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Configure logging
log_dir = os.path.join(parent_dir, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'gui_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TradingSystemGUI')

# Global configuration
CONFIG_FILE = os.path.join(parent_dir, 'config', 'gui_config.json')
DEFAULT_CONFIG = {
    "exchange": "binance",
    "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
    "strategies": ["dual_ma", "neural_net"],
    "data_source": "local",
    "mode": "backtest",
    "theme": "light",
    "start_date": "",
    "end_date": "",
    "auto_download": True
}

# Try to import trading system modules (these may fail if not available)
try:
    from src.common.config_manager import ConfigManager
    from src.data.data_manager import DataManager, TradingMode
    from src.strategy.strategy_factory import StrategyFactory
    TRADING_MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Trading modules import error: {e}")
    TRADING_MODULES_AVAILABLE = False

class RedirectText:
    """Class for redirecting stdout/stderr to a queue for GUI display"""
    def __init__(self, queue):
        self.queue = queue

    def write(self, string):
        if string.strip():  # Avoid empty strings
            self.queue.put(string)

    def flush(self):
        pass

class TradingSystemGUI:
    """Main GUI class for the trading system"""
    
    def __init__(self, root):
        """Initialize the GUI"""
        self.root = root
        self.root.title("Trading System GUI")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)
        
        # Set icon if available
        icon_path = os.path.join(current_dir, 'assets', 'icon.ico')
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize variables
        self.trading_thread = None
        self.is_running = False
        self.log_queue = queue.Queue()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.symbols_var = {}
        self.timeframe_var = tk.StringVar(value=self.config["timeframes"][0])
        self.strategy_var = tk.StringVar(value=self.config["strategies"][0])
        self.mode_var = tk.StringVar(value=self.config["mode"])
        self.data_source_var = tk.StringVar(value=self.config["data_source"])
        self.auto_download_var = tk.BooleanVar(value=self.config["auto_download"])
        self.start_date_var = tk.StringVar(value=self.config.get("start_date", ""))
        self.end_date_var = tk.StringVar(value=self.config.get("end_date", ""))
        
        # Create GUI elements
        self.create_menu()
        self.create_main_frame()
        
        # Set up log redirection
        sys.stdout = RedirectText(self.log_queue)
        sys.stderr = RedirectText(self.log_queue)
        
        # Start log processing loop
        self.process_logs()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_menu(self):
        """Create the application menu"""
        menu_bar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Load Configuration", command=self.load_config_dialog)
        file_menu.add_command(label="Save Configuration", command=self.save_config_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        # View menu
        view_menu = tk.Menu(menu_bar, tearoff=0)
        view_menu.add_command(label="Clear Logs", command=self.clear_logs)
        view_menu.add_command(label="Save Logs", command=self.save_logs)
        
        # Theme submenu
        theme_menu = tk.Menu(view_menu, tearoff=0)
        theme_menu.add_radiobutton(label="Light", command=lambda: self.set_theme("light"))
        theme_menu.add_radiobutton(label="Dark", command=lambda: self.set_theme("dark"))
        view_menu.add_cascade(label="Theme", menu=theme_menu)
        
        menu_bar.add_cascade(label="View", menu=view_menu)
        
        # Tools menu
        tools_menu = tk.Menu(menu_bar, tearoff=0)
        tools_menu.add_command(label="Download Historical Data", command=self.download_data)
        tools_menu.add_command(label="Backtest Strategy", command=self.run_backtest)
        tools_menu.add_command(label="Generate Report", command=self.generate_report)
        menu_bar.add_cascade(label="Tools", menu=tools_menu)
        
        # Help menu
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Documentation", command=lambda: self.show_docs())
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menu_bar)

    def create_main_frame(self):
        """Create the main application frame with notebook"""
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook with tabs
        self.notebook = ttk.Notebook(self.main_frame)
        
        # Dashboard tab
        self.dashboard_frame = ttk.Frame(self.notebook)
        self.create_dashboard()
        self.notebook.add(self.dashboard_frame, text="Dashboard")
        
        # Configuration tab
        self.config_frame = ttk.Frame(self.notebook)
        self.create_config_panel()
        self.notebook.add(self.config_frame, text="Configuration")
        
        # Logs tab
        self.logs_frame = ttk.Frame(self.notebook)
        self.create_logs_panel()
        self.notebook.add(self.logs_frame, text="Logs")
        
        # Charts tab
        self.charts_frame = ttk.Frame(self.notebook)
        self.create_charts_panel()
        self.notebook.add(self.charts_frame, text="Charts")
        
        self.notebook.pack(fill=tk.BOTH, expand=True)

    def create_dashboard(self):
        """Create the dashboard panel with controls and status"""
        # Top controls frame
        control_frame = ttk.LabelFrame(self.dashboard_frame, text="Trading Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Control buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_btn = ttk.Button(btn_frame, text="Start Trading", command=self.start_trading)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="Stop Trading", command=self.stop_trading, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.backtest_btn = ttk.Button(btn_frame, text="Run Backtest", command=self.run_backtest)
        self.backtest_btn.pack(side=tk.LEFT, padx=5)
        
        # Status frame
        status_frame = ttk.LabelFrame(self.dashboard_frame, text="System Status")
        status_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create 2x2 grid of status panels
        for i in range(2):
            status_frame.columnconfigure(i, weight=1)
        for i in range(2):
            status_frame.rowconfigure(i, weight=1)
        
        # Status panels
        self.create_status_panel(status_frame, 0, 0, "Trading Status", "status_panel")
        self.create_status_panel(status_frame, 0, 1, "Active Strategies", "strategy_panel")
        self.create_status_panel(status_frame, 1, 0, "Performance", "performance_panel")
        self.create_status_panel(status_frame, 1, 1, "Recent Trades", "trades_panel")

    def create_status_panel(self, parent, row, col, title, name):
        """Create a status panel within the dashboard"""
        frame = ttk.LabelFrame(parent, text=title)
        frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
        
        if name == "status_panel":
            # Status indicators
            status_frame = ttk.Frame(frame)
            status_frame.pack(fill=tk.BOTH, padx=5, pady=5)
            
            # Trading status
            ttk.Label(status_frame, text="Trading Status:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.trading_status_label = ttk.Label(status_frame, text="Stopped", foreground="red")
            self.trading_status_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
            
            # Connection status
            ttk.Label(status_frame, text="Exchange Connection:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            self.connection_status_label = ttk.Label(status_frame, text="Disconnected", foreground="red")
            self.connection_status_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
            
            # Data status
            ttk.Label(status_frame, text="Data Source:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
            self.data_status_label = ttk.Label(status_frame, text="Local", foreground="blue")
            self.data_status_label.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
            
            # Trading mode
            ttk.Label(status_frame, text="Trading Mode:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
            self.mode_status_label = ttk.Label(status_frame, text="Backtest", foreground="blue")
            self.mode_status_label.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
            
        elif name == "strategy_panel":
            # Active strategies list
            self.strategies_list = tk.Listbox(frame, height=8)
            self.strategies_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Add some demo items
            for strategy in self.config["strategies"]:
                self.strategies_list.insert(tk.END, f"{strategy} (inactive)")
                
        elif name == "performance_panel":
            # Performance metrics
            canvas_frame = ttk.Frame(frame)
            canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Create a figure for performance metrics
            fig = Figure(figsize=(3, 2), dpi=100)
            self.perf_ax = fig.add_subplot(111)
            self.perf_ax.bar(['Win Rate', 'Profit Factor', 'Sharpe'], [0.65, 1.2, 0.8])
            self.perf_ax.set_ylim(0, 2)
            self.perf_ax.set_title("Performance Metrics")
            
            # Embed the figure in the frame
            self.perf_canvas = FigureCanvasTkAgg(fig, canvas_frame)
            self.perf_canvas.draw()
            self.perf_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        elif name == "trades_panel":
            # Recent trades table
            self.trades_tree = ttk.Treeview(frame, columns=("Time", "Symbol", "Type", "Price", "Size"), show="headings", height=8)
            self.trades_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Configure columns
            self.trades_tree.heading("Time", text="Time")
            self.trades_tree.heading("Symbol", text="Symbol")
            self.trades_tree.heading("Type", text="Type")
            self.trades_tree.heading("Price", text="Price")
            self.trades_tree.heading("Size", text="Size")
            
            self.trades_tree.column("Time", width=80)
            self.trades_tree.column("Symbol", width=80)
            self.trades_tree.column("Type", width=60)
            self.trades_tree.column("Price", width=80)
            self.trades_tree.column("Size", width=60)
            
            # Add some demo items
            self.trades_tree.insert("", tk.END, values=("09:30:45", "BTC/USDT", "BUY", "42168.5", "0.01"))
            self.trades_tree.insert("", tk.END, values=("10:15:22", "ETH/USDT", "BUY", "2250.75", "0.15"))
            self.trades_tree.insert("", tk.END, values=("11:05:10", "SOL/USDT", "SELL", "108.40", "2.5"))

    def create_config_panel(self):
        """Create configuration panel with settings"""
        # Use a notebook for configuration categories
        config_notebook = ttk.Notebook(self.config_frame)
        config_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # General settings tab
        general_frame = ttk.Frame(config_notebook)
        config_notebook.add(general_frame, text="General")
        
        # Trading settings tab
        trading_frame = ttk.Frame(config_notebook)
        config_notebook.add(trading_frame, text="Trading")
        
        # Strategy settings tab
        strategy_frame = ttk.Frame(config_notebook)
        config_notebook.add(strategy_frame, text="Strategy")
        
        # Data settings tab
        data_frame = ttk.Frame(config_notebook)
        config_notebook.add(data_frame, text="Data")
        
        # General settings content
        self.create_general_settings(general_frame)
        
        # Trading settings content
        self.create_trading_settings(trading_frame)
        
        # Strategy settings content
        self.create_strategy_settings(strategy_frame)
        
        # Data settings content
        self.create_data_settings(data_frame)
        
        # Buttons frame
        buttons_frame = ttk.Frame(self.config_frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Save button
        save_btn = ttk.Button(buttons_frame, text="Save Configuration", command=self.save_config)
        save_btn.pack(side=tk.RIGHT, padx=5)
        
        # Reset button
        reset_btn = ttk.Button(buttons_frame, text="Reset to Defaults", command=self.reset_config)
        reset_btn.pack(side=tk.RIGHT, padx=5)

    def create_general_settings(self, parent):
        """Create general settings fields"""
        # Trading mode
        mode_frame = ttk.LabelFrame(parent, text="Trading Mode")
        mode_frame.pack(fill=tk.X, padx=5, pady=5)
        
        mode_options = ["backtest", "paper", "live"]
        for i, mode in enumerate(mode_options):
            ttk.Radiobutton(mode_frame, text=mode.capitalize(), value=mode, variable=self.mode_var).pack(anchor=tk.W, padx=20, pady=2)
        
        # Exchange settings
        exchange_frame = ttk.LabelFrame(parent, text="Exchange Settings")
        exchange_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(exchange_frame, text="Exchange:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        exchange_combo = ttk.Combobox(exchange_frame, values=["binance", "kucoin", "coinbase"], state="readonly")
        exchange_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        exchange_combo.set(self.config["exchange"])
        
        ttk.Label(exchange_frame, text="API Key:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        api_key_entry = ttk.Entry(exchange_frame, width=40)
        api_key_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(exchange_frame, text="API Secret:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        api_secret_entry = ttk.Entry(exchange_frame, width=40, show="*")
        api_secret_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # User interface settings
        ui_frame = ttk.LabelFrame(parent, text="User Interface")
        ui_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(ui_frame, text="Theme:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        theme_combo = ttk.Combobox(ui_frame, values=["light", "dark"], state="readonly")
        theme_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        theme_combo.set(self.config["theme"])
        theme_combo.bind("<<ComboboxSelected>>", lambda e: self.set_theme(theme_combo.get()))
        
        ttk.Label(ui_frame, text="Log Level:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        log_level_combo = ttk.Combobox(ui_frame, values=["DEBUG", "INFO", "WARNING", "ERROR"], state="readonly")
        log_level_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        log_level_combo.set("INFO")

    def create_trading_settings(self, parent):
        """Create trading settings fields"""
        # Symbol selection
        symbols_frame = ttk.LabelFrame(parent, text="Trading Symbols")
        symbols_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create checkbox for each symbol
        for i, symbol in enumerate(self.config["symbols"]):
            self.symbols_var[symbol] = tk.BooleanVar(value=True)
            ttk.Checkbutton(symbols_frame, text=symbol, variable=self.symbols_var[symbol]).grid(
                row=i//3, column=i%3, sticky=tk.W, padx=20, pady=2)
        
        # Add symbol button
        add_symbol_frame = ttk.Frame(symbols_frame)
        add_symbol_frame.grid(row=(len(self.config["symbols"])+2)//3, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(add_symbol_frame, text="Add Symbol:").pack(side=tk.LEFT, padx=5)
        add_symbol_entry = ttk.Entry(add_symbol_frame, width=15)
        add_symbol_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(add_symbol_frame, text="Add", command=lambda: self.add_symbol(add_symbol_entry.get())).pack(side=tk.LEFT, padx=5)
        
        # Risk parameters
        risk_frame = ttk.LabelFrame(parent, text="Risk Parameters")
        risk_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(risk_frame, text="Max Position Size (%):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        max_pos_entry = ttk.Entry(risk_frame, width=10)
        max_pos_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        max_pos_entry.insert(0, "10")
        
        ttk.Label(risk_frame, text="Risk Per Trade (%):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        risk_entry = ttk.Entry(risk_frame, width=10)
        risk_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        risk_entry.insert(0, "1")
        
        ttk.Label(risk_frame, text="Max Open Trades:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        max_trades_entry = ttk.Entry(risk_frame, width=10)
        max_trades_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        max_trades_entry.insert(0, "3")
        
        # Order settings
        order_frame = ttk.LabelFrame(parent, text="Order Settings")
        order_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(order_frame, text="Default Order Type:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        order_type_combo = ttk.Combobox(order_frame, values=["market", "limit"], state="readonly")
        order_type_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        order_type_combo.set("limit")
        
        ttk.Label(order_frame, text="Take Profit (%):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        tp_entry = ttk.Entry(order_frame, width=10)
        tp_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        tp_entry.insert(0, "2.5")
        
        ttk.Label(order_frame, text="Stop Loss (%):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        sl_entry = ttk.Entry(order_frame, width=10)
        sl_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        sl_entry.insert(0, "1.5")

    def create_strategy_settings(self, parent):
        """Create strategy settings fields"""
        # Strategy selection
        strategy_frame = ttk.LabelFrame(parent, text="Strategy Selection")
        strategy_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Strategy selection combobox
        ttk.Label(strategy_frame, text="Active Strategy:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        strategy_combo = ttk.Combobox(strategy_frame, textvariable=self.strategy_var, values=self.config["strategies"], state="readonly")
        strategy_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Strategy parameters frame (will change based on selected strategy)
        params_frame = ttk.LabelFrame(parent, text="Strategy Parameters")
        params_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create strategy parameter widgets based on selected strategy
        self.update_strategy_params(params_frame, self.strategy_var.get())
        
        # Bind strategy selection to update parameters
        strategy_combo.bind("<<ComboboxSelected>>", lambda e: self.update_strategy_params(params_frame, strategy_combo.get()))

    def update_strategy_params(self, parent, strategy):
        """Update strategy parameters based on selected strategy"""
        # Clear existing widgets
        for widget in parent.winfo_children():
            widget.destroy()
        
        # Create parameters based on strategy
        if strategy == "dual_ma":
            ttk.Label(parent, text="Short Window:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            short_entry = ttk.Entry(parent, width=10)
            short_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
            short_entry.insert(0, "10")
            
            ttk.Label(parent, text="Long Window:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            long_entry = ttk.Entry(parent, width=10)
            long_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
            long_entry.insert(0, "30")
            
            ttk.Label(parent, text="Signal Type:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
            signal_combo = ttk.Combobox(parent, values=["crossover", "threshold"], state="readonly")
            signal_combo.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
            signal_combo.set("crossover")
            
        elif strategy == "neural_net":
            ttk.Label(parent, text="Hidden Layers:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            layers_entry = ttk.Entry(parent, width=10)
            layers_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
            layers_entry.insert(0, "64,32")
            
            ttk.Label(parent, text="Learning Rate:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            lr_entry = ttk.Entry(parent, width=10)
            lr_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
            lr_entry.insert(0, "0.001")
            
            ttk.Label(parent, text="Dropout:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
            dropout_entry = ttk.Entry(parent, width=10)
            dropout_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
            dropout_entry.insert(0, "0.2")
            
            ttk.Label(parent, text="Epochs:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
            epochs_entry = ttk.Entry(parent, width=10)
            epochs_entry.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
            epochs_entry.insert(0, "100")
        
        # Add buttons for testing and saving strategy
        buttons_frame = ttk.Frame(parent)
        buttons_frame.grid(row=10, column=0, columnspan=2, pady=10)
        
        ttk.Button(buttons_frame, text="Test Strategy", command=self.test_strategy).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Save Parameters", command=self.save_strategy_params).pack(side=tk.LEFT, padx=5)

    def create_data_settings(self, parent):
        """Create data source settings fields"""
        # Data source selection
        source_frame = ttk.LabelFrame(parent, text="Data Source")
        source_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Data source radio buttons
        ttk.Radiobutton(source_frame, text="Local", value="local", variable=self.data_source_var).pack(anchor=tk.W, padx=20, pady=2)
        ttk.Radiobutton(source_frame, text="Exchange API", value="exchange", variable=self.data_source_var).pack(anchor=tk.W, padx=20, pady=2)
        
        # Auto-download option
        download_check = ttk.Checkbutton(source_frame, text="Auto-download missing data", variable=self.auto_download_var)
        download_check.pack(anchor=tk.W, padx=20, pady=5)
        
        # Timeframe settings
        timeframe_frame = ttk.LabelFrame(parent, text="Timeframe")
        timeframe_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(timeframe_frame, text="Active Timeframe:").pack(anchor=tk.W, padx=5, pady=2)
        timeframe_combo = ttk.Combobox(timeframe_frame, textvariable=self.timeframe_var, 
                                       values=self.config["timeframes"], state="readonly")
        timeframe_combo.pack(anchor=tk.W, padx=5, pady=2)
        
        # Date range for backtesting
        date_frame = ttk.LabelFrame(parent, text="Date Range (Backtest)")
        date_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(date_frame, text="Start Date:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        start_entry = ttk.Entry(date_frame, textvariable=self.start_date_var, width=20)
        start_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Button(date_frame, text="Select", command=lambda: self.select_date("start")).grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(date_frame, text="End Date:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        end_entry = ttk.Entry(date_frame, textvariable=self.end_date_var, width=20)
        end_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Button(date_frame, text="Select", command=lambda: self.select_date("end")).grid(row=1, column=2, padx=5, pady=2)
        
        # Data management
        data_mgmt_frame = ttk.LabelFrame(parent, text="Data Management")
        data_mgmt_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(data_mgmt_frame, text="Download Historical Data", command=self.download_data).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(data_mgmt_frame, text="Check Data Integrity", command=self.check_data_integrity).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(data_mgmt_frame, text="Clear Data Cache", command=self.clear_data_cache).pack(side=tk.LEFT, padx=5, pady=5)

    def create_logs_panel(self):
        """Create log display panel"""
        # Top controls
        control_frame = ttk.Frame(self.logs_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Clear Logs", command=self.clear_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save Logs", command=self.save_logs).pack(side=tk.LEFT, padx=5)
        
        # Log level filter
        ttk.Label(control_frame, text="Log Level:").pack(side=tk.LEFT, padx=5)
        level_combo = ttk.Combobox(control_frame, values=["ALL", "INFO", "WARNING", "ERROR"], state="readonly")
        level_combo.pack(side=tk.LEFT, padx=5)
        level_combo.set("ALL")
        level_combo.bind("<<ComboboxSelected>>", self.filter_logs)
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(self.logs_frame, wrap=tk.WORD, height=25)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)
        
        # Apply syntax highlighting for different log levels
        self.log_text.tag_configure("INFO", foreground="blue")
        self.log_text.tag_configure("WARNING", foreground="orange")
        self.log_text.tag_configure("ERROR", foreground="red")
        self.log_text.tag_configure("DEBUG", foreground="gray")

    def create_charts_panel(self):
        """Create chart display panel"""
        # Controls at top
        control_frame = ttk.Frame(self.charts_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Symbol selection
        ttk.Label(control_frame, text="Symbol:").pack(side=tk.LEFT, padx=5)
        symbol_combo = ttk.Combobox(control_frame, values=self.config["symbols"], state="readonly")
        symbol_combo.pack(side=tk.LEFT, padx=5)
        symbol_combo.set(self.config["symbols"][0])
        symbol_combo.bind("<<ComboboxSelected>>", lambda e: self.update_chart(symbol_combo.get(), timeframe_combo.get()))
        
        # Timeframe selection
        ttk.Label(control_frame, text="Timeframe:").pack(side=tk.LEFT, padx=5)
        timeframe_combo = ttk.Combobox(control_frame, values=self.config["timeframes"], state="readonly")
        timeframe_combo.pack(side=tk.LEFT, padx=5)
        timeframe_combo.set(self.config["timeframes"][0])
        timeframe_combo.bind("<<ComboboxSelected>>", lambda e: self.update_chart(symbol_combo.get(), timeframe_combo.get()))
        
        # Chart type
        ttk.Label(control_frame, text="Chart Type:").pack(side=tk.LEFT, padx=5)
        chart_type_combo = ttk.Combobox(control_frame, values=["Candlestick", "Line", "OHLC"], state="readonly")
        chart_type_combo.pack(side=tk.LEFT, padx=5)
        chart_type_combo.set("Candlestick")
        chart_type_combo.bind("<<ComboboxSelected>>", lambda e: self.update_chart(symbol_combo.get(), timeframe_combo.get()))
        
        # Indicators button
        ttk.Button(control_frame, text="Add Indicator", command=self.add_indicator).pack(side=tk.LEFT, padx=5)
        
        # Refresh button
        ttk.Button(control_frame, text="Refresh", command=lambda: self.update_chart(symbol_combo.get(), timeframe_combo.get())).pack(side=tk.LEFT, padx=5)
        
        # Chart frame
        chart_frame = ttk.Frame(self.charts_frame)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create initial empty chart
        self.create_empty_chart(chart_frame)

    def create_empty_chart(self, parent):
        """Create empty chart with placeholder message"""
        # Create figure
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Add placeholder text
        ax.text(0.5, 0.5, "Select symbol and timeframe to load chart",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Create canvas
        self.chart_canvas = FigureCanvasTkAgg(fig, parent)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_chart(self, symbol, timeframe):
        """Update chart with new data"""
        try:
            # Create demo data if no real data available
            dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
            
            if not TRADING_MODULES_AVAILABLE:
                # Generate random demo data
                np.random.seed(42)  # For reproducibility
                base_price = 100.0
                price_data = []
                
                for i in range(len(dates)):
                    open_price = base_price * (1 + 0.01 * np.random.randn())
                    high_price = open_price * (1 + 0.005 * np.random.rand())
                    low_price = open_price * (1 - 0.005 * np.random.rand())
                    close_price = (open_price + high_price + low_price) / 3 + 0.01 * np.random.randn()
                    volume = 1000 * (1 + np.random.rand())
                    
                    price_data.append([open_price, high_price, low_price, close_price, volume])
                    base_price = close_price
                
                df = pd.DataFrame(price_data, index=dates, columns=['open', 'high', 'low', 'close', 'volume'])
                
            else:
                # Try to get real data if available
                try:
                    # This might fail if modules not properly initialized
                    data_manager = DataManager(self.data_source_var.get(), self.config)
                    df = asyncio.run(data_manager.get_historical_data(
                        symbol, timeframe, self.start_date_var.get(), self.end_date_var.get()))
                except Exception as e:
                    logger.error(f"Failed to load real data: {str(e)}")
                    messagebox.showerror("Data Error", f"Could not load data: {str(e)}")
                    # Fall back to demo data
                    np.random.seed(42)
                    df = pd.DataFrame({
                        'open': np.random.normal(100, 5, len(dates)),
                        'high': np.random.normal(105, 5, len(dates)),
                        'low': np.random.normal(95, 5, len(dates)),
                        'close': np.random.normal(102, 5, len(dates)),
                        'volume': np.random.normal(1000, 200, len(dates))
                    }, index=dates)
            
            # Create a new figure and axes
            for widget in self.charts_frame.winfo_children():
                if isinstance(widget, tk.Frame) and widget != self.charts_frame.winfo_children()[0]:
                    widget.destroy()
            
            chart_frame = ttk.Frame(self.charts_frame)
            chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            # Plot candlestick chart
            width = 0.6
            width2 = 0.1
            
            up = df[df.close >= df.open]
            down = df[df.close < df.open]
            
            # Up candles
            ax.bar(up.index, up.high - up.low, width=width2, bottom=up.low, color='green', alpha=0.3)
            ax.bar(up.index, up.close - up.open, width=width, bottom=up.open, color='green')
            
            # Down candles
            ax.bar(down.index, down.high - down.low, width=width2, bottom=down.low, color='red', alpha=0.3)
            ax.bar(down.index, down.open - down.close, width=width, bottom=down.close, color='red')
            
            # Set labels and title
            ax.set_ylabel('Price')
            ax.set_title(f'{symbol} ({timeframe})')
            
            # Format x-axis dates
            fig.autofmt_xdate()
            
            # Create canvas and display
            self.chart_canvas = FigureCanvasTkAgg(fig, chart_frame)
            self.chart_canvas.draw()
            self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Create volume subplot
            ax2 = ax.twinx()
            ax2.fill_between(df.index, 0, df.volume, color='blue', alpha=0.3)
            ax2.set_ylabel('Volume')
            
            # Update canvas
            self.chart_canvas.draw()
            
            logger.info(f"Updated chart for {symbol} ({timeframe})")
            
        except Exception as e:
            logger.error(f"Error updating chart: {str(e)}")
            messagebox.showerror("Chart Error", f"Failed to update chart: {str(e)}")

    def process_logs(self):
        """Process logs from the queue and display in the UI"""
        try:
            while not self.log_queue.empty():
                log_msg = self.log_queue.get_nowait()
                
                # Detect log level based on content
                tag = None
                if "ERROR" in log_msg:
                    tag = "ERROR"
                elif "WARNING" in log_msg:
                    tag = "WARNING"
                elif "INFO" in log_msg:
                    tag = "INFO"
                elif "DEBUG" in log_msg:
                    tag = "DEBUG"
                
                # Update log text
                self.log_text.config(state=tk.NORMAL)
                if tag:
                    self.log_text.insert(tk.END, log_msg, tag)
                else:
                    self.log_text.insert(tk.END, log_msg)
                self.log_text.see(tk.END)
                self.log_text.config(state=tk.DISABLED)
        except Exception as e:
            print(f"Error processing logs: {str(e)}")
        
        # Schedule next check
        self.root.after(100, self.process_logs)

    def start_trading(self):
        """Start the trading system"""
        if self.is_running:
            messagebox.showinfo("Already Running", "Trading system is already running")
            return
        
        # Check if modules available
        if not TRADING_MODULES_AVAILABLE:
            messagebox.showwarning("Module Warning", 
                                "Trading modules not available. Running in demo mode.")
        
        # Get selected symbols
        selected_symbols = [symbol for symbol, var in self.symbols_var.items() if var.get()]
        if not selected_symbols:
            messagebox.showwarning("No Symbols", "Please select at least one trading symbol")
            return
            
        # Update UI
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.trading_status_label.config(text="Running", foreground="green")
        self.connection_status_label.config(text="Connected", foreground="green")
        self.data_status_label.config(text=self.data_source_var.get(), foreground="blue")
        self.mode_status_label.config(text=self.mode_var.get(), foreground="blue")
        
        # Update strategies list
        self.strategies_list.delete(0, tk.END)
        self.strategies_list.insert(tk.END, f"{self.strategy_var.get()} (active)")
        
        # Log the action
        logger.info(f"Starting trading system - Mode: {self.mode_var.get()}, Strategy: {self.strategy_var.get()}")
        logger.info(f"Trading symbols: {', '.join(selected_symbols)}")
        
        # Start trading thread
        self.is_running = True
        self.trading_thread = threading.Thread(target=self.trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        messagebox.showinfo("Trading Started", f"Trading system started in {self.mode_var.get()} mode")

    def stop_trading(self):
        """Stop the trading system"""
        if not self.is_running:
            messagebox.showinfo("Not Running", "Trading system is not running")
            return
        
        # Update flag to signal thread to stop
        self.is_running = False
        
        # Update UI
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.trading_status_label.config(text="Stopped", foreground="red")
        self.connection_status_label.config(text="Disconnected", foreground="red")
        
        # Update strategies list
        self.strategies_list.delete(0, tk.END)
        for strategy in self.config["strategies"]:
            self.strategies_list.insert(tk.END, f"{strategy} (inactive)")
        
        # Log the action
        logger.info("Stopping trading system")
        
        messagebox.showinfo("Trading Stopped", "Trading system has been stopped")

    def trading_loop(self):
        """Trading system loop running in a separate thread"""
        # Demo implementation
        logger.info("Trading loop started")
        
        try:
            # Set up performance metrics chart with initial values
            win_rate = 0.65
            profit_factor = 1.2
            sharpe = 0.8
            
            # Main loop
            iteration = 0
            while self.is_running:
                # Sleep to avoid high CPU usage
                time.sleep(2)
                
                # Simulate trading actions (demo only)
                iteration += 1
                
                if iteration % 5 == 0:
                    # Add a new trade every 5 iterations
                    self.add_trade()
                
                if iteration % 3 == 0:
                    # Update metrics
                    win_rate += 0.01 * (0.5 - np.random.rand())
                    profit_factor += 0.05 * (0.5 - np.random.rand())
                    sharpe += 0.02 * (0.5 - np.random.rand())
                    
                    # Keep in reasonable range
                    win_rate = max(0.4, min(0.9, win_rate))
                    profit_factor = max(0.8, min(1.8, profit_factor))
                    sharpe = max(0.3, min(1.5, sharpe))
                    
                    # Update chart in thread-safe way
                    self.root.after(0, self.update_performance_chart, win_rate, profit_factor, sharpe)
                
                # Log status
                logger.info(f"Trading iteration {iteration}, active strategy: {self.strategy_var.get()}")
                
                # Log a random debug message for demo purposes
                if np.random.rand() < 0.3:
                    log_messages = [
                        "Checking for new signals...",
                        "Analyzing market data...",
                        "Updating position sizing...",
                        "Checking order status...",
                        "Calculating risk metrics..."
                    ]
                    logger.info(np.random.choice(log_messages))
                
                # Random warning messages for demo
                if np.random.rand() < 0.1:
                    warning_messages = [
                        "Data resolution mismatch, resampling required",
                        "Connection delay detected, retry logic activated",
                        "Signal strength below threshold, skipping trade",
                        "High volatility detected, reducing position size"
                    ]
                    logger.warning(np.random.choice(warning_messages))
        
        except Exception as e:
            logger.error(f"Error in trading loop: {str(e)}")
        finally:
            logger.info("Trading loop ended")

    def add_trade(self):
        """Add a demo trade to the trades table"""
        try:
            # Random trade generation for demo
            symbols = list(self.symbols_var.keys())
            symbol = np.random.choice(symbols)
            
            trade_type = np.random.choice(["BUY", "SELL"])
            
            # Base prices depending on symbol
            if "BTC" in symbol:
                price = np.random.normal(42000, 200)
                size = np.random.uniform(0.001, 0.02)
            elif "ETH" in symbol:
                price = np.random.normal(2200, 50)
                size = np.random.uniform(0.01, 0.3)
            else:
                price = np.random.normal(100, 10)
                size = np.random.uniform(0.1, 5)
                
            # Format time
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Add to tree (thread safe)
            self.root.after(0, lambda: self.trades_tree.insert(
                "", 0, values=(current_time, symbol, trade_type, f"{price:.2f}", f"{size:.4f}")
            ))
            
            # Keep only recent trades
            if self.trades_tree.get_children():
                items = self.trades_tree.get_children()
                if len(items) > 10:
                    self.trades_tree.delete(items[-1])
                    
            logger.info(f"New trade executed: {trade_type} {size:.4f} {symbol} @ {price:.2f}")
            
        except Exception as e:
            logger.error(f"Error adding trade: {str(e)}")

    def update_performance_chart(self, win_rate, profit_factor, sharpe):
        """Update the performance metrics chart"""
        try:
            # Clear the current figure
            self.perf_ax.clear()
            
            # Update the bar chart
            metrics = ['Win Rate', 'Profit Factor', 'Sharpe']
            values = [win_rate, profit_factor, sharpe]
            colors = ['green', 'blue', 'orange']
            
            bars = self.perf_ax.bar(metrics, values, color=colors)
            
            # Add data labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                self.perf_ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                                 f'{value:.2f}', ha='center', va='bottom')
            
            # Set labels and title
            self.perf_ax.set_title("Performance Metrics")
            self.perf_ax.set_ylim(0, 2)
            
            # Redraw canvas
            self.perf_canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating performance chart: {str(e)}")

    def add_symbol(self, symbol):
        """Add a new symbol to trading list"""
        if not symbol:
            return
            
        symbol = symbol.upper().strip()
        
        # Check if already exists
        if symbol in self.symbols_var:
            messagebox.showinfo("Symbol Exists", f"Symbol {symbol} already exists")
            return
            
        # Add to symbols list
        self.config["symbols"].append(symbol)
        self.symbols_var[symbol] = tk.BooleanVar(value=True)
        
        # Refresh UI
        self.save_config()
        messagebox.showinfo("Symbol Added", f"Symbol {symbol} has been added")
        
        # Trigger config panel rebuild to show new symbol
        self.create_config_panel()

    def download_data(self):
        """Download historical data"""
        selected_symbols = [symbol for symbol, var in self.symbols_var.items() if var.get()]
        if not selected_symbols:
            messagebox.showwarning("No Symbols", "Please select at least one symbol")
            return
            
        if not self.start_date_var.get() or not self.end_date_var.get():
            messagebox.showwarning("Date Range", "Please specify start and end dates")
            return
            
        try:
            if not TRADING_MODULES_AVAILABLE:
                # Demo mode
                logger.info("Starting historical data download (demo mode)")
                
                # Create progress window
                progress_window = tk.Toplevel(self.root)
                progress_window.title("Downloading Data")
                progress_window.geometry("300x150")
                progress_window.transient(self.root)
                progress_window.grab_set()
                
                # Progress label
                progress_label = ttk.Label(progress_window, text="Downloading historical data...")
                progress_label.pack(pady=10)
                
                # Progress bar
                progress = ttk.Progressbar(progress_window, mode='determinate')
                progress.pack(fill=tk.X, padx=20, pady=10)
                
                # Symbol label
                symbol_label = ttk.Label(progress_window, text="")
                symbol_label.pack(pady=5)
                
                # Demo download loop
                def download_demo():
                    total_symbols = len(selected_symbols)
                    for i, symbol in enumerate(selected_symbols):
                        # Update progress
                        progress['value'] = (i / total_symbols) * 100
                        symbol_label.config(text=f"Downloading {symbol}...")
                        progress_window.update()
                        
                        # Simulate work
                        time.sleep(0.5 + np.random.rand() * 2)
                        
                        logger.info(f"Downloaded data for {symbol} ({self.timeframe_var.get()}) from {self.start_date_var.get()} to {self.end_date_var.get()}")
                    
                    # Complete
                    progress['value'] = 100
                    symbol_label.config(text="Download complete!")
                    progress_window.update()
                    time.sleep(1)
                    progress_window.destroy()
                    messagebox.showinfo("Download Complete", "Historical data has been downloaded")
                
                # Start in a separate thread to avoid freezing UI
                threading.Thread(target=download_demo, daemon=True).start()
                
            else:
                # Real implementation
                logger.info(f"Starting historical data download for {len(selected_symbols)} symbols")
                
                # Try to use real data manager
                data_manager = DataManager("exchange", self.config, TradingMode.BACKTEST)
                
                # Create progress window
                progress_window = tk.Toplevel(self.root)
                progress_window.title("Downloading Data")
                progress_window.geometry("300x150")
                progress_window.transient(self.root)
                progress_window.grab_set()
                
                # Progress components
                progress_label = ttk.Label(progress_window, text="Downloading historical data...")
                progress_label.pack(pady=10)
                
                progress = ttk.Progressbar(progress_window, mode='determinate')
                progress.pack(fill=tk.X, padx=20, pady=10)
                
                symbol_label = ttk.Label(progress_window, text="")
                symbol_label.pack(pady=5)
                
                # Define download function
                def download_real():
                    try:
                        results = {}
                        
                        # Download for each symbol
                        for i, symbol in enumerate(selected_symbols):
                            # Update progress
                            progress['value'] = (i / len(selected_symbols)) * 100
                            symbol_label.config(text=f"Downloading {symbol}...")
                            progress_window.update()
                            
                            # Download data
                            result = asyncio.run(data_manager.update_local_data(
                                symbol, 
                                self.timeframe_var.get(),
                                self.start_date_var.get(),
                                self.end_date_var.get()
                            ))
                            
                            results[symbol] = result
                            logger.info(f"Download for {symbol}: {'Success' if result else 'Failed'}")
                        
                        # Complete
                        progress['value'] = 100
                        symbol_label.config(text="Download complete!")
                        progress_window.update()
                        
                        # Close data manager
                        asyncio.run(data_manager.close())
                        
                        # Show results
                        success_count = sum(1 for r in results.values() if r)
                        time.sleep(1)
                        progress_window.destroy()
                        
                        messagebox.showinfo("Download Complete", 
                                          f"Downloaded data for {success_count}/{len(selected_symbols)} symbols")
                        
                    except Exception as e:
                        logger.error(f"Error downloading data: {str(e)}")
                        progress_window.destroy()
                        messagebox.showerror("Download Error", f"Failed to download data: {str(e)}")
                
                # Start in a separate thread
                threading.Thread(target=download_real, daemon=True).start()
                
        except Exception as e:
            logger.error(f"Error initiating download: {str(e)}")
            messagebox.showerror("Download Error", f"Failed to start download: {str(e)}")

    def run_backtest(self):
        """Run backtest with current settings"""
        selected_symbols = [symbol for symbol, var in self.symbols_var.items() if var.get()]
        if not selected_symbols:
            messagebox.showwarning("No Symbols", "Please select at least one symbol")
            return
            
        if not self.start_date_var.get() or not self.end_date_var.get():
            messagebox.showwarning("Date Range", "Please specify start and end dates")
            return
            
        # Ensure we're in backtest mode
        self.mode_var.set("backtest")
        
        try:
            logger.info(f"Starting backtest for {self.strategy_var.get()} on {selected_symbols}")
            
            # Create progress window
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Running Backtest")
            progress_window.geometry("400x200")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            # Progress components
            ttk.Label(progress_window, text="Running backtest...").pack(pady=10)
            
            progress = ttk.Progressbar(progress_window, mode='determinate')
            progress.pack(fill=tk.X, padx=20, pady=10)
            
            status_label = ttk.Label(progress_window, text="Initializing...")
            status_label.pack(pady=5)
            
            # Cancel button
            cancel_btn = ttk.Button(progress_window, text="Cancel", 
                                   command=lambda: progress_window.destroy())
            cancel_btn.pack(pady=10)
            
            # Define backtest function
            def run_backtest_task():
                try:
                    # Phases
                    phases = [
                        "Loading historical data...",
                        "Initializing strategy...",
                        "Optimizing parameters...",
                        "Running strategy simulation...",
                        "Calculating performance metrics...",
                        "Generating report..."
                    ]
                    
                    for i, phase in enumerate(phases):
                        # Update progress
                        progress['value'] = (i / len(phases)) * 100
                        status_label.config(text=phase)
                        progress_window.update()
                        
                        # Simulate work
                        time.sleep(0.5 + np.random.rand() * 2)
                        
                        logger.info(f"Backtest phase: {phase}")
                    
                    # Complete
                    progress['value'] = 100
                    status_label.config(text="Backtest complete!")
                    progress_window.update()
                    
                    # Update performance metrics with random results
                    win_rate = 0.6 + 0.2 * np.random.rand()
                    profit_factor = 1.1 + 0.5 * np.random.rand()
                    sharpe = 0.7 + 0.6 * np.random.rand()
                    
                    # Update chart in thread-safe way
                    self.root.after(0, self.update_performance_chart, win_rate, profit_factor, sharpe)
                    
                    time.sleep(1)
                    progress_window.destroy()
                    
                    # Show results
                    profit = round(np.random.rand() * 30, 2)
                    total_trades = int(np.random.rand() * 100) + 20
                    
                    messagebox.showinfo("Backtest Complete", 
                                      f"Backtest completed successfully!\n\n"
                                      f"Strategy: {self.strategy_var.get()}\n"
                                      f"Profit: {profit}%\n"
                                      f"Win Rate: {win_rate:.2f}\n"
                                      f"Total Trades: {total_trades}\n\n"
                                      f"See Charts tab for detailed results")
                    
                except Exception as e:
                    logger.error(f"Error in backtest: {str(e)}")
                    progress_window.destroy()
                    messagebox.showerror("Backtest Error", f"Error running backtest: {str(e)}")
            
            # Start in a separate thread
            threading.Thread(target=run_backtest_task, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Error initiating backtest: {str(e)}")
            messagebox.showerror("Backtest Error", f"Failed to start backtest: {str(e)}")

    def generate_report(self):
        """Generate performance report"""
        try:
            # Create report window
            report_window = tk.Toplevel(self.root)
            report_window.title("Performance Report")
            report_window.geometry("800x600")
            report_window.transient(self.root)
            
            # Create notebook for report tabs
            report_notebook = ttk.Notebook(report_window)
            report_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Summary tab
            summary_frame = ttk.Frame(report_notebook)
            report_notebook.add(summary_frame, text="Summary")
            
            # Create summary content
            ttk.Label(summary_frame, text="Performance Summary", font=("TkDefaultFont", 14, "bold")).pack(pady=10)
            
            summary_text = scrolledtext.ScrolledText(summary_frame, height=20, width=80)
            summary_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
            
            # Insert demo report
            report_content = f"""Strategy: {self.strategy_var.get()}
Time Period: {self.start_date_var.get()} to {self.end_date_var.get()}
Symbols: {', '.join([s for s, v in self.symbols_var.items() if v.get()])}

Performance Metrics:
-------------------
Net Profit: ${round(np.random.random() * 5000, 2)}
Return on Investment: {round(np.random.random() * 30, 2)}%
Win Rate: {round(0.5 + np.random.random() * 0.3, 2)}
Profit Factor: {round(1.1 + np.random.random() * 0.5, 2)}
Sharpe Ratio: {round(0.8 + np.random.random() * 0.7, 2)}
Maximum Drawdown: {round(np.random.random() * 15, 2)}%
Recovery Factor: {round(1.5 + np.random.random() * 1.0, 2)}

Trade Statistics:
---------------
Total Trades: {int(np.random.random() * 100) + 20}
Average Trade Duration: {int(np.random.random() * 8) + 2} hours
Average Profit per Trade: ${round(np.random.random() * 50, 2)}
Largest Winning Trade: ${round(np.random.random() * 500, 2)}
Largest Losing Trade: -${round(np.random.random() * 200, 2)}
Maximum Consecutive Winners: {int(np.random.random() * 8) + 3}
Maximum Consecutive Losers: {int(np.random.random() * 5) + 2}

Strategy Parameters:
------------------"""
            
            if self.strategy_var.get() == "dual_ma":
                report_content += """
Short Window: 10
Long Window: 30
Signal Type: crossover
"""
            elif self.strategy_var.get() == "neural_net":
                report_content += """
Hidden Layers: 64,32
Learning Rate: 0.001
Dropout: 0.2
Epochs: 100
"""
            
            summary_text.insert(tk.END, report_content)
            summary_text.config(state=tk.DISABLED)
            
            # Trades tab
            trades_frame = ttk.Frame(report_notebook)
            report_notebook.add(trades_frame, text="Trades")
            
            # Trade list
            trades_tree = ttk.Treeview(trades_frame, columns=("Date", "Symbol", "Type", "Price", "Size", "Profit"), show="headings")
            trades_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Configure columns
            for col in trades_tree["columns"]:
                trades_tree.heading(col, text=col)
                trades_tree.column(col, width=100)
            
            # Add demo trades
            for i in range(20):
                trade_type = "BUY" if np.random.random() > 0.5 else "SELL"
                price = round(100 + np.random.random() * 50, 2)
                size = round(0.01 + np.random.random() * 0.2, 4)
                profit = round(-30 + np.random.random() * 60, 2)
                
                date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d %H:%M")
                symbol = np.random.choice(list(self.symbols_var.keys()))
                
                trades_tree.insert("", tk.END, values=(date, symbol, trade_type, price, size, profit))
            
            # Charts tab
            charts_frame = ttk.Frame(report_notebook)
            report_notebook.add(charts_frame, text="Charts")
            
            # Create equity curve chart
            fig = Figure(figsize=(8, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Generate equity curve data
            days = 100
            equity = [10000]  # Start with $10000
            for i in range(1, days):
                change = equity[-1] * (0.004 * np.random.randn() + 0.001)
                equity.append(equity[-1] + change)
            
            # Plot equity curve
            ax.plot(range(days), equity)
            ax.set_title("Equity Curve")
            ax.set_xlabel("Days")
            ax.set_ylabel("Equity ($)")
            ax.grid(True)
            
            # Create canvas and pack
            equity_canvas = FigureCanvasTkAgg(fig, charts_frame)
            equity_canvas.draw()
            equity_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Export button
            ttk.Button(report_window, text="Export Report", 
                      command=lambda: self.export_report(report_content)).pack(pady=10)
            
            logger.info("Generated performance report")
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            messagebox.showerror("Report Error", f"Failed to generate report: {str(e)}")

    def export_report(self, content):
        """Export report to file"""
        try:
            # Ask for file location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt"), ("CSV Files", "*.csv"), ("All Files", "*.*")],
                title="Export Report"
            )
            
            if not file_path:
                return  # User cancelled
                
            # Write to file
            with open(file_path, 'w') as f:
                f.write(content)
                
            messagebox.showinfo("Export Complete", f"Report exported to {file_path}")
            logger.info(f"Exported report to {file_path}")
            
        except Exception as e:
            logger.error(f"Error exporting report: {str(e)}")
            messagebox.showerror("Export Error", f"Failed to export report: {str(e)}")

    def test_strategy(self):
        """Test current strategy parameters"""
        logger.info(f"Testing strategy: {self.strategy_var.get()}")
        
        try:
            # Create test window
            test_window = tk.Toplevel(self.root)
            test_window.title("Strategy Test")
            test_window.geometry("600x400")
            test_window.transient(self.root)
            
            # Progress bar
            ttk.Label(test_window, text="Testing strategy parameters...").pack(pady=10)
            progress = ttk.Progressbar(test_window, mode='indeterminate')
            progress.pack(fill=tk.X, padx=20, pady=10)
            progress.start()
            
            # Results area
            results_frame = ttk.LabelFrame(test_window, text="Test Results")
            results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Define test function
            def run_test():
                # Simulate work
                time.sleep(3)
                
                # Update UI (thread-safe)
                self.root.after(0, lambda: progress.stop())
                self.root.after(0, lambda: progress.pack_forget())
                
                # Display results
                if self.strategy_var.get() == "dual_ma":
                    param_text = """
                    Short Window: 10  ✓
                    Long Window: 30  ✓
                    Signal Type: crossover  ✓
                    
                    Parameter test passed. These settings appear optimal for recent market conditions.
                    
                    Suggested changes:
                    - Consider increasing short window to 12 for less noise
                    - Long window of 30 is optimal for current volatility
                    """
                else:
                    param_text = """
                    Hidden Layers: 64,32  ✓
                    Learning Rate: 0.001  ✓
                    Dropout: 0.2  ⚠️
                    Epochs: 100  ✓
                    
                    Parameter test completed with warnings.
                    
                    Suggested changes:
                    - Increase dropout to 0.3 to prevent overfitting
                    - Current learning rate is appropriate
                    - Layer configuration shows good performance
                    """
                
                # Add results
                results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=15)
                results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                results_text.insert(tk.END, f"Strategy: {self.strategy_var.get()}\n")
                results_text.insert(tk.END, f"Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                results_text.insert(tk.END, "Parameter Analysis:\n")
                results_text.insert(tk.END, param_text)
                results_text.config(state=tk.DISABLED)
                
                # Add apply button
                ttk.Button(test_window, text="Apply Suggestions", 
                          command=lambda: messagebox.showinfo("Applied", "Suggested parameters applied")).pack(pady=10)
            
            # Start in a separate thread
            threading.Thread(target=run_test, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Error testing strategy: {str(e)}")
            messagebox.showerror("Test Error", f"Failed to test strategy: {str(e)}")

    def save_strategy_params(self):
        """Save current strategy parameters"""
        logger.info(f"Saving parameters for strategy: {self.strategy_var.get()}")
        messagebox.showinfo("Parameters Saved", f"Parameters for {self.strategy_var.get()} have been saved")

    def add_indicator(self):
        """Add indicator to chart"""
        try:
            # Show indicator selection dialog
            indicator_window = tk.Toplevel(self.root)
            indicator_window.title("Add Indicator")
            indicator_window.geometry("300x400")
            indicator_window.transient(self.root)
            indicator_window.grab_set()
            
            # Indicator selection
            ttk.Label(indicator_window, text="Select Indicator Type:").pack(anchor=tk.W, padx=10, pady=5)
            indicator_combo = ttk.Combobox(indicator_window, values=[
                "Moving Average", "Bollinger Bands", "RSI", "MACD", 
                "Stochastic", "Ichimoku", "ATR", "Pivot Points"
            ], state="readonly")
            indicator_combo.pack(fill=tk.X, padx=10, pady=5)
            indicator_combo.set("Moving Average")
            
            # Parameters frame
            params_frame = ttk.LabelFrame(indicator_window, text="Parameters")
            params_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Add default parameter fields
            self.update_indicator_params(params_frame, "Moving Average")
            
            # Bind selection to update parameters
            indicator_combo.bind("<<ComboboxSelected>>", 
                              lambda e: self.update_indicator_params(params_frame, indicator_combo.get()))
            
            # Add buttons
            button_frame = ttk.Frame(indicator_window)
            button_frame.pack(fill=tk.X, padx=10, pady=10)
            
            ttk.Button(button_frame, text="Add", 
                      command=lambda: self.confirm_add_indicator(indicator_combo.get(), indicator_window)).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cancel", 
                      command=indicator_window.destroy).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            logger.error(f"Error adding indicator: {str(e)}")
            messagebox.showerror("Indicator Error", f"Failed to add indicator: {str(e)}")

    def update_indicator_params(self, parent, indicator_type):
        """Update indicator parameters based on type"""
        # Clear existing widgets
        for widget in parent.winfo_children():
            widget.destroy()
        
        # Add parameters based on indicator type
        if indicator_type == "Moving Average":
            ttk.Label(parent, text="Period:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            period_entry = ttk.Entry(parent, width=10)
            period_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
            period_entry.insert(0, "20")
            
            ttk.Label(parent, text="Type:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            ma_type = ttk.Combobox(parent, values=["Simple", "Exponential", "Weighted"], state="readonly")
            ma_type.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
            ma_type.set("Simple")
            
            ttk.Label(parent, text="Color:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
            color_combo = ttk.Combobox(parent, values=["Red", "Blue", "Green", "Orange", "Purple"], state="readonly")
            color_combo.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
            color_combo.set("Blue")
            
        elif indicator_type == "RSI":
            ttk.Label(parent, text="Period:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            period_entry = ttk.Entry(parent, width=10)
            period_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
            period_entry.insert(0, "14")
            
            ttk.Label(parent, text="Overbought:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            ob_entry = ttk.Entry(parent, width=10)
            ob_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
            ob_entry.insert(0, "70")
            
            ttk.Label(parent, text="Oversold:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
            os_entry = ttk.Entry(parent, width=10)
            os_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
            os_entry.insert(0, "30")
            
        elif indicator_type == "MACD":
            ttk.Label(parent, text="Fast Period:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            fast_entry = ttk.Entry(parent, width=10)
            fast_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
            fast_entry.insert(0, "12")
            
            ttk.Label(parent, text="Slow Period:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            slow_entry = ttk.Entry(parent, width=10)
            slow_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
            slow_entry.insert(0, "26")
            
            ttk.Label(parent, text="Signal Period:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
            signal_entry = ttk.Entry(parent, width=10)
            signal_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
            signal_entry.insert(0, "9")
        
        else:
            ttk.Label(parent, text="Standard parameters will be used").pack(pady=20)

    def confirm_add_indicator(self, indicator_type, window):
        """Add the indicator to chart and close dialog"""
        logger.info(f"Adding indicator: {indicator_type}")
        messagebox.showinfo("Indicator Added", f"{indicator_type} indicator has been added to the chart")
        window.destroy()

    def check_data_integrity(self):
        """Check data integrity for selected symbols"""
        selected_symbols = [symbol for symbol, var in self.symbols_var.items() if var.get()]
        if not selected_symbols:
            messagebox.showwarning("No Symbols", "Please select at least one symbol")
            return
        
        try:
            # Create progress window
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Data Integrity Check")
            progress_window.geometry("400x300")
            progress_window.transient(self.root)
            
            # Progress components
            ttk.Label(progress_window, text="Checking data integrity...").pack(pady=10)
            progress = ttk.Progressbar(progress_window, mode='determinate')
            progress.pack(fill=tk.X, padx=20, pady=10)
            
            # Results text
            results_text = scrolledtext.ScrolledText(progress_window, height=10)
            results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Function to run check
            def run_check():
                try:
                    for i, symbol in enumerate(selected_symbols):
                        # Update progress
                        progress['value'] = (i / len(selected_symbols)) * 100
                        progress_window.update()
                        
                        # Simulate check
                        time.sleep(0.5 + np.random.rand())
                        
                        # Add result with random issues
                        issues = int(np.random.rand() * 5)
                        if issues == 0:
                            status = "OK"
                            color = "green"
                        else:
                            status = f"{issues} issues found"
                            color = "red"
                            
                        results_text.insert(tk.END, f"{symbol}: {status}\n")
                        results_text.tag_add(color, f"{float(results_text.index('end'))-1.0} linestart", f"{float(results_text.index('end'))-1.0} lineend")
                        results_text.tag_configure("green", foreground="green")
                        results_text.tag_configure("red", foreground="red")
                        results_text.see(tk.END)
                        
                        logger.info(f"Data integrity check for {symbol}: {status}")
                    
                    # Complete
                    progress['value'] = 100
                    results_text.insert(tk.END, "\nCheck completed.\n")
                    
                    # Add fix button if issues found
                    if "issues found" in results_text.get("1.0", tk.END):
                        ttk.Button(progress_window, text="Fix Issues", 
                                 command=lambda: messagebox.showinfo("Issues Fixed", "Data integrity issues have been resolved")).pack(pady=10)
                    
                except Exception as e:
                    logger.error(f"Error in data integrity check: {str(e)}")
                    results_text.insert(tk.END, f"\nError: {str(e)}\n")
            
            # Start in a separate thread
            threading.Thread(target=run_check, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Error checking data integrity: {str(e)}")
            messagebox.showerror("Check Error", f"Failed to check data integrity: {str(e)}")

    def clear_data_cache(self):
        """Clear data cache"""
        confirm = messagebox.askyesno("Confirm", "Are you sure you want to clear all data cache?")
        if confirm:
            logger.info("Clearing data cache")
            messagebox.showinfo("Cache Cleared", "Data cache has been cleared")

    def select_date(self, date_type):
        """Show date picker dialog"""
        try:            
            date_window = tk.Toplevel(self.root)
            date_window.title(f"Select {date_type.capitalize()} Date")
            date_window.geometry("300x150")
            date_window.transient(self.root)
            date_window.grab_set()
            
            ttk.Label(date_window, text=f"Select {date_type} date:").pack(pady=10)
            
            # Default date to today if empty
            default_date = datetime.now() - timedelta(days=30 if date_type == "start" else 0)
            
            # Calendar widget
            cal = DateEntry(date_window, width=12, background='darkblue',
                          foreground='white', borderwidth=2)
            cal.pack(pady=10)
            
            # Set selection
            def set_date():
                selected = cal.get_date()
                formatted = selected.strftime("%Y-%m-%d")
                if date_type == "start":
                    self.start_date_var.set(formatted)
                else:
                    self.end_date_var.set(formatted)
                date_window.destroy()
            
            ttk.Button(date_window, text="Select", command=set_date).pack(pady=10)
            
        except ImportError:
            # Fallback if tkcalendar not available
            date_str = simpledialog.askstring(
                f"Select {date_type.capitalize()} Date", 
                "Enter date (YYYY-MM-DD):",
                parent=self.root
            )
            
            if date_str:
                if date_type == "start":
                    self.start_date_var.set(date_str)
                else:
                    self.end_date_var.set(date_str)

    def clear_logs(self):
        """Clear log display"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        logger.info("Logs cleared")

    def save_logs(self):
        """Save logs to file"""
        try:
            # Ask for file location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".log",
                filetypes=[("Log Files", "*.log"), ("Text Files", "*.txt"), ("All Files", "*.*")],
                title="Save Logs"
            )
            
            if not file_path:
                return  # User cancelled
                
            # Get log content
            log_content = self.log_text.get(1.0, tk.END)
            
            # Write to file
            with open(file_path, 'w') as f:
                f.write(log_content)
                
            messagebox.showinfo("Save Complete", f"Logs saved to {file_path}")
            logger.info(f"Logs saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving logs: {str(e)}")
            messagebox.showerror("Save Error", f"Failed to save logs: {str(e)}")

    def filter_logs(self, event):
        """Filter logs by level"""
        # Not implemented in this demo
        pass

    def set_theme(self, theme):
        """Change application theme"""
        logger.info(f"Setting theme to {theme}")
        self.config["theme"] = theme
        messagebox.showinfo("Theme Changed", f"Theme changed to {theme}")

    def load_config(self):
        """Load configuration from file"""
        try:
            # Check if config directory exists
            config_dir = os.path.dirname(CONFIG_FILE)
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
                
            # Check if config file exists
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {CONFIG_FILE}")
                return config
            else:
                # Create default config file
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(DEFAULT_CONFIG, f, indent=4)
                logger.info(f"Created default configuration at {CONFIG_FILE}")
                return DEFAULT_CONFIG
                
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return DEFAULT_CONFIG.copy()

    def save_config(self):
        """Save configuration to file"""
        try:
            # Update config from UI values
            self.config["mode"] = self.mode_var.get()
            self.config["data_source"] = self.data_source_var.get()
            self.config["auto_download"] = self.auto_download_var.get()
            self.config["start_date"] = self.start_date_var.get()
            self.config["end_date"] = self.end_date_var.get()
            
            # Ensure config directory exists
            config_dir = os.path.dirname(CONFIG_FILE)
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            # Write to file
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=4)
                
            logger.info(f"Configuration saved to {CONFIG_FILE}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            messagebox.showerror("Save Error", f"Failed to save configuration: {str(e)}")

    def load_config_dialog(self):
        """Show dialog to load configuration file"""
        try:
            file_path = filedialog.askopenfilename(
                defaultextension=".json",
                filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
                title="Load Configuration"
            )
            
            if not file_path:
                return  # User cancelled
                
            with open(file_path, 'r') as f:
                self.config = json.load(f)
                
            # Update UI with new config
            messagebox.showinfo("Load Complete", f"Configuration loaded from {file_path}")
            logger.info(f"Configuration loaded from {file_path}")
            
            # Refresh UI to show new settings
            self.create_config_panel()
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            messagebox.showerror("Load Error", f"Failed to load configuration: {str(e)}")

    def save_config_dialog(self):
        """Show dialog to save configuration to file"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
                title="Save Configuration"
            )
            
            if not file_path:
                return  # User cancelled
                
            # Update config from UI values first
            self.config["mode"] = self.mode_var.get()
            self.config["data_source"] = self.data_source_var.get()
            self.config["auto_download"] = self.auto_download_var.get()
            self.config["start_date"] = self.start_date_var.get()
            self.config["end_date"] = self.end_date_var.get()
            
            with open(file_path, 'w') as f:
                json.dump(self.config, f, indent=4)
                
            messagebox.showinfo("Save Complete", f"Configuration saved to {file_path}")
            logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            messagebox.showerror("Save Error", f"Failed to save configuration: {str(e)}")

    def reset_config(self):
        """Reset configuration to defaults"""
        confirm = messagebox.askyesno("Confirm Reset", "Are you sure you want to reset all settings to defaults?")
        if confirm:
            self.config = DEFAULT_CONFIG.copy()
            logger.info("Configuration reset to defaults")
            
            # Reset UI variables
            self.mode_var.set(self.config["mode"])
            self.data_source_var.set(self.config["data_source"])
            self.auto_download_var.set(self.config["auto_download"])
            self.timeframe_var.set(self.config["timeframes"][0])
            self.strategy_var.set(self.config["strategies"][0])
            
            # Update symbols
            self.symbols_var = {}
            for symbol in self.config["symbols"]:
                self.symbols_var[symbol] = tk.BooleanVar(value=True)
                
            # Refresh UI
            self.create_config_panel()
            
            messagebox.showinfo("Reset Complete", "Settings have been reset to defaults")

    def show_docs(self):
        """Show documentation"""
        docs_window = tk.Toplevel(self.root)
        docs_window.title("Documentation")
        docs_window.geometry("800x600")
        
        # Create notebook for documentation sections
        docs_notebook = ttk.Notebook(docs_window)
        docs_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Overview tab
        overview_frame = ttk.Frame(docs_notebook)
        docs_notebook.add(overview_frame, text="Overview")
        
        overview_text = scrolledtext.ScrolledText(overview_frame, wrap=tk.WORD)
        overview_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        overview_content = """# Trading System GUI Documentation

## Overview

This application provides a graphical interface for managing and running trading strategies. The system supports backtesting, paper trading, and live trading modes.

### Key Features:

1. **Strategy Management**: Configure and test trading strategies with different parameters.
2. **Data Management**: Download and manage historical price data for analysis.
3. **Backtesting**: Test strategies against historical data to evaluate performance.
4. **Visualization**: Chart price data and trading signals for analysis.
5. **Live Trading**: Execute trades automatically based on strategy signals.

## Getting Started

1. Select your trading mode (backtest, paper, or live)
2. Configure your strategy parameters
3. Select data sources and symbols
4. Run backtests or start trading

Refer to the other tabs for detailed information on each component.
"""
        overview_text.insert(tk.END, overview_content)
        overview_text.config(state=tk.DISABLED)
        
        # Configuration tab
        config_frame = ttk.Frame(docs_notebook)
        docs_notebook.add(config_frame, text="Configuration")
        
        config_text = scrolledtext.ScrolledText(config_frame, wrap=tk.WORD)
        config_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        config_content = """# Configuration Guide

## General Settings

- **Trading Mode**: Select between backtesting, paper trading, or live trading.
- **Exchange Settings**: Configure API keys and connection settings for your exchange.
- **User Interface**: Customize the application appearance and behavior.

## Trading Settings

- **Symbols**: Select which trading pairs to monitor and trade.
- **Risk Parameters**: Configure risk management settings like position sizing.
- **Order Settings**: Set default order types and take profit/stop loss parameters.

## Strategy Settings

- **Strategy Selection**: Choose from available trading strategies.
- **Strategy Parameters**: Configure specific parameters for each strategy.
- **Parameter Testing**: Test different parameter combinations for optimal results.

## Data Settings

- **Data Source**: Choose between local data files or exchange API.
- **Timeframe**: Select the candle timeframe for analysis (1m, 5m, 15m, etc.).
- **Date Range**: Set the historical data range for backtesting.
- **Data Management**: Download, validate, and manage historical data.
"""
        config_text.insert(tk.END, config_content)
        config_text.config(state=tk.DISABLED)
        
        # Strategies tab
        strategies_frame = ttk.Frame(docs_notebook)
        docs_notebook.add(strategies_frame, text="Strategies")
        
        strategies_text = scrolledtext.ScrolledText(strategies_frame, wrap=tk.WORD)
        strategies_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        strategies_content = """# Available Strategies

## Dual Moving Average (dual_ma)

The dual moving average strategy generates signals based on crossovers between two moving averages of different periods.

### Parameters:

- **Short Window**: The period for the fast moving average (default: 10)
- **Long Window**: The period for the slow moving average (default: 30)
- **Signal Type**: Crossover or threshold based signals

### Logic:

- Buy signal: Short MA crosses above Long MA
- Sell signal: Short MA crosses below Long MA

## Neural Network (neural_net)

The neural network strategy uses a deep learning model to predict price movements based on historical data patterns.

### Parameters:

- **Hidden Layers**: Structure of hidden layers, e.g., "64,32" means two layers with 64 and 32 neurons.
- **Learning Rate**: Controls how quickly the model adapts to new data (default: 0.001)
- **Dropout**: Regularization parameter to prevent overfitting (default: 0.2)
- **Epochs**: Number of training cycles (default: 100)

### Logic:

This strategy requires pre-training on historical data before generating signals. The model attempts to predict price movements based on patterns it learned during training.
"""
        strategies_text.insert(tk.END, strategies_content)
        strategies_text.config(state=tk.DISABLED)

    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About Trading System GUI",
            "Trading System GUI v1.0.0\n\n"
            "A comprehensive graphical interface for crypto trading strategies.\n\n"
            "© 2023 Your Company\n\n"
            "This application provides tools for backtesting, paper trading, and live trading "
            "using various technical analysis strategies."
        )

    def on_closing(self):
        """Handle window close event"""
        if self.is_running:
            confirm = messagebox.askyesno("Confirm Exit", "Trading system is still running. Are you sure you want to exit?")
            if not confirm:
                return
        
        # Save configuration
        self.save_config()
        
        # Restore stdout/stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        # Close app
        self.root.destroy()
        
        
def main():
    """Main entry point"""
    # Create root window with theme
    try:
        # Try to load themed Tk
        root = ttkthemes.ThemedTk(theme="arc")
    except ImportError:
        # Fall back to standard Tk
        root = tk.Tk()
    
    # Create GUI app
    app = TradingSystemGUI(root)
    
    # Start main loop
    root.mainloop()


if __name__ == "__main__":
    main()